#!/usr/bin/env python3
"""Config-driven image captioning through llama.rs's vision API.

Requires Python 3.11+, Pillow and PyYAML. Run on the host, not inside toolkit.
JSONL events on stdout are suitable for a harness; run records and server
logs live in run.state_dir. Existing .txt captions are preserved by default.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import BinaryIO, TextIO, TypedDict, cast

import yaml
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ServerConfig(TypedDict):
    runner: Path
    profile: Path
    endpoint: str
    startup_timeout: float
    request_timeout: float


class DatasetConfig(TypedDict):
    root: Path
    names: list[str]
    recursive: bool


class CaptionConfig(TypedDict):
    system: str
    prompt: str
    prefix: str
    max_side: int
    max_tokens: int
    temperature: float


class RunConfig(TypedDict):
    state_dir: Path
    retries: int


class PipelineConfig(TypedDict):
    server: ServerConfig
    datasets: DatasetConfig
    caption: CaptionConfig
    run: RunConfig


@dataclass
class Arguments(argparse.Namespace):
    profile: Path | None = None
    datasets: list[str] = field(default_factory=list)
    dataset_root: Path | None = None
    runner: str = "llama"
    state_dir: Path = field(
        default_factory=lambda: Path.home() / ".local/state/captioning"
    )
    max_side: int = 512
    max_tokens: int | None = None
    temperature: float | None = None
    prompt: str = "Caption this image."
    prefix: str = ""
    recursive: bool = False
    startup_timeout: float = 120
    request_timeout: float = 180
    retries: int = 2
    dry_run: bool = False
    limit: int | None = None
    overwrite: bool = False
    existing_server: bool = False
    json: bool = False


def mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    raw = cast(dict[object, object], value)
    result: dict[str, object] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        result[key] = item
    return result


def text_value(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def int_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def float_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    return float(value)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(args: Arguments) -> PipelineConfig:
    """Read the runner's YAML; per-run inputs belong to CLI arguments."""
    if args.profile is None:
        raise ValueError("A model profile is required")
    profile_path = args.profile.expanduser().resolve()
    with profile_path.open(encoding="utf-8") as file:
        profile = mapping(cast(object, yaml.safe_load(file)), "Model profile")
    prompt = mapping(profile.get("prompt", {}), "prompt")
    server = mapping(profile.get("server", {}), "server")
    sampling = mapping(profile.get("sampling", {}), "sampling")
    if prompt.get("system_file"):
        system = (
            resolve_path(
                text_value(prompt["system_file"], "prompt.system_file"),
                profile_path.parent,
            )
            .read_text(encoding="utf-8")
            .strip()
        )
    else:
        system = text_value(prompt.get("system", ""), "prompt.system").strip()
    if not system:
        raise ValueError("Set prompt.system or prompt.system_file in the model YAML")
    host = text_value(server.get("host", "127.0.0.1"), "server.host")
    port = int_value(server.get("port", 8080), "server.port")
    if not host or not 1 <= port <= 65535:
        raise ValueError("Invalid server.host/server.port in the model YAML")
    if args.dataset_root:
        root = args.dataset_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Dataset root does not exist: {root}")
        names = args.datasets or sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    else:
        folders = [Path(p).expanduser().resolve() for p in args.datasets]
        if not folders:
            raise ValueError("Supply dataset folders or --dataset-root")
        root = Path(os.path.commonpath([str(p.parent) for p in folders]))
        names = [str(p.relative_to(root)) for p in folders]
    max_tokens = (
        args.max_tokens
        if args.max_tokens is not None
        else int_value(sampling.get("max_tokens", 192), "sampling.max_tokens")
    )
    if args.max_side < 1 or max_tokens < 1:
        raise ValueError("Image size and token limit must be positive")
    if min(args.startup_timeout, args.request_timeout) <= 0 or args.retries < 0:
        raise ValueError("Timeouts must be positive and retries must not be negative")
    return {
        "server": {
            "runner": Path(shutil.which(args.runner) or args.runner)
            .expanduser()
            .resolve(),
            "profile": profile_path,
            "endpoint": f"http://{host}:{port}",
            "startup_timeout": args.startup_timeout,
            "request_timeout": args.request_timeout,
        },
        "datasets": {"root": root, "names": names, "recursive": args.recursive},
        "caption": {
            "system": system,
            "prompt": args.prompt,
            "prefix": args.prefix,
            "max_side": args.max_side,
            "max_tokens": max_tokens,
            "temperature": args.temperature
            if args.temperature is not None
            else float_value(sampling.get("temperature", 0.2), "sampling.temperature"),
        },
        "run": {
            "state_dir": args.state_dir.expanduser().resolve(),
            "retries": args.retries,
        },
    }


def discover(config: PipelineConfig, names: list[str] | None = None) -> list[Path]:
    root = config["datasets"]["root"]
    result: list[Path] = []
    for name in names or config["datasets"]["names"]:
        folder = (root / name).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError(
                f"Dataset must be an existing directory under {root}: {name}"
            )
        files = (
            folder.rglob("*")
            if config["datasets"].get("recursive")
            else folder.iterdir()
        )
        result.extend(
            sorted(
                p for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        )
    result = sorted(set(result))
    targets: dict[Path, Path] = {}
    for image in result:
        target = image.with_suffix(".txt")
        if target in targets:
            raise ValueError(
                f"Images share a caption basename: {targets[target]} and {image}"
            )
        targets[target] = image
    return result


def image_data_url(path: Path, max_side: int) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def api_json(
    url: str, timeout: float, payload: dict[str, object] | None = None
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with cast(BinaryIO, urllib.request.urlopen(request, timeout=timeout)) as response:
        return mapping(cast(object, json.load(response)), "API response")


def generate_caption(path: Path, config: PipelineConfig) -> str:
    caption = config["caption"]
    payload: dict[str, object] = {
        "model": "gemma4-caption",
        "stream": False,
        "messages": [
            {"role": "system", "content": caption["system"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(path, caption["max_side"])},
                    },
                    {"type": "text", "text": caption["prompt"].strip()},
                ],
            },
        ],
        "max_tokens": caption["max_tokens"],
        "temperature": caption.get("temperature", 0.2),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = api_json(
        config["server"]["endpoint"] + "/v1/chat/completions",
        config["server"].get("request_timeout", 180),
        payload,
    )
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response contains no caption choices")
    choice = mapping(cast(list[object], choices)[0], "Caption choice")
    if choice.get("finish_reason") == "length":
        raise ValueError(
            "Caption was truncated; increase sampling.max_tokens or --max-tokens"
        )
    content = mapping(choice.get("message"), "Caption message").get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned an empty caption")
    if "<think>" in content or "</think>" in content:
        raise ValueError("Model returned reasoning markup instead of a plain caption")
    return caption.get("prefix", "") + " ".join(content.split())


def save_caption(target: Path, content: str, overwrite: bool = False) -> bool:
    """Write atomically; a concurrent creator wins when overwrite is false."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=".caption-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            _ = file.write(content + "\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.chmod(0o644)
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError:
                return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Server:
    config: PipelineConfig
    log_path: Path
    existing: bool
    process: subprocess.Popen[bytes] | None
    log: TextIO | None

    def __init__(self, config: PipelineConfig, log_path: Path, existing: bool) -> None:
        self.config, self.log_path, self.existing = config, log_path, existing
        self.process = None
        self.log = None

    def start(self) -> None:
        config = self.config["server"]
        if not self.existing:
            for key, path in (
                ("runner", config["runner"]),
                ("profile", config["profile"]),
            ):
                if not path.is_file():
                    raise ValueError(f"Missing server.{key}: {path}")
            url = urllib.parse.urlparse(config["endpoint"])
            if url.hostname not in ("127.0.0.1", "localhost"):
                raise ValueError(
                    "Managed server endpoint must use localhost; use --existing-server otherwise"
                )
            try:
                with socket.create_connection(
                    (url.hostname, url.port or 80), timeout=1
                ):
                    raise ValueError(
                        "Server port is occupied; stop it or explicitly use --existing-server"
                    )
            except (ConnectionRefusedError, TimeoutError):
                pass
            self.log = self.log_path.open("w")
            self.process = subprocess.Popen(
                [str(config["runner"]), "serve", "--config", str(config["profile"])],
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=self.log_path.parent,
            )
        deadline = time.monotonic() + config.get("startup_timeout", 120)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"llama runner exited ({self.process.returncode}); see {self.log_path}"
                )
            try:
                models = api_json(config["endpoint"] + "/v1/models", 2)
                if models.get("data"):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Server did not become ready; see {self.log_path}")

    def stop(self) -> None:
        if self.process is not None:
            # llama.rs listens for Ctrl+C; terminate its whole group on fallback.
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                _ = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                _ = self.process.wait()
            # Clean up a child left behind if the runner failed before shutdown.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self.log is not None:
            self.log.close()


def main() -> int:
    def terminate(_signum: int, _frame: FrameType | None) -> None:
        # Harnesses commonly stop jobs with SIGTERM; still clean up our server.
        raise KeyboardInterrupt

    _ = signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset folder paths, or names beneath --dataset-root",
    )
    _ = parser.add_argument(
        "--profile",
        type=Path,
        default=os.environ.get("CAPTION_PROFILE"),
        help="llama.rs model YAML with prompt.system",
    )
    _ = parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Use selected names, or all non-hidden child directories",
    )
    _ = parser.add_argument(
        "--runner",
        default=os.environ.get("CAPTION_RUNNER", "llama"),
        help="llama.rs executable",
    )
    default_state = (
        Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        / "captioning"
    )
    _ = parser.add_argument("--state-dir", type=Path, default=default_state)
    _ = parser.add_argument("--max-side", type=int, default=512)
    _ = parser.add_argument(
        "--max-tokens", type=int, help="Override sampling.max_tokens in the YAML"
    )
    _ = parser.add_argument(
        "--temperature", type=float, help="Override sampling.temperature in the YAML"
    )
    _ = parser.add_argument(
        "--prompt",
        default="Caption this image.",
        help="Per-image request; system instructions come from YAML",
    )
    _ = parser.add_argument(
        "--prefix", default="", help="Optional identifier prepended to saved captions"
    )
    _ = parser.add_argument("--recursive", action="store_true")
    _ = parser.add_argument("--startup-timeout", type=float, default=120)
    _ = parser.add_argument("--request-timeout", type=float, default=180)
    _ = parser.add_argument("--retries", type=int, default=2)
    _ = parser.add_argument(
        "--dry-run", action="store_true", help="List work without launching or writing"
    )
    _ = parser.add_argument("--limit", type=int, help="Limit newly generated captions")
    _ = parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing .txt captions"
    )
    _ = parser.add_argument(
        "--existing-server",
        action="store_true",
        help="Use API without managing its lifecycle",
    )
    _ = parser.add_argument(
        "--json", action="store_true", help="Emit JSONL progress for a harness"
    )
    profile_env = os.environ.get("CAPTION_PROFILE")
    args = Arguments(
        profile=Path(profile_env) if profile_env is not None else None,
        runner=os.environ.get("CAPTION_RUNNER", "llama"),
        state_dir=default_state,
    )
    _ = parser.parse_args(namespace=args)
    if args.profile is None:
        parser.error("--profile or CAPTION_PROFILE is required")
    if not args.datasets and not args.dataset_root:
        parser.error("Supply dataset folders or --dataset-root")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    journal: TextIO | None = None

    def emit(event: str, **values: object) -> None:
        row: dict[str, object] = {"event": event, "time": time.time(), **values}
        if args.json:
            print(json.dumps(row), flush=True)
        else:
            print(
                event
                + ": "
                + ", ".join(f"{key}={value}" for key, value in values.items()),
                flush=True,
            )
        if journal is not None:
            _ = journal.write(json.dumps(row) + "\n")
            journal.flush()

    try:
        config = load_config(args)
        images = discover(config)
        pending = [
            p for p in images if args.overwrite or not p.with_suffix(".txt").exists()
        ]
        skipped = len(images) - len(pending)
        if args.limit is not None:
            pending = pending[: args.limit]
        emit(
            "plan",
            images=len(images),
            pending=len(pending),
            skipped=skipped,
            profile=str(config["server"]["profile"]),
        )
        if args.dry_run:
            for image in pending:
                emit(
                    "pending", image=str(image), caption=str(image.with_suffix(".txt"))
                )
            return 0
        if not pending:
            emit("complete", written=0, failed=0, skipped=skipped)
            return 0
        state_dir = config["run"]["state_dir"]
        state_dir.mkdir(parents=True, exist_ok=True)
        with (state_dir / "pipeline.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another captioning pipeline is running") from None
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            with (state_dir / f"{run_id}.jsonl").open("w") as record:
                journal = record
                server = Server(
                    config, state_dir / f"{run_id}.server.log", args.existing_server
                )
                written = failed = 0
                try:
                    emit("starting", run_id=run_id, server_log=str(server.log_path))
                    server.start()
                    emit("ready", endpoint=config["server"]["endpoint"])
                    for index, image in enumerate(pending, 1):
                        target = image.with_suffix(".txt")
                        if target.exists() and not args.overwrite:
                            skipped += 1
                            emit("skipped", image=str(image))
                            continue
                        start = time.monotonic()
                        content: str | None = None
                        for attempt in range(config["run"]["retries"] + 1):
                            try:
                                content = generate_caption(image, config)
                                break
                            except (OSError, ValueError, KeyError, IndexError) as error:
                                if attempt >= config["run"]["retries"]:
                                    failed += 1
                                    emit("failed", image=str(image), error=str(error))
                                    content = None
                                    break
                                emit(
                                    "retry",
                                    image=str(image),
                                    attempt=attempt + 1,
                                    error=str(error),
                                )
                                time.sleep(float(1 << min(attempt, 3)))
                        if content is None:
                            continue
                        if save_caption(target, content, args.overwrite):
                            written += 1
                            emit(
                                "captioned",
                                image=str(image),
                                caption=str(target),
                                index=index,
                                total=len(pending),
                                seconds=round(time.monotonic() - start, 2),
                            )
                        else:
                            skipped += 1
                            emit("skipped", image=str(image))
                    emit(
                        "complete",
                        run_id=run_id,
                        written=written,
                        failed=failed,
                        skipped=skipped,
                    )
                    return 1 if failed else 0
                finally:
                    server.stop()
                    journal = None
    except KeyboardInterrupt:
        emit("interrupted", message="Completed captions are preserved; rerun to resume")
        return 130
    except BrokenPipeError:
        # A harness may stop reading stdout early (for example, piping to head).
        with open(os.devnull, "w") as sink:
            _ = os.dup2(sink.fileno(), sys.stdout.fileno())
        return 1
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        yaml.YAMLError,
    ) as error:
        emit("error", error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
