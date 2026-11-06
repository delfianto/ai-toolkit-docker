"""Offline checks for caption preservation, input discovery and the API contract."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from unittest.mock import patch

from PIL import Image
from typing_extensions import override

from . import pipeline


def fixture_config(root: Path) -> pipeline.PipelineConfig:
    return {
        "server": {
            "runner": Path("/unused"),
            "profile": root / "model.yaml",
            "endpoint": "http://localhost",
            "startup_timeout": 120,
            "request_timeout": 180,
        },
        "datasets": {"root": root, "names": ["images"], "recursive": False},
        "caption": {
            "system": "Write a caption.",
            "prompt": "Describe.",
            "prefix": "",
            "max_side": 512,
            "max_tokens": 128,
            "temperature": 0.2,
        },
        "run": {"state_dir": root / "state", "retries": 2},
    }


class PipelineTests(unittest.TestCase):
    def test_preserve_existing_and_atomic_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "photo.txt"
            _ = target.write_text("original\n")
            self.assertFalse(pipeline.save_caption(target, "replacement"))
            self.assertEqual(target.read_text(), "original\n")
            self.assertTrue(
                pipeline.save_caption(target, "replacement", overwrite=True)
            )
            self.assertEqual(target.read_text(), "replacement\n")
            self.assertEqual(list(Path(directory).glob(".caption-*")), [])

    def test_discovery_rejects_same_basename_and_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "images"
            dataset.mkdir()
            (dataset / "same.jpg").touch()
            (dataset / "same.png").touch()
            config = fixture_config(root)
            with self.assertRaisesRegex(ValueError, "basename"):
                _ = pipeline.discover(config)
            with self.assertRaisesRegex(ValueError, "under"):
                _ = pipeline.discover(config, ["../"])

    def test_reject_empty_truncated_or_reasoning_caption(self) -> None:
        config = fixture_config(Path.cwd())
        for content, reason in [
            ("", "stop"),
            ("partial", "length"),
            ("<think>analysis</think>", "stop"),
        ]:
            result = {
                "choices": [{"message": {"content": content}, "finish_reason": reason}]
            }
            with (
                patch.object(pipeline, "image_data_url", return_value="data:test"),
                patch.object(pipeline, "api_json", return_value=result),
                self.assertRaises(ValueError),
            ):
                _ = pipeline.generate_caption(Path("unused.jpg"), config)

    def test_end_to_end_resume_with_existing_api(self) -> None:
        requests: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def respond(self, body: dict[str, object]) -> None:
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                _ = self.wfile.write(data)

            def do_GET(self) -> None:
                self.respond({"data": [{"id": "gemma-test"}]})

            def do_POST(self) -> None:
                requests.append(
                    pipeline.mapping(
                        cast(
                            object,
                            json.loads(
                                self.rfile.read(int(self.headers["Content-Length"]))
                            ),
                        ),
                        "request",
                    )
                )
                self.respond(
                    {
                        "choices": [
                            {
                                "message": {"content": "A red square."},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                images = root / "photos"
                images.mkdir()
                Image.new("RGB", (1024, 512), "red").save(images / "new.jpg")
                Image.new("RGB", (64, 64), "blue").save(images / "existing.jpg")
                _ = (images / "existing.txt").write_text("Keep this caption.\n")
                config = root / "model.yaml"
                _ = (root / "instructions.txt").write_text(
                    "Describe the image accurately.\n"
                )
                _ = config.write_text(f"""server:
  host: 127.0.0.1
  port: {server.server_port}
prompt:
  system: Ignored inline fallback.
  system_file: instructions.txt
sampling:
  max_tokens: 128
""")
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("pipeline.py")),
                    str(images),
                    "--existing-server",
                    "--json",
                ]
                environment = {
                    **os.environ,
                    "CAPTION_PROFILE": str(config),
                    "CAPTION_RUNNER": "/unused",
                    "XDG_STATE_HOME": str(root / "state"),
                }
                first = subprocess.run(
                    command, capture_output=True, text=True, check=True, env=environment
                )
                self.assertEqual((images / "new.txt").read_text(), "A red square.\n")
                self.assertEqual(
                    (images / "existing.txt").read_text(), "Keep this caption.\n"
                )
                self.assertEqual(
                    pipeline.mapping(
                        cast(object, json.loads(first.stdout.splitlines()[-1])),
                        "summary",
                    )["written"],
                    1,
                )
                messages = [
                    pipeline.mapping(item, "message")
                    for item in cast(list[object], requests[0]["messages"])
                ]
                content = [
                    pipeline.mapping(item, "content")
                    for item in cast(list[object], messages[1]["content"])
                ]
                image_url = pipeline.mapping(content[0]["image_url"], "image_url")
                self.assertTrue(
                    pipeline.text_value(image_url["url"], "url").startswith(
                        "data:image/jpeg;base64,"
                    )
                )
                self.assertEqual(
                    messages[0],
                    {"role": "system", "content": "Describe the image accurately."},
                )
                self.assertFalse(
                    pipeline.mapping(requests[0]["chat_template_kwargs"], "template")[
                        "enable_thinking"
                    ]
                )
                second = subprocess.run(
                    command, capture_output=True, text=True, check=True, env=environment
                )
                self.assertEqual(
                    pipeline.mapping(
                        cast(object, json.loads(second.stdout.splitlines()[-1])),
                        "summary",
                    )["written"],
                    0,
                )
                self.assertEqual(len(requests), 1)
                self.assertEqual(
                    len(list((root / "state" / "captioning").glob("*.jsonl"))), 1
                )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    _ = unittest.main()
