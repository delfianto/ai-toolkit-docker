"""Resolve upstream inputs and skip scheduled builds already published to GHCR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/ostris/ai-toolkit.git"


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE).strip()


def revision(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"Expected a full Git commit SHA, got {value!r}")
    return value


def upstream_head() -> str:
    result = command("git", "ls-remote", UPSTREAM_REPOSITORY, "HEAD")
    lines = result.splitlines()
    if len(lines) != 1 or len(lines[0].split()) != 2:
        raise ValueError("Could not resolve exactly one upstream HEAD")
    return revision(lines[0].split()[0])


def base_digest(image: str) -> str:
    result = command("docker", "buildx", "imagetools", "inspect", image)
    match = re.search(r"^Digest:\s+(sha256:[0-9a-f]{64})\s*$", result, re.MULTILINE)
    if match is None:
        raise ValueError(f"No immutable digest in registry response for {image}")
    return match.group(1)


def published(image: str) -> bool:
    result = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    message = result.stderr.lower()
    # Missing build tags are expected; auth, rate limits and outages must fail CI.
    if not any(
        word in message
        for word in ("denied", "unauthorized", "forbidden", "timeout", "429")
    ) and (
        "manifest unknown" in message
        or message.strip() == f"error: {image.lower()}: not found"
    ):
        return False
    raise RuntimeError(f"Cannot check published build {image}: {result.stderr.strip()}")


def resolve(
    image: str,
    base_image: str,
    upstream_commit: str,
    wrapper_commit: str,
    event: str,
) -> dict[str, str]:
    upstream_commit = revision(upstream_commit)
    wrapper_commit = revision(wrapper_commit)
    digest = base_digest(base_image)
    base = base_image.split("@", 1)[0] + "@" + digest
    inputs = {"upstream": upstream_commit, "base": base, "wrapper": wrapper_commit}
    fingerprint = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()[:32]
    build_tag = "build-" + fingerprint
    exists = published(image + ":" + build_tag) if event == "schedule" else False
    return {
        "upstream_commit": upstream_commit,
        "base_image": base,
        "base_digest": digest,
        "build_tag": build_tag,
        "needs_build": "false" if exists else "true",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--base-image", default="ostris/aitoolkit:latest")
    parser.add_argument("--upstream-commit", default="")
    parser.add_argument(
        "--event", default=os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    )
    parser.add_argument("--output", default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()
    result = resolve(
        image=args.image,
        base_image=args.base_image,
        upstream_commit=args.upstream_commit or upstream_head(),
        wrapper_commit=command("git", "rev-parse", "HEAD"),
        event=args.event,
    )
    print(json.dumps(result, indent=2))
    if args.output:
        with Path(args.output).open("a") as handle:
            handle.writelines(f"{key}={value}\n" for key, value in result.items())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
