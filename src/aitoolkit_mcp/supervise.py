"""Tie the UI and MCP server lifetimes to the container, with signal forwarding."""

import os
import signal
import subprocess
import time
from types import FrameType


def main() -> int:
    children: list[subprocess.Popen[bytes]] = []
    stopping = False

    def stop(_signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = 0
    try:
        for command in (
            ["bash", "/start.sh"],
            ["/opt/toolkit-mcp/bin/python", "-m", "aitoolkit_mcp.server"],
        ):
            children.append(subprocess.Popen(command, start_new_session=True))
        while not stopping:
            if any(child.poll() is not None for child in children):
                print("Toolkit UI or MCP exited; stopping container", flush=True)
                result = 1
                break
            time.sleep(0.25)
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 25
        for child in children:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
