"""Exercise a running adapter using the official MCP HTTP client."""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path

import httpx2 as httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from .adapter import JsonObject, object_data


async def check(url: str, api_url: str, token: str, exercise_writes: bool) -> None:
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + token}, timeout=30, trust_env=False
    ) as http:
        # Check that even initialize is protected, without exposing the token.
        async with httpx.AsyncClient(trust_env=False) as anonymous:
            response = await anonymous.post(url, json={})
            assert response.status_code == 401, (
                "Unauthenticated MCP access was accepted"
            )
        async with (
            streamable_http_client(url, http_client=http) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP handshake passed; {len(tools.tools)} tools")

            async def call(
                name: str, arguments: JsonObject | None = None
            ) -> JsonObject:
                result = await session.call_tool(name, arguments or {})
                if not isinstance(result, CallToolResult) or result.is_error:
                    raise RuntimeError(f"MCP tool failed: {name}")
                print(f"PASS {name}")
                return object_data(result.structured_content)

            bundles = await call("list_model_bundles")
            for bundle in bundles["bundles"]:
                await call("validate_model_bundle", {"bundle_id": bundle["id"]})
            jobs = await call("list_jobs")
            for tool in (
                "get_paths",
                "list_queues",
                "get_gpu_stats",
                "list_model_architectures",
                "list_loras",
            ):
                await call(tool)
            datasets = await call("list_datasets")
            if datasets["datasets"]:
                dataset = str(datasets["datasets"][0])
                info = await call("get_dataset_info", {"dataset": dataset})
                if info["images"]:
                    await call(
                        "get_caption", {"dataset": dataset, "image": info["images"][0]}
                    )
            if not jobs["jobs"]:
                if exercise_writes:
                    raise RuntimeError(
                        "Write test needs an existing training job to clone"
                    )
                return
            original = object_data(jobs["jobs"][0])
            job_id = str(original["id"])
            for tool in (
                "get_job",
                "get_job_logs",
                "get_job_loss",
                "get_job_files",
                "get_job_samples",
            ):
                await call(tool, {"job_id": job_id})
            if exercise_writes:
                name = "mcp_smoke_" + uuid.uuid4().hex
                draft = await call(
                    "clone_job",
                    {
                        "job_id": job_id,
                        "name": name,
                        "process_patch": {
                            "train": {"steps": 1, "disable_sampling": True}
                        },
                    },
                )
                draft_id = str(draft["id"])
                try:
                    updated = await call(
                        "update_job",
                        {
                            "job_id": draft_id,
                            "process_patch": {"save": {"save_every": 1}},
                        },
                    )
                    assert (
                        updated["job_config"]["config"]["process"][0]["train"]["steps"]
                        == 1
                    )
                    assert (
                        updated["job_config"]["config"]["process"][0]["save"][
                            "save_every"
                        ]
                        == 1
                    )
                finally:
                    # Only remove the unique, unstarted draft created by this test.
                    current = await call("get_job", {"job_id": draft_id})
                    files = await call("get_job_files", {"job_id": draft_id})
                    assert (
                        current["name"] == name
                        and current["status"] in {"created", "stopped"}
                        and not files["files"]
                    )
                    response = await http.get(
                        api_url.rstrip("/") + f"/jobs/{draft_id}/delete"
                    )
                    response.raise_for_status()
                    print("PASS temporary draft removed; no training launched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8676/mcp")
    parser.add_argument("--api-url", default="http://127.0.0.1:8675/api")
    parser.add_argument("--token-file", default="/run/secrets/AI_TOOLKIT_AUTH")
    parser.add_argument(
        "--exercise-writes",
        action="store_true",
        help="Create, edit and delete a temporary draft; never starts training",
    )
    arguments = parser.parse_args()
    token = (
        os.environ.get("AI_TOOLKIT_AUTH")
        or Path(arguments.token_file).read_text().strip()
    )
    asyncio.run(
        check(arguments.url, arguments.api_url, token, arguments.exercise_writes)
    )


if __name__ == "__main__":
    main()
