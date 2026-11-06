"""MCP protocol tests against a fake Toolkit API; no GPU or training required."""

from __future__ import annotations

import base64
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx2 as httpx
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError
from starlette.testclient import TestClient

from aitoolkit_mcp.adapter import (
    JsonObject,
    ToolkitAPI,
    child_path,
    create_server,
    merge_config,
)
from aitoolkit_mcp.server import create_app


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.dataset = self.root / "datasets" / "subject"
        self.dataset.mkdir(parents=True)
        (self.dataset / "image.png").write_bytes(b"image")
        (self.dataset / "image.txt").write_text("A portrait")
        self.config: JsonObject = {
            "job": "extension",
            "config": {
                "name": "original",
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "train": {"steps": 50, "cache_text_embeddings": True},
                        "model": {
                            "arch": "krea2",
                            "name_or_path": "/models/local.safetensors",
                        },
                        "datasets": [{"folder_path": str(self.dataset)}],
                    }
                ],
            },
        }
        self.jobs: dict[str, JsonObject] = {
            "job-1": {
                "id": "job-1",
                "name": "original",
                "gpu_ids": "0",
                "status": "completed",
                "job_config": json.dumps(self.config),
            }
        }
        self.calls: list[tuple[str, str]] = []
        self.queue_running = False
        self.http = httpx.AsyncClient(
            base_url="http://toolkit/api/", transport=httpx.MockTransport(self.handler)
        )
        self.api = ToolkitAPI(self.http)

    async def asyncTearDown(self) -> None:
        await self.http.aclose()
        self.directory.cleanup()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/")
        self.calls.append((request.method, path))
        if path == "settings":
            return httpx.Response(
                200,
                json={
                    "DATASETS_FOLDER": str(self.root / "datasets"),
                    "TRAINING_FOLDER": str(self.root / "output"),
                    "MODELS_PATH": "/models",
                    "HF_TOKEN": "must-not-leak",
                },
            )
        if path == "jobs":
            if request.method == "POST":
                body = cast(JsonObject, json.loads(request.content))
                job_id = str(body.get("id", "job-2"))
                body["id"] = job_id
                body["job_config"] = json.dumps(body["job_config"])
                body["status"] = "created"
                self.jobs[job_id] = body
                return httpx.Response(200, json=body)
            job_id = request.url.params.get("id")
            return httpx.Response(
                200,
                json=self.jobs.get(job_id)
                if job_id
                else {"jobs": list(self.jobs.values())},
            )
        if path == "datasets/list":
            return httpx.Response(200, json=["subject"])
        if path == "datasets/listImages":
            return httpx.Response(
                200, json={"root": str(self.dataset), "images": ["image.png"]}
            )
        if path == "caption/get":
            return httpx.Response(200, text=(self.dataset / "image.txt").read_text())
        if path == "img/caption":
            body = cast(JsonObject, json.loads(request.content))
            (self.dataset / "image.txt").write_text(str(body["caption"]))
            return httpx.Response(200, json={"success": True})
        if path.startswith("queue/"):
            self.queue_running = path.endswith("/start")
            return httpx.Response(
                200, json={"is_running": False}
            )  # upstream returns stale state
        if path == "queue":
            return httpx.Response(
                200,
                json={"queues": [{"gpu_ids": "0", "is_running": self.queue_running}]},
            )
        if path == "jobs/job-1/log":
            return httpx.Response(
                200, json={"log": "old\rnew\rfinal\n", "offset": 14, "reset": True}
            )
        if path.startswith("jobs/job-1/"):
            action = path.rsplit("/", 1)[1]
            if action == "start":
                self.jobs["job-1"]["status"] = "queued"
            elif action in {"stop", "mark_stopped"}:
                self.jobs["job-1"]["status"] = "stopped"
            return httpx.Response(200, json=self.jobs["job-1"])
        return httpx.Response(404)

    async def test_discovery_and_credentials_filtered(self) -> None:
        async with Client(create_server(self.api)) as client:
            tools = await client.list_tools()
            self.assertEqual(len(tools.tools), 27)
            result = await client.call_tool("get_paths", {})
            self.assertNotIn("must-not-leak", str(result))
            self.assertNotIn("HF_TOKEN", str(result))
            result = await client.call_tool("list_jobs", {})
            self.assertNotIn("job_config", str(result.structured_content))
            result = await client.call_tool("get_job", {"job_id": "job-1"})
            self.assertEqual(
                cast(JsonObject, result.structured_content)["job_config"], self.config
            )

    async def test_catalog_tools_and_queue_preflight_reject_missing_asset(self) -> None:
        weight = self.root / "arbitrary-name.safetensors"
        weight.write_bytes(b"weights")
        catalog = self.root / "models.json"
        catalog.write_text(
            json.dumps(
                {
                    "version": 1,
                    "assets": {
                        "checkpoint": {"path": str(weight)},
                        "extras": {"path": str(self.root), "kind": "directory"},
                    },
                    "bundles": {
                        "local": {
                            "profile": "sdxl",
                            "components": {
                                "checkpoint": "@checkpoint",
                                "extras": "@extras",
                            },
                            "model": {"name_or_path": "@checkpoint"},
                        }
                    },
                }
            )
        )
        self.config["config"]["process"][0]["model"] = {
            "arch": "sdxl",
            "name_or_path": "catalog:local",
        }
        self.jobs["job-1"]["job_config"] = json.dumps(self.config)
        with patch.dict(os.environ, {"AI_TOOLKIT_MODEL_CATALOG": str(catalog)}):
            async with Client(create_server(self.api)) as client:
                result = await client.call_tool("list_model_bundles", {})
                self.assertEqual(
                    cast(JsonObject, result.structured_content)["bundles"][0]["id"],
                    "local",
                )
                result = await client.call_tool(
                    "validate_model_bundle", {"bundle_id": "local"}
                )
                self.assertFalse(result.is_error)
                weight.unlink()
                result = await client.call_tool("queue_job", {"job_id": "job-1"})
                self.assertTrue(result.is_error)
                self.assertNotIn(("GET", "jobs/job-1/start"), self.calls)

    async def test_clone_update_and_active_edit_guard(self) -> None:
        async with Client(create_server(self.api)) as client:
            result = await client.call_tool(
                "clone_job",
                {
                    "job_id": "job-1",
                    "name": "new_run",
                    "process_patch": {"train": {"steps": 100}},
                },
            )
            self.assertFalse(result.is_error)
            new = cast(JsonObject, result.structured_content)
            self.assertEqual(new["job_config"]["config"]["name"], "new_run")
            self.assertEqual(
                new["job_config"]["config"]["process"][0]["train"]["steps"], 100
            )
            self.assertTrue(
                new["job_config"]["config"]["process"][0]["train"][
                    "cache_text_embeddings"
                ]
            )
            self.assertEqual(
                new["job_config"]["config"]["process"][0]["model"]["arch"], "krea2"
            )
            self.assertEqual(json.loads(self.jobs["job-1"]["job_config"]), self.config)
            self.jobs["job-1"]["status"] = "running"
            result = await client.call_tool(
                "update_job", {"job_id": "job-1", "process_patch": {}}
            )
            self.assertTrue(result.is_error)

    async def test_caption_roundtrip_and_escape_rejected(self) -> None:
        async with Client(create_server(self.api)) as client:
            result = await client.call_tool("get_dataset_info", {"dataset": "subject"})
            self.assertEqual(
                cast(JsonObject, result.structured_content)["caption_count"], 1
            )
            result = await client.call_tool(
                "set_caption",
                {"dataset": "subject", "image": "image.png", "caption": "New caption"},
            )
            self.assertFalse(result.is_error)
            result = await client.call_tool(
                "get_caption", {"dataset": "subject", "image": "image.png"}
            )
            self.assertEqual(
                cast(JsonObject, result.structured_content)["caption"], "New caption"
            )
            count = len(self.calls)
            result = await client.call_tool(
                "set_caption",
                {"dataset": "subject", "image": "../../secret.png", "caption": "No"},
            )
            self.assertTrue(result.is_error)
            self.assertNotIn(("POST", "img/caption"), self.calls[count:])

    async def test_queue_gpu_identifier_and_fresh_state(self) -> None:
        async with Client(create_server(self.api)) as client:
            result = await client.call_tool("start_queue", {"gpu_ids": "0"})
            self.assertFalse(result.is_error)
            self.assertIn(("GET", "queue/0/start"), self.calls)
            self.assertTrue(
                cast(JsonObject, result.structured_content)["queues"][0]["is_running"]
            )
            result = await client.call_tool("queue_job", {"job_id": "job-1"})
            self.assertEqual(
                cast(JsonObject, result.structured_content)["status"], "queued"
            )
            result = await client.call_tool("queue_job", {"job_id": "job-1"})
            self.assertTrue(result.is_error)
            result = await client.call_tool("stop_job", {"job_id": "job-1"})
            self.assertTrue(result.is_error)
            await client.call_tool("pause_queue", {"gpu_ids": "0"})
            result = await client.call_tool("stop_job", {"job_id": "job-1"})
            self.assertEqual(
                cast(JsonObject, result.structured_content)["status"], "stopped"
            )
            self.assertIn(("GET", "jobs/job-1/mark_stopped"), self.calls)
            self.assertNotIn(("GET", "jobs/job-1/stop"), self.calls)
            count = len(self.calls)
            await client.call_tool("stop_job", {"job_id": "job-1"})
            self.assertNotIn(("GET", "jobs/job-1/stop"), self.calls[count:])

    async def test_config_and_identifier_validation(self) -> None:
        async with Client(create_server(self.api)) as client:
            result = await client.call_tool("get_job", {"job_id": "../settings"})
            self.assertTrue(result.is_error)
            config = copy.deepcopy(self.config)
            config["config"]["process"][0]["train"]["unload_text_encoder"] = True
            result = await client.call_tool(
                "create_job", {"name": "bad", "job_config": config}
            )
            self.assertTrue(result.is_error)
            self.assertIn("Choose cache_text_embeddings", str(result.content))
            self.assertNotIn(("POST", "jobs"), self.calls)
            result = await client.call_tool(
                "get_job_logs", {"job_id": "job-1", "lines": 1}
            )
            self.assertEqual(
                cast(JsonObject, result.structured_content)["log"], "final"
            )

    async def test_sample_image_content_and_containment(self) -> None:
        folder = self.root / "output" / "original" / "samples"
        folder.mkdir(parents=True)
        pixel = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aT6kAAAAASUVORK5CYII="
        )
        (folder / "pixel.png").write_bytes(pixel)
        async with Client(create_server(self.api)) as client:
            result = await client.call_tool(
                "read_sample_image", {"job_id": "job-1", "filename": "pixel.png"}
            )
            self.assertFalse(result.is_error)
            self.assertEqual(result.content[0].type, "image")
            result = await client.call_tool(
                "read_sample_image", {"job_id": "job-1", "filename": "../../secret.png"}
            )
            self.assertTrue(result.is_error)

    def test_merge_and_symlink_containment(self) -> None:
        original: JsonObject = {
            "model": {"arch": "krea2", "quantize": True},
            "datasets": [1, 2],
        }
        merged = merge_config(original, {"model": {"quantize": False}, "datasets": [3]})
        self.assertEqual(
            merged, {"model": {"arch": "krea2", "quantize": False}, "datasets": [3]}
        )
        self.assertEqual(original["datasets"], [1, 2])
        (self.dataset / "escape").symlink_to(self.root)
        with self.assertRaises(ToolError):
            child_path(str(self.dataset), "escape/file")


class HTTPTests(unittest.TestCase):
    def test_authentication_and_wire_handshake(self) -> None:
        http = httpx.AsyncClient(
            base_url="http://toolkit/api/",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"jobs": []})
            ),
        )
        app = create_app(ToolkitAPI(http), "test-token", ["testserver", "localhost:*"])
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
            headers = {"Accept": "application/json, text/event-stream"}
            self.assertEqual(
                client.post("/mcp", json=message, headers=headers).status_code, 401
            )
            headers["Authorization"] = "Bearer test-token"
            response = client.post("/mcp", json=message, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["result"]["serverInfo"]["name"], "AI Toolkit"
            )
            headers["Host"] = "untrusted.example"
            self.assertEqual(
                client.post("/mcp", json=message, headers=headers).status_code, 421
            )


if __name__ == "__main__":
    unittest.main()
