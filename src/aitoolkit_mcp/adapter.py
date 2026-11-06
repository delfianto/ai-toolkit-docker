"""Typed tools over the AI Toolkit UI API; the UI owns jobs and GPU queues."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, cast

import httpx2 as httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations


def preflight_model(model: dict) -> None:
    if not str(model.get("name_or_path", "")).startswith("catalog:"):
        return
    from aitoolkit_models.catalog import CatalogError, configured

    try:
        catalog = configured()
        if catalog is None:
            raise CatalogError("Set AI_TOOLKIT_MODEL_CATALOG to use catalog bundles")
        catalog.resolve(model["name_or_path"][len("catalog:") :], model)
    except (CatalogError, OSError, ValueError) as error:
        raise ToolError(f"Model preflight failed: {error}") from error


JsonObject = dict[str, Any]
ACTIVE = {"running", "queued", "stopping"}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def object_data(value: object) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError("Toolkit returned an unexpected response; expected an object")
    return cast(JsonObject, value)


def name_value(value: str) -> str:
    if not NAME.fullmatch(value):
        raise ToolError(
            "Use a name starting with a letter/digit, then letters, digits, _, . or -"
        )
    return value


def job_id_value(value: str) -> str:
    # IDs are opaque API identifiers, never URL paths.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ToolError("Invalid job ID")
    return value


def gpu_value(value: str) -> str:
    if not re.fullmatch(r"\d+(?:,\d+)*", value):
        raise ToolError("gpu_ids must be comma-separated GPU indices, for example '0'")
    return value


def child_path(root: str, relative: str) -> Path:
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if Path(relative).is_absolute() or path == base or not path.is_relative_to(base):
        raise ToolError("Path must stay inside its dataset/output directory")
    return path


def merge_config(original: JsonObject, patch: JsonObject) -> JsonObject:
    """Objects merge recursively; arrays replace as a whole. Null is a value."""
    result = copy.deepcopy(original)
    for key, value in patch.items():
        old = result.get(key)
        result[key] = (
            merge_config(object_data(old), object_data(value))
            if isinstance(old, dict) and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def decode_job(value: object, include_config: bool = True) -> JsonObject:
    if value is None:
        raise ToolError("Job not found")
    job = object_data(value).copy()
    raw = job.pop("job_config", None)
    if include_config:
        job["job_config"] = (
            object_data(json.loads(raw)) if isinstance(raw, str) else raw
        )
    return job


class ToolkitAPI:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http

    async def request(
        self, path: str, body: JsonObject | None = None, *, text: bool = False
    ) -> Any:
        try:
            response = await self.http.request(
                "GET" if body is None else "POST", path, json=body
            )
        except httpx.HTTPError as error:
            raise ToolError("Toolkit UI is unreachable or timed out") from error
        # Never include request headers or the auth token in tool errors.
        if response.is_error:
            raise ToolError(
                f"Toolkit API {path.split('?')[0]} returned HTTP {response.status_code}"
            )
        if text:
            return response.text
        data: object = response.json()
        if isinstance(data, dict) and data.get("error"):
            raise ToolError(str(data["error"]))
        return data

    async def job(self, job_id: str) -> JsonObject:
        return decode_job(await self.request("jobs?id=" + job_id_value(job_id)))

    async def paths(self) -> JsonObject:
        settings = object_data(await self.request("settings"))
        # The settings endpoint can contain HF_TOKEN. Never expose the raw response.
        return {
            key: settings.get(key)
            for key in ("DATASETS_FOLDER", "TRAINING_FOLDER", "MODELS_PATH")
        }

    async def image_path(self, dataset: str, image: str) -> Path:
        paths = await self.paths()
        folder = child_path(str(paths["DATASETS_FOLDER"]), name_value(dataset))
        path = child_path(str(folder), image)
        if not path.is_file() or path.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            raise ToolError("Image does not exist or is not a supported image file")
        return path

    async def save_job(
        self, name: str, config: JsonObject, gpu_ids: str, job_id: str | None = None
    ) -> JsonObject:
        name_value(name)
        gpu_value(gpu_ids)
        if job_id:
            old = await self.job(job_id)
            if old["status"] in ACTIVE:
                raise ToolError("Stop the job before editing its configuration")
            if old["name"] != name:
                raise ToolError(
                    "Use clone_job to change a job name and keep its output history separate"
                )
        config = copy.deepcopy(config)
        if config.get("job") != "extension":
            raise ToolError("job_config.job must be 'extension'")
        inner = object_data(config.get("config"))
        processes = inner.get("process")
        if not isinstance(processes, list) or len(processes) != 1:
            raise ToolError("Expected exactly one diffusion_trainer process")
        process = object_data(processes[0])
        if process.get("type") != "diffusion_trainer":
            raise ToolError("This adapter creates diffusion_trainer jobs")
        train = object_data(process.get("train"))
        if not isinstance(train.get("steps"), int) or train["steps"] < 1:
            raise ToolError("train.steps must be a positive integer")
        if train.get("unload_text_encoder") and train.get("cache_text_embeddings"):
            raise ToolError(
                "Choose cache_text_embeddings or unload_text_encoder, not both"
            )
        datasets = process.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ToolError("At least one dataset is required")
        paths = await self.paths()
        root = Path(str(paths["DATASETS_FOLDER"])).resolve()
        for dataset in datasets:
            folder = Path(str(object_data(dataset).get("folder_path", "")))
            if (
                not folder.is_absolute()
                or not folder.resolve().is_relative_to(root)
                or not folder.is_dir()
            ):
                raise ToolError(
                    "Dataset must be an existing absolute container path under DATASETS_FOLDER"
                )
        preflight_model(object_data(process.get("model")))
        inner["name"] = name
        process["training_folder"] = paths["TRAINING_FOLDER"]
        process["sqlite_db_path"] = "/app/ai-toolkit/aitk_db.db"
        body: JsonObject = {
            "name": name,
            "gpu_ids": gpu_ids,
            "job_config": config,
            "job_type": "train",
        }
        if job_id:
            body["id"] = job_id_value(job_id)
        return decode_job(await self.request("jobs", body))


def create_server(api: ToolkitAPI) -> MCPServer:
    server = MCPServer(
        "AI Toolkit",
        version="0.1.0",
        instructions="Manage the same jobs and queues as the Toolkit UI. Use get_job before editing and clone_job for a new run. Config arrays replace as a whole. queue_job queues only; start_queue can start every pending job for that GPU selection. Container paths are required. cache_text_embeddings preserves captions; unload_text_encoder uses only the trigger/blank prompt. No tools delete datasets or checkpoints.",
    )
    read = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, open_world_hint=False
    )
    write = ToolAnnotations(
        read_only_hint=False, destructive_hint=False, open_world_hint=False
    )

    @server.tool(annotations=read)
    async def list_jobs(status: str | None = None) -> JsonObject:
        """List UI jobs with progress/status, excluding large configs. Use get_job for config."""
        data = object_data(await api.request("jobs"))
        jobs = [decode_job(job, False) for job in data["jobs"]]
        return {
            "jobs": [job for job in jobs if status is None or job["status"] == status]
        }

    @server.tool(annotations=read)
    async def get_job(job_id: str) -> JsonObject:
        """Read a job, including its complete decoded training config."""
        return await api.job(job_id)

    @server.tool(annotations=write)
    async def create_job(
        name: str, job_config: JsonObject, gpu_ids: str = "0"
    ) -> JsonObject:
        """Create an unstarted UI training job from a full {job, config, meta} configuration."""
        return await api.save_job(name, job_config, gpu_ids)

    @server.tool(annotations=write)
    async def update_job(
        job_id: str,
        process_patch: JsonObject,
        gpu_ids: str | None = None,
    ) -> JsonObject:
        """Merge model/train/sample/datasets fields into a stopped job's process. Lists replace entirely."""
        job = await api.job(job_id)
        config = copy.deepcopy(object_data(job["job_config"]))
        config["config"]["process"][0] = merge_config(
            object_data(config["config"]["process"][0]), process_patch
        )
        return await api.save_job(
            str(job["name"]), config, gpu_ids or str(job["gpu_ids"]), job_id
        )

    @server.tool(annotations=write)
    async def clone_job(
        job_id: str,
        name: str,
        process_patch: JsonObject | None = None,
        gpu_ids: str | None = None,
    ) -> JsonObject:
        """Create a new unstarted run, optionally patching its model/train/sample/datasets fields."""
        job = await api.job(job_id)
        config = copy.deepcopy(object_data(job["job_config"]))
        config["config"]["process"][0] = merge_config(
            object_data(config["config"]["process"][0]), process_patch or {}
        )
        return await api.save_job(name, config, gpu_ids or str(job["gpu_ids"]))

    @server.tool(annotations=write)
    async def queue_job(job_id: str) -> JsonObject:
        """Queue a stopped/unstarted job. It runs if its GPU queue is already enabled."""
        job = await api.job(job_id)
        if job["status"] in ACTIVE:
            raise ToolError("Job is already active")
        config = object_data(job["job_config"])
        for process in config["config"]["process"]:
            preflight_model(object_data(process["model"]))
        await api.request(f"jobs/{job_id_value(job_id)}/start")
        return await api.job(job_id)

    @server.tool(annotations=write)
    async def stop_job(job_id: str) -> JsonObject:
        """Stop a running job; cancel a queued job after pausing its queue. Inactive jobs are unchanged."""
        job = await api.job(job_id)
        if job["status"] not in ACTIVE:
            return job
        if job["status"] == "queued":
            queues = object_data(await api.request("queue"))["queues"]
            if any(
                queue["gpu_ids"] == job["gpu_ids"] and queue["is_running"]
                for queue in queues
            ):
                raise ToolError("Pause this GPU queue before cancelling a queued job")
            job = await api.job(job_id)
        # A queued job can retain the PID of an old run. Never signal that PID.
        action = "mark_stopped" if job["status"] == "queued" else "stop"
        if job["status"] not in ACTIVE:
            return job
        await api.request(f"jobs/{job_id_value(job_id)}/{action}")
        return await api.job(job_id)

    @server.tool(annotations=read)
    async def list_queues() -> JsonObject:
        """Read UI GPU queue states."""
        return object_data(await api.request("queue"))

    @server.tool(annotations=write)
    async def start_queue(gpu_ids: str = "0") -> JsonObject:
        """Enable this GPU queue; every pending job for this GPU selection may run."""
        await api.request(f"queue/{gpu_value(gpu_ids)}/start")
        return object_data(await api.request("queue"))

    @server.tool(annotations=write)
    async def pause_queue(gpu_ids: str = "0") -> JsonObject:
        """Pause dispatch of new jobs; this does not stop the currently running job."""
        await api.request(f"queue/{gpu_value(gpu_ids)}/stop")
        return object_data(await api.request("queue"))

    @server.tool(annotations=write)
    async def request_checkpoint(job_id: str) -> JsonObject:
        """Ask a running trainer to save at its next step."""
        job = await api.job(job_id)
        if job["status"] != "running":
            raise ToolError("Job is not running")
        await api.request(f"jobs/{job_id_value(job_id)}/save_now")
        return {"requested": True, "job_id": job_id}

    @server.tool(annotations=write)
    async def request_samples(job_id: str) -> JsonObject:
        """Ask a running trainer to sample at its next step, if sampling is enabled."""
        job = await api.job(job_id)
        process = job["job_config"]["config"]["process"][0]
        if job["status"] != "running" or process["train"].get("disable_sampling"):
            raise ToolError("Requires a running job with sampling enabled")
        await api.request(f"jobs/{job_id_value(job_id)}/sample_now")
        return {"requested": True, "job_id": job_id}

    @server.tool(annotations=read)
    async def get_job_logs(
        job_id: str, lines: int = 100, offset: int | None = None
    ) -> JsonObject:
        """Tail logs, with carriage-return redraws collapsed. Reuse offset for incremental reads."""
        if not 1 <= lines <= 1000 or (offset is not None and offset < 0):
            raise ToolError("lines must be 1..1000; offset must be nonnegative")
        await api.job(job_id)
        suffix = "" if offset is None else f"?offset={offset}"
        data = object_data(
            await api.request(f"jobs/{job_id_value(job_id)}/log{suffix}")
        )
        data["log"] = "\n".join(
            str(data["log"]).replace("\r", "\n").splitlines()[-lines:]
        )[-100000:]
        return data

    @server.tool(annotations=read)
    async def get_job_loss(
        job_id: str, limit: int = 100, since_step: int = 0
    ) -> JsonObject:
        """Read recorded loss points from the UI metric API."""
        if not 1 <= limit <= 2000 or since_step < 0:
            raise ToolError("limit must be 1..2000; since_step must be nonnegative")
        await api.job(job_id)
        return object_data(
            await api.request(
                f"jobs/{job_id_value(job_id)}/loss?limit={limit}&since_step={since_step}"
            )
        )

    @server.tool(annotations=read)
    async def get_job_files(job_id: str) -> JsonObject:
        """List saved checkpoints and optimizer state, with container paths and sizes."""
        await api.job(job_id)
        return object_data(await api.request(f"jobs/{job_id_value(job_id)}/files"))

    @server.tool(annotations=read)
    async def get_job_samples(job_id: str, limit: int = 4) -> JsonObject:
        """List the most recent sample paths. Use read_sample_image to view an image."""
        if not 1 <= limit <= 16:
            raise ToolError("limit must be 1..16")
        await api.job(job_id)
        data = object_data(await api.request(f"jobs/{job_id_value(job_id)}/samples"))
        data["samples"] = data["samples"][-limit:]
        return data

    @server.tool(annotations=read)
    async def read_sample_image(job_id: str, filename: str) -> Image:
        """Return an image from this job's samples as MCP image content (maximum 8 MiB)."""
        job = await api.job(job_id)
        paths = await api.paths()
        root = (
            child_path(str(paths["TRAINING_FOLDER"]), name_value(str(job["name"])))
            / "samples"
        )
        path = child_path(str(root), filename)
        if (
            path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}
            or not path.is_file()
        ):
            raise ToolError("Sample image not found")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ToolError("Sample exceeds the 8 MiB limit")
        return Image(path=path)

    @server.tool(annotations=read)
    async def list_datasets() -> JsonObject:
        """List dataset names recognized by the UI."""
        return {"datasets": await api.request("datasets/list")}

    @server.tool(annotations=read)
    async def get_dataset_info(
        dataset: str, limit: int = 100, offset: int = 0
    ) -> JsonObject:
        """Count images and nonempty .txt captions; return a paginated list of image names."""
        if not 1 <= limit <= 500 or offset < 0:
            raise ToolError("limit must be 1..500; offset must be nonnegative")
        data = object_data(
            await api.request(
                "datasets/listImages", {"datasetName": name_value(dataset)}
            )
        )
        images = cast(list[str], data["images"])
        missing: list[str] = []
        for image in images:
            path = child_path(str(data["root"]), image).with_suffix(".txt")
            if not path.is_file() or not path.read_text().strip():
                missing.append(image)
        return {
            "dataset": dataset,
            "root": data["root"],
            "image_count": len(images),
            "caption_count": len(images) - len(missing),
            "missing_captions": missing[:limit],
            "images": images[offset : offset + limit],
            "offset": offset,
            "next_offset": offset + limit if offset + limit < len(images) else None,
        }

    @server.tool(annotations=read)
    async def get_caption(dataset: str, image: str) -> JsonObject:
        """Read the .txt caption associated with a dataset image."""
        path = await api.image_path(dataset, image)
        caption = await api.request(
            "caption/get", {"imgPath": str(path), "ext": "txt"}, text=True
        )
        return {"image": image, "caption": caption}

    @server.tool(
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, open_world_hint=False
        )
    )
    async def set_caption(dataset: str, image: str, caption: str) -> JsonObject:
        """Replace one image's .txt caption through the UI API. Existing caption is overwritten."""
        if not caption.strip() or len(caption) > 32000:
            raise ToolError("Caption must be nonempty and at most 32000 characters")
        path = await api.image_path(dataset, image)
        return object_data(
            await api.request(
                "img/caption", {"imgPath": str(path), "caption": caption, "ext": "txt"}
            )
        )

    @server.tool(annotations=read)
    async def get_gpu_stats() -> JsonObject:
        """Read GPU VRAM, utilization and temperature from Toolkit."""
        return object_data(await api.request("gpu"))

    @server.tool(annotations=read)
    async def get_paths() -> JsonObject:
        """Read only dataset/output/model paths. Credentials are never returned."""
        return await api.paths()

    @server.tool(annotations=read)
    async def list_model_architectures() -> JsonObject:
        """List model architectures supported by the installed Toolkit UI."""
        return {"architectures": await api.request("model_archs")}

    @server.tool(annotations=read)
    async def list_model_bundles() -> JsonObject:
        """List registered local bundles. Use catalog:<id> as model.name_or_path."""
        from aitoolkit_models.catalog import configured

        catalog = configured()
        return (
            catalog.summary() if catalog is not None else {"version": 1, "bundles": []}
        )

    @server.tool(annotations=read)
    async def validate_model_bundle(bundle_id: str) -> JsonObject:
        """Preflight local files and metadata without loading model weights or allocating VRAM."""
        from aitoolkit_models.catalog import CatalogError, configured

        try:
            catalog = configured()
            if catalog is None:
                raise CatalogError("No model catalog is configured")
            return {"bundle": bundle_id, "model": catalog.resolve(bundle_id)}
        except (CatalogError, OSError, ValueError) as error:
            raise ToolError(f"Model preflight failed: {error}") from error

    @server.tool(annotations=read)
    async def list_loras() -> JsonObject:
        """List training outputs and shared ComfyUI LoRAs recognized by the UI."""
        return object_data(await api.request("loras"))

    return server
