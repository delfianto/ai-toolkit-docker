# AI Toolkit MCP adapter

Uses the official Python MCP SDK and Toolkit's existing authenticated UI API.
The UI owns the SQLite database, training processes and GPU queues. MCP jobs
appear in the UI, and UI jobs are available through MCP.

## Connect a harness

Streamable HTTP endpoint: `https://ostris-mcp.${TRAEFIK_ACME_DOMAIN}/mcp`.
Send `Authorization: Bearer <Toolkit UI token>` on **every** request.
The token is the existing `aitoolkit_auth` secret, not the Hugging Face key.
Use your harness's secret/environment facility; do not commit a real token.

For a client accepting an `mcpServers` JSON configuration:

```json
{
  "mcpServers": {
    "ai-toolkit": {
      "type": "http",
      "url": "https://ostris-mcp.your-domain.example/mcp",
      "headers": {"Authorization": "Bearer <Toolkit UI token>"}
    }
  }
}
```

Inside the Docker proxy network, use `http://ai-toolkit:8676/mcp` with the same
authorization header. No host port is published. `/health` checks the UI
connection without returning job data; it does not require authentication.
Host and Origin checks remain enabled. Add additional endpoint hostnames to
`TOOLKIT_MCP_ALLOWED_HOSTS` when deploying with a different reverse proxy.

## Tools

| Area | Tools |
| --- | --- |
| Jobs | `list_jobs`, `get_job`, `create_job`, `clone_job`, `update_job`, `queue_job`, `stop_job` |
| Queues | `list_queues`, `start_queue`, `pause_queue` |
| Monitoring | `get_job_logs`, `get_job_loss`, `get_gpu_stats` |
| Outputs | `get_job_files`, `get_job_samples`, `read_sample_image`, `request_checkpoint`, `request_samples`, `list_loras` |
| Datasets | `list_datasets`, `get_dataset_info`, `get_caption`, `set_caption` |
| Configuration | `get_paths`, `list_model_architectures`, `list_model_bundles`, `validate_model_bundle` |

Start with `list_jobs` and `get_job`. Clone a known working job to preserve
local checkpoint, VAE/text encoder loading, quantization and offloading.
For example, a `clone_job` call:

```json
{
  "job_id": "existing-job-id",
  "name": "my_subject_full",
  "gpu_ids": "0",
  "process_patch": {
    "train": {"steps": 1500, "cache_text_embeddings": true, "unload_text_encoder": false},
    "save": {"save_every": 250}
  }
}
```

`process_patch` merges into the first training process, so `train`, `model`,
`sample`, and `save` can be adjusted independently. Arrays such as `datasets`
and `sample.samples` replace in full. `create_job` accepts the complete
`{job, config, meta}` configuration instead. These tools create or edit drafts;
they do not queue training automatically. Running/queued jobs cannot be edited.

Call `queue_job`, then `start_queue` with the matching `gpu_ids` if the queue
is paused. Enabling a queue can run **all** jobs queued for that GPU selection.
An already-enabled queue starts newly queued jobs automatically. `pause_queue`
pauses dispatch; `stop_job` stops a particular job. GPU IDs refer to the UI's
GPU selection, not the numeric ID of a Queue database row.
Pause the matching queue before cancelling a queued job. This avoids
signalling a stale PID from an old run. The UI API has no transactional
compare-and-set; avoid concurrent queue control from multiple clients.

All dataset paths use container paths from `get_paths`. Caption updates
overwrite one existing image's `.txt`; caption coverage counts nonempty files.
Text embedding caching retains the captions while allowing the encoder to
leave VRAM. `unload_text_encoder` uses only a blank/trigger prompt instead.

The adapter does not expose arbitrary shell commands, raw API requests,
settings credentials, or dataset/checkpoint deletion. It does not wrap the
host Gemma captioning launcher. Logs are bounded; images are limited to 8 MiB.
Catalog selections receive file/dependency preflight before save and queue.
Other configuration validation is structural; Toolkit validates numerical model
compatibility during loading. The UI API is internal upstream functionality,
so run the smoke test after upgrading Toolkit.

## Container and development

The derived image installs `mcp==2.3.0` into `/opt/toolkit-mcp`, separate from
training Python packages. Source is baked into the image under `/opt/ai-toolkit/aitoolkit_mcp`.
`supervise.py` runs the upstream `/start.sh` UI and MCP server together, forwards
shutdown signals, and exits if either process dies. Systemd owns restarts.
No extra GPU process or model is loaded by MCP.
The image symlinks the upstream SQLite path into the persistent `state/`
directory so the database and its WAL/SHM files survive recreation together.

Build after changing the base image or dependencies, then deploy:

```sh
docker build -t ghcr.io/delfianto/ai-toolkit:latest .
composectl restart ai-toolkit --json
```

Run from the `ai-toolkit-docker` source checkout:

```sh
uv sync
uv run python -m unittest discover -s tests -v
basedpyright -p .
ruff check src tests
```

Live protocol smoke test, including creation/editing of a temporary draft:

```sh
docker exec ai-toolkit /opt/toolkit-mcp/bin/python -m aitoolkit_mcp.smoke_test --exercise-writes
```

The test authenticates, discovers tools, reads datasets/jobs/logs/outputs,
clones a temporary draft, edits it and removes that draft through the UI API.
It never queues training. Omit `--exercise-writes` for read-only checks.
Pass `--url https://ostris-mcp.your-domain.example/mcp` to test TLS routing.
