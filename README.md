# AI Toolkit

A reproducible Docker image for Ostris AI Toolkit, with local model catalogs,
read-only shared weights, and authenticated MCP access to the training UI.

Image: `ghcr.io/delfianto/ai-toolkit:latest` (Linux AMD64).
The source repository is named `ai-toolkit-docker`; the image and container are `ai-toolkit`.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/aitoolkit_models/` | Catalog, preflight, local repository views, runtime integration, checkpoint compatibility |
| `src/aitoolkit_mcp/` | Authenticated UI/MCP adapter and process supervisor |
| `docker/` | Entrypoint, Python startup hook, isolated MCP requirements |
| `examples/` | Catalog examples for Krea 2, SDXL, FLUX.1 and Z-Image |
| `tests/` | Catalog, runtime, compatibility and MCP regression tests |
| `tools/` | Host captioning and dataset preparation utilities |
| `docs/` | Catalog and MCP documentation |

Host Compose configuration, credentials, datasets, catalogs and model weights do
not belong in this repository. The homelab deployment is maintained separately in
[`delfianto/compose`](https://github.com/delfianto/compose), under `ai/toolkit`.

## Model selection

Your model library can use any filenames. Register exact paths and component
roles in a version 1 JSON catalog, then select a bundle in the UI's model path
field or in a job configuration:

```yaml
model:
  arch: krea2
  name_or_path: catalog:krea-lustify
  quantize: true
```

Set `AI_TOOLKIT_MODEL_CATALOG=/etc/ai-toolkit/models.json` and bind the catalog there
read-only. Both UI training workers and direct `run.py` jobs resolve `catalog:`
selections through `ModelConfig`, before transformer loading. MCP also preflights
catalog selections before saving and queueing a job.

```sh
docker exec ai-toolkit python -m aitoolkit_models list
docker exec ai-toolkit python -m aitoolkit_models validate krea-lustify
# Validate every registered bundle:
docker exec ai-toolkit python -m aitoolkit_models validate
# Inspect headers without loading tensors; output is evidence, not auto-selection:
docker exec ai-toolkit python -m aitoolkit_models inventory /models
```

The [catalog guide](docs/models.md) explains assets, architecture profiles,
repository views, component bindings and adding another model family. Start with
an [example](examples/) and adapt its paths and dependency files to your library.
Examples for families other than the host's Krea bundle are templates, not claims
that those weights are already installed or that a full training run was tested.

## Container contract

| Setting/path | Purpose |
| --- | --- |
| `AI_TOOLKIT_MODEL_CATALOG` | Optional path to the deployment's JSON catalog |
| `AI_TOOLKIT_MODEL_VIEWS` | Writable generated view directory; default `/app/ai-toolkit/state/model-views` |
| `AI_TOOLKIT_OFFLINE=1` | Default; use local catalog sources/HF cache and prevent weight downloads |
| `HF_HOME` | Existing Hugging Face cache for metadata/tokenizers; mount separately from weights |
| `MODELS_PATH` | Optional upstream ComfyUI-layout search root, e.g. `/models` |
| `/app/ai-toolkit/state` | Persistent database, SQLite WAL/SHM and generated model views |
| `/app/ai-toolkit/config` | Saved job files; catalog location is configurable |
| `/app/ai-toolkit/datasets` | Training datasets |
| `/app/ai-toolkit/output` | Training outputs |
| `AI_TOOLKIT_AUTH` | Required existing UI/MCP bearer token, supplied at runtime |
| Port `8675` | Upstream UI/API |
| Port `8676` | Authenticated MCP endpoint `/mcp`; unauthenticated readiness endpoint `/health` |

No GPUs are reserved by the image. The deployment supplies GPU devices and owns
lifecycle/restart policy. The entrypoint initializes Prisma in the mounted
persistent database directory and supervises UI and MCP together. It forwards
signals and exits if either process dies.

Mapped repositories always use their explicit local source. Unmapped repositories
use only the existing HF cache in offline mode. `AI_TOOLKIT_OFFLINE=0` enables normal
upstream behavior for unmapped requests; mount permissions still apply. Default
offline operation never needs to create `/models/.cache`.

The catalog validates declared files, metadata dependencies, component roles,
configured family/format constraints and architecture consistency. It cannot
prove numerical compatibility or trainability from a filename. Checkpoint
conversions are separate from naming resolution; unsupported formats must be
converted deliberately or rejected. Existing Krea repack/VAE conversions are
isolated in `src/aitoolkit_models/compatibility.py` and retain upstream strict loads.

## Build and publish

```sh
docker build -t ghcr.io/delfianto/ai-toolkit:latest .
# Explicit upstream upgrade, followed by tests and component-load verification:
docker build --build-arg BASE_IMAGE=ostris/aitoolkit:<tag-or-digest> \
  -t ghcr.io/delfianto/ai-toolkit:latest .
```

The default upstream base is pinned by digest in `Dockerfile`; runtime startup does
not update application code or install dependencies. Application source and MCP
requirements are baked into the image; no source bind mounts are needed.

GitHub Actions tests every pull request and builds the image without publishing.
Pushes to `main`, `v*` tags and manual dispatches publish to
`ghcr.io/<owner>/ai-toolkit`, using the workflow's `GITHUB_TOKEN` with `packages:write`.
Tags include `latest` on the default branch, a commit SHA tag, and release versions.
A manual dispatch can choose an upstream image override. Configure package
visibility/access in GitHub as appropriate for the deployment.

## Development

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s tools -t . -v
ruff check src tests tools
basedpyright -p .
```

Tests cover resolution across the four example architectures, arbitrary filenames,
missing assets/configs, format/family constraints, symlink views, offline guards,
UI configuration expansion, MCP access and checkpoint metadata. No training GPU
is required for these tests. Validate upgrades with the image and actual component
loads as well; upstream loader APIs are an explicit integration dependency.

See [MCP usage](docs/mcp.md), [captioning](tools/captioning/README.md), and
[dataset preparation](tools/dataset_prep/README.md).
