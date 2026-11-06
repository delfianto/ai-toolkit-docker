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

GitHub Actions checks upstream every night at **02:23 WIB** (`19:23 UTC`). It
resolves `ostris/ai-toolkit` HEAD and `ostris/aitoolkit:latest` to an exact source
commit and runtime-image digest. Releases and upstream tags are not required.

A published `build-<fingerprint>` tag records the combination of upstream source,
runtime base and this wrapper's commit. If that tag already exists, the scheduled
run skips tests and image building. Source changes, base-image changes, wrapper
changes, or a previously failed publication require a new build. Registry access
errors fail the check rather than being mistaken for a missing image.

Builds fetch the selected upstream commit, refresh Python requirements while
preserving the base's CUDA accelerator versions, and reinstall/rebuild the UI.
The image imports the common integration APIs before publication, so detected
upstream API/dependency incompatibilities fail the build. Updating accelerator
versions belongs to the runtime base; a requirement conflict fails explicitly.
No dependencies or source updates happen when a deployed container starts.

Successful default-branch builds publish `ghcr.io/delfianto/ai-toolkit:latest`, a
fingerprint tag for rollback, and a wrapper commit tag. OCI labels record the
upstream commit and base digest. GitHub Actions uses `GITHUB_TOKEN` with
`packages:write`; the package must grant the source repository Actions write
access. Pull requests test/build without publishing. Pushes to `main` and manual
dispatches always build; a manual dispatch may override the base image or choose
a full upstream commit SHA. The optional `v*` tags refer to this wrapper's own
versions, not upstream releases.

For a local build using the nightly selection logic:

```sh
python3 tools/upstream/check.py --image ghcr.io/delfianto/ai-toolkit
# Use the printed immutable base_image and upstream_commit:
docker build --build-arg BASE_IMAGE='<base_image>' \
  --build-arg UPSTREAM_COMMIT='<upstream_commit>' \
  -t ghcr.io/delfianto/ai-toolkit:latest .
```

A plain `docker build` retains the pinned Dockerfile base and its bundled source,
which is useful for reproducing the baseline without resolving a new upstream
commit. Application source and MCP requirements are baked into the image; no
source bind mounts are needed. Publishing a nightly image does not restart a
running deployment; use `composectl update ai-toolkit --json` when ready.

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
