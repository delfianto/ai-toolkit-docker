# Vision captioning pipeline

The pipeline launches `llama serve --config <model.yaml>`, sends image requests
through its OpenAI-compatible API, writes matching `.txt` captions, and stops
the server. AI Toolkit reads the saved files directly.

## Run

```sh
python3 pipeline.py --profile /path/to/model.yaml /path/to/dataset --json

# Discover all non-hidden dataset folders beneath a root.
python3 pipeline.py --profile /path/to/model.yaml --dataset-root /path/to/datasets --json

# Select one folder beneath that root.
python3 pipeline.py --profile /path/to/model.yaml --dataset-root /path/to/datasets my_subject
```

Dataset paths are run arguments, not committed configuration. A local launcher
can hold host defaults. `CAPTION_PROFILE` and `CAPTION_RUNNER` can supply the
profile and launcher executable; `--runner` overrides the latter. There is no
pipeline TOML file.

## Model profile

Copy `gemma4-captioning-sample.yaml` to `gemma4-captioning.yaml` in this folder
and adapt it to your machine. The latter is gitignored; only the sample belongs
in GitHub. It uses the existing llama.rs schema:

- `paths.model_dir` and `paths.model_file`: main model location.
- `extra_args: [--mmproj, filename.gguf, ...]`: vision projector. Use a runner
  version that resolves its filename beside `model_dir`; absolute paths also work.
- `prompt.system`: captioning instructions. `prompt.system_file` takes priority
  and resolves relative to the YAML file.
- `server.host`/`server.port`: API endpoint.
- `sampling.max_tokens`/`sampling.temperature`: generation defaults.

The pipeline explicitly sends the YAML instructions as a system message.
The runner's OpenAI API is a passthrough and does not inject them automatically.
`--prompt` controls the per-image user request (default: `Caption this image.`).
`--max-tokens` and `--temperature` override YAML sampling values.

For a 12 GB GPU, the example uses CPU MoE expert offloading. The vision
projector is required. Adjust GPU and memory settings for your hardware.

## Resume and automation

Existing `.txt` files are preserved by default. Ctrl+C or SIGTERM stops the
managed server; rerun to resume. `--overwrite` explicitly replaces captions.
`--limit 3` limits new captions; `--dry-run` only lists work. `--max-side 512`
sets the image preview size, and `--recursive` includes nested image folders.
`--prefix` can prepend an identity identifier. Original images are unchanged.

`--json` emits JSONL events including `plan`, `starting`, `ready`, `captioned`,
`retry`, `failed`, `skipped`, `complete`, `interrupted` and `error`.
`complete` reports written, failed and skipped counts. Exit codes: 0 success,
1 failure, 130 interrupted, 2 argument error. Empty, truncated, or reasoning-only
outputs are rejected. Writes are atomic and do not replace concurrently created
captions unless `--overwrite` is set.

Logs default to `$XDG_STATE_HOME/captioning` (or `~/.local/state/captioning`).
Use `--state-dir` to choose a local directory. A lock prevents concurrent runs
sharing that directory. Managed mode refuses an occupied server port; explicitly
use `--existing-server` to attach and leave lifecycle to its owner.

## Dependencies and checks

Requires Python 3.11+, Pillow, PyYAML, typing_extensions, your Rust llama.rs runner, and a compatible
llama.cpp/ik_llama.cpp server binary. Use a runner with `model_dir`/`model_file`
and `--mmproj` path resolution support.

For development, use the shared environment from the parent Toolkit service
directory. Standalone deployments can still install `captioning/requirements.txt`.

```sh
uv sync
uv run python captioning/pipeline.py --help
uv run python -m unittest captioning.test_pipeline -v
basedpyright -p .
ruff check captioning sitecustomize.py
```
