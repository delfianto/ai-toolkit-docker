# Local model catalogs

The catalog separates model identity from filenames. It is deployment data,
mounted read-only into the image. Weights remain in the existing model library;
no discovery step renames, moves, downloads or substitutes them.

## Assets and bundles

An asset identifies one exact file or directory, with optional family and format
labels. A directory's `required_files` lists every file its loader will need,
including metadata, tokenizer files, weight indexes and every referenced shard.
Use container paths, not host paths:

```json
{
  "version": 1,
  "assets": {
    "my-checkpoint": {
      "path": "/models/checkpoints/anything-you-call-it.safetensors",
      "family": "sdxl",
      "format": "original"
    },
    "sdxl-metadata": {
      "path": "/model-metadata/sdxl",
      "kind": "directory",
      "required_files": ["model_index.json", "tokenizer/tokenizer_config.json"]
    }
  },
  "bundles": {
    "my-sdxl": {
      "profile": "sdxl",
      "components": {"checkpoint": "@my-checkpoint", "extras": "@sdxl-metadata"},
      "model": {
        "arch": "sdxl",
        "name_or_path": "@my-checkpoint",
        "extras_name_or_path": "@sdxl-metadata"
      }
    }
  }
}
```

This shortened example illustrates the schema. The full SDXL template lists more
metadata dependencies. Some single-file loaders select a default upstream config
repository internally; map that repository through `repositories` to the intended
local metadata directory. `extras_name_or_path` alone is not honored uniformly by
all upstream architectures.

`@asset-name` references are expanded recursively in native model fields,
including `model_kwargs`, `model_paths` and lists. `@view:name` references a
generated repository view. Non-reference values pass through to upstream.

`catalog:<bundle-id>` is resolved when Toolkit constructs `ModelConfig`. The job's
architecture must agree with the catalog. Job training options such as
`quantize` can override defaults, but catalog-defined component path fields and
`model_kwargs`/`model_paths` cannot silently change in the job. Edit the catalog
or create another bundle for a different component selection.

Ordinary local paths and Hub IDs retain upstream semantics. In offline mode,
uncataloged jobs receive the shared download guards, but do not receive complete
bundle preflight. Migrate jobs to catalog selections to check their declared
dependencies before loading large weights.

## Profiles and formats

Built-in dependency profiles are `sdxl`, `flux`, `zimage` and `krea2`. SDXL requires
`checkpoint` and `extras`; FLUX/Z-Image require `transformer` and `extras`; Krea
requires `transformer`, `text_encoder`, `vae`, `text_config` and `vae_config`.

Profiles describe required roles. Catalog-specific profiles may additionally
constrain the declared family/format of each asset:

```json
{
  "profiles": {
    "my-family": {
      "arch": "upstream_architecture_id",
      "required": ["transformer", "extras"],
      "formats": {"transformer": ["diffusers-bf16"]},
      "families": {"transformer": ["my-family"]},
      "model": {"quantize": false}
    }
  }
}
```

New upstream architectures can use the same catalog machinery by adding a data
profile and native model configuration. No filename-specific Python edit is
necessary. A genuinely new checkpoint serialization or tensor layout can still
need an upstream loader/converter; adding a manifest entry does not implement one.

Labels are explicit operator assertions, not content certification. Use the
header-only inventory command to inspect tensor names, dtypes and quantization
markers before registering a file. Never treat arbitrary `.safetensors` files,
GGUF encoders, or differently quantized repacks as interchangeable.

## Local repository views

Some upstream loaders need a Diffusers/Transformers directory instead of a single
checkpoint. A view creates canonical loader-facing names under writable
`AI_TOOLKIT_MODEL_VIEWS`, pointing to unchanged source files:

```json
{
  "views": {
    "flux-full": {
      "base": "@flux-metadata-and-extras",
      "files": {
        "transformer/diffusion_pytorch_model.safetensors": "@my-flux-transformer"
      },
      "required_files": [
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors"
      ]
    }
  }
}
```

The base must already supply the matching configs and other components. `files`
may link files or directories. Views are generated atomically in immutable,
content-addressed directories; old view directories remain valid for running
jobs. Weights are not copied, and model sources need no write access. Use
`@view:flux-full` for the model path. Views do not reserialize tensors or change
checkpoint formats: a Comfy quantized repack cannot simply be linked under a
Diffusers filename unless that loader supports its tensor representation.

Directory assets can refer to an already populated Hugging Face cache:

```json
{
  "cache_repo": "Qwen/Qwen3-VL-4B-Instruct",
  "revision": "main",
  "kind": "directory",
  "required_files": ["config.json", "tokenizer.json", "tokenizer_config.json"]
}
```

Use a cached commit hash for a fixed snapshot. Branch revisions resolve only via
existing cache refs; the resolver makes no network calls.

## Repository aliases and single-file bindings

`repositories` maps upstream metadata/tokenizer repository IDs to local directory
assets or views. HF file and snapshot requests use those exact local sources;
missing entries do not trigger downloads. This also covers upstream loaders with
hardcoded configuration repository IDs.

```json
{
  "repositories": {"Qwen/Qwen3-VL-4B-Instruct": "@qwen-config"},
  "component_bindings": [
    {
      "repo": "Qwen/Qwen3-VL-4B-Instruct",
      "loader": "toolkit.models.v2.text_encoders.qwen3_vl.Qwen3VLTextEncoder",
      "asset": "my-encoder"
    }
  ]
}
```

Bindings select an exact single-file asset at the upstream common component
loader boundary. Optional `subfolder` and fully qualified `loader` fields
disambiguate a repository containing several components. Multiple matching
bindings fail explicitly. They apply catalog-wide; use distinct local/native
paths or repository identities when a process needs different versions of a
shared component. Declare the repositories used by each bundle in its
`repositories` list so preflight checks their metadata and associated bindings.

Binding selection is independent of upstream's expected basename. Existing
Krea/VAE tensor conversions are isolated in the compatibility module; repository
aliases and bindings themselves never convert formats.

## Adding SDXL, FLUX or Z-Image

1. Inspect and register the exact checkpoint and all required supporting assets.
2. Start from the corresponding example and select the representation its
   upstream loader accepts (full SDXL checkpoint; FLUX directory/view; supported
   Z-Image single file with complete extras).
3. Supply matching configs/tokenizers and every shard. List these in
   `required_files`; map any internally selected config repositories.
4. Validate the bundle through the CLI or MCP `validate_model_bundle`.
5. Select `catalog:<id>` and the matching architecture in the UI or MCP job.
6. Verify a small training run before a long job. Catalog validation does not
   replace a numerical model-load/training compatibility test.

No model downloads happen during these steps. Populate approved assets and
metadata in a separate, deliberate acquisition step, then register them.
