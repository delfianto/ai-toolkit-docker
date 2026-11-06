"""Load local Krea 2 components and compatible community checkpoint repacks.

Imported by ai-toolkit-models.pth. Stays lazy: torch is not loaded until
ai-toolkit imports the VAE / text-encoder modules.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import json
import sys
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Protocol, cast

from typing_extensions import override

StateDict = dict[str, object]
WeightCandidates = dict[str | tuple[str, str], list[str]]


class CheckpointLoader(Protocol):
    load_file: Callable[..., StateDict]


def _restore_quantization_markers(
    state_dict: StateDict, metadata: dict[str, str]
) -> StateDict:
    """Translate header-only Comfy quantization into Toolkit's tensor markers."""
    raw = metadata.get("_quantization_metadata")
    if raw is None:
        return state_dict
    config = json.loads(raw)
    if (
        not isinstance(config, dict)
        or config.get("format_version") != "1.0"
        or not isinstance(config.get("layers"), dict)
    ):
        raise ValueError("Unsupported Krea2 quantization metadata schema")
    # Keep imports lazy: the .pth hook is also loaded by non-training tools.
    torch = importlib.import_module("torch")
    converted = dict(state_dict)
    for prefix, layer_config in config["layers"].items():
        if not isinstance(layer_config, dict) or layer_config.get("format") not in {
            "int8_tensorwise",
            "nvfp4",
            "float8_e4m3fn",
        }:
            raise ValueError(f"Unsupported Krea2 quantization metadata for {prefix}")
        for suffix in (".weight", ".weight_scale"):
            if prefix + suffix not in converted:
                raise ValueError(f"Missing quantized Krea2 tensor: {prefix + suffix}")
        marker = prefix + ".comfy_quant"
        if marker in converted:
            existing = json.loads(
                bytes(cast(QuantizedTensor, converted[marker]).tolist()).decode()
            )
            if existing != layer_config:
                raise ValueError(f"Conflicting Krea2 quantization metadata: {prefix}")
        else:
            converted[marker] = torch.tensor(
                list(json.dumps(layer_config).encode()), dtype=torch.uint8
            )
    print(
        f"Restored quantization markers for {len(config['layers'])} Krea2 layers from header",
        flush=True,
    )
    return converted


def _patch_krea_loader(mod: ModuleType) -> None:
    loader = cast(CheckpointLoader, mod)
    original = loader.load_file

    def load_file(path: str, *args: object, **kwargs: object) -> StateDict:
        safetensors = importlib.import_module("safetensors")
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        return _restore_quantization_markers(original(path, *args, **kwargs), metadata)

    # Only Krea's file reader changes; other safetensors consumers are untouched.
    loader.load_file = load_file


class BoundConverter(Protocol):
    @property
    def __func__(self) -> Callable[[type[object], StateDict], StateDict]: ...


class LocalModelClass(Protocol):
    aitk_comfy_repo: str | None
    aitk_comfy_weight_names: WeightCandidates
    convert_state_dict_on_load: BoundConverter


class ExecutableLoader(Protocol):
    exec_module: Callable[[ModuleType], None]


class FloatingTensor(Protocol):
    def __mul__(self, other: object) -> object: ...


class QuantizedTensor(Protocol):
    def float(self) -> FloatingTensor: ...
    def tolist(self) -> list[int]: ...


def _patch_vae(mod: ModuleType) -> None:
    cls = cast(LocalModelClass, mod.QwenImageVAE)
    orig = cls.convert_state_dict_on_load.__func__

    def convert(c: type[object], state_dict: StateDict) -> StateDict:
        if any(
            k == "conv1.weight" or k.startswith("decoder.middle") for k in state_dict
        ):
            # This dependency exists in the toolkit container, not on the host.
            module = importlib.import_module("diffusers.loaders.single_file_utils")
            converter = cast(
                Callable[[StateDict], StateDict],
                module.convert_wan_vae_to_diffusers,
            )
            return converter(state_dict)
        return orig(c, state_dict)

    cls.convert_state_dict_on_load = classmethod(convert)


def _patch_krea(mod: ModuleType) -> None:
    cls = cast(LocalModelClass, mod.SingleStreamDiT)
    orig = cls.convert_state_dict_on_load.__func__

    def convert(c: type[object], state_dict: StateDict) -> StateDict:
        state_dict = orig(c, state_dict)
        # Some RAW repacks carry these auxiliary matrices (and quant markers).
        # Neither the official LastLayer nor ComfyUI uses them. The quantized
        # importer otherwise tries to attach them as nonexistent Linear layers.
        # Preserve every other key so the upstream strict load still checks
        # missing weights, shapes and unexpected tensors.
        skipped = {
            key for key in state_dict if key.startswith(("last.down.", "last.up."))
        }
        if skipped:
            print(
                f"Skipping {len(skipped)} auxiliary Krea2 last.down/last.up tensors",
                flush=True,
            )
        converted = {
            key: value for key, value in state_dict.items() if key not in skipped
        }
        # This 12 -> 1 layer is explicitly excluded from toolkit quantization,
        # but a repack may already quantize it. _int_mm cannot accept K=12.
        # Decode its plain per-row INT8 weights exactly; leave all large layers
        # in their encoded format. The checkpoint on disk remains unchanged.
        prefix = "txtfusion.projector"
        marker = prefix + ".comfy_quant"
        if marker in converted:
            quant_config = json.loads(
                bytes(cast(QuantizedTensor, converted[marker]).tolist()).decode()
            )
            if quant_config.get("format") != "int8_tensorwise" or quant_config.get(
                "convrot"
            ):
                raise ValueError(
                    "Unsupported quantization for Krea2's tiny text projector"
                )
            weight_key, scale_key = prefix + ".weight", prefix + ".weight_scale"
            converted[weight_key] = (
                cast(QuantizedTensor, converted[weight_key]).float()
                * converted[scale_key]
            )
            del converted[scale_key], converted[marker]
            print("Decoded tiny Krea2 text projector to floating point", flush=True)
        return converted

    cls.convert_state_dict_on_load = classmethod(convert)


_PATCHERS: dict[str, Callable[[ModuleType], None]] = {
    "toolkit.models.v2.vae.qwen_image": _patch_vae,
    "extensions_built_in.diffusion_models.krea2.src.mmdit": _patch_krea,
    "extensions_built_in.diffusion_models.krea2.krea2": _patch_krea_loader,
}


class _Finder(importlib.abc.MetaPathFinder):
    @override
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _PATCHERS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        loader = cast(ExecutableLoader, spec.loader)
        orig_exec = loader.exec_module
        patch = _PATCHERS[fullname]

        def exec_module(module: ModuleType) -> None:
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
