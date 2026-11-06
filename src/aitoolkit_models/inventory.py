"""Read safetensors headers without importing Torch or trusting filenames."""

import json
import struct
from pathlib import Path


def inspect_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if size > 64 * 1024 * 1024 or size > path.stat().st_size - 8:
            raise ValueError("Invalid safetensors header size")
        data = json.loads(handle.read(size))
    tensors = {k: v for k, v in data.items() if k != "__metadata__"}
    return {
        "metadata": data.get("__metadata__", {}),
        "tensor_count": len(tensors),
        "dtypes": sorted({value["dtype"] for value in tensors.values()}),
        "sample_keys": list(tensors)[:12],
        "quantization_markers": [
            key
            for key in tensors
            if key.endswith((".comfy_quant", ".weight_scale", ".scale_weight"))
        ][:16],
    }


def inventory(root: str) -> dict:
    result = []
    for path in sorted(Path(root).rglob("*.safetensors")):
        try:
            result.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    **inspect_header(path),
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.append({"path": str(path), "error": str(error)})
    return {
        "files": result,
        "note": "Discovery evidence only; register component roles, families and formats explicitly.",
    }
