"""Regression checks for header-only checkpoint quantization metadata."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aitoolkit_models.compatibility import _restore_quantization_markers


class Marker:
    def __init__(self, values: list[int]):
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


class QuantizationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.layer = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }
        self.metadata = {
            "_quantization_metadata": json.dumps(
                {"format_version": "1.0", "layers": {"blocks.0.attn.wq": self.layer}}
            )
        }
        self.weights = {
            "blocks.0.attn.wq.weight": object(),
            "blocks.0.attn.wq.weight_scale": object(),
            "unquantized.weight": object(),
        }
        # The host dev environment deliberately has no Torch dependency.
        fake_torch = SimpleNamespace(
            tensor=lambda values, **kwargs: Marker(values), uint8=object()
        )
        self.mock = patch.dict("sys.modules", {"torch": fake_torch})
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_preserves_encoded_weights_and_rotation(self):
        result = _restore_quantization_markers(self.weights, self.metadata)
        for key, value in self.weights.items():
            self.assertIs(result[key], value)
        marker = result["blocks.0.attn.wq.comfy_quant"]
        self.assertEqual(json.loads(bytes(marker.tolist())), self.layer)
        self.assertNotIn("blocks.0.attn.wq.comfy_quant", self.weights)

    def test_unquantized_checkpoint_is_unchanged(self):
        self.assertIs(_restore_quantization_markers(self.weights, {}), self.weights)

    def test_rejects_missing_scale_instead_of_loading_as_plain_int8(self):
        del self.weights["blocks.0.attn.wq.weight_scale"]
        with self.assertRaisesRegex(ValueError, "Missing quantized"):
            _restore_quantization_markers(self.weights, self.metadata)

    def test_rejects_conflicting_rotation(self):
        self.weights["blocks.0.attn.wq.comfy_quant"] = Marker(
            list(json.dumps({**self.layer, "convrot": False}).encode())
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            _restore_quantization_markers(self.weights, self.metadata)

    def test_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            _restore_quantization_markers(
                self.weights,
                {"_quantization_metadata": '{"format_version": "2.0", "layers": {}}'},
            )


if __name__ == "__main__":
    unittest.main()
