import json
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from aitoolkit_models.catalog import CatalogError
from aitoolkit_models.runtime import (
    _patch_config,
    _patch_hub,
    _patch_mixin,
    _patch_resolver,
)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weight = self.root / "arbitrary-name.safetensors"
        self.weight.write_bytes(b"weight")
        (self.root / "config.json").write_text("{}")
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(
            json.dumps(
                {
                    "version": 1,
                    "assets": {
                        "weight": {"path": str(self.weight)},
                        "repo": {"path": str(self.root), "kind": "directory"},
                    },
                    "repositories": {"Owner/Repo": "@repo"},
                    "component_bindings": [{"repo": "Owner/Repo", "asset": "weight"}],
                    "bundles": {
                        "test": {
                            "profile": "sdxl",
                            "components": {"checkpoint": "@weight", "extras": "@repo"},
                            "model": {"name_or_path": "@weight"},
                        }
                    },
                }
            )
        )
        self.env = patch.dict(
            os.environ,
            {"AI_TOOLKIT_MODEL_CATALOG": str(self.catalog), "AI_TOOLKIT_OFFLINE": "1"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_ui_model_config_preflights_and_expands_catalog_selection(self):
        module = ModuleType("toolkit.config_modules")

        class Config:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        module.ModelConfig = Config
        _patch_config(module)
        result = Config(name_or_path="catalog:test", arch="sdxl", quantize=False)
        self.assertEqual(result.kwargs["name_or_path"], str(self.weight))
        self.weight.unlink()
        with self.assertRaisesRegex(CatalogError, "missing"):
            Config(name_or_path="catalog:test", arch="sdxl")

    def test_mixin_binding_ignores_upstream_filenames_and_enforces_offline(self):
        module = ModuleType("toolkit.models.v2._mixin")
        calls = []

        class Mixin:
            @classmethod
            def resolve_comfy_weights(cls, name_or_path, **kwargs):
                calls.append(kwargs)

        module.OstrisModelMixin = Mixin
        _patch_mixin(module)
        self.assertEqual(Mixin.resolve_comfy_weights("Owner/Repo"), str(self.weight))
        self.assertIsNone(Mixin.resolve_comfy_weights("Other/Repo"))
        self.assertTrue(calls[0]["local_only"])

    def test_hub_metadata_uses_repo_alias_without_download(self):
        module = ModuleType("huggingface_hub.file_download")

        def original(
            repo_id, filename, *, subfolder=None, local_dir=None, local_files_only=False
        ):
            self.fail("Mapped metadata must not invoke the Hub downloader")

        module.hf_hub_download = original
        _patch_hub(module)
        self.assertEqual(
            module.hf_hub_download(
                "Owner/Repo",
                "config.json",
                local_dir="/models",
                proxies=None,
                resume_download=False,
                local_dir_use_symlinks=False,
            ),
            str(self.root / "config.json"),
        )
        self.assertFalse((self.root / ".cache").exists())

    def test_hub_offline_fallback_never_uses_read_only_local_dir(self):
        module = ModuleType("huggingface_hub.file_download")

        def original(repo_id, filename, *, local_dir=None, local_files_only=False):
            self.assertIsNone(local_dir)
            self.assertTrue(local_files_only)
            return "cached-file"

        module.hf_hub_download = original
        _patch_hub(module)
        self.assertEqual(
            module.hf_hub_download("Other/Repo", "weights", local_dir="/models"),
            "cached-file",
        )

    def test_snapshot_patch_does_not_accidentally_wrap_imported_file_downloader(self):
        module = ModuleType("huggingface_hub._snapshot_download")

        def snapshot(repo_id, *, local_dir=None, local_files_only=False):
            self.fail("Mapped snapshot must not download")

        module.snapshot_download = snapshot
        module.hf_hub_download = lambda: None
        _patch_hub(module)
        self.assertEqual(module.snapshot_download("Owner/Repo"), str(self.root))

    def test_resolver_does_not_attempt_download_on_local_miss(self):
        module = ModuleType("toolkit.models.v2.resolver")
        module.comfy_precision_rank = lambda filename, qtype=None: 0

        def original(files, repo_id, **kwargs):
            self.assertTrue(kwargs["local_only"])

        module.resolve_comfy_candidates = original
        _patch_resolver(module)
        self.assertIsNone(
            module.resolve_comfy_candidates(["missing.safetensors"], "Other/Repo")
        )


if __name__ == "__main__":
    unittest.main()
