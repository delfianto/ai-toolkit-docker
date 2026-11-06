import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aitoolkit_models.catalog import Catalog, CatalogError


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "totally-custom-name.safetensors"
        self.weights.write_bytes(b"weights")
        self.meta = self.root / "metadata"
        self.meta.mkdir()
        (self.meta / "config.json").write_text("{}")
        self.data = {
            "version": 1,
            "assets": {
                "weights": {
                    "path": str(self.weights),
                    "format": "bf16",
                    "family": "test",
                },
                "meta": {
                    "path": str(self.meta),
                    "kind": "directory",
                    "required_files": ["config.json"],
                },
            },
            "views": {
                "extras": {
                    "base": "@meta",
                    "files": {"vae/custom.safetensors": "@weights"},
                    "required_files": ["config.json", "vae/custom.safetensors"],
                }
            },
            "repositories": {"owner/repo": "@view:extras"},
            "component_bindings": [
                {"repo": "owner/repo", "subfolder": "vae", "asset": "weights"}
            ],
            "bundles": {},
        }

    def catalog(self):
        path = self.root / "catalog.json"
        path.write_text(json.dumps(self.data))
        return Catalog(path, self.root / "views")

    def test_same_resolution_path_supports_four_architectures(self):
        for profile, primary in [
            ("sdxl", "checkpoint"),
            ("flux", "transformer"),
            ("zimage", "transformer"),
            ("krea2", "transformer"),
        ]:
            with self.subTest(profile=profile):
                components = {primary: "@weights", "extras": "@view:extras"}
                if profile == "krea2":
                    components.update(
                        {
                            "text_encoder": "@weights",
                            "vae": "@weights",
                            "text_config": "@meta",
                            "vae_config": "@meta",
                        }
                    )
                self.data["bundles"][profile] = {
                    "profile": profile,
                    "components": components,
                    "model": {
                        "name_or_path": "@weights",
                        "extras_name_or_path": "@view:extras",
                    },
                }
                result = self.catalog().resolve(
                    profile, {"name_or_path": "catalog:" + profile, "quantize": False}
                )
                self.assertEqual(result["arch"], profile)
                self.assertEqual(result["name_or_path"], str(self.weights))
                self.assertFalse(result["quantize"])

    def test_repository_view_uses_links_without_copying_or_mutating_weights(self):
        catalog = self.catalog()
        view = catalog.repository("owner/repo")
        self.assertTrue((view / "vae/custom.safetensors").is_symlink())
        self.assertEqual((view / "vae/custom.safetensors").resolve(), self.weights)
        self.assertEqual((view / "config.json").read_text(), "{}")
        self.assertEqual(view, catalog.repository("owner/repo"))
        self.assertEqual(list(self.meta.iterdir()), [self.meta / "config.json"])

    def test_nested_view_preserves_linked_component_directories(self):
        self.data["views"]["parent"] = {"files": {"tokenizer": "@meta"}}
        self.data["views"]["child"] = {"base": "@view:parent"}
        view = self.catalog().view("child")
        self.assertEqual((view / "tokenizer/config.json").read_text(), "{}")

    def test_missing_file_fails_even_after_materializing_view(self):
        catalog = self.catalog()
        catalog.view("extras")
        self.weights.unlink()
        with self.assertRaisesRegex(CatalogError, "weights is missing"):
            catalog.view("extras")

    def test_missing_metadata_is_explicit(self):
        (self.meta / "config.json").unlink()
        with self.assertRaisesRegex(CatalogError, "incomplete"):
            self.catalog().asset("meta")

    def test_no_filename_fallback_or_silent_encoder_substitution(self):
        self.weights.rename(self.root / "similar-name.safetensors")
        with self.assertRaisesRegex(
            CatalogError, "Register an existing compatible file"
        ):
            self.catalog().asset("weights")

    def test_new_architecture_is_a_data_profile(self):
        self.data["profiles"] = {
            "future": {
                "arch": "future-model",
                "required": ["encoder"],
                "formats": {"encoder": ["bf16"]},
                "families": {"encoder": ["test"]},
            }
        }
        self.data["bundles"]["new"] = {
            "profile": "future",
            "components": {"encoder": "@weights"},
            "model": {"name_or_path": "@weights"},
        }
        self.assertEqual(self.catalog().resolve("new")["arch"], "future-model")
        self.data["assets"]["weights"]["format"] = "gguf"
        with self.assertRaisesRegex(CatalogError, "not accepted"):
            self.catalog().resolve("new")
        self.data["assets"]["weights"].update(format="bf16", family="wrong")
        with self.assertRaisesRegex(CatalogError, "incompatible"):
            self.catalog().resolve("new")

    def test_missing_role_fails_before_expansion(self):
        self.data["bundles"]["bad"] = {
            "profile": "flux",
            "components": {"transformer": "@weights"},
            "model": {"name_or_path": "@weights"},
        }
        with self.assertRaisesRegex(CatalogError, "required component role extras"):
            self.catalog().resolve("bad")

    def test_job_cannot_silently_change_bundle_architecture_or_paths(self):
        self.data["bundles"]["one"] = {
            "profile": "sdxl",
            "components": {"checkpoint": "@weights", "extras": "@meta"},
            "model": {"name_or_path": "@weights", "vae_path": "@weights"},
        }
        with self.assertRaisesRegex(CatalogError, "conflicts"):
            self.catalog().resolve("one", {"arch": "flux"})
        with self.assertRaisesRegex(CatalogError, "change vae_path"):
            self.catalog().resolve("one", {"vae_path": "/other.safetensors"})

    def test_view_rejects_path_escape_and_cycles(self):
        original = copy.deepcopy(self.data["views"])
        for path in ("../escape", "/absolute"):
            self.data["views"]["extras"]["files"] = {path: "@weights"}
            with self.assertRaisesRegex(CatalogError, "relative path"):
                self.catalog().view("extras")
        self.data["views"] = original
        self.data["views"]["extras"]["base"] = "@view:extras"
        with self.assertRaisesRegex(CatalogError, "Cyclic"):
            self.catalog().view("extras")

    def test_binding_is_exact_and_ambiguity_fails(self):
        catalog = self.catalog()
        self.assertEqual(catalog.binding("owner/repo", "vae", "Loader"), self.weights)
        self.assertIsNone(catalog.binding("owner/repo", "text_encoder", "Loader"))
        self.data["component_bindings"].append({"repo": "owner/repo", "asset": "meta"})
        with self.assertRaisesRegex(CatalogError, "Ambiguous"):
            self.catalog().binding("owner/repo", "vae", "Loader")

    def test_cached_metadata_resolves_revision_without_hub_calls(self):
        cache = self.root / "hub/models--owner--repo"
        (cache / "refs").mkdir(parents=True)
        commit = "a" * 40
        (cache / "refs/main").write_text(commit)
        (cache / "snapshots" / commit).mkdir(parents=True)
        self.data["assets"]["cached"] = {
            "cache_repo": "owner/repo",
            "kind": "directory",
        }
        with patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "hub")}):
            self.assertEqual(
                self.catalog().asset("cached"), cache / "snapshots" / commit
            )


if __name__ == "__main__":
    unittest.main()
