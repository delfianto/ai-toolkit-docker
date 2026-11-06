"""Explicit model identities and local repository views; no Torch or downloads."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


class CatalogError(ValueError):
    """An incomplete or incompatible local model selection."""


# These describe dependencies, not filename patterns or checkpoint conversions.
PROFILES = {
    "sdxl": {"arch": "sdxl", "required": ["checkpoint", "extras"]},
    "flux": {"arch": "flux", "required": ["transformer", "extras"]},
    "zimage": {"arch": "zimage", "required": ["transformer", "extras"]},
    "krea2": {
        "arch": "krea2",
        "required": ["transformer", "text_encoder", "vae", "text_config", "vae_config"],
    },
}


def merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value == ".":
        raise CatalogError(f"Repository entry must be a relative path: {value!r}")
    return Path(*path.parts)


class Catalog:
    def __init__(self, path: str | Path, view_root: str | Path | None = None):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data.get("version") != 1:
            raise CatalogError("Expected model catalog version 1")
        for section in ("assets", "bundles", "profiles", "views", "repositories"):
            if not isinstance(self.data.get(section, {}), dict):
                raise CatalogError(f"Catalog {section} must be an object")
        if not isinstance(self.data.get("component_bindings", []), list):
            raise CatalogError("component_bindings must be a list")
        self.view_root = Path(
            view_root
            or os.environ.get(
                "AI_TOOLKIT_MODEL_VIEWS", "/app/ai-toolkit/state/model-views"
            )
        )
        self._building: set[str] = set()

    def asset(self, name: str) -> Path:
        asset = self.data.get("assets", {}).get(name)
        if not isinstance(asset, dict):
            raise CatalogError(f"Unknown model asset: {name}")
        if "cache_repo" in asset:
            repo = asset["cache_repo"]
            if not isinstance(repo, str) or len(repo.split("/")) != 2:
                raise CatalogError(f"Asset {name}: cache_repo must be owner/repository")
            home = Path(
                os.environ.get(
                    "HF_HUB_CACHE",
                    str(
                        Path(
                            os.environ.get("HF_HOME", "~/.cache/huggingface")
                        ).expanduser()
                        / "hub"
                    ),
                )
            )
            cache = home / ("models--" + repo.replace("/", "--"))
            revision = asset.get("revision", "main")
            revision_path = relative(revision)
            ref = cache / "refs" / revision_path
            commit = ref.read_text().strip() if ref.is_file() else revision
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise CatalogError(f"Asset {name}: uncached revision {repo}@{revision}")
            path = cache / "snapshots" / commit
            if asset.get("subfolder"):
                path /= relative(asset["subfolder"])
        else:
            raw = asset.get("path")
            if not isinstance(raw, str):
                raise CatalogError(f"Asset {name}: path or cache_repo is required")
            path = Path(os.path.expandvars(raw)).expanduser()
            if not path.is_absolute():
                raise CatalogError(f"Asset {name}: use an absolute container path")
        if not path.exists():
            raise CatalogError(
                f"Asset {name} is missing: {path}. Register an existing compatible file; downloads are never implicit."
            )
        kind = asset.get("kind", "file")
        if (
            kind not in {"file", "directory"}
            or (kind == "file" and not path.is_file())
            or (kind == "directory" and not path.is_dir())
        ):
            raise CatalogError(f"Asset {name}: expected {kind}, found {path}")
        for required in asset.get("required_files", []):
            if not (path / relative(required)).is_file():
                raise CatalogError(
                    f"Asset {name} is incomplete: missing {required} in {path}"
                )
        return path

    def reference(self, value: str) -> Path:
        if value.startswith("@view:"):
            return self.view(value[6:])
        if value.startswith("@"):
            return self.asset(value[1:])
        raise CatalogError(f"Expected an @asset or @view:name reference: {value!r}")

    def expand(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("@"):
            return str(self.reference(value))
        if isinstance(value, dict):
            return {key: self.expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.expand(item) for item in value]
        return copy.deepcopy(value)

    def view(self, name: str) -> Path:
        if name in self._building:
            raise CatalogError(f"Cyclic repository view: {name}")
        spec = self.data.get("views", {}).get(name)
        if not isinstance(spec, dict):
            raise CatalogError(f"Unknown repository view: {name}")
        self._building.add(name)
        try:
            base = self.reference(spec["base"]) if spec.get("base") else None
            if base is not None and not base.is_dir():
                raise CatalogError(f"View {name}: base must be a directory")
            sources = {
                str(relative(key)): self.reference(value)
                for key, value in spec.get("files", {}).items()
            }
            # Include directory entries and links in the identity. Never copy weights.
            entries = (
                {}
                if base is None
                else {
                    str(p.relative_to(base)): p
                    for p in base.rglob("*")
                    if p.is_file() or (p.is_symlink() and p.is_dir())
                }
            )
            for target, source in sources.items():
                entries = {
                    key: value
                    for key, value in entries.items()
                    if key != target and not key.startswith(target + "/")
                }
                entries[target] = source
            if not entries:
                raise CatalogError(f"View {name} has no files")
            identity = json.dumps(
                {key: str(value) for key, value in sorted(entries.items())},
                sort_keys=True,
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()
            target = self.view_root / digest
            if not target.is_dir():
                self.view_root.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.view_root))
                try:
                    for rel, source in entries.items():
                        link = staging / relative(rel)
                        link.parent.mkdir(parents=True, exist_ok=True)
                        link.symlink_to(source, target_is_directory=source.is_dir())
                    try:
                        staging.rename(target)
                    except OSError:
                        if not target.is_dir():
                            raise
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            for required in spec.get("required_files", []):
                if not (target / relative(required)).is_file():
                    raise CatalogError(f"View {name}: missing required file {required}")
            return target
        finally:
            self._building.remove(name)

    def repository(self, repo: str) -> Path | None:
        ref = self.data.get("repositories", {}).get(repo)
        if ref is None:
            return None
        path = self.reference(ref)
        if not path.is_dir():
            raise CatalogError(f"Repository {repo} must reference a directory or view")
        return path

    def binding(self, repo: str, subfolder: str | None, loader: str) -> Path | None:
        matches = [
            item
            for item in self.data.get("component_bindings", [])
            if item.get("repo") == repo
            and ("subfolder" not in item or item["subfolder"] == (subfolder or ""))
            and ("loader" not in item or item["loader"] == loader)
        ]
        if len(matches) > 1:
            raise CatalogError(
                f"Ambiguous component bindings for {repo}, {subfolder}, {loader}"
            )
        return self.asset(matches[0]["asset"]) if matches else None

    def resolve(self, bundle_id: str, overrides: dict | None = None) -> dict:
        bundle = self.data.get("bundles", {}).get(bundle_id)
        if not isinstance(bundle, dict):
            raise CatalogError(f"Unknown model bundle: {bundle_id}")
        profile_id = bundle.get("profile")
        if not isinstance(profile_id, str):
            raise CatalogError(f"Bundle {bundle_id}: profile is required")
        profile = self.data.get("profiles", {}).get(
            profile_id, PROFILES.get(profile_id)
        )
        if not isinstance(profile, dict):
            raise CatalogError(
                f"Bundle {bundle_id}: unknown profile {profile_id!r}. Add a data profile for another upstream architecture."
            )
        components = bundle.get("components", {})
        for role in profile.get("required", []):
            if role not in components:
                raise CatalogError(
                    f"Bundle {bundle_id}: missing required component role {role}"
                )
        for role, ref in components.items():
            self.reference(ref)
            if ref.startswith("@") and not ref.startswith("@view:"):
                asset = self.data["assets"][ref[1:]]
                allowed = profile.get("formats", {}).get(role)
                if allowed and asset.get("format") not in allowed:
                    raise CatalogError(
                        f"Bundle {bundle_id}: {role} format {asset.get('format')!r} is not accepted by profile {profile_id}: {allowed}"
                    )
                families = profile.get("families", {}).get(role)
                if families and asset.get("family") not in families:
                    raise CatalogError(
                        f"Bundle {bundle_id}: incompatible {role} family {asset.get('family')!r}"
                    )
        model = merge(profile.get("model", {}), bundle.get("model", {}))
        model.setdefault("arch", profile.get("arch"))
        # Training options may override defaults; model identity remains the bundle's.
        overrides = copy.deepcopy(overrides or {})
        requested_arch = overrides.pop("arch", None)
        if (
            requested_arch
            and requested_arch.split(":")[0] != str(model.get("arch", "")).split(":")[0]
        ):
            raise CatalogError(
                f"Bundle {bundle_id}: job architecture {requested_arch} conflicts with {model.get('arch')}"
            )
        overrides.pop("name_or_path", None)
        # Paths are pinned by the bundle. Silent path changes defeat preflight.
        pinned = {
            key
            for key in model
            if key.endswith("_path") or key in {"model_kwargs", "model_paths"}
        }
        for key in pinned & overrides.keys():
            if overrides[key] != model[key] and overrides[key] != self.expand(
                model[key]
            ):
                raise CatalogError(
                    f"Bundle {bundle_id}: change {key} in the catalog rather than overriding its component selection"
                )
            overrides.pop(key)
        model = self.expand(merge(model, overrides))
        if not model.get("arch") or not model.get("name_or_path"):
            raise CatalogError(
                f"Bundle {bundle_id}: model.arch and model.name_or_path are required"
            )
        # Check declared metadata and bindings as well as the primary component.
        for repo in bundle.get("repositories", []):
            if self.repository(repo) is None:
                raise CatalogError(
                    f"Bundle {bundle_id}: unregistered metadata repository {repo}"
                )
        for binding in self.data.get("component_bindings", []):
            if binding.get("repo") in bundle.get("repositories", []):
                self.asset(binding["asset"])
        return model

    def summary(self) -> dict:
        return {
            "version": 1,
            "bundles": [
                {
                    "id": name,
                    "profile": spec.get("profile"),
                    "description": spec.get("description", ""),
                }
                for name, spec in self.data.get("bundles", {}).items()
            ],
        }


def configured() -> Catalog | None:
    path = os.environ.get("AI_TOOLKIT_MODEL_CATALOG")
    return Catalog(path) if path else None
