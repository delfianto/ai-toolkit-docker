"""One lazy integration boundary for UI workers and command-line training."""

from __future__ import annotations

import importlib
import inspect
import os
from functools import wraps
from types import ModuleType

from .catalog import CatalogError, configured, relative


def offline() -> bool:
    return os.environ.get("AI_TOOLKIT_OFFLINE", "1") == "1"


def _patch_config(module: ModuleType) -> None:
    original = module.ModelConfig.__init__

    @wraps(original)
    def initialize(self, **kwargs):
        catalog = configured()
        selected = kwargs.get("name_or_path", "")
        if isinstance(selected, str) and selected.startswith("catalog:"):
            if catalog is None:
                raise CatalogError(
                    "catalog: selections require AI_TOOLKIT_MODEL_CATALOG"
                )
            bundle_id = selected[len("catalog:") :]
            kwargs = catalog.resolve(bundle_id, kwargs)
            print(f"Model preflight passed: catalog:{bundle_id}", flush=True)
        original(self, **kwargs)

    module.ModelConfig.__init__ = initialize


def _patch_mixin(module: ModuleType) -> None:
    cls = module.OstrisModelMixin
    original = cls.resolve_comfy_weights.__func__

    @classmethod
    @wraps(original)
    def resolve(model_class, name_or_path, subfolder=None, local_only=False, **kwargs):
        catalog = configured()
        if catalog is not None:
            loader = model_class.__module__ + "." + model_class.__name__
            path = catalog.binding(name_or_path, subfolder, loader)
            if path is not None:
                return str(path)
        # Never let upstream's automatic download write into a shared model tree.
        return original(
            model_class,
            name_or_path,
            subfolder=subfolder,
            local_only=local_only or offline(),
            **kwargs,
        )

    cls.resolve_comfy_weights = resolve


def _patch_hub(module: ModuleType) -> None:
    """Alias repo requests to local views; keep normal HF cache fallback offline."""
    name = (
        "hf_hub_download"
        if module.__name__ == "huggingface_hub.file_download"
        else "snapshot_download"
    )
    original = getattr(module, name)
    signature = inspect.signature(original)

    @wraps(original)
    def download(*args, **kwargs):
        # Older Diffusers passes retired Hub arguments. The Hub decorator accepts
        # and removes these even though inspect.signature exposes the new API.
        kwargs = dict(kwargs)
        for retired in (
            "proxies",
            "resume_download",
            "force_filename",
            "local_dir_use_symlinks",
        ):
            if retired not in signature.parameters:
                kwargs.pop(retired, None)
        bound = signature.bind(*args, **kwargs)
        repo = bound.arguments.get("repo_id")
        catalog = configured()
        root = (
            catalog.repository(repo)
            if catalog is not None
            and isinstance(repo, str)
            and bound.arguments.get("repo_type", "model") in {None, "model"}
            else None
        )
        if root is not None:
            # A repository alias is a pinned local source; it never falls back.
            filename = bound.arguments.get("filename")
            if filename is None:
                return str(root)
            subfolder = bound.arguments.get("subfolder")
            target = root / relative(subfolder) if subfolder else root
            target /= relative(filename)
            if target.is_file():
                return str(target)
            error_class = importlib.import_module(
                "huggingface_hub.errors"
            ).LocalEntryNotFoundError
            raise error_class(
                f"Local repository {repo} is missing {target.relative_to(root)}. Add it to the catalog view; downloads are disabled for mapped repositories."
            )
        if offline():
            bound.arguments["local_files_only"] = True
            # HF local_dir creates .cache metadata BEFORE its offline check.
            # Read from its ordinary writable cache instead.
            bound.arguments["local_dir"] = None
        return original(*bound.args, **bound.kwargs)

    setattr(module, name, download)


def _patch_resolver(module: ModuleType) -> None:
    original = module.resolve_comfy_candidates

    @wraps(original)
    def candidates(files, repo_id, **kwargs):
        files = list(files)
        catalog = configured()
        root = catalog.repository(repo_id) if catalog is not None else None
        if root is not None:
            ordered = sorted(
                files,
                key=lambda value: module.comfy_precision_rank(
                    value, qtype=kwargs.get("qtype")
                ),
            )
            for value in ordered:
                path = root / relative(value)
                if path.is_file():
                    return str(path)
        kwargs["local_only"] = kwargs.get("local_only", False) or offline()
        return original(files, repo_id, **kwargs)

    module.__dict__["resolve_comfy_candidates"] = candidates


def install() -> None:
    # Reuse the lazy hook implementation; no training packages are imported here.
    from . import compatibility

    compatibility._PATCHERS.update(
        {
            "toolkit.config_modules": _patch_config,
            "toolkit.models.v2._mixin": _patch_mixin,
            "toolkit.models.v2.resolver": _patch_resolver,
            "huggingface_hub.file_download": _patch_hub,
            "huggingface_hub._snapshot_download": _patch_hub,
        }
    )
    if offline():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


install()
