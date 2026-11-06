"""Inspect and preflight a catalog without importing Torch or allocating VRAM."""

import argparse
import json
import sys

from .catalog import Catalog, CatalogError, configured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    check = sub.add_parser("validate")
    check.add_argument("bundle", nargs="?")
    sub.add_parser("inventory").add_argument("root")
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            from .inventory import inventory

            print(json.dumps(inventory(args.root), indent=2))
            return 0
        catalog = Catalog(args.catalog) if args.catalog else configured()
        if catalog is None:
            raise CatalogError("Set AI_TOOLKIT_MODEL_CATALOG or pass --catalog")
        if args.command == "list":
            result = catalog.summary()
        else:
            names = (
                [args.bundle] if args.bundle else list(catalog.data.get("bundles", {}))
            )
            result = {name: catalog.resolve(name) for name in names}
        print(json.dumps(result, indent=2))
        return 0
    except (CatalogError, OSError, ValueError) as error:
        print(f"Model preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
