"""CLI entry point for the private source export DVC stage."""

from __future__ import annotations

import argparse
import os

from .export import export_senses
from .profiles import load_profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.environ.get("LEXI_SOURCE_DB"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-definition-chars", type=int, default=3)
    args = parser.parse_args(argv)
    if not args.source:
        parser.error("--source or LEXI_SOURCE_DB is required")
    profiles = load_profiles(args.profiles)
    result = export_senses(args.source, args.out, min_definition_chars=args.min_definition_chars)
    if result.pool_rows < args.min_rows:
        parser.error(f"pool has {result.pool_rows} rows; expected at least {args.min_rows}")
    print(
        f"exported {result.pool_rows} senses ({result.target_groups} targets); "
        f"{len(profiles.profiles)} profiles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
