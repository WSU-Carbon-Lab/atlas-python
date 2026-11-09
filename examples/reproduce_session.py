"""Record a session in a lockfile and replay it at the pinned revisions.

Run from ``extensions/python``:

    uv run python examples/reproduce_session.py d/abcd2345 --models m/k7m2xq4n \
        --lock build/analysis.lock
    uv run python examples/reproduce_session.py --restore --lock build/analysis.lock

Recording pulls everything first and then writes one lockfile with the dataset versions and the
model versions plus checksums. Restoring re-fetches exactly those revisions: a pinned dataset
version that is no longer current or cached fails instead of silently moving, and a model whose
checksum changed fails as well. The lockfile pins the Atlas host and contains no token or ORCID.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import xrayatlas as xa
from _shared import add_host_argument, open_client


def main() -> int:
    """Either record the listed pulls or restore a lockfile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="*", help="Datasets to pull while recording.")
    parser.add_argument(
        "--models", nargs="*", default=[], help="Model tags to pull while recording."
    )
    parser.add_argument("--lock", type=Path, default=Path("atlas.lock"), help="Lockfile path.")
    parser.add_argument(
        "--restore", action="store_true", help="Replay the lockfile instead of recording."
    )
    add_host_argument(parser)
    args = parser.parse_args()

    if args.restore:
        xa.restore(args.lock)
        print(f"restored {args.lock} (the lockfile pins the host)")
        return 0

    if not args.refs:
        parser.error("give at least one dataset reference, or pass --restore")
    with closing(open_client(args)) as client:
        for ref in args.refs:
            print(xa.datasets.get(ref, client=client))
        for tag in args.models:
            print(xa.models.get(tag, client=client))
        path = xa.lock(args.lock)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
