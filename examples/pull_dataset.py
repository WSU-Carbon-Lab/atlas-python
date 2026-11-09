"""Export one dataset's spectrum to CSV.

Run from ``extensions/python``:

    uv run python examples/pull_dataset.py d/abcd2345 --out build/abcd2345.csv

Metadata comes from the same export as the spectrum. Auxiliary files are a separate download; see
``xa.datasets.Dataset.aux_files``.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import xrayatlas as xa
from _shared import add_host_argument, open_client, write_frame


def main() -> int:
    """Load the dataset, print its metadata, and write the requested CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="Atlas dataset id (d/abcd2345) or experiment UUID.")
    parser.add_argument("--version", help="Dataset version M.m.p; default is the current version.")
    parser.add_argument("--raw", action="store_true", help="Export only raw_abs and i0.")
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="Accept the server's 100000-row export cap instead of raising.",
    )
    parser.add_argument("--bibtex", type=Path, help="Write the dataset BibTeX record here.")
    parser.add_argument("--out", type=Path, default=Path("dataset.csv"), help="CSV destination.")
    add_host_argument(parser)
    args = parser.parse_args()

    with closing(open_client(args)) as client:
        dataset = xa.datasets.get(args.ref, args.version, client=client)
        print(dataset)
        print(f"version: {dataset.version or 'current'}")
        print(f"molecule: {dataset.molecule_name or '-'} ({dataset.molecule_id or '-'})")
        print(f"edge: {dataset.edge or '-'}; instrument: {dataset.instrument or '-'}")
        print(f"facility: {dataset.facility or '-'}; type: {dataset.experiment_type or '-'}")
        print(f"created: {dataset.created_at or '-'}")
        print(f"DOIs: {', '.join(dataset.publication_dois) or '-'}")
        print("geometries (theta, phi):")
        for theta, phi in dataset.geometries():
            print(f"  theta={theta}  phi={phi}")
        frame = dataset.spectra(raw=args.raw, allow_truncated=args.allow_truncated)
        write_frame(frame, args.out)
        if args.bibtex is not None:
            args.bibtex.parent.mkdir(parents=True, exist_ok=True)
            args.bibtex.write_text(dataset.bibtex(), encoding="utf-8")
            print(f"wrote {args.bibtex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
