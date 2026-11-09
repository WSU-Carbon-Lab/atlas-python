"""Compare two or more models by aligned peaks and normalized-channel differences.

Run from ``extensions/python``:

    uv run python examples/compare_models.py m/k7m2xq4n m/p8q9r2s3 \
        --dataset d/abcd2345 --channel mass_absorption --out build/compare

Peaks are aligned by single-linkage tolerance, and each model's server-normalized channel is
merged on exact ``(theta_deg, phi_deg, energy_ev)`` keys, so no resampling or interpolation happens
here. Differences are computed against the first model in the argument list.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import xrayatlas as xa
from _shared import add_host_argument, open_client, write_frame


def main() -> int:
    """Load the models, print the aligned peaks, and write both comparison tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+", help="Two or more model tags.")
    parser.add_argument(
        "--dataset", help="Atlas id or UUID both models link; default is each link."
    )
    parser.add_argument(
        "--channel",
        choices=("mass_absorption", "od", "beta"),
        default="mass_absorption",
        help="Normalized channel to compare.",
    )
    parser.add_argument("--tolerance", type=float, default=0.2, help="Peak tolerance in eV.")
    parser.add_argument("--out", type=Path, default=Path("build/compare"), help="Output prefix.")
    add_host_argument(parser)
    args = parser.parse_args()

    if len(args.tags) < 2:
        parser.error("give at least two model tags")

    with closing(open_client(args)) as client:
        models = [xa.models.get(tag, client=client) for tag in args.tags]
        comparison = xa.compare(models, tolerance_ev=args.tolerance)
        peaks = comparison.peaks
        print(f"aligned peak groups: {len(peaks)}")
        if not peaks.empty:
            print(peaks.to_string(index=False))
        write_frame(peaks, args.out.with_name(f"{args.out.name}-peaks.csv"))

        spectra = comparison.spectra(args.dataset, channel=args.channel)
        print(f"aligned channel rows: {len(spectra)}")
        for column in spectra.columns:
            if "_minus_" in column:
                values = spectra[column].dropna()
                print(f"  {column}: {len(values)} finite rows, max |diff| {values.abs().max():.6g}")
        write_frame(spectra, args.out.with_name(f"{args.out.name}-{args.channel}.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
