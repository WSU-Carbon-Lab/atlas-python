"""Read one published optical model and export its tables.

Run from ``extensions/python``:

    uv run python examples/read_model.py m/k7m2xq4n --version 2 --out build/model

Peaks are always written next to ``--out``. Pass ``--normalized`` to also export the
server-computed channels of a linked dataset, and ``--bibtex`` to write the model citation. The
tensor, when present, is exported as one column per named component: it is a container of named
parameter columns such as ``delta`` and ``beta``, not a physical tensor. The stored energy axis is
described by ``manifest.energy_ev``; ``load()`` returns the values only, so the CSV labels rows by
sample index and prints the grid bounds rather than inventing an energy column.
"""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

import xrayatlas as xa
from _shared import add_host_argument, open_client, write_frame


def _version(value: str) -> int | str:
    """Parse ``latest``, ``draft``, or a positive published version number."""
    if value in {"latest", "draft"}:
        return value
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use latest, draft, or a positive version number") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("the version number must be positive")
    return number


def _tensor_frame(handle: xa.models.TensorHandle) -> pd.DataFrame | None:
    """Return the tensor as columns, or ``None`` when it is not a 2-D payload."""
    array = handle.load()
    manifest = handle.manifest
    if manifest is None:
        raise SystemExit("The tensor has no manifest, so its components are unknown.")
    labels = list(manifest.components)
    if array.ndim != 2:
        print(f"tensor shape {array.shape}: only 2-D payloads export as columns")
        return None
    if array.shape[1] != len(labels):
        raise SystemExit(
            f"tensor axis 1 has {array.shape[1]} entries but the manifest names {len(labels)}."
        )
    columns: dict[str, npt.NDArray[np.float64]] = {}
    if np.iscomplexobj(array):
        for index, label in enumerate(labels):
            columns[f"{label}.real"] = np.asarray(array[:, index].real, dtype=np.float64)
            columns[f"{label}.imag"] = np.asarray(array[:, index].imag, dtype=np.float64)
        print("complex payload: each .real column is delta and each .imag column is beta")
    else:
        for index, label in enumerate(labels):
            columns[label] = np.asarray(array[:, index], dtype=np.float64)
    return pd.DataFrame(columns)


def main() -> int:
    """Load the model, print its metadata, and write the requested CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Model tag such as m/k7m2xq4n.")
    parser.add_argument("--version", type=_version, default="latest", help="latest, draft, or N.")
    parser.add_argument("--out", type=Path, default=Path("build/model"), help="Output prefix.")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Also export the server-computed channels of a linked dataset.",
    )
    parser.add_argument("--dataset", help="Dataset for --normalized; default is the first link.")
    parser.add_argument("--bibtex", type=Path, help="Write the published version's BibTeX here.")
    add_host_argument(parser)
    args = parser.parse_args()

    with closing(open_client(args)) as client:
        model = xa.models.get(args.tag, args.version, client=client)
        print(f"{model.tag} {model.title!r}")
        print(f"version: {model.version_number or 'draft'}; status: {model.status}")
        print(f"owner: {model.owner_orcid}; visibility: {model.visibility}")
        print(f"checksum: {model.checksum or '-'}")
        print(f"lineage: {len(model.lineage().ancestors)} visible ancestors")
        if model.normalization is not None:
            print("normalization:")
            print(json.dumps(model.normalization.to_wire(), indent=2))
        print("linked datasets:")
        for link in model.datasets:
            print(f"  {link}")

        peaks = model.peaks
        print(f"peaks: {len(peaks)}")
        if not peaks.empty:
            print(peaks.to_string(index=False))
        write_frame(peaks, args.out.with_name(f"{args.out.name}-peaks.csv"))

        if args.normalized:
            dataset = (
                args.dataset
                if args.dataset is not None
                else model.datasets[0].load()
                if model.datasets
                else None
            )
            if dataset is None:
                raise SystemExit("The model links no dataset; pass --dataset.")
            channels = model.normalized(dataset)
            write_frame(channels, args.out.with_name(f"{args.out.name}-normalized.csv"))

        if model.tensor is not None:
            manifest = model.tensor.manifest
            print(f"tensor: {model.tensor.byte_size} bytes, sha256 {model.tensor.sha256[:12]}")
            if manifest is not None:
                grid = manifest.energy_ev
                print(
                    f"  grid: {grid.count} samples from {grid.min} to {grid.max} eV; "
                    f"components {', '.join(manifest.components)}; frame {manifest.frame}; "
                    f"units {manifest.units}"
                )
            frame = _tensor_frame(model.tensor)
            if frame is not None:
                path = args.out.with_name(f"{args.out.name}-tensor.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index_label="sample")
                print(f"wrote {path} ({len(frame)} samples)")

        if args.bibtex is not None:
            if model.status != "published":
                raise SystemExit("Only a published version can be cited; pass a published version.")
            args.bibtex.parent.mkdir(parents=True, exist_ok=True)
            args.bibtex.write_text(model.cite(), encoding="utf-8")
            print(f"wrote {args.bibtex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
