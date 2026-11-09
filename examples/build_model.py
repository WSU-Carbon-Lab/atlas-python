"""Build a model draft from a dataset and publish it when asked.

Run from ``extensions/python``:

    uv run python examples/build_model.py d/abcd2345 --title "My model" \
        --pre 280 283 --post 325 340 --add-peak 285.4 pi-star \
        --tensor-csv build/tensor.csv --out build/model --publish

The new model starts from the dataset's default model (or ``--from-model``), so peaks and
normalization you do not change are carried over. Nothing is written until the ``with`` block exits
cleanly, and publishing only happens with ``--publish``. ``--tensor-csv`` is a CSV whose first
column is ``energy_ev`` and whose remaining columns are named components, for example
``delta,beta``. The payload is a container of named parameter columns; a full physical tensor is
future work.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

import xrayatlas as xa
from _shared import add_host_argument, open_client


def _pair(
    parser: argparse.ArgumentParser, values: list[float] | None, name: str
) -> tuple[float, float] | None:
    """Validate an ordered two-value energy window."""
    if values is None:
        return None
    low, high = values
    if low >= high:
        parser.error(f"{name} needs low < high")
    return low, high


def _tensor_from_csv(path: Path) -> xa.Tensor:
    """Build a tensor from a CSV with ``energy_ev`` plus one column per component."""
    frame = pd.read_csv(path)
    if "energy_ev" not in frame.columns:
        raise SystemExit(f"{path} must have an energy_ev column.")
    components = [column for column in frame.columns if column != "energy_ev"]
    if not components:
        raise SystemExit(f"{path} must have at least one component column after energy_ev.")
    return xa.Tensor.from_array(
        frame[components].to_numpy(dtype=np.float64),
        energy_ev=frame["energy_ev"].to_numpy(dtype=np.float64),
        components=components,
        frame="lab",
        units="1",
    )


def main() -> int:
    """Build the draft, print its tag, and publish it when ``--publish`` was given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", help="Dataset the model links; the first link is primary.")
    parser.add_argument(
        "--title", help="Title of the new model; the source title is reused when omitted."
    )
    parser.add_argument("--description", help="Description of the new model.")
    parser.add_argument(
        "--from-model", help="Model tag to start from instead of the dataset default."
    )
    parser.add_argument(
        "--from-version", type=int, help="Published version of --from-model to start from."
    )
    parser.add_argument(
        "--pre", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Pre-edge window in eV."
    )
    parser.add_argument(
        "--post", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Post-edge window in eV."
    )
    parser.add_argument(
        "--add-peak",
        nargs=2,
        metavar=("EV", "KIND"),
        help="Append one peak; KIND is pi-star, sigma-star, or other.",
    )
    parser.add_argument(
        "--tensor-csv",
        type=Path,
        help="CSV with energy_ev plus component columns (for example delta,beta).",
    )
    parser.add_argument("--publish", action="store_true", help="Publish the saved draft.")
    add_host_argument(parser)
    args = parser.parse_args()

    if args.from_version is not None and args.from_model is None:
        parser.error("--from-version requires --from-model")
    if (args.pre is None) != (args.post is None):
        parser.error("give both --pre and --post, or neither")
    if args.add_peak is not None and args.add_peak[1] not in {"pi-star", "sigma-star", "other"}:
        parser.error("--add-peak KIND must be pi-star, sigma-star, or other")

    with closing(open_client(args)) as client:
        dataset = xa.datasets.get(args.ref, client=client)
        builder = xa.model(
            datasets=[dataset],
            title=args.title,
            from_model=args.from_model,
            from_version=args.from_version,
            client=client,
        )
        with builder:
            if args.pre is not None:
                pre = _pair(parser, args.pre, "--pre")
                post = _pair(parser, args.post, "--post")
                assert pre is not None and post is not None
                builder.normalization = xa.Normalization(pre=pre, post=post)
            if args.description is not None:
                builder.description = args.description
            if args.add_peak is not None:
                center, kind = args.add_peak
                extra = pd.DataFrame({"center_ev": [float(center)], "kind": [kind]})
                builder.peaks = pd.concat([builder.peaks, extra], ignore_index=True)
            if args.tensor_csv is not None:
                builder.tensor = _tensor_from_csv(args.tensor_csv)

        draft = builder.model
        assert draft is not None
        print(f"saved draft {draft.tag} (status {draft.status})")
        if not args.publish:
            print("not published; pass --publish to mint the next immutable version")
            return 0
        published = builder.publish()
        print(f"published {published.tag} version {published.version_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
