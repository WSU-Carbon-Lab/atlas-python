"""Helpers shared by the example scripts.

Run every example from the package root, for example
``uv run python examples/pull_dataset.py d/abcd2345``. Each script keeps its own argument surface;
this module only centralizes the connection option and the CSV writer so the examples stay short.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import pandas as pd

import xrayatlas as xa


class HostArgs(Protocol):
    """The parsed ``--host`` option every example accepts."""

    host: str | None


def add_host_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--host`` to ``parser``."""
    parser.add_argument(
        "--host",
        help=(
            "Atlas origin such as https://xrayatlas.wsu.edu. Defaults to XRAYATLAS_HOST, then the "
            "credential file written by `xatlas login`, then the public site."
        ),
    )


def open_client(args: HostArgs) -> xa.Client:
    """Create a client for the parsed ``--host``; ``None`` keeps the default resolution."""
    return xa.Client(host=args.host)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` as CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")
