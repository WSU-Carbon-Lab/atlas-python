"""Session registry and reproducible lockfile of pulled datasets and models.

Every `Dataset` and every published `Model` obtained through this package records itself here, so
`lock()` can pin exactly what a session used. Registration is in-memory only and never writes
credentials or tokens. `restore()` re-fetches each pin at exactly its recorded revision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from xrayatlas._credentials import normalize_host
from xrayatlas.client import Client, default_client
from xrayatlas.errors import ConflictError, ValidationError
from xrayatlas.validate import parse_model

__all__ = ["lock", "reset", "restore"]


@dataclass(frozen=True)
class _DatasetEntry:
    """One dataset version recorded in the session registry."""

    host: str
    experiment_id: str
    version: str


@dataclass(frozen=True)
class _ModelEntry:
    """One published model version recorded in the session registry."""

    host: str
    tag: str
    version: int
    checksum: str


_DATASETS: dict[tuple[str, str, str], _DatasetEntry] = {}
_MODELS: dict[tuple[str, str, int], _ModelEntry] = {}


def register_dataset(*, host: str, experiment_id: str, version: str) -> None:
    """Record one dataset version in the session registry.

    Called by `xrayatlas.datasets.Dataset` when it is constructed. A call that repeats an entry
    already present with the same identity is ignored.

    Args:
        host: Atlas origin the dataset was pulled from.
        experiment_id: Experiment UUID of the dataset.
        version: Dataset version ``M.m.p``.
    """
    _DATASETS.setdefault(
        (host, experiment_id, version),
        _DatasetEntry(host=host, experiment_id=experiment_id, version=version),
    )


def register_model(*, host: str, tag: str, version: int, checksum: str) -> None:
    """Record one published model version in the session registry.

    Called by `xrayatlas.models.Model` when it is constructed from a published version. A call
    that repeats an entry already present with the same identity is ignored, and the first checksum
    recorded for a ``(host, tag, version)`` wins; versions are immutable and `reset` clears the
    registry, so a drifted checksum never overwrites one already seen in this process.

    Args:
        host: Atlas origin the model was pulled from.
        tag: The model tag.
        version: The published version number.
        checksum: The version's content checksum.
    """
    _MODELS.setdefault(
        (host, tag, version),
        _ModelEntry(host=host, tag=tag, version=version, checksum=checksum),
    )


def reset() -> None:
    """Clear the registry; for tests and long-lived sessions."""
    _DATASETS.clear()
    _MODELS.clear()


class _DatasetPin(BaseModel):
    """One dataset pin in a lockfile document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str


class _ModelPin(BaseModel):
    """One model pin in a lockfile document."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    version: int
    checksum: str


class _Lockfile(BaseModel):
    """The lockfile document `restore` reads, exactly as `lock` writes it."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    host: str
    datasets: list[_DatasetPin]
    models: list[_ModelPin]

    @field_validator("version", mode="before")
    @classmethod
    def _exactly_one(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("version must be the integer 1.")
        return value

    @field_validator("host")
    @classmethod
    def _origin(cls, value: str) -> str:
        normalized = normalize_host(value)
        if normalized is None:
            raise ValueError("host must be an https origin (http is allowed only for localhost).")
        return normalized


def lock(path: str | Path = "atlas.lock") -> Path:
    """Write the session registry to a lockfile and return the path written.

    Datasets sort by ``(id, version)`` and models by ``(tag, version)``, with duplicates removed.
    The lockfile holds no token, no Authorization header, and no user's ORCID.

    Args:
        path: File to write; parent directories are created when missing.

    Returns:
        The path written.

    Raises:
        ValidationError: The registry holds entries from more than one host; a lockfile pins one
            Atlas host.
    """
    hosts = {entry.host for entry in _DATASETS.values()}
    hosts |= {entry.host for entry in _MODELS.values()}
    if len(hosts) > 1:
        raise ValidationError(
            "The session registry holds entries from more than one host "
            f"({', '.join(sorted(hosts))}); a lockfile pins one Atlas host."
        )
    payload = {
        "version": 1,
        "host": next(iter(hosts)) if hosts else default_client().host,
        "datasets": [
            {"id": entry.experiment_id, "version": entry.version}
            for entry in sorted(
                _DATASETS.values(), key=lambda item: (item.experiment_id, item.version)
            )
        ],
        "models": [
            {"tag": entry.tag, "version": entry.version, "checksum": entry.checksum}
            for entry in sorted(_MODELS.values(), key=lambda item: (item.tag, item.version))
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def restore(path: str | Path) -> None:
    """Re-fetch every pin in a lockfile at exactly its recorded revision.

    The file is read and validated first (``version`` must be exactly ``1``, ``host`` an origin,
    and both pin lists present). One `Client` is built for the lockfile's host; a token is
    attached only when the shared credential was issued for that host. Each dataset pin is
    fetched with `xrayatlas.datasets.get` and each model pin with `xrayatlas.models.get`; the
    hooks re-register what was fetched.

    Args:
        path: Lockfile to read, as written by `lock`.

    Raises:
        ValidationError: The file is not JSON or not a valid lockfile document.
        FileNotFoundError: No file exists at ``path``.
        ConflictError: A pinned dataset version is neither current nor cached, or a pinned model
            version's checksum no longer matches the pinned one.
        ApiError: The server refused a request.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise ValidationError(f"{path} is not valid JSON.") from exc
    lockfile = parse_model(_Lockfile, document)
    client = Client(host=lockfile.host)
    # Deferred so the module-level imports of `datasets` and `models` (which import this module)
    # never form a cycle.
    from xrayatlas import datasets, models

    for dataset_pin in lockfile.datasets:
        datasets.get(dataset_pin.id, dataset_pin.version, client=client)
    for model_pin in lockfile.models:
        model = models.get(model_pin.tag, model_pin.version, client=client)
        if model.checksum is None:
            raise ConflictError(
                f"Model {model_pin.tag} version {model_pin.version} reports no checksum, but "
                f"{path} pins {model_pin.checksum}.",
                status=409,
                code="CONFLICT",
            )
        if model.checksum != model_pin.checksum:
            raise ConflictError(
                f"Model {model_pin.tag} version {model_pin.version} has checksum "
                f"{model.checksum}, but {path} pins {model_pin.checksum}.",
                status=409,
                code="CONFLICT",
            )
