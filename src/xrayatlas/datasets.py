"""Dataset pull: spectra, metadata, auxiliary files, citation, and the models on a dataset.

`get` returns a `Dataset` pinned to one dataset version and cached by (host, dataset, version), so
a pinned dataset is reproducible even after the server moves on. The server cannot replay old
versions yet, so a pinned version that is neither current nor cached raises `ConflictError`; the
client never silently substitutes another version. `search` pages lightweight summaries and loads
spectra only on request. Geometry identity is the ordered ``(theta_deg, phi_deg)`` pair.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from xrayatlas._http import JsonValue
from xrayatlas.client import Client, default_client
from xrayatlas.errors import (
    AtlasError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from xrayatlas.lock import register_dataset
from xrayatlas.types import ModelSummary
from xrayatlas.validate import is_finite_number, parse_model

EXPORT_LIMIT = 100_000
SPECTRUM_COLUMNS = (
    "energy_ev",
    "theta_deg",
    "phi_deg",
    "raw_abs",
    "i0",
    "od",
    "mass_absorption",
    "beta",
    "delta",
)
RAW_COLUMNS = ("energy_ev", "theta_deg", "phi_deg", "raw_abs", "i0")
_NUMERIC = (
    "energy_ev",
    "theta_deg",
    "phi_deg",
    "raw_abs",
    "i0",
    "od",
    "mass_absorption",
    "beta",
    "delta",
)
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_ATLAS_ID = re.compile(r"^[0-9a-hjkmnp-tv-z]{8}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_MAX_ARCHIVE_BYTES = 1024**3
Cell = str | float | int | None


class _ExportPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    row_count: int = Field(0, alias="rowCount")
    dataset_versions: dict[str, str] = Field(default_factory=dict, alias="datasetVersions")
    rows: list[dict[str, Cell]] = Field(default_factory=list)


def parse_dataset_ref(ref: str) -> tuple[str, str]:
    """Interpret a dataset reference.

    Args:
        ref: An experiment UUID, an 8-character Atlas dataset id, or ``d/{id}`` / ``/d/{id}``.

    Returns:
        ``("datasetId", uuid)`` or ``("atlasDatasetId", id)``, lowercase.

    Raises:
        ValidationError: ``ref`` is none of those forms.
    """
    text = ref.strip()
    lowered = text.lower()
    for prefix in ("/d/", "d/"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    if _UUID.match(lowered):
        return "datasetId", lowered
    if _ATLAS_ID.match(lowered):
        return "atlasDatasetId", lowered
    raise ValidationError(
        f"{ref!r} is not a dataset reference; use an Atlas id such as 'abcd2345', 'd/abcd2345', or a UUID."  # noqa: E501
    )


def _as_object(value: JsonValue, what: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise AtlasError(f"Unexpected {what} response from the server.")
    return value


def _parse_export(data: dict[str, JsonValue]) -> _ExportPayload:
    """Validate an export payload, including every numeric channel, or raise `ValidationError`."""
    payload = parse_model(_ExportPayload, data, strict=False)
    _check_numeric_rows(payload.rows)
    return payload


def _check_numeric_rows(rows: list[dict[str, Cell]]) -> None:
    """Raise `ValidationError` when a channel value is not a finite number.

    ``None`` stays a missing value. Booleans, strings, and non-finite floats are errors naming the
    row index and column, so a server change cannot silently become an all-NaN channel.
    """
    for index, row in enumerate(rows):
        for column in _NUMERIC:
            value = row.get(column)
            if value is None:
                continue
            if not is_finite_number(value):
                raise ValidationError(
                    f"row {index}: {column} must be a finite number, not {value!r}."
                )


def _export(
    client: Client,
    selector: Mapping[str, str],
    *,
    limit: int,
    version: str | None = None,
) -> tuple[_ExportPayload, bytes]:
    query: dict[str, str | int | float | bool] = {**selector, "limit": limit, "format": "json"}
    if version is not None:
        query["version"] = version
    try:
        body = client.http.get_json("/api/v1/datasets/export", query)
    except ValidationError as exc:
        if "historical version" in str(exc):
            raise ConflictError(str(exc), status=409, code="CONFLICT") from exc
        raise
    payload = _parse_export(_as_object(body, "export"))
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    return payload, raw


class AuxFiles:
    """Auxiliary files of a dataset, served inside the public dataset archive."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def download(
        self,
        dest: str | Path,
        kind: Literal["experiment", "sample"] | None = None,
    ) -> list[Path]:
        """Download and extract the dataset's auxiliary files.

        The whole archive is fetched, every member path is checked first (absolute paths, ``..``
        segments, links, and archives over 1 GiB are rejected before anything is written), and only
        regular files under ``experiment-aux/`` and ``sample-aux/`` are written, keeping their
        folder names. The archive is buffered in memory before extraction, so the 1 GiB cap bounds
        the extracted size, not the peak memory.

        Args:
            dest: Destination directory, created when missing.
            kind: ``"experiment"`` or ``"sample"`` to restrict, or ``None`` for both.

        Returns:
            Paths of the files written.

        Raises:
            AtlasError: The archive holds an unsafe path or exceeds the size cap.
            ApiError: The server answered with a non-2xx status.
        """
        data = self._dataset.client.http.get_bytes(
            f"/api/v1/datasets/{self._dataset.experiment_id}/bundle"
        )
        wanted = {
            None: ("experiment-aux", "sample-aux"),
            "experiment": ("experiment-aux",),
            "sample": ("sample-aux",),
        }[kind]
        root = Path(dest)
        written: list[Path] = []
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
            total = 0
            for member in members:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk():
                    raise AtlasError(f"Unsafe path in dataset archive: {member.name!r}.")
                total += max(member.size, 0)
            if total > _MAX_ARCHIVE_BYTES:
                raise AtlasError("The dataset archive is larger than 1 GiB.")
            for member in members:
                pure = PurePosixPath(member.name)
                if not member.isfile() or not pure.parts or pure.parts[0] not in wanted:
                    continue
                target = root.joinpath(*pure.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                target.write_bytes(handle.read())
                written.append(target)
        return written


class Dataset:
    """One dataset at one version, with its spectra loaded.

    The export endpoint returns at most 100 000 rows, the server's cap, so a result may be
    incomplete; `spectra` raises then unless ``allow_truncated``. The bundle endpoint
    ``/api/v1/datasets/{id}/bundle`` carries the full spectrum and the auxiliary files.
    """

    def __init__(
        self,
        client: Client,
        experiment_id: str,
        version: str | None,
        rows: list[dict[str, Cell]],
    ) -> None:
        self.client = client
        self.experiment_id = experiment_id
        self.version = version
        if version is not None:
            register_dataset(host=client.host, experiment_id=experiment_id, version=version)
        self._rows = rows
        self._frame: pd.DataFrame | None = None
        first = rows[0] if rows else {}
        self.atlas_id = _text(first.get("atlas_dataset_id"))
        self.molecule_id = _text(first.get("molecule_id"))
        self.molecule_name = _text(first.get("molecule_iupac_name"))
        self.edge = _text(first.get("edge_label"))
        self.instrument = _text(first.get("instrument_name"))
        self.facility = _text(first.get("facility_name"))
        self.experiment_type = _text(first.get("experiment_type"))
        self.created_at = _text(first.get("created_at"))
        dois = _text(first.get("publication_dois")) or ""
        self.publication_dois = tuple(
            part for part in (chunk.strip() for chunk in dois.split(";")) if part
        )
        self.truncated = len(rows) >= EXPORT_LIMIT
        self.aux_files = AuxFiles(self)

    def spectra(self, raw: bool = False, *, allow_truncated: bool = False) -> pd.DataFrame:
        """The dataset's spectrum points.

        Rows are sorted by ``(theta_deg, phi_deg, energy_ev)`` with rows of unknown geometry last.
        Geometry identity is the ordered pair, so ``(90, 45)`` and ``(45, 90)`` are different
        geometries.

        Args:
            raw: When true return only the uploaded raw channel and I0
                (``energy_ev, theta_deg, phi_deg, raw_abs, i0``); otherwise also ``od``,
                ``mass_absorption``, ``beta``, and ``delta`` where the dataset has them.
            allow_truncated: Accept the capped result instead of raising.

        Returns:
            A new DataFrame (callers may modify it).

        Raises:
            AtlasError: The export hit the server row cap and ``allow_truncated`` is false.
        """
        if self.truncated and not allow_truncated:
            raise AtlasError(
                f"The export returned {EXPORT_LIMIT} rows, the server's cap, so the spectrum may "  # noqa: E501
                "be incomplete; pass allow_truncated=True to accept the partial result or download "
                "the dataset bundle for the full spectrum."
            )
        if self._frame is None:
            frame = pd.DataFrame(self._rows)
            for column in SPECTRUM_COLUMNS:
                if column not in frame.columns:
                    frame[column] = pd.NA
            for column in _NUMERIC:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.sort_values(
                ["theta_deg", "phi_deg", "energy_ev"], na_position="last", kind="stable"
            ).reset_index(drop=True)
            self._frame = frame[list(SPECTRUM_COLUMNS)]
        columns = RAW_COLUMNS if raw else SPECTRUM_COLUMNS
        return self._frame[list(columns)].copy()

    def geometries(self) -> list[tuple[float | None, float | None]]:
        """Distinct ``(theta_deg, phi_deg)`` pairs in sorted order; ``None`` marks unknown angles."""  # noqa: E501
        frame = self.spectra(raw=True)
        pairs = frame[["theta_deg", "phi_deg"]].drop_duplicates()
        return [
            (None if pd.isna(t) else float(t), None if pd.isna(p) else float(p))
            for t, p in pairs.itertuples(index=False, name=None)
        ]

    def bibtex(self) -> str:
        """The dataset's BibTeX record, as served by the Atlas."""
        data = self.client.http.get_bytes(f"/api/v1/datasets/{self.experiment_id}/bibtex")
        return data.decode("utf-8")

    def models(
        self, scope: Literal["public", "mine", "shared", "all"] = "all"
    ) -> list[ModelSummary]:
        """Optical models visible to the caller that link this dataset.

        Args:
            scope: ``public``, ``mine``, ``shared`` (shared with you), or ``all``.
        """
        body = _as_object(
            self.client.http.get_json(
                "/api/v1/models", {"dataset": self.experiment_id, "scope": scope}
            ),
            "models",
        )
        items = body.get("items")
        if not isinstance(items, list):
            raise AtlasError("Unexpected models response from the server.")
        parsed: list[ModelSummary] = []
        for item in items:
            if not isinstance(item, dict):
                raise AtlasError("Unexpected models response from the server.")
            parsed.append(parse_model(ModelSummary, cast("dict[str, object]", item), strict=False))
        return parsed

    def __repr__(self) -> str:
        return (
            f"Dataset(atlas_id={self.atlas_id!r}, version={self.version!r}, "
            f"molecule={self.molecule_name!r}, edge={self.edge!r}, rows={len(self._rows)})"
        )


def _text(value: Cell) -> str | None:
    return None if value is None else str(value)


def get(ref: str, version: str | None = None, *, client: Client | None = None) -> Dataset:
    """Load a dataset, pinned to one version and cached.

    The current version is probed with a one-row export. With ``version=None`` the current version
    is loaded. A pinned ``version`` is served from the cache when present (even after the server has
    moved on); otherwise it is fetched only if it is the current version.

    Args:
        ref: An Atlas id (``abcd2345``, ``d/abcd2345``) or an experiment UUID.
        version: Dataset version ``M.m.p``, or ``None`` for the current version.
        client: Client to use; defaults to the shared client.

    Returns:
        The dataset with its spectra loaded.

    Raises:
        ValidationError: ``ref`` is malformed, or ``version`` does not look like ``M.m.p``.
        NotFoundError: No such dataset, or it has no spectrum rows.
        ConflictError: ``version`` is neither current nor cached.
        ApiError: The server answered with another error.
    """
    c = client or default_client()
    selector_key, selector_value = parse_dataset_ref(ref)
    if version is not None:
        version = version.strip()
        if not _VERSION.match(version):
            raise ValidationError(f"version must look like '1.0.2', not {version!r}.")
    probe, _ = _export(c, {selector_key: selector_value}, limit=1)
    if not probe.rows:
        raise NotFoundError(f"Dataset {ref!r} was not found or has no spectrum rows.")
    experiment_id = str(probe.rows[0]["dataset_id"])
    current = probe.dataset_versions.get(experiment_id)
    target = current if version is None else version
    cache_key = f"{c.host}|dataset|{experiment_id}|{target}"
    cached = c.cache.get(cache_key) if target else None
    if cached is not None:
        try:
            payload = _parse_export(_as_object(json.loads(cached), "export"))
        except (ValueError, AtlasError):
            c.cache.delete(cache_key)
        else:
            return Dataset(c, experiment_id, target, payload.rows)
    if version is not None and version != current:
        raise ConflictError(
            f"Dataset {ref!r} is at version {current}; version {version} is not cached and the "
            "server cannot replay historical versions.",
            status=409,
            code="CONFLICT",
        )
    payload, raw = _export(c, {"datasetId": experiment_id}, limit=EXPORT_LIMIT, version=target)
    if target:
        c.cache.put(cache_key, raw)
    return Dataset(c, experiment_id, target, payload.rows)


def resolve_experiment_id(ref: str, *, client: Client | None = None) -> str:
    """Experiment UUID for a dataset reference.

    A UUID is returned as is (lowercase). An Atlas id is resolved with a one-row export probe.

    Raises:
        ValidationError: ``ref`` is malformed.
        NotFoundError: No such dataset, or it has no spectrum rows.
    """
    key, value = parse_dataset_ref(ref)
    if key == "datasetId":
        return value
    probe, _ = _export(client or default_client(), {key: value}, limit=1)
    if not probe.rows:
        raise NotFoundError(f"Dataset {ref!r} was not found or has no spectrum rows.")
    return str(probe.rows[0]["dataset_id"])


@dataclass(frozen=True)
class DatasetSummary:
    """A dataset as listed by `search`; spectra are loaded only by `load`."""

    dataset_id: str
    atlas_dataset_id: str | None
    molecule_name: str | None
    edge_label: str | None
    experiment_type: str | None
    spectrum_point_count: int | None
    dataset_doi: str | None
    created_at: str | None
    _client: Client

    def load(self, version: str | None = None) -> Dataset:
        """Load this dataset's spectra (see `get`)."""
        return get(self.dataset_id, version, client=self._client)


def _opt_str(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def search(
    *,
    molecule_id: str | None = None,
    edge_id: str | None = None,
    doi: str | None = None,
    atlas_dataset_id: str | None = None,
    page_size: int = 50,
    client: Client | None = None,
) -> Iterator[DatasetSummary]:
    """Iterate dataset summaries that match the filters, newest first.

    Pages are fetched lazily as the iterator advances. Nothing downloads spectra; call
    `DatasetSummary.load` for the datasets you want.

    Args:
        molecule_id: Molecule UUID.
        edge_id: Edge UUID.
        doi: A linked publication DOI.
        atlas_dataset_id: An Atlas dataset id.
        page_size: Summaries per request, 1 to 200.
        client: Client to use; defaults to the shared client.

    Yields:
        One `DatasetSummary` per dataset.

    Raises:
        ValidationError: ``page_size`` is outside 1 to 200.
        AtlasError: A summary is missing its ``datasetId``.
    """
    c = client or default_client()
    if not 1 <= page_size <= 200:
        raise ValidationError("page_size must be 1 to 200.")
    offset = 0
    while True:
        params: dict[str, str | int | float | bool] = {"limit": page_size, "offset": offset}
        for key, value in (
            ("moleculeId", molecule_id),
            ("edgeId", edge_id),
            ("doi", doi),
            ("atlasDatasetId", atlas_dataset_id),
        ):
            if value is not None:
                params[key] = value
        body = _as_object(c.http.get_json("/api/v1/datasets", params), "datasets")
        data = body.get("data")
        if not isinstance(data, list):
            raise AtlasError("Unexpected datasets response from the server.")
        for item in data:
            if not isinstance(item, dict):
                continue
            dataset_id = item.get("datasetId")
            if not isinstance(dataset_id, str):
                raise AtlasError("A dataset summary is missing its datasetId.")
            molecule = item.get("molecule")
            edge = item.get("edge")
            count = item.get("spectrumPointCount")
            yield DatasetSummary(
                dataset_id=dataset_id,
                atlas_dataset_id=_opt_str(item.get("atlasDatasetId")),
                molecule_name=_opt_str(molecule.get("iupacName"))
                if isinstance(molecule, dict)
                else None,
                edge_label=_opt_str(edge.get("label")) if isinstance(edge, dict) else None,
                experiment_type=_opt_str(item.get("experimentType")),
                spectrum_point_count=count if isinstance(count, int) else None,
                dataset_doi=_opt_str(item.get("datasetDoi")),
                created_at=_opt_str(item.get("createdAt")),
                _client=c,
            )
        pagination = body.get("pagination")
        next_offset = pagination.get("nextOffset") if isinstance(pagination, dict) else None
        if not isinstance(next_offset, int):
            return
        if next_offset <= offset:
            return
        offset = next_offset


__all__ = [
    "AuxFiles",
    "Dataset",
    "DatasetSummary",
    "get",
    "parse_dataset_ref",
    "resolve_experiment_id",
    "search",
]
