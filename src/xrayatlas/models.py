"""Reading optical models: pull a model, its peaks, linked datasets, lineage, normalized channels,
tensor, and citation.

A `Model` is one version of a model (the latest published version by default). Model JSON is never
cached because visibility can change; tensor blobs are cached by SHA-256 and verified on every read.
Normalization is always computed by the server. Writing models lives in `xrayatlas.publish`.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from xrayatlas import datasets
from xrayatlas._http import JsonValue
from xrayatlas.client import Client, default_client
from xrayatlas.errors import AtlasError, ConflictError, ValidationError
from xrayatlas.lock import register_model
from xrayatlas.tensor import TensorArray, widen_tensor_dtype
from xrayatlas.types import (
    Author,
    DatasetLinkOut,
    Lineage,
    ModelDetail,
    ModelSummary,
    ModelVersion,
    Normalization,
    TensorManifest,
)
from xrayatlas.validate import clean_tag, is_finite_number, parse_model

if TYPE_CHECKING:
    from xrayatlas import publish

VersionSelector = int | Literal["latest", "draft"]

PEAK_COLUMNS = (
    "ordinal",
    "center_ev",
    "kind",
    "transition",
    "amplitude",
    "width",
    "shape",
    "center_err",
    "amplitude_err",
    "width_err",
    "experiment_id",
    "polarization_id",
    "params",
    "fit_provenance",
)
CHANNEL_COLUMNS = (
    "energy_ev",
    "od",
    "mass_absorption",
    "beta",
    "polarization_id",
    "theta_deg",
    "phi_deg",
)


def _object(value: JsonValue, what: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise AtlasError(f"Unexpected {what} response from the server.")
    return value


def _version_param(version: VersionSelector) -> str:
    return str(version)


def _major(version: str) -> int:
    return int(version.split(".")[0])


class ModelDatasetLink:
    """A dataset a model version links, pinned to a dataset version."""

    def __init__(self, client: Client, link: DatasetLinkOut) -> None:
        self._client = client
        self.experiment_id = link.experiment_id
        self.atlas_dataset_id = link.atlas_dataset_id
        self.pinned = link.pinned
        self.role = link.role

    def load(self) -> datasets.Dataset:
        """The dataset at exactly the pinned version (from the cache or the server)."""
        return datasets.get(self.experiment_id, self.pinned, client=self._client)

    def spectra(self, raw: bool = False) -> pd.DataFrame:
        """The linked dataset's spectrum.

        Args:
            raw: When true, the original uploaded channel and I0. These do not change between
                minor and patch versions, so the current version is loaded and a `ConflictError`
                is raised only when its major version differs from the pinned one. When false, the
                derived channels at exactly the pinned version.

        Raises:
            ConflictError: The dataset's raw data may differ from the pinned version, or the pinned
                version is neither current nor cached.
        """
        if not raw:
            return self.load().spectra()
        current = datasets.get(self.experiment_id, client=self._client)
        if current.version is not None and _major(current.version) != _major(self.pinned):
            raise ConflictError(
                f"Dataset {self.experiment_id} is at {current.version}; the model pins {self.pinned} "  # noqa: E501
                "and the raw data may differ across major versions.",
                status=409,
                code="CONFLICT",
            )
        return current.spectra(raw=True)

    def __repr__(self) -> str:
        return (
            f"ModelDatasetLink(atlas_dataset_id={self.atlas_dataset_id!r}, "
            f"pinned={self.pinned!r}, role={self.role!r})"
        )


class TensorHandle:
    """A model version's stored tensor."""

    def __init__(
        self,
        client: Client,
        tag: str,
        version: VersionSelector,
        sha256: str,
        byte_size: int,
        manifest: TensorManifest | None,
    ) -> None:
        self._client = client
        self._tag = tag
        self._version = version
        self.sha256 = sha256
        self.byte_size = byte_size
        self.manifest = manifest

    def load(self) -> TensorArray:
        """Download (or read from the cache) and return the tensor array.

        Every call first resolves a fresh signed URL from the server (one small request); the
        cache avoids only re-downloading the tensor blob itself, so ``load`` still needs the
        network. The blob bytes are verified against the SHA-256 and size the server reports on
        every read, and the array is checked against the stored manifest when one exists, so a
        corrupt or swapped cache entry is evicted and repaired rather than served. The dtype is
        whatever the publisher stored, ``float64`` or ``complex128``. Read the result as named
        parameter columns, not as a physical tensor: axis 1 follows the manifest ``components``,
        and a complex payload holds delta in its real part and beta in its imaginary part as a
        storage convenience. The archive is loaded without pickle support.

        Raises:
            AtlasError: The digest or size does not match, the array does not match its manifest
                shape or dtype, the archive has no ``tensor`` array, or the stored format is not
                ``npz``.
        """
        if self.manifest is not None and self.manifest.format != "npz":
            raise AtlasError(
                f"Tensor format {self.manifest.format!r} is not supported by this client."
            )
        body = _object(
            self._client.http.get_json(
                f"/api/v1/models/{self._tag}/tensor", {"version": _version_param(self._version)}
            ),
            "tensor",
        )
        url = body.get("url")
        if not isinstance(url, str):
            raise AtlasError("Unexpected tensor response from the server.")
        key = f"{self._client.host}|tensor|{self.sha256}"
        data = self._client.cache.get(key)
        if data is not None and (
            hashlib.sha256(data).hexdigest() != self.sha256 or len(data) != self.byte_size
        ):
            self._client.cache.delete(key)
            data = None
        if data is None:
            data = self._client.http.get_bytes(url)
            if hashlib.sha256(data).hexdigest() != self.sha256 or len(data) != self.byte_size:
                raise AtlasError(
                    "The downloaded tensor does not match its recorded digest or size."
                )
            self._client.cache.put(key, data)
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if "tensor" not in archive.files:
                raise AtlasError("The tensor archive has no 'tensor' array.")
            stored = archive["tensor"]
            stored_dtype = str(stored.dtype)
            if self.manifest is not None and (
                [int(size) for size in stored.shape] != self.manifest.shape
                or stored_dtype != self.manifest.dtype
            ):
                raise AtlasError("The tensor does not match its manifest shape or dtype.")
            array = widen_tensor_dtype(stored)
        if array is None:
            raise AtlasError(
                f"The stored tensor has dtype {stored_dtype}, which this client cannot read; "
                "expected float or complex data."
            )
        return array

    def __repr__(self) -> str:
        return f"TensorHandle(sha256={self.sha256[:12]!r}, byte_size={self.byte_size})"


class Model:
    """One version of an optical model."""

    def __init__(self, client: Client, detail: ModelDetail) -> None:
        self._client = client
        self._detail = detail
        version: ModelVersion = detail.version
        self.tag = detail.tag
        self.title = detail.title
        self.description = detail.description
        self.owner_orcid = detail.owner_orcid
        self.visibility = detail.visibility
        self.is_default = detail.is_default
        self.forked_from = detail.forked_from
        self.versions = list(detail.versions)
        self.has_draft = detail.has_draft
        self.version_number = version.version_number
        self.status = version.status
        self.checksum = version.checksum
        if version.version_number is not None and version.checksum is not None:
            register_model(
                host=client.host,
                tag=detail.tag,
                version=version.version_number,
                checksum=version.checksum,
            )
        self.normalization: Normalization | None = version.normalization
        self.authors: list[Author] = list(version.authors)
        self._selector: VersionSelector = (
            version.version_number if version.version_number is not None else "draft"
        )
        self.datasets = [ModelDatasetLink(client, link) for link in version.datasets]
        tensor = version.tensor
        self.tensor = (
            TensorHandle(
                client, self.tag, self._selector, tensor.sha256, tensor.byte_size, tensor.manifest
            )
            if tensor is not None
            else None
        )

    @property
    def peaks(self) -> pd.DataFrame:
        """The version's peaks, one row per peak in stored order, with the columns in `PEAK_COLUMNS`."""  # noqa: E501
        records = [
            {column: getattr(peak, column) for column in PEAK_COLUMNS}
            for peak in self._detail.version.peaks
        ]
        frame = pd.DataFrame(records, columns=list(PEAK_COLUMNS))
        return frame.sort_values("ordinal", kind="stable").reset_index(drop=True)

    def lineage(self) -> Lineage:
        """Visible ancestors of this version (newest first) and the count the caller cannot see."""
        body = self._client.http.get_json(
            f"/api/v1/models/{self.tag}/lineage", {"version": _version_param(self._selector)}
        )
        return parse_model(
            Lineage, cast("dict[str, object]", _object(body, "lineage")), strict=False
        )

    def normalized(self, dataset: datasets.Dataset | str | None = None) -> pd.DataFrame:
        """Channels of one linked dataset under this version's normalization, computed by the
        server.

        Args:
            dataset: A `Dataset`, an Atlas id or ``d/`` reference, or a dataset UUID. Required
                when the version links several datasets; optional when it links one.

        Returns:
            A DataFrame with columns `CHANNEL_COLUMNS`; missing values are ``NaN``.

        Raises:
            ValidationError: The version has no normalization (``NORMALIZATION_UNDEFINED``),
                ``dataset`` was not given or is not a valid dataset reference, or a channel value
                is present but is neither a finite number nor null (numbers) or a string or null
                (``polarization_id``).
            AtlasError: The channels response is missing a channel array or its arrays do not all
                have the same length.
        """
        params: dict[str, str | int | float | bool] = {"version": _version_param(self._selector)}
        if isinstance(dataset, datasets.Dataset):
            params["dataset"] = dataset.experiment_id
        elif dataset is not None:
            _, selector = datasets.parse_dataset_ref(dataset)
            params["dataset"] = selector
        body = _object(
            self._client.http.get_json(f"/api/v1/models/{self.tag}/normalized", params),
            "normalized",
        )
        channels = body.get("channels")
        if not isinstance(channels, dict):
            raise AtlasError("Unexpected normalized channels from the server.")
        mapping = {
            "energy_ev": "energyEv",
            "od": "od",
            "mass_absorption": "massAbsorption",
            "beta": "beta",
            "polarization_id": "polarizationId",
            "theta_deg": "thetaDeg",
            "phi_deg": "phiDeg",
        }
        arrays: dict[str, list[JsonValue]] = {}
        length: int | None = None
        for column, wire in mapping.items():
            values = channels.get(wire)
            if not isinstance(values, list):
                raise AtlasError("Unexpected normalized channels from the server.")
            if length is None:
                length = len(values)
            elif len(values) != length:
                raise AtlasError("Unexpected normalized channels from the server.")
            arrays[column] = values
        columns: dict[str, list[object]] = {}
        for column, values in arrays.items():
            if column == "polarization_id":
                labels: list[object] = []
                for index, value in enumerate(values):
                    if value is None or isinstance(value, str):
                        labels.append(value)
                    else:
                        raise ValidationError(
                            f"channel polarization_id index {index} must be a string, "  # noqa: E501
                            f"not {value!r}."
                        )
                columns[column] = labels
            else:
                numbers: list[object] = []
                for index, value in enumerate(values):
                    if value is None:
                        numbers.append(np.nan)
                    elif (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and is_finite_number(value)
                    ):
                        numbers.append(float(value))
                    else:
                        raise ValidationError(
                            f"channel {column} index {index} must be a finite number, "  # noqa: E501
                            f"not {value!r}."
                        )
                columns[column] = numbers
        frame = pd.DataFrame(columns)
        return frame[list(CHANNEL_COLUMNS)]

    def cite(self) -> str:
        """BibTeX of this published version (the version's signed authors, parents, and datasets)."""  # noqa: E501
        if self.version_number is None:
            raise AtlasError("Only published versions can be cited.")
        data = self._client.http.get_bytes(
            f"/api/v1/models/{self.tag}/bibtex?version={self.version_number}"
        )
        return data.decode("utf-8")

    def __repr__(self) -> str:
        return (
            f"Model(tag={self.tag!r}, version={self.version_number!r}, status={self.status!r}, "
            f"visibility={self.visibility!r}, title={self.title!r})"
        )

    def fork(self, title: str | None = None) -> publish.ModelBuilder:
        """Start a new model from this version (its datasets, peaks, and normalization).

        The prefill is read from exactly this version and the fork body carries its number, so
        forking version 3 reproduces version 3's content even when a newer version exists.

        Raises:
            ValidationError: This object is a draft; fork a published version instead.
        """
        from xrayatlas import publish as _publish

        if self.version_number is None:
            raise ValidationError("fork a published version, not a draft.")
        return _publish.model(
            [link.experiment_id for link in self.datasets],
            title,
            self.tag,
            from_version=self.version_number,
            client=self._client,
        )

    def publish(self) -> Model:
        """Publish this model's draft as its next version (owner only)."""
        from xrayatlas import publish as _publish

        return _publish.publish(self.tag, client=self._client)

    def set_visibility(self, visibility: Literal["private", "shared", "public"]) -> None:
        """Set visibility (owner only)."""
        from xrayatlas import publish as _publish

        _publish.set_visibility(self.tag, visibility, client=self._client)

    def share(self, with_: Sequence[str]) -> None:
        """Share with ORCID iDs (owner only)."""
        from xrayatlas import publish as _publish

        _publish.share(self.tag, with_, client=self._client)

    def unshare(self, orcid: str) -> None:
        """Remove a share (owner only)."""
        from xrayatlas import publish as _publish

        _publish.unshare(self.tag, orcid, client=self._client)

    def review(
        self,
        dataset: datasets.Dataset | str,
        state: Literal["approved", "rejected"],
        note: str | None = None,
    ) -> None:
        """Approve or reject this model on a dataset you can edit."""
        from xrayatlas import publish as _publish

        _publish.review(self.tag, dataset, state, note, client=self._client)

    def delete(self) -> None:
        """Delete this model if it was never published (owner only)."""
        from xrayatlas import publish as _publish

        _publish.delete(self.tag, client=self._client)


def get(tag: str, version: VersionSelector = "latest", *, client: Client | None = None) -> Model:
    """Load a model.

    Args:
        tag: The model tag (8 characters; ``m/`` prefixes are accepted).
        version: A published version number, ``"latest"`` (default), or ``"draft"`` (owner only).
        client: Client to use; defaults to the shared client.

    Returns:
        The model at that version.

    Raises:
        ValidationError: ``tag`` is not a model tag, or ``version`` is neither ``"latest"``,
            ``"draft"``, nor a positive integer, and no request was sent.
        NotFoundError: The model or version does not exist or is not visible to the caller.
    """
    c = client or default_client()
    if version not in ("latest", "draft") and not (
        isinstance(version, int) and not isinstance(version, bool) and version > 0
    ):
        raise ValidationError("version must be 'latest', 'draft', or a positive integer.")
    clean = clean_tag(tag)
    body = c.http.get_json(f"/api/v1/models/{clean}", {"version": _version_param(version)})
    detail = parse_model(
        ModelDetail, cast("dict[str, object]", _object(body, "model")), strict=False
    )
    return Model(c, detail)


def _pages(c: Client, scope: str, page_size: int) -> Iterator[ModelSummary]:
    cursor: str | None = None
    while True:
        params: dict[str, str | int | float | bool] = {"scope": scope, "limit": page_size}
        if cursor is not None:
            params["cursor"] = cursor
        body = _object(c.http.get_json("/api/v1/models", params), "models")
        items = body.get("items")
        if not isinstance(items, list):
            raise AtlasError("Unexpected models response from the server.")
        for item in items:
            if isinstance(item, dict):
                yield parse_model(ModelSummary, cast("dict[str, object]", item), strict=False)
        next_cursor = body.get("nextCursor")
        if not isinstance(next_cursor, str):
            return
        cursor = next_cursor


def mine(*, page_size: int = 50, client: Client | None = None) -> Iterator[ModelSummary]:
    """Iterate the models you own, newest first (needs a token).

    Args:
        page_size: Summaries per request, 1 to 200.

    Raises:
        ValidationError: ``page_size`` is outside 1 to 200, and no request was sent.
    """
    if not 1 <= page_size <= 200:
        raise ValidationError("page_size must be 1 to 200.")
    return _pages(client or default_client(), "mine", page_size)


def shared_with_me(*, page_size: int = 50, client: Client | None = None) -> Iterator[ModelSummary]:
    """Iterate the published models other people shared with you (needs a token).

    Args:
        page_size: Summaries per request, 1 to 200.

    Raises:
        ValidationError: ``page_size`` is outside 1 to 200, and no request was sent.
    """
    if not 1 <= page_size <= 200:
        raise ValidationError("page_size must be 1 to 200.")
    return _pages(client or default_client(), "shared", page_size)


__all__ = [
    "CHANNEL_COLUMNS",
    "Model",
    "ModelDatasetLink",
    "PEAK_COLUMNS",
    "TensorHandle",
    "get",
    "mine",
    "shared_with_me",
]
