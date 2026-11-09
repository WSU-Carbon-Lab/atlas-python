"""Typed models of optical model content and server responses.

Fields are snake_case in Python and camelCase on the wire. Request-side models (``_Req``) are
strict: they reject unknown fields, reject wrong Python types (for example a string or a bool where
a number is expected), and enforce the same limits as the server (finite numbers, ordered windows,
size caps, UUID identifiers) so mistakes fail before any HTTP call. Response-side models (``_Wire``)
ignore unknown fields so older clients keep working. Server responses are parsed with
``parse_model(cls, data, strict=False)``, which relaxes type coercion for that call only. Use
`xrayatlas.validate.parse_model` to turn a pydantic failure into `ValidationError`. This module does
no I/O.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

PEAK_KINDS = ("pi-star", "sigma-star", "other")
TENSOR_FRAMES = ("lab", "molecular", "principal")
TENSOR_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz", "delta", "beta")
VISIBILITIES = ("private", "shared", "public")

PeakKind = Literal["pi-star", "sigma-star", "other"]
Visibility = Literal["private", "shared", "public"]
ReviewState = Literal["unreviewed", "approved", "rejected"]

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(allow_inf_nan=False, ge=0)]


def _as_pair(value: Any) -> Any:
    """Coerce a 2-element list, tuple, or 1-D numpy array to a tuple.

    Anything else is returned unchanged so pydantic rejects it.
    """
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(value)
    return value


Window = Annotated[tuple[FiniteFloat, FiniteFloat], BeforeValidator(_as_pair)]


class _Wire(BaseModel):
    """Base model for responses: camelCase aliases, either name accepted, unknown fields ignored."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class _Req(BaseModel):
    """Base model for requests: camelCase aliases, strict types, and unknown fields rejected."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


def _check_uuid(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("must be a UUID") from exc
    return value


def _check_pair(pre: Window, post: Window, label: str) -> None:
    if pre[0] >= pre[1]:
        raise ValueError(f"{label}pre window must have low < high")
    if post[0] >= post[1]:
        raise ValueError(f"{label}post window must have low < high")
    if pre[1] > post[0]:
        raise ValueError(f"{label}pre window must end at or before the post window starts")


class WindowPair(_Req):
    """Pre-edge and post-edge energy windows in eV, each ``(low, high)``."""

    pre: Window
    post: Window

    @model_validator(mode="after")
    def _ordered(self) -> WindowPair:
        _check_pair(self.pre, self.post, "")
        return self


class Normalization(_Req):
    """Normalization of a model.

    ``scope="unified"`` (the default) applies one ``pre`` and ``post`` window to every channel.
    ``scope="per_channel"`` gives ``od``, ``massabsorption``, and ``beta`` each its own
    `WindowPair`. Windows are in eV, each requires ``low < high``, and each pre window must end at
    or before its post window starts.

    Call `to_wire` to get the JSON object the server expects, with defaults such as
    ``schemaVersion`` and ``scope`` included.
    """

    schema_version: Literal[1] = 1
    scope: Literal["unified", "per_channel"] = "unified"
    pre: Window | None = None
    post: Window | None = None
    od: WindowPair | None = None
    massabsorption: WindowPair | None = None
    beta: WindowPair | None = None

    @model_validator(mode="after")
    def _shape(self) -> Normalization:
        if self.scope == "unified":
            if self.pre is None or self.post is None:
                raise ValueError("unified normalization needs both pre and post")
            if any(x is not None for x in (self.od, self.massabsorption, self.beta)):
                raise ValueError("unified normalization must not set od, massabsorption, or beta")
            _check_pair(self.pre, self.post, "")
        else:
            if self.pre is not None or self.post is not None:
                raise ValueError("per_channel normalization must not set pre or post")
            if self.od is None or self.massabsorption is None or self.beta is None:
                raise ValueError("per_channel normalization needs od, massabsorption, and beta")
        return self

    def to_wire(self) -> dict[str, Any]:
        """JSON object for this normalization, camelCase, with unset members dropped.

        Returns:
            The object with every non-None member, including schema defaults, in JSON-compatible
            form (windows as two-element lists).
        """
        return self.model_dump(by_alias=True, mode="json", exclude_none=True)


class FitProvenance(_Req):
    """Which tool produced a peak fit."""

    tool: Annotated[str, Field(min_length=1, max_length=100)]
    version: Annotated[str, Field(max_length=100)] | None = None
    code_hash: Annotated[str, Field(max_length=100)] | None = None


ParamValue = FiniteFloat | Annotated[str, Field(max_length=200)] | bool


class Peak(_Req):
    """One assigned peak from the user's own fitting code. Energies are eV.

    ``experiment_id`` and ``polarization_id`` must be UUID strings when set.
    """

    ordinal: int | None = None
    center_ev: FiniteFloat
    kind: PeakKind = "other"
    transition: Annotated[str, Field(max_length=200)] | None = None
    amplitude: FiniteFloat | None = None
    width: NonNegativeFloat | None = None
    shape: Annotated[str, Field(max_length=60)] | None = None
    center_err: NonNegativeFloat | None = None
    amplitude_err: NonNegativeFloat | None = None
    width_err: NonNegativeFloat | None = None
    experiment_id: str | None = None
    polarization_id: str | None = None
    params: dict[str, ParamValue] | None = None
    fit_provenance: FitProvenance | None = None

    @field_validator("experiment_id", "polarization_id")
    @classmethod
    def _uuid(cls, value: str | None) -> str | None:
        return _check_uuid(value)

    @field_validator("params")
    @classmethod
    def _params_bounds(cls, value: dict[str, ParamValue] | None) -> dict[str, ParamValue] | None:
        if value is None:
            return None
        if len(value) > 50:
            raise ValueError("params may hold at most 50 keys")
        for key in value:
            if not 1 <= len(key) <= 60:
                raise ValueError("params keys must be 1 to 60 characters")
        return value

    def to_wire(self) -> dict[str, Any]:
        """JSON object for this peak, camelCase, with ``ordinal`` and unset members dropped.

        Returns:
            The object with every non-None member except ``ordinal``, including the ``kind``
            default.
        """
        return self.model_dump(by_alias=True, mode="json", exclude_none=True, exclude={"ordinal"})


class EnergyGrid(_Req):
    """Bounds and sample count of a tensor's energy axis."""

    min: FiniteFloat
    max: FiniteFloat
    count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _ordered(self) -> EnergyGrid:
        if self.min >= self.max:
            raise ValueError("energyEv.min must be less than max")
        return self


class TensorManifest(_Req):
    """Manifest describing a stored parameter blob (columns plus energy axis)."""

    schema_version: Literal[1] = 1
    format: Literal["npz", "parquet"]
    energy_ev: EnergyGrid
    components: Annotated[
        list[Literal["xx", "yy", "zz", "xy", "xz", "yz", "delta", "beta"]],
        Field(min_length=1),
    ]
    frame: Literal["lab", "molecular", "principal"]
    units: Annotated[str, Field(min_length=1, max_length=40)]
    dtype: Annotated[str, Field(min_length=1, max_length=20)]
    shape: Annotated[list[Annotated[int, Field(gt=0)]], Field(min_length=1, max_length=6)]


class DatasetLinkIn(_Req):
    """Link from a draft to one dataset (experiment UUID)."""

    experiment_id: str
    role: Annotated[str, Field(min_length=1, max_length=60)] = "primary"

    @field_validator("experiment_id")
    @classmethod
    def _uuid(cls, value: str) -> str:
        checked = _check_uuid(value)
        assert checked is not None
        return checked


class DraftContent(_Req):
    """Fields of a draft to replace. Fields never set are left unchanged on the server.

    Use `to_wire` as the JSON body. It sends only the fields that were set. A field set explicitly
    to ``None`` (other than ``title``) is sent as ``null`` and clears that field on the server.
    """

    title: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    description: Annotated[str, Field(max_length=5000)] | None = None
    normalization: Normalization | None = None
    peaks: Annotated[list[Peak], Field(max_length=2000)] | None = None
    datasets: Annotated[list[DatasetLinkIn], Field(min_length=1, max_length=50)] | None = None

    @field_validator("datasets")
    @classmethod
    def _unique_datasets(cls, value: list[DatasetLinkIn] | None) -> list[DatasetLinkIn] | None:
        if value is not None and len({d.experiment_id for d in value}) != len(value):
            raise ValueError("datasets must not repeat an experimentId")
        return value

    def to_wire(self) -> dict[str, Any]:
        """JSON body for the draft route.

        Only fields that were set are included, in camelCase. A non-None ``normalization`` and each
        peak are dumped with `Normalization.to_wire` and `Peak.to_wire`, so the schema defaults the
        server requires are present and nested ``None`` members are dropped. An explicitly set
        top-level ``None`` (for example ``description=None``) stays as ``null``.

        Returns:
            A JSON-compatible dict ready to send as the request body.
        """
        body: dict[str, Any] = self.model_dump(by_alias=True, mode="json", exclude_unset=True)
        if self.normalization is not None:
            body["normalization"] = self.normalization.to_wire()
        if self.peaks is not None:
            body["peaks"] = [peak.to_wire() for peak in self.peaks]
        return body


class ForkedFrom(_Wire):
    """Parent model and version a model was forked from."""

    tag: str
    version_number: int


class ModelSummary(_Wire):
    """One model in a list."""

    tag: str
    title: str
    owner_orcid: str
    visibility: Visibility
    latest_version: int | None = None
    review_state: ReviewState | None = None
    is_default: bool = False
    forked_from: ForkedFrom | None = None


class DatasetLinkOut(_Wire):
    """A dataset link of a version, pinned to a dataset version such as ``1.0.2``."""

    experiment_id: str
    atlas_dataset_id: str | None = None
    pinned: str
    role: str


class Author(_Wire):
    """An author of a model version."""

    orcid: str
    name: str | None = None
    role: Literal["owner", "coauthor"]
    signed: bool


class TensorInfo(_Wire):
    """Digest, size, and manifest of a version's tensor."""

    sha256: str
    byte_size: int
    manifest: TensorManifest | None = None


class ModelVersion(_Wire):
    """One version of a model."""

    version_number: int | None = None
    status: Literal["draft", "published"]
    normalization: Normalization | None = None
    peaks: list[Peak] = Field(default_factory=list)
    datasets: list[DatasetLinkOut] = Field(default_factory=list)
    tensor: TensorInfo | None = None
    authors: list[Author] = Field(default_factory=list)
    checksum: str | None = None
    published_at: str | None = None


class ModelDetail(ModelSummary):
    """A model with one selected version."""

    description: str | None = None
    version: ModelVersion
    versions: list[int] = Field(default_factory=list)
    has_draft: bool = False


class LineageNode(_Wire):
    """One visible ancestor of a version."""

    tag: str
    version_number: int
    owner_orcid: str
    title: str
    kind: Literal["fork", "merge"]


class Lineage(_Wire):
    """Visible ancestors, newest first, and the count the caller cannot see."""

    ancestors: list[LineageNode] = Field(default_factory=list)
    hidden_ancestors: int = 0
