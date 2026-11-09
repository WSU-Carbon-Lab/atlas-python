"""Building, publishing, sharing, and reviewing optical models.

`model()` returns a `ModelBuilder` context manager: it reads the source model on enter and writes
only on a clean exit (fork, then draft, then tensor); publishing is always a separate explicit call.
All input is validated before any request. Publishing is never retried automatically. The functions
here take a model tag and are also exposed as methods on `xrayatlas.models.Model`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from types import TracebackType
from typing import Literal, cast

import pandas as pd

from xrayatlas import datasets as _datasets
from xrayatlas import models as _models
from xrayatlas._http import JsonValue
from xrayatlas.client import Client, default_client
from xrayatlas.errors import AtlasError, ValidationError
from xrayatlas.tensor import Tensor
from xrayatlas.types import DraftContent, Normalization, Peak
from xrayatlas.validate import clean_tag, parse_model, validate_peaks

_ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
Visibility = Literal["private", "shared", "public"]
ReviewDecision = Literal["approved", "rejected"]
_PEAK_FRAME_COLUMNS = (
    "center_ev",
    "kind",
    "transition",
    "amplitude",
    "width",
    "shape",
    "center_err",
    "amplitude_err",
    "width_err",
    "dataset",
    "polarization",
    "fit_tool",
    "fit_version",
    "fit_code_hash",
)


def _object(value: JsonValue, what: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise AtlasError(f"Unexpected {what} response from the server.")
    return value


def _peaks_frame(peaks: Sequence[Peak]) -> pd.DataFrame:
    """Editable peak table: validate_peaks columns, with params expanded into columns."""
    rows: list[dict[str, object]] = []
    for peak in peaks:
        params = peak.params or {}
        for key in params:
            if key in _PEAK_FRAME_COLUMNS:
                raise ValidationError(f"peak parameter {key!r} collides with a peak column.")
        row: dict[str, object] = dict(params)
        row["center_ev"] = peak.center_ev
        row["kind"] = peak.kind
        row["transition"] = peak.transition
        row["amplitude"] = peak.amplitude
        row["width"] = peak.width
        row["shape"] = peak.shape
        row["center_err"] = peak.center_err
        row["amplitude_err"] = peak.amplitude_err
        row["width_err"] = peak.width_err
        row["dataset"] = peak.experiment_id
        row["polarization"] = peak.polarization_id
        if peak.fit_provenance is not None:
            row["fit_tool"] = peak.fit_provenance.tool
            row["fit_version"] = peak.fit_provenance.version
            row["fit_code_hash"] = peak.fit_provenance.code_hash
        rows.append(row)
    return pd.DataFrame(rows, columns=list(_PEAK_FRAME_COLUMNS)) if not rows else pd.DataFrame(rows)


class ModelBuilder:
    """Context manager that builds a new model from a source model.

    Attributes you may set inside the ``with`` block: ``normalization`` (a `Normalization` or
    ``None``), ``peaks`` (a DataFrame, see `xrayatlas.validate.validate_peaks`), ``description``,
    ``tensor`` (a `Tensor`), and ``title``. After a clean exit ``tag`` and ``model`` identify the
    new private draft.
    """

    def __init__(
        self,
        client: Client,
        source: _models.Model,
        links: list[tuple[str, str]],
        title: str | None,
        *,
        from_version: int | None = None,
    ) -> None:
        self._client = client
        self._source = source
        self._links = links
        self._from_version = from_version
        self._saved = False
        self.title = title
        self.normalization: Normalization | None = source.normalization
        self.peaks: pd.DataFrame = _peaks_frame(
            [
                parse_model(
                    Peak,
                    {
                        "center_ev": r.center_ev,
                        "kind": r.kind,
                        "transition": r.transition,
                        "amplitude": r.amplitude,
                        "width": r.width,
                        "shape": r.shape,
                        "center_err": r.center_err,
                        "amplitude_err": r.amplitude_err,
                        "width_err": r.width_err,
                        "experiment_id": r.experiment_id,
                        "polarization_id": r.polarization_id,
                        "params": r.params,
                        "fit_provenance": r.fit_provenance,
                    },
                    strict=False,
                )
                for r in source._detail.version.peaks  # noqa: SLF001
            ]
        )
        self.description: str | None = source.description
        self.tensor: Tensor | None = None
        self.tag: str | None = None
        self.model: _models.Model | None = None

    def __enter__(self) -> ModelBuilder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is None:
            self.save()
        return False

    def save(self) -> _models.Model:
        """Validate, fork the source, replace the draft, and upload the tensor. Does not publish.

        Idempotent: when the draft (and tensor, if any) were already saved successfully, the same
        draft is returned without writing again. When an earlier call failed between the fork and
        the end, the forked tag is reused instead of forking a second time.

        Raises:
            ValidationError: Any input is invalid; nothing was sent.
            ApiError: The server refused a step.
        """
        if self._saved:
            model = self.model
            assert model is not None
            return model
        peaks = validate_peaks(self.peaks) if len(self.peaks) > 0 else []
        fields: dict[str, object] = {
            "normalization": self.normalization,
            "peaks": peaks,
            "datasets": [
                {"experiment_id": experiment_id, "role": role}
                for experiment_id, role in self._links
            ],
        }
        if self.title is not None:
            fields["title"] = self.title
        if self.description is not None:
            fields["description"] = self.description
        content = parse_model(DraftContent, fields)
        http = self._client.http
        if self.tag is None:
            fork_body: dict[str, JsonValue] = {"title": self.title} if self.title else {}
            if self._from_version is not None:
                fork_body["fromVersion"] = self._from_version
            forked = _object(
                http.send_json("POST", f"/api/v1/models/{self._source.tag}/fork", fork_body),
                "fork",
            )
            tag = forked.get("tag")
            if not isinstance(tag, str):
                raise AtlasError("Unexpected fork response from the server.")
            self.tag = tag
        draft_tag = self.tag
        http.send_json(
            "PUT", f"/api/v1/models/{draft_tag}/draft", cast("JsonValue", content.to_wire())
        )
        if self.tensor is not None:
            _upload_tensor(self._client, draft_tag, self.tensor)
        self.model = _models.get(draft_tag, "draft", client=self._client)
        self._saved = True
        return self.model

    def publish(self) -> _models.Model:
        """Save (if not saved yet) and publish the draft as the next immutable version.

        The draft is saved first and `save` is idempotent, so an in-block failure is re-raised by
        `save` instead of publishing partial server content.
        """
        model = self.save()
        return publish(model.tag, client=self._client)


def _upload_tensor(client: Client, tag: str, tensor: Tensor) -> None:
    http = client.http
    grant = _object(
        http.send_json(
            "POST",
            f"/api/v1/models/{tag}/tensor/upload-url",
            {"sizeBytes": tensor.size_bytes, "sha256": tensor.sha256},
        ),
        "tensor upload",
    )
    upload_url = grant.get("uploadUrl")
    storage_path = grant.get("storagePath")
    if not isinstance(upload_url, str) or not isinstance(storage_path, str):
        raise AtlasError("Unexpected tensor upload response from the server.")
    http.put_bytes(upload_url, tensor.blob, "application/octet-stream")
    http.send_json(
        "POST",
        f"/api/v1/models/{tag}/tensor/commit",
        {
            "storagePath": storage_path,
            "sizeBytes": tensor.size_bytes,
            "sha256": tensor.sha256,
            "manifest": cast("JsonValue", tensor.manifest.model_dump(by_alias=True, mode="json")),
        },
    )


def _links(client: Client, refs: Sequence[_datasets.Dataset | str]) -> list[tuple[str, str]]:
    """Dataset links for refs: every reference is parsed and validated before any is resolved.

    A `Dataset` contributes its experiment id; a valid UUID string resolves locally; an Atlas id
    is resolved with a one-row export probe in the second pass, so a malformed reference raises
    `ValidationError` with zero HTTP calls.

    Raises:
        ValidationError: No ref was given or a reference is malformed.
        NotFoundError: An Atlas id has no dataset.
    """
    if not refs:
        raise ValidationError("at least one dataset is required.")
    parsed: list[tuple[_datasets.Dataset | None, str, str]] = []
    for ref in refs:
        if isinstance(ref, _datasets.Dataset):
            parsed.append((ref, "datasetId", ref.experiment_id))
        else:
            key, value = _datasets.parse_dataset_ref(ref)
            parsed.append((None, key, value))
    resolved: list[str] = []
    for dataset, key, value in parsed:
        if dataset is not None:
            experiment_id = dataset.experiment_id
        elif key == "datasetId":
            experiment_id = value
        else:
            experiment_id = _datasets.resolve_experiment_id(value, client=client)
        resolved.append(experiment_id)
    return [
        (experiment_id, "primary" if position == 0 else "additional")
        for position, experiment_id in enumerate(resolved)
    ]


def model(
    datasets: Sequence[_datasets.Dataset | str],
    title: str | None = None,
    from_model: str | None = None,
    from_version: int | None = None,
    *,
    client: Client | None = None,
) -> ModelBuilder:
    """Start a new model on one or more datasets.

    The source is ``from_model`` when given, otherwise the first dataset's default model. With
    ``from_version`` set, the source is that version of ``from_model`` and the fork body carries
    ``fromVersion``; without it, the latest published version is read. Reading the source happens
    now; nothing is written until the ``with`` block exits cleanly.

    Args:
        datasets: Datasets (objects, Atlas ids, or UUIDs) the model links; the first is primary.
        title: Title of the new model; the server derives one from the source when omitted.
        from_model: Tag of the model to fork instead of the default model.
        from_version: Published version of ``from_model`` to prefill from and fork from; the
            latest published version is read when ``None``.
        client: Client to use; defaults to the shared client.

    Returns:
        The builder, usable as a context manager.

    Raises:
        ValidationError: No dataset was given, a reference is malformed, ``from_model`` is not a
            model tag, or ``from_version`` is not a positive integer.
        AtlasError: The first dataset has no default model.
    """
    c = client or default_client()
    links = _links(c, datasets)
    if from_model is not None:
        source = _models.get(
            from_model, from_version if from_version is not None else "latest", client=c
        )
    else:
        listed = _object(
            c.http.get_json("/api/v1/models", {"dataset": links[0][0], "scope": "public"}), "models"
        )
        items = listed.get("items")
        default_tag: str | None = None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("isDefault") is True:
                    tag_value = item.get("tag")
                    if isinstance(tag_value, str):
                        default_tag = tag_value
                        break
        if default_tag is None:
            raise AtlasError("The first dataset has no default model; pass from_model=<tag>.")
        source = _models.get(default_tag, client=c)
    return ModelBuilder(c, source, links, title, from_version=from_version)


def publish(tag: str, *, client: Client | None = None) -> _models.Model:
    """Publish a model's draft as its next immutable version (owner only).

    Never retried: a 409 (no draft, or a concurrent publish won) raises `ConflictError`.

    Returns:
        The model at the new version. The response itself is relied on for the minted version
        number: when reading the model back afterwards fails, the raised error names the version
        that was published so a success is never lost.

    Raises:
        AtlasError: The publish succeeded but reading the model back failed (chained from the
            original error).
        ApiError: The server refused the publish.
    """
    c = client or default_client()
    clean = clean_tag(tag)
    response = _object(c.http.send_json("POST", f"/api/v1/models/{clean}/publish"), "publish")
    version_number = response.get("versionNumber")
    try:
        return _models.get(clean, "latest", client=c)
    except AtlasError as error:
        if isinstance(version_number, int) and not isinstance(version_number, bool):
            raise AtlasError(
                f"Model {clean} was published as version {version_number}, but reading it back "
                f"failed: {error}"
            ) from error
        raise


def set_visibility(tag: str, visibility: Visibility, *, client: Client | None = None) -> None:
    """Set a model's visibility to ``private``, ``shared``, or ``public`` (owner only).

    Raises:
        ValidationError: ``visibility`` is not one of the three values or the tag is malformed;
            nothing is sent.
        ForbiddenError: The server refuses ``public`` because an ancestor model is not public.
    """
    if visibility not in ("private", "shared", "public"):
        raise ValidationError("visibility must be 'private', 'shared', or 'public'.")
    c = client or default_client()
    c.http.send_json(
        "PUT", f"/api/v1/models/{clean_tag(tag)}/visibility", {"visibility": visibility}
    )


def _check_orcid(orcid: str) -> str:
    cleaned = orcid.strip().upper()
    if not _ORCID.match(cleaned):
        raise ValidationError(f"{orcid!r} is not an ORCID iD such as 0000-0002-1825-0097.")
    return cleaned


def share(tag: str, orcids: Sequence[str], *, client: Client | None = None) -> None:
    """Share a model with ORCID iDs; a private model becomes ``shared`` (owner only).

    The tag and every ORCID iD are validated before the first request.

    Raises:
        ValidationError: The tag is malformed or an ORCID iD is invalid; nothing is sent.
    """
    cleaned = [_check_orcid(o) for o in orcids]
    c = client or default_client()
    for orcid in cleaned:
        c.http.send_json("PUT", f"/api/v1/models/{clean_tag(tag)}/shares", {"orcid": orcid})


def unshare(tag: str, orcid: str, *, client: Client | None = None) -> None:
    """Remove a share (owner only). Idempotent.

    Raises:
        ValidationError: The tag or the ORCID iD is malformed; nothing is sent.
    """
    cleaned = _check_orcid(orcid)
    c = client or default_client()
    c.http.send_json("DELETE", f"/api/v1/models/{clean_tag(tag)}/shares", {"orcid": cleaned})


def review(
    tag: str,
    dataset: _datasets.Dataset | str,
    state: ReviewDecision,
    note: str | None = None,
    *,
    client: Client | None = None,
) -> None:
    """Approve or reject a model on a dataset you can edit. A rejection never hides the model.

    Args:
        tag: The model tag.
        dataset: The dataset (object, Atlas id, or UUID) the verdict applies to.
        state: ``approved`` or ``rejected``.
        note: A note of 1 to 1000 characters sent with the verdict; ``None`` sends none.

    Raises:
        ValidationError: ``state`` is not ``approved`` or ``rejected``, ``note`` is not ``None``
            and not 1 to 1000 characters (empty or whitespace-only counts as empty), the tag is
            malformed, or the dataset reference is malformed; nothing is sent.
        ForbiddenError: You cannot edit that dataset.
    """
    if state not in ("approved", "rejected"):
        raise ValidationError("state must be 'approved' or 'rejected'.")
    if note is not None:
        trimmed = note.strip()
        if not 1 <= len(trimmed) <= 1000:
            raise ValidationError("note must be 1 to 1000 characters.")
    clean = clean_tag(tag)
    c = client or default_client()
    experiment_id = (
        dataset.experiment_id
        if isinstance(dataset, _datasets.Dataset)
        else _datasets.resolve_experiment_id(dataset, client=c)
    )
    body: dict[str, JsonValue] = {"experimentId": experiment_id, "state": state}
    if note is not None:
        body["note"] = note.strip()
    c.http.send_json("POST", f"/api/v1/models/{clean}/review", body)


def delete(tag: str, *, client: Client | None = None) -> None:
    """Delete a model that was never published (owner only); a published model is 409.

    Raises:
        ValidationError: The tag is malformed; nothing is sent.
    """
    c = client or default_client()
    c.http.send_json("DELETE", f"/api/v1/models/{clean_tag(tag)}")


__all__ = [
    "ModelBuilder",
    "delete",
    "model",
    "publish",
    "review",
    "set_visibility",
    "share",
    "unshare",
]
