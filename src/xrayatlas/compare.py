"""Compare optical models by aligned peaks and normalized-channel differences.

Pure local computation over already-fetched `Model` objects. The only network access is
`Model.normalized`, which is computed by the server; this module never normalizes, fits, or
interpolates. Peaks group under the single-linkage rule described in `Comparison.peaks`, and
spectra align on exact keys, holding ``NaN`` wherever a model lacks a value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from xrayatlas.errors import ValidationError
from xrayatlas.models import Model

if TYPE_CHECKING:
    from xrayatlas.datasets import Dataset

__all__ = ["Comparison", "compare"]

Channel = Literal["mass_absorption", "od", "beta"]
_CHANNELS: tuple[Channel, ...] = ("mass_absorption", "od", "beta")
_RESERVED_COLUMNS = frozenset({"center_ev", "count", "energy_ev", "theta_deg", "phi_deg"})
_Entry = tuple[float, int]


def _labels(models: Sequence[Model]) -> list[str]:
    """Unique ``tag#version`` column labels; ``#2``, ``#3``, ... appended per repeated label.

    The suffix loop keeps the labels unique even for a tag that itself contains ``#``, so the
    result never collides with an earlier label.
    """
    seen: dict[str, int] = {}
    labels: list[str] = []
    for model in models:
        version = model.version_number if model.version_number is not None else "draft"
        base = f"{model.tag}#{version}"
        occurrence = seen.get(base, 0) + 1
        candidate = base if occurrence == 1 else f"{base}#{occurrence}"
        while candidate in labels:
            occurrence += 1
            candidate = f"{base}#{occurrence}"
        seen[base] = occurrence
        labels.append(candidate)
    return labels


def _finite_centers(model: Model) -> list[float]:
    """The model's peak centers that parse as finite eV values, in stored peak order.

    Raises:
        ValidationError: A ``center_ev`` value is not numeric. Numeric non-finite centers
            (``NaN`` or infinity) are dropped instead of raised.
    """
    try:
        numbers = pd.to_numeric(model.peaks["center_ev"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Model {model.tag} has a non-numeric center_ev value: {exc}"
        ) from exc
    return [float(value) for value in numbers if math.isfinite(float(value))]


def _groups(entries: Sequence[_Entry], tolerance_ev: float) -> list[list[_Entry]]:
    """Greedily chain center-sorted entries into groups under the single-linkage tolerance."""
    groups: list[list[_Entry]] = []
    current: list[_Entry] = []
    widest = 0.0
    for center, model_index in sorted(entries, key=lambda entry: entry[0]):
        if current and center - widest <= tolerance_ev:
            current.append((center, model_index))
            widest = max(widest, center)
            continue
        if current:
            groups.append(current)
        current = [(center, model_index)]
        widest = center
    if current:
        groups.append(current)
    return groups


def _aligned_peaks(
    models: Sequence[Model], labels: Sequence[str], tolerance_ev: float
) -> pd.DataFrame:
    """One row per aligned peak group with one center column per model label."""
    entries: list[_Entry] = []
    for model_index, model in enumerate(models):
        entries.extend((center, model_index) for center in _finite_centers(model))
    rows: list[dict[str, float | int]] = []
    for group in _groups(entries, tolerance_ev):
        centers = [center for center, _ in group]
        mean = sum(centers) / len(centers)
        nearest: dict[int, tuple[float, float]] = {}
        for center, model_index in group:
            candidate = (abs(center - mean), center)
            previous = nearest.get(model_index)
            if previous is None or candidate < previous:
                nearest[model_index] = candidate
        winners = sorted(nearest)
        selected = [nearest[index][1] for index in winners]
        row: dict[str, float | int] = {
            "center_ev": sum(selected) / len(selected),
            "count": len(selected),
        }
        for model_index, center in zip(winners, selected, strict=True):
            row[labels[model_index]] = center
        rows.append(row)
    columns = ["center_ev", "count", *labels]
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column != "count":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["count"] = frame["count"].astype(int)
    return frame.sort_values("center_ev", kind="stable").reset_index(drop=True)


def _tolerance(value: object) -> float:
    """``value`` as a positive finite float, or raise `ValidationError`."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValidationError("tolerance_ev must be a positive finite number of eV.")
    return float(value)


def _packed(frame: pd.DataFrame, keys: Sequence[str], channel: str, label: str) -> pd.DataFrame:
    """Numeric key columns plus ``channel`` renamed to ``label``; absent columns become ``NaN``.

    Raises:
        ValidationError: A present key or channel value is not numeric, or the frame holds more
            than one row for the same key.
    """
    missing = pd.Series(np.nan, index=frame.index, dtype=float)
    values: dict[str, pd.Series[float]] = {}
    for column in (*keys, channel):
        if column not in frame.columns:
            values[column] = missing
            continue
        try:
            values[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label} has a non-numeric {column} value: {exc}") from exc
    packed = pd.DataFrame(values)[[*keys, channel]].rename(columns={channel: label})
    duplicated = packed.duplicated(subset=list(keys), keep=False)
    if duplicated.any():
        sample = packed.loc[duplicated, list(keys)].iloc[0]
        key_text = ", ".join(f"{key}={value!r}" for key, value in zip(keys, sample, strict=True))
        raise ValidationError(
            f"{label} has more than one row at the same key ({key_text}); compare needs one row "
            "per key per model."
        )
    return packed


def _has_geometry(frames: Sequence[pd.DataFrame]) -> bool:
    """Whether any frame carries a finite ``theta_deg`` or ``phi_deg`` value.

    Raises:
        ValidationError: A present geometry column holds a value that is not numeric, or an
            infinite one. ``NaN`` and ``None`` mean the geometry is absent.
    """
    found = False
    for frame in frames:
        for column in ("theta_deg", "phi_deg"):
            if column not in frame.columns:
                continue
            try:
                values = pd.to_numeric(frame[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{column} must hold numeric angles: {exc}") from exc
            if np.isinf(values.to_numpy(dtype=float)).any():
                raise ValidationError(f"{column} must hold finite angles, not infinity.")
            if np.isfinite(values.to_numpy(dtype=float)).any():
                found = True
    return found


class Comparison:
    """Peaks and normalized channels of several models on shared, aligned axes.

    Create with `compare`; the models and tolerance are fixed at construction, and `spectra` is
    safe to call more than once. Direct construction validates the same tolerance and requires one
    unique label per model that is not already a frame column name: ``center_ev``, ``count``,
    ``energy_ev``, ``theta_deg``, and ``phi_deg`` are reserved, and a label must not equal another
    model's derived ``{label}_minus_{first_label}`` column. Omit ``labels`` to derive
    ``tag#version`` labels.

    Raises:
        ValidationError: ``models`` is empty or holds a non-`Model` entry, ``labels`` is not one
            unique string per model or collides with a frame column name, or ``tolerance_ev`` is
            not a positive finite number of eV.
    """

    def __init__(
        self,
        models: Sequence[Model],
        labels: Sequence[str] | None = None,
        tolerance_ev: float = 0.2,
    ) -> None:
        entries = tuple(models)
        if not entries or not all(isinstance(entry, Model) for entry in entries):
            raise ValidationError("models must be one or more Model objects.")
        resolved = tuple(labels) if labels is not None else tuple(_labels(entries))
        derived = {f"{label}_minus_{resolved[0]}" for label in resolved[1:]} if resolved else set()
        if (
            len(resolved) != len(entries)
            or not all(isinstance(label, str) for label in resolved)
            or len(set(resolved)) != len(resolved)
            or set(resolved) & (_RESERVED_COLUMNS | derived)
        ):
            raise ValidationError(
                "labels must be one unique string per model, and must not collide with a frame "
                "column."
            )
        self._models = entries
        self._labels = resolved
        self._tolerance_ev = _tolerance(tolerance_ev)
        self._peaks: pd.DataFrame | None = None

    @property
    def peaks(self) -> pd.DataFrame:
        """One row per aligned peak group, sorted by group center ascending.

        Centers chain into one group while each next center is at most the tolerance past the
        group's widest center. Each model contributes at most one peak per group: the one nearest
        the group's mean center, ties to the lower center. Columns are `center_ev` (mean of the
        contributing peaks), `count` (how many models contributed), then one center column per
        model labeled ``tag#version`` (``tag#draft`` for an unpublished version). A model with no
        peak in a group holds ``NaN``. Repeated labels gain a ``#2``, ``#3``, ... suffix
        (``tag#1#2`` for a second read of ``tag#1``). No peaks at all still yields those columns
        with zero rows. The frame is a copy: editing it does not change later reads.
        """
        if self._peaks is None:
            self._peaks = _aligned_peaks(self._models, self._labels, self._tolerance_ev)
        return self._peaks.copy()

    def spectra(
        self,
        dataset: Dataset | str | None = None,
        *,
        channel: Channel = "mass_absorption",
    ) -> pd.DataFrame:
        """Normalized channels of every model on one shared axis, plus differences from the first.

        Args:
            dataset: A `Dataset`, an Atlas id, or a UUID passed through to `Model.normalized`
                (server-side normalization; never recomputed here). ``None`` lets each model use
                its single linked dataset.
            channel: The normalized channel to compare: ``mass_absorption`` (default), ``od``, or
                ``beta``.

        Returns:
            A DataFrame holding, in order: the alignment keys (`theta_deg`, `phi_deg`, and
            `energy_ev`; the geometry keys only when at least one frame has a finite ``theta_deg``
            or ``phi_deg`` value, energy alone otherwise), one channel column per model label as
            in `peaks`, then one ``{label}_minus_{first_label}`` difference column for every model
            after the first. Rows sort by the keys exactly like `Dataset.spectra` (missing
            geometry last). A key one model lacks holds ``NaN`` in that model's column, and
            difference columns never use filled or interpolated values: a model whose geometry
            labels differ from every other model's yields an all-``NaN`` difference column rather
            than a cross-geometry subtraction. Energy grids must agree exactly: two dataset
            versions whose sample energies differ even in the last bit produce disjoint rows and
            ``NaN`` differences instead of a resampled comparison.

        Raises:
            ValidationError: ``channel`` is not one of the supported channels; a model reports a
                non-numeric key or channel value; or one model reports two rows at the same key.
                The key is the ordered geometry pair plus the energy, so one model must not hold
                two rows for the same geometry and energy (``polarization_id`` is not part of it).
        """
        if channel not in _CHANNELS:
            raise ValidationError("channel must be 'mass_absorption', 'od', or 'beta'.")
        raw = [model.normalized(dataset) for model in self._models]
        keys = ["theta_deg", "phi_deg", "energy_ev"] if _has_geometry(raw) else ["energy_ev"]
        packed = _packed(raw[0], keys, channel, self._labels[0])
        for index in range(1, len(raw)):
            packed = packed.merge(
                _packed(raw[index], keys, channel, self._labels[index]), on=keys, how="outer"
            )
        merged = packed[list(keys) + list(self._labels)]
        merged = merged.sort_values(keys, na_position="last", kind="stable").reset_index(drop=True)
        for label in self._labels[1:]:
            merged[f"{label}_minus_{self._labels[0]}"] = merged[label] - merged[self._labels[0]]
        return merged


def compare(models: Sequence[Model], *, tolerance_ev: float = 0.2) -> Comparison:
    """Compare models by aligned peaks and normalized-channel differences.

    Args:
        models: At least two `Model` objects, each fetched at the version to compare.
        tolerance_ev: The single-linkage tolerance in eV; a difference strictly greater than this
            starts a new peak group.

    Returns:
        A `Comparison` whose `peaks` frame has one row per aligned peak group and whose `spectra`
        aligns server-computed channels on exact keys.

    Raises:
        ValidationError: Fewer than two entries were given or an entry is not a `Model`, or
            ``tolerance_ev`` is not a positive finite number of eV. `Comparison.spectra` raises
            for an unknown channel.
    """
    entries = list(models)
    if len(entries) < 2 or not all(isinstance(entry, Model) for entry in entries):
        raise ValidationError("compare needs at least two models.")
    return Comparison(entries, tolerance_ev=_tolerance(tolerance_ev))
