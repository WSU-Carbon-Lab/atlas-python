"""Client-side validation helpers.

Turns pydantic failures into `ValidationError` with the offending field path, and converts a peak
table (a pandas DataFrame) into validated `Peak` objects. Does no I/O.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Any, TypeVar

import pandas as pd
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from xrayatlas.errors import ValidationError
from xrayatlas.types import Peak

ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_PEAKS = 2000
_MODEL_TAG = re.compile(r"^[0-9a-z]{8}$")
_NUMERIC = ("center_ev", "amplitude", "width", "center_err", "amplitude_err", "width_err")
_TEXT = ("kind", "transition", "shape")
_PROVENANCE = ("fit_tool", "fit_version", "fit_code_hash")
_KNOWN = (*_NUMERIC, *_TEXT, *_PROVENANCE, "dataset", "polarization")


def parse_model(cls: type[ModelT], data: dict[str, Any], *, strict: bool = True) -> ModelT:
    """Build ``cls`` from ``data`` or raise `ValidationError` naming each failing field.

    Request models are parsed strictly by default: wrong Python types (a string where a number is
    expected, a list where a tuple is expected) and unknown keys are errors. Pass ``strict=False``
    only when parsing a server response, so that JSON-decoded lists coerce to tuples and numeric
    strings are accepted; the response models in `xrayatlas.types` still reject non-finite values.

    Args:
        cls: A pydantic model class from `xrayatlas.types`.
        data: Field values, snake_case or camelCase.
        strict: ``True`` (the default) for request input; ``False`` for server responses.

    Returns:
        The validated model.

    Raises:
        ValidationError: One or more fields failed; the message lists ``path: reason`` lines.
    """
    try:
        return cls.model_validate(data, strict=strict)
    except PydanticValidationError as exc:
        lines = [
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ValidationError("; ".join(lines)) from exc


def is_finite_number(value: object) -> bool:
    """Whether ``value`` is a finite int or float (booleans are not numbers)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_tag(tag: str) -> str:
    """Normalize a model tag, or raise `ValidationError`.

    Strips surrounding whitespace, lowercases, drops one leading ``m/`` or ``/``, and requires
    eight lowercase ASCII letters or digits. The Atlas mints base32-style tags (its alphabet
    omits ``i``, ``l``, ``o``, ``u``), but the client accepts any eight lowercase alphanumerics so
    tests and future generators keep working; the server's own schema is laxer (1 to 32
    characters). Typical valid tags: ``k7m2xq4n``, ``newmodel``.

    Returns:
        The cleaned tag.

    Raises:
        ValidationError: The cleaned tag is not eight lowercase ASCII letters or digits.
    """
    cleaned = tag.strip().lower().removeprefix("/").removeprefix("m/")
    if not _MODEL_TAG.match(cleaned):
        raise ValidationError(
            f"{tag!r} is not a model tag; use eight lowercase letters or digits such as 'k7m2xq4n'."
        )
    return cleaned


def _native(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _identifier(row_index: int, column: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"row {row_index}: {column} must be a UUID string.")
    return value


def validate_peaks(df: pd.DataFrame) -> list[Peak]:
    """Convert a peak table into validated peaks.

    Columns: ``center_ev`` (required, eV), and optionally ``kind``, ``transition``, ``amplitude``,
    ``width``, ``shape``, ``center_err``, ``amplitude_err``, ``width_err``, ``dataset`` (experiment
    UUID string), ``polarization`` (polarization UUID string), and the fit provenance columns
    ``fit_tool`` with ``fit_version`` and ``fit_code_hash`` (which require ``fit_tool``); these map
    to each peak's ``fit_provenance``. Every other column is folded into each peak's ``params``
    (strings, finite numbers, or booleans). A missing value (NaN or None) in an optional column
    means "not set"; infinity anywhere, a boolean in a numeric column, or a missing or non-finite
    ``center_ev`` is an error. Row order becomes peak order. An empty table (no rows, with a
    ``center_ev`` column) returns ``[]``.

    Args:
        df: The peak table. Columns must be flat and unique.

    Returns:
        One `Peak` per row.

    Raises:
        ValidationError: ``df`` is not a DataFrame, has MultiIndex or duplicate column names, lacks
            a ``center_ev`` column (the message suggests near matches), has more than 2000 rows, or
            holds an invalid value; the message names the row and column.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("validate_peaks expects a pandas DataFrame.")
    if isinstance(df.columns, pd.MultiIndex):
        raise ValidationError("validate_peaks requires flat column names, not a MultiIndex.")
    if not df.columns.is_unique:
        duplicates = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        raise ValidationError(f"Duplicate column names: {duplicates}.")
    if "center_ev" not in df.columns:
        close = difflib.get_close_matches("center_ev", [str(c) for c in df.columns], n=3)
        hint = f" Did you mean {close}?" if close else ""
        found = [str(c) for c in df.columns]
        raise ValidationError(f"Missing required column 'center_ev'. Found columns: {found}.{hint}")
    if len(df) > MAX_PEAKS:
        raise ValidationError(f"At most {MAX_PEAKS} peaks are allowed; the table has {len(df)}.")

    extra = [c for c in df.columns if c not in _KNOWN]
    peaks: list[Peak] = []
    for row_index, record in enumerate(df.to_dict(orient="records")):
        payload: dict[str, Any] = {}
        for column in _NUMERIC:
            if column not in record:
                continue
            value = _native(record[column])
            if column == "center_ev":
                if not is_finite_number(value):
                    raise ValidationError(f"row {row_index}: center_ev must be a finite number.")
                payload[column] = float(value)
            elif not _missing(value):
                if not is_finite_number(value):
                    raise ValidationError(f"row {row_index}: {column} must be a finite number.")
                payload[column] = float(value)
        for column in _TEXT:
            if column in record and not _missing(_native(record[column])):
                payload[column] = str(_native(record[column]))
        if "dataset" in record and not _missing(_native(record["dataset"])):
            payload["experiment_id"] = _identifier(row_index, "dataset", _native(record["dataset"]))
        if "polarization" in record and not _missing(_native(record["polarization"])):
            payload["polarization_id"] = _identifier(
                row_index, "polarization", _native(record["polarization"])
            )
        fit_tool = _native(record["fit_tool"]) if "fit_tool" in record else None
        if "fit_tool" in record and not _missing(fit_tool):
            provenance: dict[str, Any] = {"tool": str(fit_tool)}
            for column, key in (("fit_version", "version"), ("fit_code_hash", "code_hash")):
                if column in record and not _missing(_native(record[column])):
                    provenance[key] = str(_native(record[column]))
            payload["fit_provenance"] = provenance
        elif ("fit_version" in record and not _missing(_native(record["fit_version"]))) or (
            "fit_code_hash" in record and not _missing(_native(record["fit_code_hash"]))
        ):
            raise ValidationError(f"row {row_index}: fit_version/fit_code_hash need fit_tool.")
        params: dict[str, Any] = {}
        for column in extra:
            value = _native(record[column])
            if _missing(value):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                raise ValidationError(f"row {row_index}: {column} must be finite.")
            if not isinstance(value, (bool, int, float, str)):
                raise ValidationError(
                    f"row {row_index}: {column} must be a number, string, or bool."
                )
            params[str(column)] = (
                float(value)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else value
            )
        if params:
            payload["params"] = params
        try:
            peak = parse_model(Peak, payload)
        except ValidationError as exc:
            raise ValidationError(f"row {row_index}: {exc}") from exc
        peaks.append(peak.model_copy(update={"ordinal": row_index}))
    return peaks
