"""Stored optical-model parameter payloads.

A `Tensor` is built from the user's own array with `Tensor.from_array`, which validates it, writes
a compressed ``.npz`` (arrays ``tensor`` and ``energy_ev``), computes its SHA-256, and builds the
manifest the server stores. The client never computes optical constants. Does no network I/O.

The stored object is a container for named parameter columns, not a physical tensor: axis 1 is
labelled by `TENSOR_COMPONENTS`, and the values a researcher works with today are the optical
constants ``delta`` and ``beta`` (or directional columns such as ``xx``) as separate series on a
shared energy axis. A complex array is a storage convenience for carrying delta in the real part
and beta in the imaginary part; the client stores complex arrays as ``complex128`` (widening
``complex64`` exactly) and floats as ``float64``, and it never narrows complex data to a real
dtype. A real tensor with coordinate transforms and physical invariants is future work, so do not
read the array shape as one.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

from xrayatlas.errors import ValidationError
from xrayatlas.types import TENSOR_COMPONENTS, TENSOR_FRAMES, TensorManifest
from xrayatlas.validate import parse_model

MAX_TENSOR_BYTES = 128 * 1024 * 1024

TensorArray: TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.complex128]


def widen_tensor_dtype(array: npt.ArrayLike) -> TensorArray | None:
    """Return ``array`` as ``float64`` or ``complex128``, widening narrow floats exactly.

    ``float16`` and ``float32`` widen to ``float64``, and ``complex64`` widens to ``complex128``;
    both are exact and never lose precision. A ``float64`` or ``complex128`` array is returned
    unchanged (without a copy).

    Args:
        array: Any array-like value.

    Returns:
        The array with an accepted float or complex dtype, or ``None`` when the dtype is integer,
        boolean, object, string, or otherwise not a supported float or complex type.
    """
    values = np.asarray(array)
    dtype = values.dtype
    if dtype == np.dtype(np.complex128):
        return cast("npt.NDArray[np.complex128]", values)
    if dtype == np.dtype(np.float64):
        return cast("npt.NDArray[np.float64]", values)
    if dtype in (np.dtype(np.float16), np.dtype(np.float32)):
        return cast("npt.NDArray[np.float64]", values.astype(np.float64))
    if dtype == np.dtype(np.complex64):
        return cast("npt.NDArray[np.complex128]", values.astype(np.complex128))
    return None


@dataclass(frozen=True)
class Tensor:
    """A validated, serialized parameter payload ready to upload."""

    blob: bytes
    sha256: str
    manifest: TensorManifest

    @property
    def size_bytes(self) -> int:
        """Size of the serialized ``.npz`` in bytes."""
        return len(self.blob)

    @classmethod
    def from_array(
        cls,
        array: npt.ArrayLike,
        *,
        energy_ev: npt.ArrayLike,
        components: Sequence[str],
        frame: str,
        units: str,
    ) -> Tensor:
        """Validate named parameter columns and serialize them.

        The dtype policy is lossless: ``float16``, ``float32``, ``float64``, ``complex64``, and
        ``complex128`` are accepted. ``float16`` and ``float32`` widen to ``float64`` and
        ``complex64`` widens to ``complex128`` (exact, no precision loss) before serializing, so
        the stored dtype is always ``float64`` or ``complex128``. Integer, boolean, object, and
        string dtypes are rejected so a silently widened or coerced tensor cannot hide a mistake;
        an integer array must be cast explicitly by the caller.

        Args:
            array: Numbers of an accepted float or complex dtype shaped ``(n_energy, n_components,
                ...)`` with 2 to 6 dimensions, all finite.
            energy_ev: Strictly increasing finite 1-D energies in eV, ``n_energy >= 2``.
            components: Labels for axis 1, each in `TENSOR_COMPONENTS`, no duplicates.
            frame: One of `TENSOR_FRAMES`.
            units: Units label, 1 to 40 characters.

        Returns:
            The tensor with its blob, digest, and manifest.

        Raises:
            ValidationError: Any of the constraints above is violated, or the serialized size
                exceeds ``MAX_TENSOR_BYTES`` (128 MiB).
        """
        values = widen_tensor_dtype(array)
        if values is None:
            raise ValidationError(
                "tensor array must be float or complex; cast integers explicitly with .astype(float64)."  # noqa: E501
            )
        axis = np.asarray(energy_ev)
        if not 2 <= values.ndim <= 6:
            raise ValidationError("tensor array must have 2 to 6 dimensions.")
        if axis.ndim != 1 or axis.size < 2 or axis.dtype.kind not in "fiu":
            raise ValidationError("energy_ev must be a 1-D numeric array with at least 2 values.")
        axis = axis.astype(float)
        if not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
            raise ValidationError("energy_ev must be finite and strictly increasing.")
        if values.shape[0] != axis.size:
            raise ValidationError(
                f"tensor axis 0 has {values.shape[0]} entries but energy_ev has {axis.size}."
            )
        labels = tuple(components)
        if not labels or len(set(labels)) != len(labels):
            raise ValidationError("components must be non-empty and unique.")
        for label in labels:
            if label not in TENSOR_COMPONENTS:
                raise ValidationError(f"component {label!r} is not one of {TENSOR_COMPONENTS}.")
        if values.shape[1] != len(labels):
            raise ValidationError(
                f"tensor axis 1 has {values.shape[1]} entries but {len(labels)} components were given."  # noqa: E501
            )
        if frame not in TENSOR_FRAMES:
            raise ValidationError(f"frame must be one of {TENSOR_FRAMES}.")
        if not np.isfinite(values).all():
            raise ValidationError("tensor array must be finite (no NaN or infinity).")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, tensor=values, energy_ev=axis)
        blob = buffer.getvalue()
        if len(blob) > MAX_TENSOR_BYTES:
            raise ValidationError(
                f"serialized tensor is {len(blob)} bytes; the limit is {MAX_TENSOR_BYTES}."
            )
        manifest = parse_model(
            TensorManifest,
            {
                "format": "npz",
                "energy_ev": {
                    "min": float(axis[0]),
                    "max": float(axis[-1]),
                    "count": int(axis.size),
                },
                "components": list(labels),
                "frame": frame,
                "units": units,
                "dtype": str(values.dtype),
                "shape": [int(n) for n in values.shape],
            },
        )
        return cls(blob=blob, sha256=hashlib.sha256(blob).hexdigest(), manifest=manifest)
