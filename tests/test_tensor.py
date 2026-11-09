import io

import numpy as np
import pytest

from xrayatlas import Tensor, ValidationError
from xrayatlas import tensor as tensor_module

AXIS = [280.0, 290.0, 300.0]


def good() -> np.ndarray:
    return np.ones((3, 2))


def test_valid_tensor_has_manifest_digest_and_blob() -> None:
    t = Tensor.from_array(good(), energy_ev=AXIS, components=["xx", "yy"], frame="lab", units="1")
    assert t.manifest.shape == [3, 2] and t.manifest.energy_ev.count == 3
    assert t.manifest.energy_ev.min == 280.0 and t.manifest.energy_ev.max == 300.0
    assert len(t.sha256) == 64 and t.size_bytes == len(t.blob)
    with np.load(__import__("io").BytesIO(t.blob), allow_pickle=False) as archive:
        assert archive["tensor"].shape == (3, 2) and list(archive["energy_ev"]) == AXIS


def test_complex_arrays_are_allowed() -> None:
    t = Tensor.from_array(
        good() * (1 + 2j),
        energy_ev=AXIS,
        components=["xx", "yy"],
        frame="principal",
        units="1",
    )
    assert t.manifest.dtype == "complex128"


def test_complex64_widens_to_complex128() -> None:
    t = Tensor.from_array(
        np.ones((3, 2), dtype=np.complex64) * (1 + 2j),
        energy_ev=AXIS,
        components=["xx", "yy"],
        frame="principal",
        units="1",
    )
    assert t.manifest.dtype == "complex128"
    with np.load(io.BytesIO(t.blob), allow_pickle=False) as archive:
        assert archive["tensor"].dtype == np.complex128


def test_float32_widens_to_float64() -> None:
    t = Tensor.from_array(
        np.ones((3, 2), dtype=np.float32),
        energy_ev=AXIS,
        components=["xx", "yy"],
        frame="lab",
        units="1",
    )
    assert t.manifest.dtype == "float64"
    with np.load(io.BytesIO(t.blob), allow_pickle=False) as archive:
        assert archive["tensor"].dtype == np.float64


def test_int64_array_is_rejected_with_cast_hint() -> None:
    with pytest.raises(ValidationError, match="float or complex"):
        Tensor.from_array(
            np.ones((3, 2), dtype=np.int64),
            energy_ev=AXIS,
            components=["xx", "yy"],
            frame="lab",
            units="1",
        )


@pytest.mark.parametrize(
    ("array", "axis", "components", "frame", "message"),
    [
        (np.ones((4, 2)), AXIS, ["xx", "yy"], "lab", "axis 0"),
        (np.ones((3, 3)), AXIS, ["xx", "yy"], "lab", "axis 1"),
        (np.full((3, 2), np.nan), AXIS, ["xx", "yy"], "lab", "finite"),
        (np.full((3, 2), np.inf), AXIS, ["xx", "yy"], "lab", "finite"),
        (np.ones((3, 2)), [280.0, 280.0, 300.0], ["xx", "yy"], "lab", "increasing"),
        (np.ones((3, 2)), [280.0, np.nan, 300.0], ["xx", "yy"], "lab", "finite"),
        (np.ones((3, 2)), AXIS, ["xx", "xx"], "lab", "unique"),
        (np.ones((3, 2)), AXIS, ["xx", "qq"], "lab", "qq"),
        (np.ones((3, 2)), AXIS, ["xx", "yy"], "sideways", "frame"),
        (np.ones(3), AXIS, ["xx"], "lab", "dimensions"),
        (np.array([["a", "b"]] * 3), AXIS, ["xx", "yy"], "lab", "float or complex"),
        (np.ones((3, 2)), [280.0], ["xx", "yy"], "lab", "at least 2"),
    ],
)
def test_invalid_inputs_are_rejected(
    array: np.ndarray, axis: list[float], components: list[str], frame: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Tensor.from_array(array, energy_ev=axis, components=components, frame=frame, units="1")


def test_oversized_tensor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tensor_module, "MAX_TENSOR_BYTES", 10)
    with pytest.raises(ValidationError, match="limit"):
        Tensor.from_array(good(), energy_ev=AXIS, components=["xx", "yy"], frame="lab", units="1")
