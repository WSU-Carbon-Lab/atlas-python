"""Aligned-peak grouping and normalized-channel comparison (`xrayatlas.compare`)."""

import httpx
import numpy as np
import pandas as pd
import pytest

from xrayatlas.client import Client
from xrayatlas.compare import Comparison, compare
from xrayatlas.datasets import Dataset
from xrayatlas.errors import ValidationError
from xrayatlas.models import Model
from xrayatlas.types import ModelDetail
from xrayatlas.validate import parse_model

HOST = "https://atlas.test"
EXPERIMENT = "123e4567-e89b-12d3-a456-426614174000"
TAGS = ("k7m2xq4n", "p8q9r2s3", "t5u6v7w8")


def detail(
    tag: str,
    *,
    version: int | None = 1,
    peaks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """A `ModelDetail` server response for ``tag`` at ``version`` with optional peaks."""
    return {
        "tag": tag,
        "title": "My model",
        "ownerOrcid": "0000-0002-1825-0097",
        "visibility": "public",
        "latestVersion": version,
        "reviewState": None,
        "isDefault": False,
        "forkedFrom": {"tag": "p8q9r2s3", "versionNumber": 1},
        "description": "d",
        "versions": [version] if version is not None else [],
        "hasDraft": version is None,
        "version": {
            "versionNumber": version,
            "status": "published" if version is not None else "draft",
            "normalization": {
                "schemaVersion": 1,
                "scope": "unified",
                "pre": [280, 283],
                "post": [325, 340],
            },
            "peaks": peaks if peaks is not None else [],
            "datasets": [
                {
                    "experimentId": EXPERIMENT,
                    "atlasDatasetId": "abcd2345",
                    "pinned": "1.0.0",
                    "role": "primary",
                }
            ],
            "tensor": None,
            "authors": [
                {"orcid": "0000-0002-1825-0097", "name": "A", "role": "owner", "signed": True}
            ],
            "checksum": "a" * 64,
            "publishedAt": "2026-10-25T00:00:00.000Z",
        },
    }


def wire_channels(energies: list[float], values: list[float]) -> dict[str, list[object]]:
    """A normalized `channels` payload on one geometry (55 deg, 0 deg), mass absorption only."""
    return {
        "energyEv": energies,
        "od": [None] * len(energies),
        "massAbsorption": values,
        "beta": [None] * len(energies),
        "polarizationId": ["p1"] * len(energies),
        "thetaDeg": [55.0] * len(energies),
        "phiDeg": [0.0] * len(energies),
    }


def wire_channels_full(
    energies: list[float],
    *,
    od: list[float | None] | None = None,
    mass_absorption: list[float | None] | None = None,
    beta: list[float | None] | None = None,
    theta: list[float | None] | None = None,
    phi: list[float | None] | None = None,
) -> dict[str, list[object]]:
    """A normalized `channels` payload with every channel column stated explicitly.

    Absent arguments become all-``None`` at the payload's length, so a test can leave a channel or
    a geometry column entirely unset while another channel carries values.
    """
    count = len(energies)

    def column(values: list[float | None] | None) -> list[object]:
        return list(values) if values is not None else [None] * count

    return {
        "energyEv": energies,
        "od": column(od),
        "massAbsorption": column(mass_absorption),
        "beta": column(beta),
        "polarizationId": ["p1"] * count,
        "thetaDeg": column(theta),
        "phiDeg": column(phi),
    }


def client_for(
    channels: dict[str, dict[str, list[object]]] | None = None,
    requests: list[httpx.Request] | None = None,
) -> Client:
    """A client whose mock serves the per-tag normalized `channels` mapping (404 otherwise)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        parts = [part for part in request.url.path.split("/") if part]
        body = (channels or {}).get(parts[-2] if len(parts) >= 2 else "")
        if body is None:
            return httpx.Response(
                404, json={"error": {"code": "NOT_FOUND", "message": "no such channels"}}
            )
        return httpx.Response(200, json={"channels": body})

    return Client(host=HOST, token="xra_test", transport=httpx.MockTransport(handler))


def model(
    tag: str,
    *,
    version: int | None = 1,
    centers: list[float] | None = None,
    channels: dict[str, dict[str, list[object]]] | None = None,
    client: Client | None = None,
) -> Model:
    """A `Model` built directly from a detail dict; `channels` serves `normalized` per tag."""
    peaks = [{"ordinal": index, "centerEv": center} for index, center in enumerate(centers or [])]
    parsed = parse_model(ModelDetail, detail(tag, version=version, peaks=peaks), strict=False)
    return Model(client or client_for(channels), parsed)


def test_peaks_within_tolerance_group_together() -> None:
    result = compare(
        [
            model(TAGS[0], centers=[285.0]),
            model(TAGS[1], centers=[285.15]),
            model(TAGS[2], centers=[285.30]),
        ],
        tolerance_ev=0.2,
    )
    assert isinstance(result, Comparison)
    frame = result.peaks
    assert len(frame) == 1
    assert list(frame.columns) == [
        "center_ev",
        "count",
        f"{TAGS[0]}#1",
        f"{TAGS[1]}#1",
        f"{TAGS[2]}#1",
    ]
    assert frame["count"].iat[0] == 3
    assert frame[f"{TAGS[0]}#1"].iat[0] == pytest.approx(285.0)
    assert frame[f"{TAGS[1]}#1"].iat[0] == pytest.approx(285.15)
    assert frame[f"{TAGS[2]}#1"].iat[0] == pytest.approx(285.30)
    assert frame["center_ev"].iat[0] == pytest.approx(285.15)


def test_peaks_outside_tolerance_stay_separate() -> None:
    frame = compare(
        [model(TAGS[0], centers=[285.0]), model(TAGS[1], centers=[285.4])], tolerance_ev=0.2
    ).peaks
    assert list(frame["center_ev"].to_list()) == pytest.approx([285.0, 285.4])
    assert frame[f"{TAGS[0]}#1"].to_list() == pytest.approx([285.0, np.nan], nan_ok=True)
    assert frame[f"{TAGS[1]}#1"].to_list() == pytest.approx([np.nan, 285.4], nan_ok=True)


def test_group_order_is_by_energy() -> None:
    frame = compare(
        [
            model(TAGS[0], centers=[290.0]),
            model(TAGS[1], centers=[285.0]),
            model(TAGS[2], centers=[287.5]),
        ]
    ).peaks
    assert frame["center_ev"].to_list() == pytest.approx([285.0, 287.5, 290.0])
    assert list(frame.columns)[2:] == [f"{tag}#1" for tag in TAGS]


def test_model_without_peak_in_group_is_nan() -> None:
    frame = compare(
        [
            model(TAGS[0], centers=[285.0]),
            model(TAGS[1], centers=[285.1]),
            model(TAGS[2]),
        ],
        tolerance_ev=0.5,
    ).peaks
    assert len(frame) == 1
    assert frame["count"].iat[0] == 2
    assert frame[f"{TAGS[0]}#1"].iat[0] == pytest.approx(285.0)
    assert frame[f"{TAGS[1]}#1"].iat[0] == pytest.approx(285.1)
    assert np.isnan(frame[f"{TAGS[2]}#1"].iat[0])


def test_nearest_peak_to_group_mean_wins() -> None:
    frame = compare(
        [model(TAGS[0], centers=[285.0, 285.3]), model(TAGS[1], centers=[285.2])],
        tolerance_ev=0.2,
    ).peaks
    assert len(frame) == 1
    assert frame["count"].iat[0] == 2
    assert frame[f"{TAGS[0]}#1"].iat[0] == pytest.approx(285.3)
    assert frame[f"{TAGS[1]}#1"].iat[0] == pytest.approx(285.2)
    assert frame["center_ev"].iat[0] == pytest.approx(285.25)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_tolerance_must_be_positive_and_finite(tolerance: float) -> None:
    models = [model(TAGS[0], centers=[285.0]), model(TAGS[1], centers=[285.4])]
    with pytest.raises(ValidationError, match="tolerance_ev"):
        compare(models, tolerance_ev=tolerance)


def test_compare_needs_two_models() -> None:
    with pytest.raises(ValidationError, match="at least two models"):
        compare([])
    with pytest.raises(ValidationError, match="at least two models"):
        compare([model(TAGS[0], centers=[285.0])])
    with pytest.raises(ValidationError, match="at least two models"):
        compare([model(TAGS[0], centers=[285.0]), "not a model"])  # type: ignore[list-item]


def test_spectra_aligns_exactly_and_subtracts_first() -> None:
    channels = {
        TAGS[0]: wire_channels([285.0, 286.0], [0.2, 0.3]),
        TAGS[1]: wire_channels([286.0, 287.0], [0.4, 0.5]),
    }
    frame = compare(
        [model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)]
    ).spectra()
    first, second = f"{TAGS[0]}#1", f"{TAGS[1]}#1"
    assert list(frame.columns) == [
        "theta_deg",
        "phi_deg",
        "energy_ev",
        first,
        second,
        f"{second}_minus_{first}",
    ]
    assert frame["energy_ev"].to_list() == pytest.approx([285.0, 286.0, 287.0])
    assert frame["theta_deg"].to_list() == pytest.approx([55.0, 55.0, 55.0])
    assert frame["phi_deg"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert frame[first].to_list() == pytest.approx([0.2, 0.3, np.nan], nan_ok=True)
    assert frame[second].to_list() == pytest.approx([np.nan, 0.4, 0.5], nan_ok=True)
    assert frame[f"{second}_minus_{first}"].to_list() == pytest.approx(
        [np.nan, 0.1, np.nan], nan_ok=True
    )


def test_spectra_does_not_interpolate() -> None:
    channels = {
        TAGS[0]: wire_channels([285.0, 286.0], [0.2, 0.3]),
        TAGS[1]: wire_channels([287.0], [0.5]),
    }
    frame = compare(
        [model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)]
    ).spectra()
    assert len(frame) == 3
    assert frame["energy_ev"].to_list() == pytest.approx([285.0, 286.0, 287.0])
    left = np.asarray(frame[f"{TAGS[0]}#1"], dtype=float)
    right = np.asarray(frame[f"{TAGS[1]}#1"], dtype=float)
    assert np.allclose(left, [0.2, 0.3, np.nan], equal_nan=True)
    assert np.allclose(right, [np.nan, np.nan, 0.5], equal_nan=True)


def test_spectra_rejects_unknown_channel() -> None:
    comparison = compare([model(TAGS[0], centers=[285.0]), model(TAGS[1], centers=[285.4])])
    with pytest.raises(ValidationError, match="channel must be"):
        comparison.spectra(channel="delta")


def test_duplicate_model_labels_are_disambiguated() -> None:
    frame = compare(
        [model(TAGS[0], centers=[285.0]), model(TAGS[0], centers=[285.4])], tolerance_ev=0.2
    ).peaks
    assert list(frame.columns) == ["center_ev", "count", f"{TAGS[0]}#1", f"{TAGS[0]}#1#2"]
    assert frame[f"{TAGS[0]}#1"].to_list() == pytest.approx([285.0, np.nan], nan_ok=True)
    assert frame[f"{TAGS[0]}#1#2"].to_list() == pytest.approx([np.nan, 285.4], nan_ok=True)
    draft_frame = compare([model(TAGS[0], centers=[285.0]), model(TAGS[0], version=None)]).peaks
    assert list(draft_frame.columns)[2:] == [f"{TAGS[0]}#1", f"{TAGS[0]}#draft"]


def test_peaks_tie_breaks_to_the_lower_center() -> None:
    frame = compare(
        [model(TAGS[0], centers=[285.0, 285.4]), model(TAGS[1], centers=[285.2])],
        tolerance_ev=0.2,
    ).peaks
    assert len(frame) == 1
    assert frame[f"{TAGS[0]}#1"].iat[0] == pytest.approx(285.0)
    assert frame[f"{TAGS[1]}#1"].iat[0] == pytest.approx(285.2)
    assert frame["center_ev"].iat[0] == pytest.approx(285.1)


def test_empty_peaks_keep_the_column_contract() -> None:
    frame = compare([model(TAGS[0]), model(TAGS[1])]).peaks
    assert list(frame.columns) == ["center_ev", "count", f"{TAGS[0]}#1", f"{TAGS[1]}#1"]
    assert len(frame) == 0
    assert frame["count"].dtype == np.dtype("int64")
    assert frame["center_ev"].dtype == np.dtype("float64")
    assert frame[f"{TAGS[0]}#1"].dtype == np.dtype("float64")


def test_peaks_returns_a_copy() -> None:
    comparison = compare([model(TAGS[0], centers=[285.0]), model(TAGS[1], centers=[285.4])])
    edited = comparison.peaks
    edited.loc[0, "center_ev"] = -1.0
    assert comparison.peaks["center_ev"].iat[0] == pytest.approx(285.0)


def test_spectra_selects_the_requested_channel() -> None:
    channels = {
        TAGS[0]: wire_channels_full(
            [285.0, 286.0], od=[1.0, 2.0], mass_absorption=[0.1, 0.2], beta=[3.0, 4.0]
        ),
        TAGS[1]: wire_channels_full(
            [285.0, 286.0], od=[5.0, 6.0], mass_absorption=[0.5, 0.6], beta=[7.0, 8.0]
        ),
    }
    comparison = compare([model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)])
    first, second = f"{TAGS[0]}#1", f"{TAGS[1]}#1"
    absorption = comparison.spectra()
    assert absorption[first].to_list() == pytest.approx([0.1, 0.2])
    assert absorption[second].to_list() == pytest.approx([0.5, 0.6])
    optical_density = comparison.spectra(channel="od")
    assert optical_density[first].to_list() == pytest.approx([1.0, 2.0])
    assert optical_density[second].to_list() == pytest.approx([5.0, 6.0])
    beta = comparison.spectra(channel="beta")
    assert beta[first].to_list() == pytest.approx([3.0, 4.0])
    assert beta[second].to_list() == pytest.approx([7.0, 8.0])


def test_spectra_falls_back_to_energy_when_no_frame_has_geometry() -> None:
    channels = {
        TAGS[0]: wire_channels_full(
            [285.0, 286.0], mass_absorption=[0.2, 0.3], theta=[None, None], phi=[None, None]
        ),
        TAGS[1]: wire_channels_full(
            [286.0, 287.0], mass_absorption=[0.4, 0.5], theta=[None, None], phi=[None, None]
        ),
    }
    frame = compare(
        [model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)]
    ).spectra()
    first, second = f"{TAGS[0]}#1", f"{TAGS[1]}#1"
    assert list(frame.columns) == ["energy_ev", first, second, f"{second}_minus_{first}"]
    assert frame["energy_ev"].to_list() == pytest.approx([285.0, 286.0, 287.0])
    assert frame[f"{second}_minus_{first}"].to_list() == pytest.approx(
        [np.nan, 0.1, np.nan], nan_ok=True
    )


def test_spectra_uses_geometry_keys_when_only_theta_is_labeled() -> None:
    channels = {
        TAGS[0]: wire_channels_full([285.0], mass_absorption=[0.2], theta=[55.0], phi=[None]),
        TAGS[1]: wire_channels_full([285.0], mass_absorption=[0.9], theta=[75.0], phi=[None]),
    }
    frame = compare(
        [model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)]
    ).spectra()
    first, second = f"{TAGS[0]}#1", f"{TAGS[1]}#1"
    assert list(frame.columns)[:3] == ["theta_deg", "phi_deg", "energy_ev"]
    assert len(frame) == 2
    assert frame[first].to_list() == pytest.approx([0.2, np.nan], nan_ok=True)
    assert frame[second].to_list() == pytest.approx([np.nan, 0.9], nan_ok=True)
    assert frame[f"{second}_minus_{first}"].isna().all()


def test_spectra_never_subtracts_across_labeled_and_unlabeled_geometry() -> None:
    channels = {
        TAGS[0]: wire_channels_full([285.0], mass_absorption=[0.2], theta=[55.0], phi=[0.0]),
        TAGS[1]: wire_channels_full([285.0], mass_absorption=[0.9], theta=[None], phi=[None]),
    }
    frame = compare(
        [model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)]
    ).spectra()
    first, second = f"{TAGS[0]}#1", f"{TAGS[1]}#1"
    assert len(frame) == 2
    assert frame["theta_deg"].iat[0] == pytest.approx(55.0)
    assert np.isnan(frame["theta_deg"].iat[-1]) and np.isnan(frame["phi_deg"].iat[-1])
    assert frame[f"{second}_minus_{first}"].isna().all()


def test_spectra_rejects_duplicate_keys_in_one_model() -> None:
    channels = {
        TAGS[0]: wire_channels_full([285.0, 285.0], mass_absorption=[0.1, 0.2]),
        TAGS[1]: wire_channels_full([285.0], mass_absorption=[0.5]),
    }
    comparison = compare([model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)])
    with pytest.raises(ValidationError, match="more than one row at the same key"):
        comparison.spectra()


def test_spectra_propagates_a_non_numeric_channel_from_normalized() -> None:
    channels = {
        TAGS[0]: wire_channels_full([285.0], mass_absorption=["oops"]),  # type: ignore[list-item]
        TAGS[1]: wire_channels_full([285.0], mass_absorption=[0.5]),
    }
    comparison = compare([model(TAGS[0], channels=channels), model(TAGS[1], channels=channels)])
    with pytest.raises(ValidationError, match="finite number"):
        comparison.spectra()


def test_spectra_rejects_non_numeric_values_from_a_normalized_override() -> None:
    class Overriding(Model):
        """A `Model` whose ``normalized`` returns a non-numeric channel value."""

        def normalized(self, dataset: Dataset | str | None = None) -> pd.DataFrame:
            return pd.DataFrame({"energy_ev": [285.0], "mass_absorption": ["oops"]})

    first = Overriding(client_for(), parse_model(ModelDetail, detail(TAGS[0]), strict=False))
    channels = {TAGS[1]: wire_channels([285.0], [0.5])}
    comparison = compare([first, model(TAGS[1], channels=channels)])
    with pytest.raises(ValidationError, match="non-numeric mass_absorption"):
        comparison.spectra()


@pytest.mark.parametrize(
    ("theta", "message"),
    [
        ("55 deg", "theta_deg must hold numeric angles"),
        (np.inf, "theta_deg must hold finite angles"),
    ],
)
def test_spectra_rejects_invalid_geometry_from_a_normalized_override(
    theta: object, message: str
) -> None:
    class Overriding(Model):
        """A `Model` whose ``normalized`` returns an unusable geometry angle."""

        def normalized(self, dataset: Dataset | str | None = None) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "energy_ev": [285.0],
                    "theta_deg": [theta],
                    "phi_deg": [0.0],
                    "mass_absorption": [0.2],
                }
            )

    first = Overriding(client_for(), parse_model(ModelDetail, detail(TAGS[0]), strict=False))
    channels = {TAGS[1]: wire_channels([285.0], [0.5])}
    comparison = compare([first, model(TAGS[1], channels=channels)])
    with pytest.raises(ValidationError, match=message):
        comparison.spectra()


def test_peaks_rejects_non_numeric_centers_from_a_peaks_override() -> None:
    class Overriding(Model):
        """A `Model` whose ``peaks`` frame holds a non-numeric center."""

        @property
        def peaks(self) -> pd.DataFrame:
            return pd.DataFrame({"center_ev": ["oops"], "kind": ["pi-star"]})

    first = Overriding(client_for(), parse_model(ModelDetail, detail(TAGS[0]), strict=False))
    with pytest.raises(ValidationError, match="non-numeric center_ev"):
        _ = compare([first, model(TAGS[1], centers=[285.4])]).peaks


def test_derived_labels_stay_unique_for_tags_containing_hash() -> None:
    frame = compare(
        [
            model("x", centers=[285.0]),
            model("x", centers=[285.1]),
            model("x#1", version=2, centers=[285.2]),
        ]
    ).peaks
    assert list(frame.columns)[2:] == ["x#1", "x#1#2", "x#1#2#2"]
    assert frame["count"].to_list() == [3]


def test_spectra_passes_the_dataset_reference_through() -> None:
    channels = {
        TAGS[0]: wire_channels([285.0], [0.2]),
        TAGS[1]: wire_channels([285.0], [0.5]),
    }
    requests: list[httpx.Request] = []
    client = client_for(channels, requests)
    comparison = compare(
        [
            model(TAGS[0], channels=channels, client=client),
            model(TAGS[1], channels=channels, client=client),
        ]
    )
    comparison.spectra("d/abcd2345")
    assert len(requests) == 2
    assert [request.url.params["dataset"] for request in requests] == ["abcd2345", "abcd2345"]


def test_comparison_constructor_validates_models_labels_and_tolerance() -> None:
    first = model(TAGS[0], centers=[285.0])
    second = model(TAGS[1], centers=[285.4])
    third = model(TAGS[2], centers=[285.8])
    with pytest.raises(ValidationError, match="Model objects"):
        Comparison([])
    with pytest.raises(ValidationError, match="Model objects"):
        Comparison([first, "not a model"])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="per model"):
        Comparison([first, second], ["only-one"])
    with pytest.raises(ValidationError, match="per model"):
        Comparison([first, second], ["a", "b", "c"])
    with pytest.raises(ValidationError, match="per model"):
        Comparison([first, second], ["a", "a"])
    with pytest.raises(ValidationError, match="per model"):
        Comparison([first, second], ["a", 1])  # type: ignore[list-item]
    for reserved in ("center_ev", "count", "energy_ev", "theta_deg", "phi_deg"):
        with pytest.raises(ValidationError, match="frame column"):
            Comparison([first, second], [reserved, "b"])
    with pytest.raises(ValidationError, match="frame column"):
        Comparison([first, second, third], ["a", "b", "b_minus_a"])
    with pytest.raises(ValidationError, match="tolerance_ev"):
        Comparison([first, second], tolerance_ev=-1.0)
    with pytest.raises(ValidationError, match="tolerance_ev"):
        Comparison([first, second], tolerance_ev="0.2")  # type: ignore[arg-type]
    assert list(Comparison([first, second]).peaks.columns) == [
        "center_ev",
        "count",
        f"{TAGS[0]}#1",
        f"{TAGS[1]}#1",
    ]


@pytest.mark.parametrize("tolerance", ["0.2", None, [0.2]])
def test_non_numeric_tolerance_raises_validation_error(tolerance: object) -> None:
    models = [model(TAGS[0], centers=[285.0]), model(TAGS[1], centers=[285.4])]
    with pytest.raises(ValidationError, match="tolerance_ev"):
        compare(models, tolerance_ev=tolerance)  # type: ignore[arg-type]
