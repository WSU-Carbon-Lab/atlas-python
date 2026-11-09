import math

import numpy as np
import pandas as pd
import pytest

from xrayatlas import (
    DatasetLinkIn,
    DraftContent,
    ModelDetail,
    Normalization,
    Peak,
    TensorManifest,
    ValidationError,
    WindowPair,
    parse_model,
    validate_peaks,
)

W = {"pre": (280.0, 283.0), "post": (325.0, 340.0)}
UID = "123e4567-e89b-12d3-a456-426614174000"


def test_unified_normalization_accepts_valid_windows() -> None:
    norm = parse_model(Normalization, W)
    assert norm.scope == "unified" and norm.pre == (280.0, 283.0)


def test_normalization_rejects_equal_low_high() -> None:
    with pytest.raises(ValidationError, match="pre window must have low < high"):
        parse_model(Normalization, {"pre": (285.0, 285.0), "post": (325.0, 340.0)})


def test_normalization_rejects_overlapping_windows() -> None:
    with pytest.raises(ValidationError, match="end at or before"):
        parse_model(Normalization, {"pre": (280.0, 300.0), "post": (290.0, 320.0)})


def test_normalization_rejects_non_finite() -> None:
    with pytest.raises(ValidationError, match="post"):
        parse_model(Normalization, {"pre": (280.0, 283.0), "post": (math.inf, 340.0)})


def test_per_channel_needs_all_three_channels() -> None:
    with pytest.raises(ValidationError, match="od, massabsorption, and beta"):
        parse_model(Normalization, {"scope": "per_channel", "od": W, "massabsorption": W})


def test_per_channel_valid() -> None:
    norm = parse_model(
        Normalization, {"scope": "per_channel", "od": W, "massabsorption": W, "beta": W}
    )
    assert norm.beta is not None and norm.beta.post == (325.0, 340.0)


def test_per_channel_to_wire_shape() -> None:
    norm = parse_model(
        Normalization, {"scope": "per_channel", "od": W, "massabsorption": W, "beta": W}
    )
    assert norm.to_wire() == {
        "schemaVersion": 1,
        "scope": "per_channel",
        "od": {"pre": [280.0, 283.0], "post": [325.0, 340.0]},
        "massabsorption": {"pre": [280.0, 283.0], "post": [325.0, 340.0]},
        "beta": {"pre": [280.0, 283.0], "post": [325.0, 340.0]},
    }


def test_unified_rejects_channel_fields() -> None:
    with pytest.raises(ValidationError, match="must not set od, massabsorption, or beta"):
        parse_model(Normalization, {**W, "od": W})


def test_normalization_rejects_string_windows() -> None:
    with pytest.raises(ValidationError, match="pre"):
        parse_model(Normalization, {"pre": ("1", "2"), "post": (3, 4)})


def test_peak_rejects_negative_width() -> None:
    with pytest.raises(ValidationError, match="width"):
        parse_model(Peak, {"center_ev": 285.0, "width": -1.0})


def test_peak_kind_must_be_known() -> None:
    with pytest.raises(ValidationError, match="kind"):
        parse_model(Peak, {"center_ev": 285.0, "kind": "rydberg"})


def test_peak_rejects_string_center() -> None:
    with pytest.raises(ValidationError, match="center_ev"):
        parse_model(Peak, {"center_ev": "285"})


def test_peak_rejects_bool_center() -> None:
    with pytest.raises(ValidationError, match="center_ev"):
        parse_model(Peak, {"center_ev": True})


def test_peak_rejects_unknown_field_typo() -> None:
    with pytest.raises(ValidationError, match="widht"):
        parse_model(Peak, {"center_ev": 1, "widht": 2})


def test_peak_rejects_non_uuid_experiment_id() -> None:
    with pytest.raises(ValidationError, match="experiment_id: Value error, must be a UUID"):
        parse_model(Peak, {"center_ev": 285.0, "experiment_id": "abc"})


def test_draft_content_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="titel"):
        parse_model(DraftContent, {"titel": "x"})


def test_dataset_link_rejects_non_uuid() -> None:
    with pytest.raises(ValidationError, match="experiment_id: Value error, must be a UUID"):
        parse_model(DatasetLinkIn, {"experiment_id": "abc"})


def test_validate_peaks_names_missing_center_column_and_suggests() -> None:
    with pytest.raises(ValidationError, match="center_ev") as info:
        validate_peaks(pd.DataFrame({"center": [285.0]}))
    assert "Did you mean" in str(info.value)


def test_validate_peaks_rejects_nan_center() -> None:
    with pytest.raises(ValidationError, match="row 1: center_ev"):
        validate_peaks(pd.DataFrame({"center_ev": [285.0, float("nan")]}))


def test_validate_peaks_rejects_bool_center() -> None:
    with pytest.raises(ValidationError, match="row 0: center_ev must be a finite number"):
        validate_peaks(pd.DataFrame({"center_ev": [True]}))


def test_validate_peaks_rejects_numpy_bool_center() -> None:
    with pytest.raises(ValidationError, match="row 0: center_ev must be a finite number"):
        validate_peaks(pd.DataFrame({"center_ev": np.array([np.True_])}))


def test_validate_peaks_rejects_infinite_amplitude() -> None:
    with pytest.raises(ValidationError, match="amplitude"):
        validate_peaks(pd.DataFrame({"center_ev": [285.0], "amplitude": [math.inf]}))


def test_validate_peaks_rejects_bool_in_numeric_column() -> None:
    with pytest.raises(ValidationError, match="row 0: width must be a finite number"):
        validate_peaks(pd.DataFrame({"center_ev": [285.0], "width": [True]}))


def test_validate_peaks_nan_in_optional_column_means_unset() -> None:
    peaks = validate_peaks(pd.DataFrame({"center_ev": [285.0], "width": [float("nan")]}))
    assert peaks[0].width is None


def test_validate_peaks_folds_extra_columns_into_params_and_numbers_rows() -> None:
    df = pd.DataFrame(
        {
            "center_ev": [285.4, 288.6],
            "kind": ["pi-star", "sigma-star"],
            "chi2": [1.2, 0.9],
            "fitter": ["lmfit", "lmfit"],
        }
    )
    peaks = validate_peaks(df)
    assert [p.ordinal for p in peaks] == [0, 1]
    assert peaks[0].params == {"chi2": 1.2, "fitter": "lmfit"}
    assert peaks[1].kind == "sigma-star"


def test_validate_peaks_empty_table_returns_empty_list() -> None:
    assert validate_peaks(pd.DataFrame({"center_ev": pd.Series([], dtype=float)})) == []


def test_validate_peaks_rejects_non_string_dataset() -> None:
    with pytest.raises(ValidationError, match="row 0: dataset must be a UUID string"):
        validate_peaks(pd.DataFrame({"center_ev": [285.0], "dataset": [5]}))


def test_validate_peaks_rejects_non_uuid_dataset() -> None:
    with pytest.raises(ValidationError, match="row 0: experiment_id: Value error, must be a UUID"):
        validate_peaks(pd.DataFrame({"center_ev": [285.0], "dataset": ["abc"]}))


def test_validate_peaks_rejects_duplicate_column_names() -> None:
    df = pd.DataFrame([[285.0, 1.0, 2.0]], columns=["center_ev", "chi2", "chi2"])
    with pytest.raises(ValidationError, match="Duplicate column names: \\['chi2'\\]"):
        validate_peaks(df)


def test_validate_peaks_rejects_multiindex_columns() -> None:
    df = pd.DataFrame(
        [[285.0, 1.0]], columns=pd.MultiIndex.from_tuples([("center_ev", ""), ("chi2", "")])
    )
    with pytest.raises(ValidationError, match="MultiIndex"):
        validate_peaks(df)


def test_validate_peaks_rejects_non_dataframe() -> None:
    with pytest.raises(ValidationError, match="expects a pandas DataFrame"):
        validate_peaks({"center_ev": [285.0]})  # type: ignore[arg-type]


def test_validate_peaks_rejects_more_than_2000_rows() -> None:
    with pytest.raises(ValidationError, match="At most 2000 peaks"):
        validate_peaks(pd.DataFrame({"center_ev": [285.0] * 2001}))


def test_draft_content_rejects_duplicate_datasets() -> None:
    with pytest.raises(ValidationError, match="must not repeat an experimentId"):
        parse_model(DraftContent, {"datasets": [{"experiment_id": UID}, {"experiment_id": UID}]})


def test_draft_content_to_wire_sends_nested_defaults_and_only_set_fields() -> None:
    draft = DraftContent(
        normalization=Normalization(pre=(280.0, 283.0), post=(325.0, 340.0)),
        peaks=[Peak(center_ev=285.0)],
    )
    assert draft.to_wire() == {
        "normalization": {
            "schemaVersion": 1,
            "scope": "unified",
            "pre": [280.0, 283.0],
            "post": [325.0, 340.0],
        },
        "peaks": [{"centerEv": 285.0, "kind": "other"}],
    }


def test_draft_content_to_wire_keeps_explicit_nulls() -> None:
    draft = DraftContent(title="t", description=None, normalization=None)
    assert draft.to_wire() == {"title": "t", "description": None, "normalization": None}


def test_tensor_manifest_requires_ordered_energy_grid() -> None:
    base = {
        "format": "npz",
        "components": ["xx"],
        "frame": "lab",
        "units": "1",
        "dtype": "float64",
        "shape": [2, 1],
    }
    with pytest.raises(ValidationError, match="less than max"):
        parse_model(TensorManifest, {**base, "energy_ev": {"min": 5.0, "max": 5.0, "count": 2}})
    ok = parse_model(
        TensorManifest, {**base, "energy_ev": {"min": 280.0, "max": 340.0, "count": 2}}
    )
    assert ok.schema_version == 1


def test_model_detail_parses_server_json_and_ignores_unknown_fields() -> None:
    wire = {
        "tag": "k7m2xq4n",
        "title": "Default model",
        "ownerOrcid": "0000-0002-1825-0097",
        "visibility": "public",
        "latestVersion": 1,
        "reviewState": None,
        "isDefault": True,
        "forkedFrom": None,
        "description": None,
        "versions": [1],
        "hasDraft": False,
        "futureField": 7,
        "version": {
            "versionNumber": 1,
            "status": "published",
            "normalization": {
                "schemaVersion": 1,
                "scope": "unified",
                "pre": [280, 283],
                "post": [325, 340],
            },
            "peaks": [],
            "datasets": [
                {
                    "experimentId": "e",
                    "atlasDatasetId": "abcd2345",
                    "pinned": "1.0.0",
                    "role": "primary",
                }
            ],
            "tensor": None,
            "authors": [
                {"orcid": "0000-0002-1825-0097", "name": None, "role": "owner", "signed": True}
            ],
            "checksum": "a" * 64,
            "publishedAt": "2026-10-25T00:00:00.000Z",
        },
    }
    detail = parse_model(ModelDetail, wire, strict=False)
    assert detail.version.normalization is not None
    assert detail.version.normalization.pre == (280.0, 283.0)
    assert detail.version.datasets[0].pinned == "1.0.0"
    assert detail.is_default is True


def test_normalization_accepts_list_windows_equal_to_tuple_form() -> None:
    from_lists = parse_model(Normalization, {"pre": [280, 283], "post": [325, 340]})
    from_tuples = parse_model(Normalization, {"pre": (280, 283), "post": (325, 340)})
    assert from_lists == from_tuples
    assert from_lists.to_wire() == from_tuples.to_wire()


def test_normalization_accepts_numpy_array_windows() -> None:
    norm = parse_model(
        Normalization,
        {"pre": np.array([280.0, 283.0]), "post": np.array([325.0, 340.0])},
    )
    assert norm.pre == (280.0, 283.0) and norm.post == (325.0, 340.0)


def test_normalization_rejects_string_element_in_list_window() -> None:
    with pytest.raises(ValidationError, match="pre"):
        parse_model(Normalization, {"pre": [280, "283"], "post": [325, 340]})


def test_normalization_rejects_three_element_window() -> None:
    with pytest.raises(ValidationError, match="pre"):
        parse_model(Normalization, {"pre": [280, 283, 290], "post": [325, 340]})


def test_window_pair_accepts_lists() -> None:
    pair = parse_model(WindowPair, {"pre": [1, 2], "post": [3, 4]})
    assert pair.pre == (1.0, 2.0) and pair.post == (3.0, 4.0)


def test_validate_peaks_fit_provenance_columns() -> None:
    df = pd.DataFrame(
        {
            "center_ev": [285.4, 288.6],
            "kind": ["pi-star", "sigma-star"],
            "fit_tool": ["lmfit", "myfit"],
            "fit_version": ["1.3.2", float("nan")],
            "fit_code_hash": [float("nan"), "deadbeef"],
        }
    )
    peaks = validate_peaks(df)
    first, second = peaks[0].fit_provenance, peaks[1].fit_provenance
    assert first is not None and first.tool == "lmfit" and first.version == "1.3.2"
    assert first.code_hash is None
    assert second is not None and second.tool == "myfit" and second.code_hash == "deadbeef"
    assert second.version is None


def test_validate_peaks_fit_version_without_fit_tool_raises() -> None:
    with pytest.raises(ValidationError, match="row 0: fit_version/fit_code_hash need fit_tool"):
        validate_peaks(pd.DataFrame({"center_ev": [285.4], "fit_version": ["1.3"]}))
