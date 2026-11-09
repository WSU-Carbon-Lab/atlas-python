import hashlib
import io
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx

from xrayatlas import Client, ConflictError, NotFoundError, ValidationError, models
from xrayatlas.errors import AtlasError

HOST = "https://atlas.test"
EXPERIMENT = "123e4567-e89b-12d3-a456-426614174000"


def make_client(tmp_path: Path) -> Client:
    return Client(host=HOST, token="xra_test", cache_dir=tmp_path / "c")


def detail(
    *,
    version: int | None = 1,
    tensor: dict[str, object] | None = None,
    peaks: list[dict[str, object]] | None = None,
    pinned: str = "1.0.0",
    normalization: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "tag": "k7m2xq4n",
        "title": "My model",
        "ownerOrcid": "0000-0002-1825-0097",
        "visibility": "public",
        "latestVersion": 1,
        "reviewState": None,
        "isDefault": False,
        "forkedFrom": {"tag": "p8q9r2s3", "versionNumber": 1},
        "description": "d",
        "versions": [1],
        "hasDraft": False,
        "version": {
            "versionNumber": version,
            "status": "published" if version is not None else "draft",
            "normalization": normalization
            or {"schemaVersion": 1, "scope": "unified", "pre": [280, 283], "post": [325, 340]},
            "peaks": peaks if peaks is not None else [],
            "datasets": [
                {
                    "experimentId": EXPERIMENT,
                    "atlasDatasetId": "abcd2345",
                    "pinned": pinned,
                    "role": "primary",
                }
            ],
            "tensor": tensor,
            "authors": [
                {"orcid": "0000-0002-1825-0097", "name": "A", "role": "owner", "signed": True}
            ],
            "checksum": "a" * 64,
            "publishedAt": "2026-10-25T00:00:00.000Z",
        },
    }


PEAKS = [
    {"ordinal": 1, "centerEv": 288.6, "kind": "sigma-star", "width": 1.2},
    {"ordinal": 0, "centerEv": 285.4, "kind": "pi-star", "amplitude": 3.0, "params": {"chi2": 1.1}},
]


def test_get_parses_detail_and_orders_peaks(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(peaks=PEAKS))
        )
        model = models.get("m/K7M2XQ4N", client=make_client(tmp_path))
        assert route.calls[0].request.url.params["version"] == "latest"
    assert model.tag == "k7m2xq4n" and model.version_number == 1 and model.forked_from is not None
    frame = model.peaks
    assert list(frame["ordinal"]) == [0, 1]
    assert list(frame["center_ev"]) == [285.4, 288.6]
    assert list(frame.columns)[:3] == ["ordinal", "center_ev", "kind"]
    assert frame.loc[0, "params"] == {"chi2": 1.1}


def test_empty_peaks_still_have_columns(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
    assert model.peaks.empty and "center_ev" in model.peaks.columns


def test_numeric_and_draft_versions_are_sent(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(version=None))
        )
        models.get("k7m2xq4n", 3, client=make_client(tmp_path))
        models.get("k7m2xq4n", "draft", client=make_client(tmp_path))
    assert [c.request.url.params["version"] for c in route.calls] == ["3", "draft"]


def test_invisible_model_is_not_found(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(
                404, json={"error": {"code": "NOT_FOUND", "message": "Model not found."}}
            )
        )
        with pytest.raises(NotFoundError):
            models.get("k7m2xq4n", client=make_client(tmp_path))


def test_model_detail_is_not_cached(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail())
        )
        client = make_client(tmp_path)
        models.get("k7m2xq4n", client=client)
        models.get("k7m2xq4n", client=client)
    assert route.call_count == 2


def test_normalized_returns_aligned_channels_and_sends_dataset(tmp_path: Path) -> None:
    channels = {
        "energyEv": [285.0, 286.0],
        "od": [0.1, None],
        "massAbsorption": [0.2, 0.3],
        "beta": [None, None],
        "polarizationId": ["p1", "p1"],
        "thetaDeg": [55.0, 55.0],
        "phiDeg": [0.0, 0.0],
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        route = router.get("/api/v1/models/k7m2xq4n/normalized").mock(
            return_value=httpx.Response(
                200, json={"experimentId": EXPERIMENT, "pinned": "1.0.0", "channels": channels}
            )
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        frame = model.normalized("abcd2345")
    assert route.calls[0].request.url.params["dataset"] == "abcd2345"
    assert route.calls[0].request.url.params["version"] == "1"
    assert list(frame.columns) == [
        "energy_ev",
        "od",
        "mass_absorption",
        "beta",
        "polarization_id",
        "theta_deg",
        "phi_deg",
    ]
    assert np.isnan(frame.loc[1, "od"]) and frame.loc[1, "mass_absorption"] == 0.3
    assert frame["beta"].isna().all()


def test_normalized_rejects_non_numeric_channel_value(tmp_path: Path) -> None:
    channels = {
        "energyEv": [285.0, 286.0],
        "od": ["n/a", 0.2],
        "massAbsorption": [0.2, 0.3],
        "beta": [None, 0.1],
        "polarizationId": ["p1", "p1"],
        "thetaDeg": [55.0, 55.0],
        "phiDeg": [0.0, 0.0],
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        router.get("/api/v1/models/k7m2xq4n/normalized").mock(
            return_value=httpx.Response(200, json={"channels": channels})
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        with pytest.raises(ValidationError, match="channel od index 0 must be a finite number"):
            model.normalized()


def test_normalized_rejects_shorter_channel_arrays(tmp_path: Path) -> None:
    channels = {
        "energyEv": [285.0, 286.0],
        "od": [0.1],
        "massAbsorption": [0.2, 0.3],
        "beta": [None, 0.1],
        "polarizationId": ["p1", "p1"],
        "thetaDeg": [55.0, 55.0],
        "phiDeg": [0.0, 0.0],
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        router.get("/api/v1/models/k7m2xq4n/normalized").mock(
            return_value=httpx.Response(200, json={"channels": channels})
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        with pytest.raises(AtlasError, match="Unexpected normalized channels"):
            model.normalized()


def test_normalized_null_channel_values_become_nan(tmp_path: Path) -> None:
    channels = {
        "energyEv": [285.0, 286.0],
        "od": [0.1, None],
        "massAbsorption": [0.2, 0.3],
        "beta": [None, 0.1],
        "polarizationId": ["p1", "p1"],
        "thetaDeg": [55.0, 55.0],
        "phiDeg": [0.0, 0.0],
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        router.get("/api/v1/models/k7m2xq4n/normalized").mock(
            return_value=httpx.Response(200, json={"channels": channels})
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        frame = model.normalized()
    assert np.isnan(frame.loc[0, "beta"]) and frame.loc[1, "beta"] == 0.1


def test_normalized_validates_the_dataset_reference(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        with pytest.raises(ValidationError, match="not a dataset reference"):
            model.normalized("nope")


def test_normalized_undefined_surfaces_as_validation_error(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        router.get("/api/v1/models/k7m2xq4n/normalized").mock(
            return_value=httpx.Response(
                422,
                json={
                    "error": {"code": "UNPROCESSABLE_ENTITY", "message": "NORMALIZATION_UNDEFINED"}
                },
            )
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        with pytest.raises(ValidationError, match="NORMALIZATION_UNDEFINED"):
            model.normalized()


def test_lineage_and_cite(tmp_path: Path) -> None:
    lineage = {
        "ancestors": [
            {
                "tag": "p8q9r2s3",
                "versionNumber": 1,
                "ownerOrcid": "0000-0001-0000-0001",
                "title": "Root",
                "kind": "fork",
            }
        ],
        "hiddenAncestors": 2,
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        router.get("/api/v1/models/k7m2xq4n/lineage").mock(
            return_value=httpx.Response(200, json=lineage)
        )
        router.get("/api/v1/models/k7m2xq4n/bibtex").mock(
            return_value=httpx.Response(200, text="@misc{x}\n")
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        found = model.lineage()
        text = model.cite()
    assert found.hidden_ancestors == 2 and found.ancestors[0].tag == "p8q9r2s3"
    assert text == "@misc{x}\n"


def test_draft_cannot_be_cited(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(version=None))
        )
        model = models.get("k7m2xq4n", "draft", client=make_client(tmp_path))
        with pytest.raises(AtlasError, match="published"):
            model.cite()


def _npz(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, tensor=array)
    return buffer.getvalue()


def _tensor_info(data: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byteSize": len(data),
        "manifest": {
            "schemaVersion": 1,
            "format": "npz",
            "energyEv": {"min": 280.0, "max": 340.0, "count": 2},
            "components": ["xx"],
            "frame": "lab",
            "units": "1",
            "dtype": "float64",
            "shape": [2, 1],
        },
    }


def test_tensor_load_verifies_digest_and_caches(tmp_path: Path) -> None:
    data = _npz(np.array([[1.0], [2.0]]))
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=_tensor_info(data)))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/blob?sig=1",
                    "sha256": "x",
                    "byteSize": 0,
                    "manifest": None,
                },
            )
        )
        blob = router.get("https://storage.test/blob?sig=1").mock(
            return_value=httpx.Response(200, content=data)
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        assert model.tensor is not None
        first = model.tensor.load()
        second = model.tensor.load()
    assert first.shape == (2, 1) and np.array_equal(first, second)
    assert blob.call_count == 1
    assert "authorization" not in blob.calls[0].request.headers


def test_tensor_with_wrong_digest_is_rejected(tmp_path: Path) -> None:
    data = _npz(np.array([[1.0], [2.0]]))
    info = _tensor_info(data)
    info["sha256"] = "0" * 64
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=info))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/b",
                    "sha256": "0" * 64,
                    "byteSize": len(data),
                    "manifest": None,
                },
            )
        )
        router.get("https://storage.test/b").mock(return_value=httpx.Response(200, content=data))
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        assert model.tensor is not None
        with pytest.raises(AtlasError, match="digest"):
            model.tensor.load()


def test_corrupt_cache_entry_is_evicted_and_repaired(tmp_path: Path) -> None:
    data = _npz(np.array([[1.0], [2.0]]))
    fake = _npz(np.array([[9.0], [9.0]]))
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=_tensor_info(data)))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/blob?sig=1",
                    "sha256": "x",
                    "byteSize": 0,
                    "manifest": None,
                },
            )
        )
        blob = router.get("https://storage.test/blob?sig=1").mock(
            return_value=httpx.Response(200, content=data)
        )
        client = make_client(tmp_path)
        key = f"{HOST}|tensor|{hashlib.sha256(data).hexdigest()}"
        client.cache.put(key, fake)
        model = models.get("k7m2xq4n", client=client)
        assert model.tensor is not None
        first = model.tensor.load()
        second = model.tensor.load()
    assert np.array_equal(first, np.array([[1.0], [2.0]])) and np.array_equal(first, second)
    assert blob.call_count == 1
    assert client.cache.get(key) == data


def test_tensor_shape_and_dtype_are_checked_against_manifest(tmp_path: Path) -> None:
    data = _npz(np.array([[1.0], [2.0]]))
    info = _tensor_info(data)
    manifest = info["manifest"]
    assert isinstance(manifest, dict)
    info["manifest"] = {**manifest, "shape": [3, 1]}
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=info))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/blob?sig=1",
                    "sha256": "x",
                    "byteSize": 0,
                    "manifest": None,
                },
            )
        )
        router.get("https://storage.test/blob?sig=1").mock(
            return_value=httpx.Response(200, content=data)
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        assert model.tensor is not None
        with pytest.raises(AtlasError, match="manifest shape"):
            model.tensor.load()


def test_tensor_dtype_mismatch_is_checked_against_manifest(tmp_path: Path) -> None:
    data = _npz(np.array([[1.0], [2.0]]))
    info = _tensor_info(data)
    manifest = info["manifest"]
    assert isinstance(manifest, dict)
    info["manifest"] = {**manifest, "dtype": "complex128"}
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=info))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/blob?sig=1",
                    "sha256": "x",
                    "byteSize": 0,
                    "manifest": None,
                },
            )
        )
        router.get("https://storage.test/blob?sig=1").mock(
            return_value=httpx.Response(200, content=data)
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        assert model.tensor is not None
        with pytest.raises(AtlasError, match="manifest shape"):
            model.tensor.load()


def test_tensor_load_widens_a_stored_float32_array(tmp_path: Path) -> None:
    stored = np.array([[1.0], [2.0]], dtype=np.float32)
    data = _npz(stored)
    info = _tensor_info(data)
    manifest = info["manifest"]
    assert isinstance(manifest, dict)
    info["manifest"] = {**manifest, "dtype": "float32"}
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=info))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/blob?sig=1",
                    "sha256": "x",
                    "byteSize": 0,
                    "manifest": None,
                },
            )
        )
        router.get("https://storage.test/blob?sig=1").mock(
            return_value=httpx.Response(200, content=data)
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        assert model.tensor is not None
        array = model.tensor.load()
    assert array.dtype == np.dtype(np.float64)
    assert np.array_equal(array, np.array([[1.0], [2.0]]))


def test_tensor_load_rejects_a_stored_integer_array(tmp_path: Path) -> None:
    data = _npz(np.array([[1], [2]], dtype=np.int64))
    info = _tensor_info(data)
    manifest = info["manifest"]
    assert isinstance(manifest, dict)
    info["manifest"] = {**manifest, "dtype": "int64"}
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(tensor=info))
        )
        router.get("/api/v1/models/k7m2xq4n/tensor").mock(
            return_value=httpx.Response(
                200,
                json={
                    "url": "https://storage.test/blob?sig=1",
                    "sha256": "x",
                    "byteSize": 0,
                    "manifest": None,
                },
            )
        )
        router.get("https://storage.test/blob?sig=1").mock(
            return_value=httpx.Response(200, content=data)
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        assert model.tensor is not None
        with pytest.raises(AtlasError, match="dtype int64"):
            model.tensor.load()


def test_model_without_tensor_has_none(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(return_value=httpx.Response(200, json=detail()))
        assert models.get("k7m2xq4n", client=make_client(tmp_path)).tensor is None


def _export_payload(version: str) -> dict[str, object]:
    row = {
        "dataset_id": EXPERIMENT,
        "atlas_dataset_id": "abcd2345",
        "experiment_type": "TOTAL_ELECTRON_YIELD",
        "sample_id": "s",
        "sample_identifier": "S",
        "molecule_id": "m",
        "molecule_iupac_name": "benzene",
        "edge_id": "e",
        "edge_label": "C K",
        "instrument_id": "i",
        "instrument_name": "BL",
        "facility_id": "f",
        "facility_name": "ALS",
        "publication_dois": "",
        "energy_ev": 285.0,
        "raw_abs": 1.0,
        "i0": 2.0,
        "od": 0.5,
        "mass_absorption": 0.6,
        "beta": 0.7,
        "delta": None,
        "created_at": "2026-01-01T00:00:00.000Z",
        "polarization_id": None,
        "theta_deg": 55.0,
        "phi_deg": 0.0,
    }
    return {
        "columns": [],
        "rowCount": 1,
        "datasetVersions": {EXPERIMENT: version},
        "rows": [row],
        "filters": {},
    }


def test_raw_recall_uses_current_version_within_same_major(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(pinned="1.0.0"))
        )
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=_export_payload("1.2.3"))
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        frame = model.datasets[0].spectra(raw=True)
    assert list(frame.columns) == ["energy_ev", "theta_deg", "phi_deg", "raw_abs", "i0"]


def test_raw_recall_refuses_a_different_major(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(pinned="1.0.0"))
        )
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=_export_payload("2.0.0"))
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        with pytest.raises(ConflictError, match="major"):
            model.datasets[0].spectra(raw=True)


def test_derived_spectra_require_the_pinned_version(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models/k7m2xq4n").mock(
            return_value=httpx.Response(200, json=detail(pinned="1.0.0"))
        )
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=_export_payload("1.0.1"))
        )
        model = models.get("k7m2xq4n", client=make_client(tmp_path))
        with pytest.raises(ConflictError):
            model.datasets[0].spectra()


def test_mine_and_shared_with_me_follow_cursors(tmp_path: Path) -> None:
    def item(tag: str) -> dict[str, object]:
        return {
            "tag": tag,
            "title": tag,
            "ownerOrcid": "0000-0002-1825-0097",
            "visibility": "private",
            "latestVersion": None,
            "reviewState": None,
            "isDefault": False,
            "forkedFrom": None,
        }

    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/models").mock(
            side_effect=[
                httpx.Response(200, json={"items": [item("aaaaaaaa")], "nextCursor": "c1"}),
                httpx.Response(200, json={"items": [item("bbbbbbbb")], "nextCursor": None}),
                httpx.Response(200, json={"items": [item("cccccccc")], "nextCursor": None}),
            ]
        )
        client = make_client(tmp_path)
        mine = [m.tag for m in models.mine(page_size=1, client=client)]
        shared = [m.tag for m in models.shared_with_me(client=client)]
    assert mine == ["aaaaaaaa", "bbbbbbbb"] and shared == ["cccccccc"]
    assert route.calls[0].request.url.params["scope"] == "mine"
    assert route.calls[1].request.url.params["cursor"] == "c1"
    assert route.calls[2].request.url.params["scope"] == "shared"


def test_read_inputs_are_validated_before_any_request(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        with pytest.raises(ValidationError, match="not a model tag"):
            models.get("", client=client)
        with pytest.raises(ValidationError, match="version must be"):
            models.get("k7m2xq4n", 0, client=client)
        with pytest.raises(ValidationError, match="version must be"):
            models.get("k7m2xq4n", "v3", client=client)  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="page_size"):
            models.mine(page_size=0, client=client)
        with pytest.raises(ValidationError, match="page_size"):
            models.shared_with_me(page_size=201, client=client)
        assert len(router.calls) == 0
