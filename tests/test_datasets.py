import io
import json
import tarfile
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from xrayatlas import Client, ConflictError, NotFoundError, ValidationError, datasets
from xrayatlas.errors import AtlasError

HOST = "https://atlas.test"
UUID = "123e4567-e89b-12d3-a456-426614174000"


def row(
    energy: float, theta: float | None, phi: float | None, **extra: object
) -> dict[str, object]:
    base: dict[str, object] = {
        "dataset_id": UUID,
        "atlas_dataset_id": "abcd2345",
        "experiment_type": "TOTAL_ELECTRON_YIELD",
        "sample_id": "s",
        "sample_identifier": "S1",
        "molecule_id": "m",
        "molecule_iupac_name": "benzene",
        "edge_id": "e",
        "edge_label": "C K",
        "instrument_id": "i",
        "instrument_name": "BL 5.3.2.2",
        "facility_id": "f",
        "facility_name": "ALS",
        "publication_dois": "10.1/a;10.1/b",
        "energy_ev": energy,
        "raw_abs": 1.0,
        "i0": 2.0,
        "od": 0.5,
        "mass_absorption": 0.6,
        "beta": 0.7,
        "delta": None,
        "created_at": "2026-01-01T00:00:00.000Z",
        "polarization_id": None,
        "theta_deg": theta,
        "phi_deg": phi,
    }
    base.update(extra)
    return base


def payload(rows: list[dict[str, object]], version: str = "1.0.0") -> dict[str, object]:
    return {
        "columns": [],
        "rowCount": len(rows),
        "datasetVersions": {UUID: version},
        "rows": rows,
        "filters": {},
    }


def make_client(tmp_path: Path) -> Client:
    return Client(host=HOST, token=None, cache_dir=tmp_path / "c")


ROWS = [
    row(290.0, 90.0, 45.0),
    row(285.0, 90.0, 45.0),
    row(285.0, 45.0, 90.0),
    row(285.0, 45.0, 0.0),
]


@pytest.mark.parametrize(
    ("ref", "param", "value"),
    [
        ("abcd2345", "atlasDatasetId", "abcd2345"),
        ("d/abcd2345", "atlasDatasetId", "abcd2345"),
        ("/d/ABCD2345", "atlasDatasetId", "abcd2345"),
        (UUID.upper(), "datasetId", UUID),
    ],
)
def test_get_resolves_references(tmp_path: Path, ref: str, param: str, value: str) -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        datasets.get(ref, client=make_client(tmp_path))
        assert route.calls[0].request.url.params[param] == value


def test_get_rejects_malformed_reference(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        datasets.get("nope", client=make_client(tmp_path))


def test_unknown_dataset_raises_not_found(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload([]))
        )
        with pytest.raises(NotFoundError):
            datasets.get("abcd2345", client=make_client(tmp_path))


def test_spectra_sorted_by_geometry_then_energy_and_geometries_distinct(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    frame = ds.spectra()
    assert list(frame["theta_deg"]) == [45.0, 45.0, 90.0, 90.0]
    assert list(frame["phi_deg"]) == [0.0, 90.0, 45.0, 45.0]
    assert list(frame["energy_ev"][2:]) == [285.0, 290.0]
    assert ds.geometries() == [(45.0, 0.0), (45.0, 90.0), (90.0, 45.0)]
    assert (90.0, 45.0) in ds.geometries() and (45.0, 90.0) in ds.geometries()


def test_unknown_geometry_sorts_last(tmp_path: Path) -> None:
    rows = [row(285.0, None, None), row(285.0, 55.0, 0.0)]
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(rows))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert ds.geometries() == [(55.0, 0.0), (None, None)]


def test_raw_returns_only_raw_channel_and_i0(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert list(ds.spectra(raw=True).columns) == [
        "energy_ev",
        "theta_deg",
        "phi_deg",
        "raw_abs",
        "i0",
    ]
    assert "od" in ds.spectra().columns


def test_metadata_fields(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert ds.atlas_id == "abcd2345" and ds.version == "1.0.0" and ds.edge == "C K"
    assert ds.publication_dois == ("10.1/a", "10.1/b")


def test_second_get_does_not_refetch_full_export(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        datasets.get("abcd2345", client=client)
        datasets.get("abcd2345", client=client)
        limits = [call.request.url.params["limit"] for call in route.calls]
    assert limits.count("100000") == 1
    assert limits.count("1") == 2


def test_corrupted_cache_entry_is_refetched(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        datasets.get("abcd2345", client=client)
        data_path, _meta = client.cache._paths(f"{HOST}|dataset|{UUID}|1.0.0")
        data_path.write_bytes(b"{}")
        datasets.get("abcd2345", client=client)
        limits = [call.request.url.params["limit"] for call in route.calls]
    assert limits.count("100000") == 2


def test_pinned_version_is_served_from_cache_after_server_moves_on(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/datasets/export")
        route.mock(return_value=httpx.Response(200, json=payload(ROWS, "1.0.0")))
        datasets.get("abcd2345", client=client)
        route.mock(return_value=httpx.Response(200, json=payload(ROWS[:1], "1.0.1")))
        pinned = datasets.get("abcd2345", "1.0.0", client=client)
        latest = datasets.get("abcd2345", client=client)
    assert pinned.version == "1.0.0" and len(pinned.spectra()) == 4
    assert latest.version == "1.0.1" and len(latest.spectra()) == 1


def test_pinned_version_missing_from_cache_raises_conflict(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS, "1.0.1"))
        )
        with pytest.raises(ConflictError, match="1.0.1"):
            datasets.get("abcd2345", "1.0.0", client=make_client(tmp_path))


def test_server_422_historical_version_becomes_conflict(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            side_effect=[
                httpx.Response(200, json=payload(ROWS, "1.0.0")),
                httpx.Response(
                    422,
                    json={
                        "error": {
                            "code": "UNPROCESSABLE_ENTITY",
                            "message": "historical version replay is not available yet (requested 1.0.0, current 1.0.1).",  # noqa: E501
                        }
                    },
                ),
            ]
        )
        with pytest.raises(ConflictError):
            datasets.get("abcd2345", "1.0.0", client=make_client(tmp_path))


def test_bibtex_and_models(tmp_path: Path) -> None:
    summary = {
        "tag": "k7m2xq4n",
        "title": "Default model",
        "ownerOrcid": "0000-0002-1825-0097",
        "visibility": "public",
        "latestVersion": 1,
        "reviewState": "approved",
        "isDefault": True,
        "forkedFrom": None,
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        router.get(f"/api/v1/datasets/{UUID}/bibtex").mock(
            return_value=httpx.Response(200, text="@dataset{x}\n")
        )
        models_route = router.get("/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": [summary], "nextCursor": None})
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
        assert ds.bibtex() == "@dataset{x}\n"
        found = ds.models(scope="public")
    assert found[0].tag == "k7m2xq4n" and found[0].is_default is True
    assert models_route.calls[0].request.url.params["scope"] == "public"
    assert models_route.calls[0].request.url.params["dataset"] == UUID


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_aux_download_writes_only_aux_files(tmp_path: Path) -> None:
    bundle = _archive(
        {
            "spectrum-all-polarizations.csv": b"a,b\n",
            "experiment-aux/notes.txt": b"n",
            "sample-aux/prep.pdf": b"p",
        }
    )
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        router.get(f"/api/v1/datasets/{UUID}/bundle").mock(
            return_value=httpx.Response(200, content=bundle)
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
        written = ds.aux_files.download(tmp_path / "out")
        only_sample = ds.aux_files.download(tmp_path / "out2", kind="sample")
    assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in written) == [
        "experiment-aux/notes.txt",
        "sample-aux/prep.pdf",
    ]
    assert not (tmp_path / "out" / "spectrum-all-polarizations.csv").exists()
    assert [p.name for p in only_sample] == ["prep.pdf"]


def test_aux_download_rejects_unsafe_paths_before_writing(tmp_path: Path) -> None:
    bundle = _archive({"experiment-aux/ok.txt": b"ok", "../evil.txt": b"x"})
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        router.get(f"/api/v1/datasets/{UUID}/bundle").mock(
            return_value=httpx.Response(200, content=bundle)
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
        with pytest.raises(AtlasError, match="Unsafe path"):
            ds.aux_files.download(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_search_pages_lazily_and_loads_on_demand(tmp_path: Path) -> None:
    def page(ids: list[str], next_offset: int | None) -> dict[str, object]:
        return {
            "data": [
                {
                    "datasetId": UUID,
                    "atlasDatasetId": i,
                    "experimentType": "TOTAL_ELECTRON_YIELD",
                    "edge": {"label": "C K"},
                    "molecule": {"iupacName": "benzene"},
                    "spectrumPointCount": 4,
                    "datasetDoi": None,
                    "createdAt": "2026-01-01T00:00:00.000Z",
                }
                for i in ids
            ],
            "pagination": {"limit": 1, "offset": 0, "nextOffset": next_offset},
        }

    client = make_client(tmp_path)
    with respx.mock(base_url=HOST) as router:
        listing = router.get("/api/v1/datasets").mock(
            side_effect=[
                httpx.Response(200, json=page(["abcd2345"], 1)),
                httpx.Response(200, json=page(["efgh6789"], None)),
            ]
        )
        export = router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        results = datasets.search(molecule_id="m", page_size=1, client=client)
        first = next(results)
        assert listing.call_count == 1 and export.call_count == 0
        rest = list(results)
        assert listing.call_count == 2 and export.call_count == 0
        loaded = first.load()
    assert [first.atlas_dataset_id, rest[0].atlas_dataset_id] == ["abcd2345", "efgh6789"]
    assert isinstance(loaded.spectra(), pd.DataFrame)
    assert listing.calls[0].request.url.params["moleculeId"] == "m"


def test_non_numeric_od_is_an_error(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload([row(285.0, 90.0, 45.0, od="n/a")]))
        )
        with pytest.raises(ValidationError, match="od"):
            datasets.get("abcd2345", client=make_client(tmp_path))


def test_string_energy_is_an_error_naming_energy_ev(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(
                200, json=payload([row(0.0, 90.0, 45.0, energy_ev="285.5")])
            )
        )
        with pytest.raises(ValidationError, match="energy_ev"):
            datasets.get("abcd2345", client=make_client(tmp_path))


def test_none_channel_value_stays_missing(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload([row(285.0, 90.0, 45.0, od=None)]))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert pd.isna(ds.spectra()["od"]).all()


def test_capped_export_raises_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two_rows = [row(285.0, 90.0, 45.0), row(290.0, 90.0, 45.0)]
    monkeypatch.setattr(datasets, "EXPORT_LIMIT", 2)
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(two_rows))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert ds.truncated is True
    with pytest.raises(AtlasError, match="cap"):
        ds.spectra()
    allowed = ds.spectra(allow_truncated=True)
    assert len(allowed) == 2


def test_short_export_is_not_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets, "EXPORT_LIMIT", 2)
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS[:1]))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert ds.truncated is False
    assert len(ds.spectra()) == 1


def test_version_is_validated_before_any_request(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        for bad in ("", "1.2"):
            with pytest.raises(ValidationError, match="version must look like"):
                datasets.get("abcd2345", bad, client=make_client(tmp_path))
        assert len(router.calls) == 0


def test_publication_dois_strip_whitespace_after_separator(tmp_path: Path) -> None:
    rows = [row(285.0, 90.0, 45.0, publication_dois="10.1/a; 10.1/b ")]
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(rows))
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
    assert ds.publication_dois == ("10.1/a", "10.1/b")


def test_search_rejects_page_size_out_of_range(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        router.get("/api/v1/datasets").mock(return_value=httpx.Response(200, json={}))
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        for page_size in (0, 201):
            with pytest.raises(ValidationError, match="page_size"):
                list(datasets.search(page_size=page_size, client=make_client(tmp_path)))
        assert len(router.calls) == 0


def test_search_summary_without_dataset_id_raises(tmp_path: Path) -> None:
    body: dict[str, object] = {
        "data": [{"experimentType": "TOTAL_ELECTRON_YIELD"}],
        "pagination": {},
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets").mock(return_value=httpx.Response(200, json=body))
        with pytest.raises(AtlasError, match="datasetId"):
            list(datasets.search(client=make_client(tmp_path)))


def test_search_echoed_next_offset_terminates(tmp_path: Path) -> None:
    body: dict[str, object] = {
        "data": [],
        "pagination": {"limit": 50, "offset": 0, "nextOffset": 0},
    }
    with respx.mock(base_url=HOST) as router:
        listing = router.get("/api/v1/datasets").mock(return_value=httpx.Response(200, json=body))
        assert list(datasets.search(client=make_client(tmp_path))) == []
        assert listing.call_count == 1


def test_models_non_dict_item_raises(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        router.get("/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": ["not a dict"], "nextCursor": None})
        )
        ds = datasets.get("abcd2345", client=make_client(tmp_path))
        with pytest.raises(AtlasError, match="Unexpected models response"):
            ds.models()


def test_cache_bytes_never_contain_the_token(tmp_path: Path) -> None:
    client = Client(host=HOST, token="xra_secret_value", cache_dir=tmp_path / "c")
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        datasets.get("abcd2345", client=client)
    for path in client.cache.root.rglob("*"):
        if path.is_file():
            assert b"xra_secret_value" not in path.read_bytes()


def test_cache_is_host_scoped(tmp_path: Path) -> None:
    shared = tmp_path / "c"
    client_a = Client(host=HOST, cache_dir=shared)
    client_b = Client(host="https://other.test", cache_dir=shared)
    with respx.mock(base_url=HOST) as first:
        first.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        ds_a = datasets.get("abcd2345", client=client_a)
    with respx.mock(base_url="https://other.test") as second:
        route_b = second.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS[:1], "1.0.0"))
        )
        ds_b = datasets.get("abcd2345", client=client_b)
        again = datasets.get("abcd2345", client=client_b)
    assert len(ds_a.spectra()) == 4
    assert len(ds_b.spectra()) == 1
    assert len(again.spectra()) == 1
    assert route_b.call_count == 3  # two for the first get (probe + full), one probe when cached


def test_pinned_cache_hit_does_not_refetch_the_export(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS, "1.0.0"))
        )
        datasets.get("abcd2345", client=client)
        route.mock(return_value=httpx.Response(200, json=payload(ROWS[:1], "1.0.1")))
        pinned = datasets.get("abcd2345", "1.0.0", client=client)
    assert pinned.version == "1.0.0" and len(pinned.spectra()) == 4
    full_exports = [call for call in route.calls if call.request.url.params["limit"] == "100000"]
    assert len(full_exports) == 1


def test_corrupt_cache_entry_is_evicted_and_refetched(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    key = f"{HOST}|dataset|{UUID}|1.0.0"
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=payload(ROWS))
        )
        datasets.get("abcd2345", client=client)
        client.cache.put(key, b'{"rows": [broken')
        refreshed = datasets.get("abcd2345", client=client)
    assert len(refreshed.spectra()) == 4
    stored = client.cache.get(key)
    assert stored is not None
    assert b"broken" not in stored
    assert json.loads(stored)["rowCount"] == 4
