"""Tests for the session lockfile: registry, JSON shape, and pinned restore."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from xrayatlas import Client, ConflictError, ValidationError, datasets, models
from xrayatlas.lock import lock, register_dataset, register_model, reset, restore

HOST = "https://atlas.test"
EXPERIMENT = "123e4567-e89b-12d3-a456-426614174000"
TAG = "k7m2xq4n"
DATASET_VERSION = "1.0.2"
CHECKSUM = "a" * 64
OTHER_CHECKSUM = "b" * 64
ORCID = "0000-0002-1825-0097"


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    reset()


def make_client(tmp_path: Path, token: str = "xra_test") -> Client:
    return Client(host=HOST, token=token, cache_dir=tmp_path / "c")


def export_payload(version: str) -> dict[str, object]:
    return {
        "rowCount": 1,
        "datasetVersions": {EXPERIMENT: version},
        "rows": [
            {
                "dataset_id": EXPERIMENT,
                "atlas_dataset_id": "abcd2345",
                "energy_ev": 285.0,
                "raw_abs": 1.0,
                "i0": 2.0,
            }
        ],
    }


def model_detail(*, version: int | None = 1, checksum: str | None = CHECKSUM) -> dict[str, object]:
    return {
        "tag": TAG,
        "title": "My model",
        "ownerOrcid": ORCID,
        "visibility": "public",
        "latestVersion": version,
        "version": {
            "versionNumber": version,
            "status": "published" if version is not None else "draft",
            "checksum": checksum,
        },
    }


def test_lock_records_pinned_versions_only(tmp_path: Path) -> None:
    path = tmp_path / "atlas.lock"
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=export_payload(DATASET_VERSION))
        )
        router.get(f"/api/v1/models/{TAG}").mock(
            side_effect=[
                httpx.Response(200, json=model_detail(version=1)),
                httpx.Response(200, json=model_detail(version=None)),
            ]
        )
        client = make_client(tmp_path)
        datasets.get("d/abcd2345", client=client)
        models.get(TAG, client=client)
        models.get(TAG, "draft", client=client)
        lock(path)
    assert (
        path.read_text(encoding="utf-8")
        == json.dumps(
            {
                "version": 1,
                "host": HOST,
                "datasets": [{"id": EXPERIMENT, "version": DATASET_VERSION}],
                "models": [{"tag": TAG, "version": 1, "checksum": CHECKSUM}],
            },
            indent=2,
        )
        + "\n"
    )


def test_lock_contains_no_token(tmp_path: Path) -> None:
    path = tmp_path / "atlas.lock"
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=export_payload(DATASET_VERSION))
        )
        router.get(f"/api/v1/models/{TAG}").mock(
            return_value=httpx.Response(200, json=model_detail())
        )
        client = make_client(tmp_path, token="xra_secret_value")
        datasets.get("d/abcd2345", client=client)
        models.get(TAG, client=client)
        lock(path)
    text = path.read_text(encoding="utf-8")
    assert "xra_secret_value" not in text
    assert "authorization" not in text.lower()
    assert ORCID not in text
    payload = json.loads(text)
    assert payload["models"] == [{"tag": TAG, "version": 1, "checksum": CHECKSUM}]
    assert payload["datasets"] == [{"id": EXPERIMENT, "version": DATASET_VERSION}]


def test_lock_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.lock"
    second = tmp_path / "second.lock"
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version="1.0.2")
    register_model(host=HOST, tag=TAG, version=2, checksum=CHECKSUM)
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version="0.9.0")
    register_model(host=HOST, tag=TAG, version=1, checksum=OTHER_CHECKSUM)
    lock(first)
    reset()
    register_model(host=HOST, tag=TAG, version=2, checksum=CHECKSUM)
    register_model(host=HOST, tag=TAG, version=2, checksum=CHECKSUM)
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version="1.0.2")
    register_model(host=HOST, tag=TAG, version=1, checksum=OTHER_CHECKSUM)
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version="0.9.0")
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version="0.9.0")
    lock(second)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["datasets"] == [
        {"id": EXPERIMENT, "version": "0.9.0"},
        {"id": EXPERIMENT, "version": "1.0.2"},
    ]
    assert payload["models"] == [
        {"tag": TAG, "version": 1, "checksum": OTHER_CHECKSUM},
        {"tag": TAG, "version": 2, "checksum": CHECKSUM},
    ]


def test_lock_rejects_entries_from_two_hosts(tmp_path: Path) -> None:
    register_dataset(host="https://a.test", experiment_id=EXPERIMENT, version=DATASET_VERSION)
    register_model(host="https://b.test", tag=TAG, version=1, checksum=CHECKSUM)
    path = tmp_path / "atlas.lock"
    with pytest.raises(ValidationError, match="one Atlas host"):
        lock(path)
    assert not path.exists()


def test_restore_pins_each_version(tmp_path: Path) -> None:
    path = tmp_path / "atlas.lock"
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version=DATASET_VERSION)
    register_model(host=HOST, tag=TAG, version=2, checksum=CHECKSUM)
    lock(path)
    reset()
    with respx.mock(base_url=HOST) as router:
        export_route = router.get("/api/v1/datasets/export").mock(
            return_value=httpx.Response(200, json=export_payload(DATASET_VERSION))
        )
        model_route = router.get(f"/api/v1/models/{TAG}").mock(
            return_value=httpx.Response(200, json=model_detail(version=2))
        )
        assert restore(path) is None
    assert [call.request.url.params.get("version") for call in export_route.calls] == [
        None,
        DATASET_VERSION,
    ]
    assert export_route.calls[1].request.url.params["datasetId"] == EXPERIMENT
    assert model_route.calls[0].request.url.params["version"] == "2"
    # Restoring re-registered what it fetched, so a second lock is byte-identical.
    replay = tmp_path / "replay.lock"
    lock(replay)
    assert replay.read_bytes() == path.read_bytes()


def test_restore_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "atlas.lock"
    register_model(host=HOST, tag=TAG, version=2, checksum=CHECKSUM)
    lock(path)
    reset()
    with respx.mock(base_url=HOST) as router:
        router.get(f"/api/v1/models/{TAG}").mock(
            return_value=httpx.Response(200, json=model_detail(version=2, checksum=OTHER_CHECKSUM))
        )
        with pytest.raises(ConflictError, match="checksum") as excinfo:
            restore(path)
    assert CHECKSUM in str(excinfo.value) and OTHER_CHECKSUM in str(excinfo.value)


def test_restore_rejects_a_bad_lockfile(tmp_path: Path) -> None:
    wrong_version = tmp_path / "wrong-version.lock"
    wrong_version.write_text(
        json.dumps({"version": 2, "host": HOST, "datasets": [], "models": []}), encoding="utf-8"
    )
    missing_host = tmp_path / "missing-host.lock"
    missing_host.write_text(
        json.dumps({"version": 1, "datasets": [], "models": []}), encoding="utf-8"
    )
    bad_host = tmp_path / "bad-host.lock"
    bad_host.write_text(
        json.dumps({"version": 1, "host": "not-an-origin", "datasets": [], "models": []}),
        encoding="utf-8",
    )
    not_json = tmp_path / "not-json.lock"
    not_json.write_text("not a lockfile", encoding="utf-8")
    true_version = tmp_path / "true-version.lock"
    true_version.write_text(
        json.dumps({"version": True, "host": HOST, "datasets": [], "models": []}), encoding="utf-8"
    )
    float_version = tmp_path / "float-version.lock"
    float_version.write_text(
        json.dumps({"version": 1.0, "host": HOST, "datasets": [], "models": []}), encoding="utf-8"
    )
    with pytest.raises(ValidationError, match="version"):
        restore(wrong_version)
    with pytest.raises(ValidationError, match="host"):
        restore(missing_host)
    with pytest.raises(ValidationError, match="host"):
        restore(bad_host)
    with pytest.raises(ValidationError, match="not valid JSON"):
        restore(not_json)
    with pytest.raises(ValidationError, match="version"):
        restore(true_version)
    with pytest.raises(ValidationError, match="version"):
        restore(float_version)


def test_restore_normalizes_the_lockfile_host(tmp_path: Path) -> None:
    path = tmp_path / "atlas.lock"
    path.write_text(
        json.dumps({"version": 1, "host": "HTTPS://Atlas.Test/", "datasets": [], "models": []}),
        encoding="utf-8",
    )
    assert restore(path) is None


def test_published_version_without_checksum_is_not_registered(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get(f"/api/v1/models/{TAG}").mock(
            return_value=httpx.Response(200, json=model_detail(checksum=None))
        )
        models.get(TAG, client=make_client(tmp_path))
    path = lock(tmp_path / "atlas.lock")
    assert json.loads(path.read_text(encoding="utf-8"))["models"] == []


def test_registry_dedupes_and_resets(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "atlas.lock"
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version=DATASET_VERSION)
    register_dataset(host=HOST, experiment_id=EXPERIMENT, version=DATASET_VERSION)
    register_model(host=HOST, tag=TAG, version=1, checksum=CHECKSUM)
    register_model(host=HOST, tag=TAG, version=1, checksum=CHECKSUM)
    lock(path)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["datasets"] == [{"id": EXPERIMENT, "version": DATASET_VERSION}]
    assert payload["models"] == [{"tag": TAG, "version": 1, "checksum": CHECKSUM}]
    reset()
    empty = lock(tmp_path / "empty.lock")
    assert json.loads(empty.read_text(encoding="utf-8")) == {
        "version": 1,
        "host": "https://xrayatlas.wsu.edu",
        "datasets": [],
        "models": [],
    }
