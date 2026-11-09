import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from xrayatlas import Client, ConflictError, Normalization, Tensor, ValidationError, models
from xrayatlas import publish as publish_module
from xrayatlas.errors import ApiError, AtlasError

HOST = "https://atlas.test"
STORAGE = "https://storage.test"
EXPERIMENT = "123e4567-e89b-12d3-a456-426614174000"
ORCID = "0000-0002-1825-0097"


def make_client(tmp_path: Path) -> Client:
    return Client(host=HOST, token="xra_test", cache_dir=tmp_path / "c")


def detail(
    tag: str, *, version: int | None = 1, peaks: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "tag": tag,
        "title": "Source",
        "ownerOrcid": ORCID,
        "visibility": "public",
        "latestVersion": 1,
        "reviewState": None,
        "isDefault": tag == "defaultm",
        "forkedFrom": None,
        "description": "from source",
        "versions": [1],
        "hasDraft": False,
        "version": {
            "versionNumber": version,
            "status": "published" if version else "draft",
            "normalization": {
                "schemaVersion": 1,
                "scope": "unified",
                "pre": [280, 283],
                "post": [325, 340],
            },
            "peaks": peaks or [],
            "datasets": [
                {
                    "experimentId": EXPERIMENT,
                    "atlasDatasetId": "abcd2345",
                    "pinned": "1.0.0",
                    "role": "primary",
                }
            ],
            "tensor": None,
            "authors": [],
            "checksum": "a" * 64,
            "publishedAt": None,
        },
    }


SOURCE_PEAKS = [
    {
        "ordinal": 0,
        "centerEv": 285.4,
        "kind": "pi-star",
        "params": {"chi2": 1.1},
        "fitProvenance": {"tool": "lmfit", "version": "1.3"},
    }
]


class Atlas:
    """Registers the routes the builder flow uses and records the writes."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.router = router
        router.get("/api/v1/models", params={"dataset": EXPERIMENT, "scope": "public"}).mock(
            return_value=httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "tag": "defaultm",
                            "title": "d",
                            "ownerOrcid": ORCID,
                            "visibility": "public",
                            "latestVersion": 1,
                            "reviewState": None,
                            "isDefault": True,
                            "forkedFrom": None,
                        }
                    ],
                    "nextCursor": None,
                },
            )
        )
        router.get("/api/v1/models/defaultm").mock(
            return_value=httpx.Response(200, json=detail("defaultm", peaks=SOURCE_PEAKS))
        )
        self.fork = router.post("/api/v1/models/defaultm/fork").mock(
            return_value=httpx.Response(201, json={"tag": "newmodel"})
        )
        self.draft = router.put("/api/v1/models/newmodel/draft").mock(
            return_value=httpx.Response(200, json={"tag": "newmodel"})
        )
        self.upload = router.post("/api/v1/models/newmodel/tensor/upload-url").mock(
            return_value=httpx.Response(
                200, json={"uploadUrl": f"{STORAGE}/put?sig=1", "storagePath": "m/u.bin"}
            )
        )
        self.put = router.put(f"{STORAGE}/put?sig=1").mock(return_value=httpx.Response(200))
        self.commit = router.post("/api/v1/models/newmodel/tensor/commit").mock(
            return_value=httpx.Response(200, json={"tag": "newmodel", "storagePath": "m/u.bin"})
        )
        self.read_draft = router.get("/api/v1/models/newmodel").mock(
            return_value=httpx.Response(200, json=detail("newmodel", version=None))
        )
        self.publish = router.post("/api/v1/models/newmodel/publish").mock(
            return_value=httpx.Response(200, json={"versionNumber": 1, "checksum": "b" * 64})
        )


def peaks_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"center_ev": [285.4, 288.6], "kind": ["pi-star", "sigma-star"], "chi2": [1.0, 2.0]}
    )


def test_builder_writes_only_on_clean_exit_and_does_not_publish(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        with publish_module.model(
            [EXPERIMENT], title="Mine", client=make_client(tmp_path)
        ) as builder:
            assert builder.normalization is not None and len(builder.peaks) == 1
            assert atlas.fork.call_count == 0 and atlas.draft.call_count == 0
            builder.normalization = Normalization(pre=[281, 284], post=[326, 341])
            builder.peaks = peaks_df()
        assert atlas.fork.call_count == 1 and atlas.draft.call_count == 1
        assert atlas.publish.call_count == 0
        assert json.loads(atlas.fork.calls[0].request.content) == {"title": "Mine"}
        body = json.loads(atlas.draft.calls[0].request.content)
        assert body["title"] == "Mine" and body["description"] == "from source"
        assert body["normalization"] == {
            "schemaVersion": 1,
            "scope": "unified",
            "pre": [281.0, 284.0],
            "post": [326.0, 341.0],
        }
        assert [p["centerEv"] for p in body["peaks"]] == [285.4, 288.6]
        assert body["peaks"][0]["params"] == {"chi2": 1.0} and body["peaks"][0]["kind"] == "pi-star"
        assert body["datasets"] == [{"experimentId": EXPERIMENT, "role": "primary"}]
    assert (
        builder.tag == "newmodel" and builder.model is not None and builder.model.status == "draft"
    )


def test_builder_prefills_peaks_with_params_and_fit_provenance(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        Atlas(router)
        builder = publish_module.model([EXPERIMENT], client=make_client(tmp_path))
    frame = builder.peaks
    assert frame.loc[0, "center_ev"] == 285.4 and frame.loc[0, "chi2"] == 1.1
    assert frame.loc[0, "fit_tool"] == "lmfit" and frame.loc[0, "fit_version"] == "1.3"


def test_exception_in_block_writes_nothing(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        with pytest.raises(RuntimeError):
            with publish_module.model([EXPERIMENT], client=make_client(tmp_path)) as builder:
                builder.peaks = peaks_df()
                raise RuntimeError("fit failed")
        assert atlas.fork.call_count == 0 and atlas.draft.call_count == 0
        assert atlas.upload.call_count == 0


def test_invalid_peaks_never_send_a_request(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        with pytest.raises(ValidationError, match="center_ev"):
            with publish_module.model([EXPERIMENT], client=make_client(tmp_path)) as builder:
                builder.peaks = pd.DataFrame({"center_ev": [285.0, float("nan")]})
        assert atlas.fork.call_count == 0


def test_tensor_upload_flow_and_no_token_to_storage(tmp_path: Path) -> None:
    tensor = Tensor.from_array(
        np.ones((3, 2)),
        energy_ev=[280.0, 290.0, 300.0],
        components=["xx", "yy"],
        frame="lab",
        units="1",
    )
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        with publish_module.model([EXPERIMENT], client=make_client(tmp_path)) as builder:
            builder.tensor = tensor
        upload = json.loads(atlas.upload.calls[0].request.content)
        assert upload == {"sizeBytes": tensor.size_bytes, "sha256": tensor.sha256}
        assert atlas.put.calls[0].request.content == tensor.blob
        assert "authorization" not in atlas.put.calls[0].request.headers
        assert atlas.put.calls[0].request.headers["content-type"] == "application/octet-stream"
        commit = json.loads(atlas.commit.calls[0].request.content)
        assert commit["storagePath"] == "m/u.bin" and commit["sha256"] == tensor.sha256
        assert commit["manifest"]["format"] == "npz" and commit["manifest"]["shape"] == [3, 2]
        order = [c.request.url.path for c in router.calls if c.request.method in {"POST", "PUT"}]
        assert (
            order.index("/api/v1/models/newmodel/tensor/upload-url")
            < order.index("/put")
            < order.index("/api/v1/models/newmodel/tensor/commit")
        )


def test_failed_commit_raises_and_nothing_is_published(tmp_path: Path) -> None:
    tensor = Tensor.from_array(
        np.ones((3, 2)),
        energy_ev=[280.0, 290.0, 300.0],
        components=["xx", "yy"],
        frame="lab",
        units="1",
    )
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        atlas.commit.mock(
            return_value=httpx.Response(
                422,
                json={
                    "error": {
                        "code": "UNPROCESSABLE_ENTITY",
                        "message": "Uploaded content does not match the declared SHA-256.",
                    }
                },
            )
        )
        with pytest.raises(ValidationError, match="SHA-256"):
            with publish_module.model([EXPERIMENT], client=make_client(tmp_path)) as builder:
                builder.tensor = tensor
        assert atlas.publish.call_count == 0
        assert atlas.fork.call_count == 1
        assert atlas.upload.call_count == 1
        assert atlas.commit.call_count == 1
        with pytest.raises(ValidationError, match="SHA-256"):
            builder.publish()
        assert atlas.fork.call_count == 1
        assert atlas.publish.call_count == 0


def test_publish_409_raises_conflict_without_retry(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        atlas.publish.mock(
            return_value=httpx.Response(
                409,
                json={
                    "error": {
                        "code": "CONFLICT",
                        "message": "The draft was published at the same time.",
                    }
                },
            )
        )
        with pytest.raises(ConflictError, match="same time"):
            publish_module.publish("newmodel", client=make_client(tmp_path))
        assert atlas.publish.call_count == 1


def test_builder_publish_saves_then_publishes(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        atlas.router.get("/api/v1/models/newmodel").mock(
            return_value=httpx.Response(200, json=detail("newmodel", version=1))
        )
        builder = publish_module.model([EXPERIMENT], client=make_client(tmp_path))
        result = builder.publish()
        assert atlas.fork.call_count == 1 and atlas.publish.call_count == 1
    assert result.version_number == 1


def test_with_block_publish_forks_and_puts_the_draft_once(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        with publish_module.model([EXPERIMENT], client=make_client(tmp_path)) as builder:
            builder.publish()
        assert atlas.fork.call_count == 1 and atlas.draft.call_count == 1


def test_failed_save_reuses_the_forked_tag_instead_of_forking_again(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        atlas.draft.mock(
            return_value=httpx.Response(
                500, json={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "boom"}}
            )
        )
        builder = publish_module.model([EXPERIMENT], client=make_client(tmp_path))
        with pytest.raises(ApiError):
            builder.save()
        assert builder.tag == "newmodel"
        with pytest.raises(ApiError):
            builder.publish()
        assert atlas.fork.call_count == 1
        assert atlas.publish.call_count == 0


def test_publish_reports_the_version_when_reading_it_back_fails(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        atlas.publish.mock(
            return_value=httpx.Response(200, json={"versionNumber": 4, "checksum": "c" * 64})
        )
        atlas.read_draft.mock(
            return_value=httpx.Response(
                500, json={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "boom"}}
            )
        )
        with pytest.raises(AtlasError, match="published as version 4"):
            publish_module.publish("newmodel", client=make_client(tmp_path))
        assert atlas.publish.call_count == 1


def test_missing_default_model_is_an_error(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/models").mock(
            return_value=httpx.Response(200, json={"items": [], "nextCursor": None})
        )
        with pytest.raises(AtlasError, match="default model"):
            publish_module.model([EXPERIMENT], client=make_client(tmp_path))


def test_a_dataset_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="at least one dataset"):
        publish_module.model([], client=make_client(tmp_path))


def test_share_validates_every_orcid_before_sending(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        route = router.put("/api/v1/models/newmodel/shares").mock(
            return_value=httpx.Response(200, json={})
        )
        with pytest.raises(ValidationError, match="ORCID"):
            publish_module.share("newmodel", [ORCID, "not-an-orcid"], client=make_client(tmp_path))
        assert route.call_count == 0
        publish_module.share("m/NEWMODEL", [ORCID.lower()], client=make_client(tmp_path))
        assert json.loads(route.calls[0].request.content) == {"orcid": ORCID}


def test_unshare_and_visibility_and_delete(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST) as router:
        unshare = router.delete("/api/v1/models/newmodel/shares").mock(
            return_value=httpx.Response(200, json={})
        )
        vis = router.put("/api/v1/models/newmodel/visibility").mock(
            return_value=httpx.Response(200, json={})
        )
        gone = router.delete("/api/v1/models/newmodel").mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )
        client = make_client(tmp_path)
        publish_module.unshare("newmodel", ORCID, client=client)
        publish_module.set_visibility("newmodel", "public", client=client)
        publish_module.delete("newmodel", client=client)
        with pytest.raises(ValidationError):
            publish_module.set_visibility("newmodel", "secret", client=client)  # type: ignore[arg-type]
    assert json.loads(unshare.calls[0].request.content) == {"orcid": ORCID}
    assert json.loads(vis.calls[0].request.content) == {"visibility": "public"}
    assert gone.call_count == 1


def test_review_resolves_atlas_id_and_validates_state(tmp_path: Path) -> None:
    export = {
        "columns": [],
        "rowCount": 1,
        "datasetVersions": {EXPERIMENT: "1.0.0"},
        "rows": [{"dataset_id": EXPERIMENT}],
        "filters": {},
    }
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/datasets/export").mock(return_value=httpx.Response(200, json=export))
        route = router.post("/api/v1/models/newmodel/review").mock(
            return_value=httpx.Response(200, json={})
        )
        client = make_client(tmp_path)
        publish_module.review("newmodel", "abcd2345", "rejected", "needs work", client=client)
        with pytest.raises(ValidationError):
            publish_module.review("newmodel", "abcd2345", "maybe", client=client)  # type: ignore[arg-type]
    assert json.loads(route.calls[0].request.content) == {
        "experimentId": EXPERIMENT,
        "state": "rejected",
        "note": "needs work",
    }


def test_model_write_wrappers_delegate_and_fork_prefills(tmp_path: Path) -> None:
    export = {
        "columns": [],
        "rowCount": 1,
        "datasetVersions": {EXPERIMENT: "1.0.0"},
        "rows": [{"dataset_id": EXPERIMENT}],
        "filters": {},
    }
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        router.get("/api/v1/datasets/export").mock(return_value=httpx.Response(200, json=export))
        visibility = router.put("/api/v1/models/defaultm/visibility").mock(
            return_value=httpx.Response(200, json={})
        )
        share_route = router.put("/api/v1/models/defaultm/shares").mock(
            return_value=httpx.Response(200, json={})
        )
        unshare = router.delete("/api/v1/models/defaultm/shares").mock(
            return_value=httpx.Response(200, json={})
        )
        review_route = router.post("/api/v1/models/defaultm/review").mock(
            return_value=httpx.Response(200, json={})
        )
        gone = router.delete("/api/v1/models/defaultm").mock(
            return_value=httpx.Response(200, json={"deleted": True})
        )
        client = make_client(tmp_path)
        source = models.get("defaultm", client=client)
        publish_defaultm = router.post("/api/v1/models/defaultm/publish").mock(
            return_value=httpx.Response(200, json={"versionNumber": 1, "checksum": "b" * 64})
        )
        published = source.publish()
        source.set_visibility("public")
        source.share([ORCID])
        source.unshare(ORCID)
        source.review("abcd2345", "approved", "fit looks right")
        builder = source.fork(title="Variant")
        source.delete()
        assert builder.title == "Variant" and len(builder.peaks) == 1
        assert atlas.fork.call_count == 0
        assert published.version_number == 1
        assert publish_defaultm.call_count == 1
    assert publish_defaultm.calls[0].request.method == "POST"
    assert publish_defaultm.calls[0].request.url.path == "/api/v1/models/defaultm/publish"
    assert json.loads(visibility.calls[0].request.content) == {"visibility": "public"}
    assert json.loads(share_route.calls[0].request.content) == {"orcid": ORCID}
    assert json.loads(unshare.calls[0].request.content) == {"orcid": ORCID}
    assert unshare.calls[0].request.method == "DELETE"
    assert json.loads(review_route.calls[0].request.content) == {
        "experimentId": EXPERIMENT,
        "state": "approved",
        "note": "fit looks right",
    }
    assert gone.calls[0].request.method == "DELETE"
    assert gone.calls[0].request.url.path == "/api/v1/models/defaultm"


def test_model_fork_prefills_from_the_requested_version(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        router.get("/api/v1/models/defaultm").mock(
            return_value=httpx.Response(200, json=detail("defaultm", version=3, peaks=SOURCE_PEAKS))
        )
        client = make_client(tmp_path)
        source = models.get("defaultm", 3, client=client)
        builder = source.fork(title="Variant")
        assert builder.title == "Variant" and len(builder.peaks) == 1
        assert builder.peaks.loc[0, "center_ev"] == 285.4
        builder.save()
        assert atlas.fork.call_count == 1
        assert json.loads(atlas.fork.calls[0].request.content) == {
            "title": "Variant",
            "fromVersion": 3,
        }


def test_model_fork_rejects_a_draft(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        Atlas(router)
        router.get("/api/v1/models/defaultm").mock(
            return_value=httpx.Response(200, json=detail("defaultm", version=None))
        )
        source = models.get("defaultm", "draft", client=make_client(tmp_path))
        assert source.version_number is None
        with pytest.raises(ValidationError, match="draft"):
            source.fork()


def test_malformed_dataset_reference_makes_no_request(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        with pytest.raises(ValidationError, match="not a dataset reference"):
            publish_module.model(["abcd2345", "not-a-ref"], client=make_client(tmp_path))
        assert len(router.calls) == 0


def test_source_peak_params_cannot_collide_with_peak_columns(tmp_path: Path) -> None:
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        atlas = Atlas(router)
        router.get("/api/v1/models/defaultm").mock(
            return_value=httpx.Response(
                200,
                json=detail(
                    "defaultm",
                    peaks=[{"ordinal": 0, "centerEv": 285.4, "params": {"width": 2.0}}],
                ),
            )
        )
        client = make_client(tmp_path)
        with pytest.raises(ValidationError, match="collides"):
            publish_module.model([EXPERIMENT], client=client)
        assert atlas.fork.call_count == 0 and atlas.draft.call_count == 0


def test_review_rejects_blank_or_overlong_note_before_requests(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    with respx.mock(base_url=HOST, assert_all_called=False) as router:
        for note in ("", "   ", "x" * 1001):
            with pytest.raises(ValidationError, match="note must be 1 to 1000 characters"):
                publish_module.review("newmodel", "abcd2345", "rejected", note, client=client)
        assert len(router.calls) == 0
