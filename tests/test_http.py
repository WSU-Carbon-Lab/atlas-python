import httpx
import pytest
import respx

from xrayatlas import (
    ApiError,
    AuthError,
    ConflictError,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    ValidationError,
    _http,
)
from xrayatlas._http import HttpClient

HOST = "https://atlas.test"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_http, "_sleep", lambda _seconds: None)


def test_401_raises_auth_error_without_retry() -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        with pytest.raises(AuthError) as info:
            HttpClient(HOST, "xra_x").get_json("/api/v1/models")
        assert route.call_count == 1
        assert "xatlas login" in str(info.value)


@pytest.mark.parametrize(
    ("status", "cls"),
    [(403, ForbiddenError), (404, NotFoundError), (409, ConflictError), (422, ValidationError)],
)
def test_error_envelope_maps_status_to_subclass(status: int, cls: type[Exception]) -> None:
    with respx.mock(base_url=HOST) as router:
        router.post("/api/v1/x").mock(
            return_value=httpx.Response(status, json={"error": {"code": "CODE", "message": "boom"}})
        )
        with pytest.raises(cls) as info:
            HttpClient(HOST, "xra_x").send_json("POST", "/api/v1/x", {})
        assert str(info.value) == "boom"
        assert info.value.code == "CODE"


def test_get_retries_5xx_then_succeeds() -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/x").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
        )
        assert HttpClient(HOST, None).get_json("/api/v1/x") == {"ok": True}
        assert route.call_count == 2


def test_post_is_never_retried() -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.post("/api/v1/x").mock(return_value=httpx.Response(503))
        with pytest.raises(ApiError):
            HttpClient(HOST, "xra_x").send_json("POST", "/api/v1/x", {})
        assert route.call_count == 1


def test_bearer_header_sent_to_atlas_but_not_to_signed_url() -> None:
    with respx.mock() as router:
        atlas = router.get(f"{HOST}/api/v1/x").mock(return_value=httpx.Response(200, json={}))
        signed = router.put("https://storage.test/upload?sig=1").mock(
            return_value=httpx.Response(200)
        )
        client = HttpClient(HOST, "xra_secret")
        client.get_json("/api/v1/x")
        client.put_bytes("https://storage.test/upload?sig=1", b"data", "application/octet-stream")
        assert atlas.calls[0].request.headers["authorization"] == "Bearer xra_secret"
        assert "authorization" not in signed.calls[0].request.headers


@pytest.mark.parametrize(
    "foreign",
    [
        "https://atlas.test.evil.com/x",
        "https://atlas.test@evil.com/x",
        "HTTPS://evil.com/x",
        "http://atlas.test/x",
        "https://atlas.test:8443/x",
    ],
)
def test_token_never_sent_to_lookalike_origins(foreign: str) -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.get(url__regex=r".*").mock(return_value=httpx.Response(200, content=b"x"))
        HttpClient(HOST, "xra_secret").get_bytes(foreign)
        assert route.call_count == 1
        assert "authorization" not in route.calls[0].request.headers


def test_same_origin_with_explicit_default_port_gets_token() -> None:
    with respx.mock(assert_all_called=False) as router:
        route = router.get(url__regex=r".*").mock(return_value=httpx.Response(200, json={}))
        HttpClient(HOST, "xra_secret").get_json("https://atlas.test:443/api/v1/x")
        assert route.calls[0].request.headers["authorization"] == "Bearer xra_secret"


def test_redirect_is_an_error_not_a_success() -> None:
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/x").mock(
            return_value=httpx.Response(302, headers={"location": "https://evil.test/"})
        )
        with pytest.raises(ApiError) as info:
            HttpClient(HOST, "xra_x").get_json("/api/v1/x")
        assert info.value.status == 302


def test_empty_body_returns_none_and_bad_json_raises_api_error() -> None:
    with respx.mock(base_url=HOST) as router:
        router.delete("/api/v1/a").mock(return_value=httpx.Response(204))
        router.get("/api/v1/b").mock(return_value=httpx.Response(200, content=b"<html>"))
        client = HttpClient(HOST, "xra_x")
        assert client.send_json("DELETE", "/api/v1/a") is None
        with pytest.raises(ApiError):
            client.get_json("/api/v1/b")


def test_get_429_is_retried_three_times_then_raises() -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/x").mock(return_value=httpx.Response(429))
        with pytest.raises(ApiError):
            HttpClient(HOST, None).get_json("/api/v1/x")
        assert route.call_count == 3


def test_retry_after_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr(_http, "_sleep", waits.append)
    with respx.mock(base_url=HOST) as router:
        router.get("/api/v1/x").mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "2"}),
                httpx.Response(200, json={}),
            ]
        )
        HttpClient(HOST, None).get_json("/api/v1/x")
    assert waits == [2.0]


def test_transport_error_becomes_network_error_after_retries() -> None:
    with respx.mock(base_url=HOST) as router:
        route = router.get("/api/v1/x").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(NetworkError):
            HttpClient(HOST, None).get_json("/api/v1/x")
        assert route.call_count == 3


def test_put_bytes_sets_content_type_and_rejects_relative_url() -> None:
    with respx.mock() as router:
        route = router.put("https://storage.test/up").mock(return_value=httpx.Response(200))
        client = HttpClient(HOST, "xra_x")
        client.put_bytes("https://storage.test/up", b"d", "application/octet-stream")
        assert route.calls[0].request.headers["content-type"] == "application/octet-stream"
        with pytest.raises(ValidationError):
            client.put_bytes("/relative", b"d", "application/octet-stream")


def test_storage_401_does_not_blame_atlas_login() -> None:
    with respx.mock() as router:
        router.get("https://storage.test/x").mock(return_value=httpx.Response(401))
        with pytest.raises(ApiError) as info:
            HttpClient(HOST, "xra_x").get_bytes("https://storage.test/x")
        assert "xatlas login" not in str(info.value)
