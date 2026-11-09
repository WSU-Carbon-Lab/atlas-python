import json
from pathlib import Path

import pytest

from xrayatlas import AuthError, Client, credential_path, load_credential


def _write(root: Path, body: dict[str, object]) -> None:
    path = root / "xatlas" / "credentials.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("XRAYATLAS_TOKEN", raising=False)
    monkeypatch.delenv("XRAYATLAS_HOST", raising=False)


def test_credential_path_uses_xdg(tmp_path: Path) -> None:
    assert credential_path() == tmp_path / "xatlas" / "credentials.json"


def test_load_credential_reads_cli_file(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "version": 1,
            "host": "https://xrayatlas.wsu.edu/",
            "token": "xra_abc",
            "orcid": "0000-0002-1825-0097",
            "name": "A",
            "scopes": ["read", "write"],
            "tokenName": "laptop",
            "expiresAt": "2027-01-01T00:00:00Z",
        },
    )
    cred = load_credential()
    assert cred is not None
    assert cred.host == "https://xrayatlas.wsu.edu"
    assert cred.token == "xra_abc"
    assert cred.orcid == "0000-0002-1825-0097"
    assert cred.scopes == ("read", "write")
    assert cred.expires_at == "2027-01-01T00:00:00Z"


def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(
        tmp_path, {"version": 1, "host": "https://a.test", "token": "xra_file", "scopes": ["read"]}
    )
    monkeypatch.setenv("XRAYATLAS_TOKEN", "xra_env")
    monkeypatch.setenv("XRAYATLAS_HOST", "https://b.test/path")
    cred = load_credential()
    assert cred is not None
    assert cred.token == "xra_env"
    assert cred.host == "https://b.test"
    assert cred.orcid is None


def test_wrong_file_version_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, {"version": 2, "host": "https://a.test", "token": "xra_abc"})
    assert load_credential() is None


def test_token_without_prefix_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, {"version": 1, "host": "https://a.test", "token": "abc"})
    assert load_credential() is None


def test_missing_file_gives_none() -> None:
    assert load_credential() is None


def test_invalid_env_host_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRAYATLAS_TOKEN", "xra_env")
    monkeypatch.setenv("XRAYATLAS_HOST", "atlas.test")
    with pytest.raises(AuthError):
        load_credential()


def test_file_token_ignored_when_env_host_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path,
        {"version": 1, "host": "https://a.test", "token": "xra_file", "scopes": ["read"]},
    )
    monkeypatch.setenv("XRAYATLAS_HOST", "https://b.test")
    assert load_credential() is None


def test_http_host_rejected_except_localhost(tmp_path: Path) -> None:
    _write(tmp_path, {"version": 1, "host": "http://evil.test", "token": "xra_abc"})
    assert load_credential() is None


def test_localhost_http_allowed_and_bad_port_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, {"version": 1, "host": "http://localhost:3000", "token": "xra_abc"})
    cred = load_credential()
    assert cred is not None and cred.host == "http://localhost:3000"


def test_corrupt_port_does_not_crash(tmp_path: Path) -> None:
    _write(tmp_path, {"version": 1, "host": "https://a.test:abc", "token": "xra_abc"})
    assert load_credential() is None


def test_ipv6_host_keeps_brackets() -> None:
    from xrayatlas._credentials import normalize_host

    assert normalize_host("http://[::1]:8080/x") == "http://[::1]:8080"


def test_env_host_applies_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XRAYATLAS_HOST", "https://staging.test")
    client = Client()
    assert client.host == "https://staging.test"
    assert client.authenticated is False


def test_client_context_manager_closes_http(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Client()
    closed: list[bool] = []
    monkeypatch.setattr(client.http, "close", lambda: closed.append(True))
    with client as entered:
        assert entered is client
    assert closed == [True]


def test_file_token_not_sent_to_a_different_env_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, {"version": 1, "host": "https://a.test", "token": "xra_file"})
    monkeypatch.setenv("XRAYATLAS_HOST", "https://b.test")
    client = Client()
    assert client.host == "https://b.test"
    assert client.authenticated is False
