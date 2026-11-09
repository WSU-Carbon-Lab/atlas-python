import pytest

from xrayatlas import AuthError, login


def test_login_without_cli_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(AuthError):
        login()
