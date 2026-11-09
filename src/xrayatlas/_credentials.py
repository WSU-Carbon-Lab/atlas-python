"""Credential lookup shared with the ``xatlas`` CLI.

Reads ``$XDG_CONFIG_HOME/xatlas/credentials.json`` (default ``~/.config/xatlas/credentials.json``),
a version 1 JSON file written by ``xatlas login``. Environment variables ``XRAYATLAS_TOKEN`` and
``XRAYATLAS_HOST`` override the file. This module only reads credentials; it never writes them and
never prints a token.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from xrayatlas.errors import AuthError

DEFAULT_HOST = "https://xrayatlas.wsu.edu"
_FILE_VERSION = 1
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Credential:
    """A host and bearer token with the account fields the CLI stored."""

    host: str
    token: str
    orcid: str | None
    scopes: tuple[str, ...]
    expires_at: str | None


def credential_path() -> Path:
    """Path of the CLI credential file.

    Returns:
        ``$XDG_CONFIG_HOME/xatlas/credentials.json``, or the same path under ``~/.config``.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg) if xdg else Path.home() / ".config"
    return root / "xatlas" / "credentials.json"


def normalize_host(value: str) -> str | None:
    """Reduce a URL to its http(s) origin.

    Plain ``http`` is accepted only for localhost, 127.0.0.1, and ::1, so a bearer token never
    travels in cleartext to a remote host. Userinfo is rejected.

    Args:
        value: Candidate host such as ``https://xrayatlas.wsu.edu/`` .

    Returns:
        ``scheme://host[:port]`` without path, query, or userinfo, or ``None`` when ``value`` is not
        an acceptable origin.
    """
    try:
        parsed = urlparse(value.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host:
        return None
    if parsed.username or parsed.password:
        return None
    if parsed.scheme == "http" and host not in _LOCAL_HOSTS:
        return None
    shown = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{shown}{f':{port}' if port else ''}"


def _read_file() -> Credential | None:
    try:
        data = json.loads(credential_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != _FILE_VERSION:
        return None
    token = data.get("token")
    host = data.get("host")
    if not isinstance(token, str) or not token.startswith("xra_"):
        return None
    if not isinstance(host, str):
        return None
    origin = normalize_host(host)
    if origin is None:
        return None
    scopes_raw = data.get("scopes")
    scopes = (
        tuple(s for s in scopes_raw if isinstance(s, str)) if isinstance(scopes_raw, list) else ()
    )
    orcid = data.get("orcid")
    expires = data.get("expiresAt")
    return Credential(
        host=origin,
        token=token,
        orcid=orcid if isinstance(orcid, str) else None,
        scopes=scopes,
        expires_at=expires if isinstance(expires, str) else None,
    )


def resolve_host() -> str | None:
    """Host the environment and credential file select.

    ``XRAYATLAS_HOST`` wins when set; otherwise the CLI file's host; otherwise ``None`` (the caller
    falls back to `DEFAULT_HOST`).

    Raises:
        AuthError: ``XRAYATLAS_HOST`` is set but is not an acceptable origin.
    """
    env_host_raw = os.environ.get("XRAYATLAS_HOST", "").strip()
    if env_host_raw:
        env_host = normalize_host(env_host_raw)
        if env_host is None:
            raise AuthError(
                "XRAYATLAS_HOST must be an https origin (http is allowed only for localhost)."
            )
        return env_host
    stored = _read_file()
    return stored.host if stored else None


def load_credential() -> Credential | None:
    """Resolve the credential to use.

    ``XRAYATLAS_TOKEN`` overrides the CLI file; its host is ``XRAYATLAS_HOST``, else the CLI file's
    host, else the default. A CLI file token is used only for the host it was issued for: when
    ``XRAYATLAS_HOST`` names a different origin the file token is ignored.

    Returns:
        The credential, or ``None`` when no usable token exists.

    Raises:
        AuthError: ``XRAYATLAS_HOST`` is set but is not an acceptable origin.
    """
    stored = _read_file()
    env_token = os.environ.get("XRAYATLAS_TOKEN", "").strip()
    env_host_raw = os.environ.get("XRAYATLAS_HOST", "").strip()
    env_host: str | None = None
    if env_host_raw:
        env_host = normalize_host(env_host_raw)
        if env_host is None:
            raise AuthError(
                "XRAYATLAS_HOST must be an https origin (http is allowed only for localhost)."
            )
    if env_token:
        host = env_host or (stored.host if stored else DEFAULT_HOST)
        return Credential(host=host, token=env_token, orcid=None, scopes=(), expires_at=None)
    if stored is None:
        return None
    if env_host is not None and env_host != stored.host:
        return None
    return stored
