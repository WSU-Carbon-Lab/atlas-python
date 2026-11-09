"""The client object that ties host, token, HTTP, and cache together.

`Client` resolves its host and token from explicit arguments, the shared credential file, and the
environment. A module-level default client serves the ``xa.datasets`` and ``xa.models`` helpers.
A token is attached only for the host it was issued for. Does no network I/O at construction.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import httpx

from xrayatlas._cache import Cache
from xrayatlas._credentials import DEFAULT_HOST, load_credential, normalize_host, resolve_host
from xrayatlas._http import HttpClient
from xrayatlas.errors import ValidationError


class Client:
    """Atlas connection: host, optional bearer token, HTTP layer, and verified cache."""

    def __init__(
        self,
        host: str | None = None,
        token: str | None = None,
        cache_dir: Path | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a client.

        Args:
            host: Atlas origin. Defaults to the environment host, then the credential's host, then
                the public site.
            token: Bearer token. Defaults to the credential's token when its host equals ``host``;
                otherwise the client reads public data anonymously.
            cache_dir: Cache directory, or ``None`` for the default.
            transport: Optional httpx transport, used by tests.

        Raises:
            ValidationError: ``host`` is not an acceptable origin.
            AuthError: ``XRAYATLAS_HOST`` is set but invalid.
        """
        credential = load_credential()
        if host is None:
            resolved_host = resolve_host() or DEFAULT_HOST
            resolved_token = token or (
                credential.token if credential and credential.host == resolved_host else None
            )
        else:
            normalized = normalize_host(host)
            if normalized is None:
                raise ValidationError(
                    "host must be an https origin (http is allowed only for localhost)."
                )
            resolved_host = normalized
            resolved_token = token or (
                credential.token if credential and credential.host == normalized else None
            )
        self.host = resolved_host
        self.authenticated = resolved_token is not None
        self.http = HttpClient(resolved_host, resolved_token, transport=transport)
        self.cache = Cache(cache_dir)

    def close(self) -> None:
        """Release network resources."""
        self.http.close()

    def __enter__(self) -> Client:
        """Return this client for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the client when leaving a ``with`` block."""
        self.close()


_default: Client | None = None


def default_client() -> Client:
    """The shared client, created on first use from the environment and credential file."""
    global _default
    if _default is None:
        _default = Client()
    return _default


def set_default_client(client: Client | None) -> None:
    """Replace the shared client (``None`` makes the next use create a fresh one)."""
    global _default
    _default = client
