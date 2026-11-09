"""HTTP layer for the Atlas REST API.

Owns request sending, bearer authentication, the ``{ "error": { "code", "message" } }`` envelope,
retry of idempotent GETs, and the rule that the bearer token is attached only to requests whose
origin (scheme, host, port) exactly matches the Atlas origin. Redirects are never followed. Does not
know any route. A 401 from Atlas is never retried and never falls back to an anonymous request.
Tokens are never logged.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import httpx

from xrayatlas.errors import ApiError, AtlasError, AuthError, ValidationError, error_for_status

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Params: TypeAlias = Mapping[str, str | int | float | bool]
JsonBody: TypeAlias = Mapping[str, object] | Sequence[object] | str | int | float | bool | None

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_TIMEOUT_SECONDS = 30.0
_MAX_RETRY_AFTER_SECONDS = 30.0
_sleep = time.sleep

AUTH_HINT = "Authentication failed. Run `xatlas login` or set XRAYATLAS_TOKEN."


class NetworkError(AtlasError):
    """The server could not be reached (connection failure or timeout)."""


def _origin_of(url: httpx.URL) -> tuple[str, str, int]:
    scheme = url.scheme.lower()
    host = (url.host or "").lower()
    port = url.port if url.port is not None else (443 if scheme == "https" else 80)
    return scheme, host, port


def _error_from_response(response: httpx.Response, *, atlas_origin: bool) -> ApiError:
    status = response.status_code
    if not atlas_origin:
        return error_for_status(
            status,
            f"The storage service answered HTTP {status}; a signed URL may have expired.",
        )
    if 300 <= status < 400:
        return error_for_status(
            status,
            f"Unexpected redirect (HTTP {status}); the client does not follow redirects.",
        )
    code: str | None = None
    app_code: str | None = None
    message = f"Request failed with status {status}."
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            raw_code = error.get("code")
            raw_message = error.get("message")
            raw_app = error.get("appCode")
            code = raw_code if isinstance(raw_code, str) else None
            app_code = raw_app if isinstance(raw_app, str) else None
            if isinstance(raw_message, str) and raw_message:
                message = raw_message
        elif isinstance(error, str) and error:
            message = error
    if status == 401:
        return AuthError(AUTH_HINT, code=code or "UNAUTHORIZED", app_code=app_code)
    return error_for_status(status, message, code, app_code)


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("retry-after", "").strip()
        try:
            seconds = float(raw)
        except ValueError:
            seconds = -1.0
        if seconds >= 0:
            return min(seconds, _MAX_RETRY_AFTER_SECONDS)
    return 0.5 * (2.0**attempt)


def _json_body(response: httpx.Response) -> JsonValue:
    if not response.content:
        return None
    try:
        value: JsonValue = response.json()
    except ValueError as exc:
        raise ApiError(
            "The server returned a response that is not valid JSON.",
            status=response.status_code,
        ) from exc
    return value


class HttpClient:
    """Synchronous client for one Atlas origin."""

    def __init__(
        self,
        host: str,
        token: str | None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a client.

        Args:
            host: Atlas origin without a path, for example ``https://xrayatlas.wsu.edu``.
            token: Bearer token, or ``None`` for anonymous reads of public data.
            transport: Optional httpx transport, used by tests.
        """
        self._base = httpx.URL(host)
        self._origin = _origin_of(self._base)
        self._token = token
        self._client = httpx.Client(
            timeout=_TIMEOUT_SECONDS, transport=transport, follow_redirects=False
        )

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._client.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Params | None = None,
        json: JsonBody = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        retry: bool,
    ) -> httpx.Response:
        resolved = self._base.join(url)
        atlas_origin = _origin_of(resolved) == self._origin and not resolved.userinfo
        # httpx turns URL userinfo into a Basic Authorization header; strip it so no credential
        # embedded in a caller-supplied URL is sent anywhere.
        resolved = resolved.copy_with(userinfo=b"")
        request_headers: dict[str, str] = {"accept": "application/json"}
        if headers:
            request_headers.update(headers)
        if atlas_origin and self._token:
            request_headers["authorization"] = f"Bearer {self._token}"
        attempts = _MAX_ATTEMPTS if retry else 1
        for attempt in range(attempts):
            last = attempt + 1 >= attempts
            try:
                response = self._client.request(
                    method,
                    resolved,
                    params=dict(params) if params else None,
                    json=json,
                    content=content,
                    headers=request_headers,
                )
            except httpx.RequestError as exc:
                if not last:
                    _sleep(_retry_delay(None, attempt))
                    continue
                raise NetworkError(
                    f"Could not reach {resolved.host}: {type(exc).__name__}."
                ) from exc
            if 200 <= response.status_code < 300:
                return response
            if response.status_code in _RETRY_STATUSES and not last and retry:
                _sleep(_retry_delay(response, attempt))
                continue
            raise _error_from_response(response, atlas_origin=atlas_origin)
        raise AssertionError("unreachable")  # pragma: no cover

    def get_json(self, path: str, params: Params | None = None) -> JsonValue:
        """GET a path and parse the JSON body. Retries 429, 5xx, and transport errors up to 3 times.

        Returns:
            The parsed body, or ``None`` when the body is empty.

        Raises:
            ApiError: The server answered with a non-2xx status or a non-JSON body.
            NetworkError: The server could not be reached after the retries.
        """
        return _json_body(self._request("GET", path, params=params, retry=True))

    def send_json(self, method: str, path: str, body: JsonBody = None) -> JsonValue:
        """Send a JSON body with POST, PUT, PATCH, or DELETE and parse the JSON response.

        Never retried, because the call may not be idempotent.

        Returns:
            The parsed body, or ``None`` when the body is empty (for example HTTP 204).

        Raises:
            ApiError: The server answered with a non-2xx status or a non-JSON body.
            NetworkError: The server could not be reached.
        """
        return _json_body(self._request(method, path, json=body, retry=False))

    def get_bytes(self, url: str) -> bytes:
        """GET a path or absolute URL and return the raw body. Retries like `get_json`.

        The Atlas bearer token is attached only when the URL's origin equals the Atlas origin.

        Raises:
            ApiError: The server answered with a non-2xx status.
            NetworkError: The server could not be reached after the retries.
        """
        return self._request("GET", url, retry=True).content

    def put_bytes(self, url: str, data: bytes, content_type: str) -> None:
        """PUT raw bytes to an absolute URL (a signed upload URL). Never retried.

        The Atlas bearer token is attached only when the URL's origin equals the Atlas origin.

        Raises:
            ValidationError: ``url`` is not absolute.
            ApiError: The server answered with a non-2xx status.
            NetworkError: The server could not be reached.
        """
        if not url.startswith(("http://", "https://")):
            raise ValidationError("put_bytes requires an absolute URL.")
        self._request("PUT", url, content=data, headers={"content-type": content_type}, retry=False)
