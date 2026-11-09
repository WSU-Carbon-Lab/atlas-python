"""Error types raised by the X-ray Atlas client.

All client errors derive from `AtlasError`. HTTP failures derive from `ApiError`, one subclass per
status the server uses. The module performs no I/O.
"""

from __future__ import annotations


class AtlasError(Exception):
    """Base class of every error the client raises."""


class ApiError(AtlasError):
    """A failed API call or a request the client refused before sending.

    Attributes:
        status: HTTP status of the response, or the status the failure corresponds to.
        code: Server error code such as ``CONFLICT``, or ``None``.
        app_code: Application code such as ``CONTRIBUTION_AGREEMENT_REQUIRED``, or ``None``.
    """

    default_status = 500

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        app_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status if status is not None else self.default_status
        self.code = code
        self.app_code = app_code


class AuthError(ApiError):
    """HTTP 401: the token is missing, expired, or revoked."""

    default_status = 401


class ForbiddenError(ApiError):
    """HTTP 403: the token lacks a scope or the action is not allowed."""

    default_status = 403


class NotFoundError(ApiError):
    """HTTP 404: the object does not exist or is not visible to the caller."""

    default_status = 404


class ConflictError(ApiError):
    """HTTP 409: the object changed or another request won a race."""

    default_status = 409


class ValidationError(ApiError):
    """HTTP 400 or 422, or input the client rejected before sending."""

    default_status = 422


_BY_STATUS: dict[int, type[ApiError]] = {
    400: ValidationError,
    401: AuthError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
}


def error_for_status(
    status: int,
    message: str,
    code: str | None = None,
    app_code: str | None = None,
) -> ApiError:
    """Build the error that matches an HTTP status.

    Args:
        status: HTTP status of the failed response.
        message: Message safe to show to a user.
        code: Server error code, if the response carried one.
        app_code: Application code, if the response carried one.

    Returns:
        An instance of the subclass registered for ``status``, or plain `ApiError`.
    """
    cls = _BY_STATUS.get(status, ApiError)
    return cls(message, status=status, code=code, app_code=app_code)
