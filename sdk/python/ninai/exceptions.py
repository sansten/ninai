"""
Ninai SDK Exceptions
====================

Custom exceptions for the Ninai Python SDK.
"""


class NinaiError(Exception):
    """Base exception for all Ninai SDK errors."""
    
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        self.message = message
        self.status_code = status_code
        self.response = response or {}
        super().__init__(self.message)


class AuthenticationError(NinaiError):
    """Raised when authentication fails (401)."""
    pass


class AuthorizationError(NinaiError):
    """Raised when access is forbidden (403)."""
    pass


class EnterpriseFeatureRequired(AuthorizationError):
    """Raised when an endpoint requires an Enterprise license.

    The server returned 403 with ``{"error": "feature_not_enabled"}``.
    Upgrade at https://sansten.com/ninai/pricing or contact sales@sansten.com.

    Attributes:
        feature: The feature flag that was not enabled (e.g. ``"enterprise.scim"``).
    """

    def __init__(self, message: str, feature: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.feature = feature

    def __str__(self) -> str:
        base = self.message
        if self.feature:
            base = f"{base} (feature={self.feature})"
        return f"{base} — upgrade at https://sansten.com/ninai/pricing"


class NotFoundError(NinaiError):
    """Raised when a resource is not found (404)."""
    pass


class ValidationError(NinaiError):
    """Raised when request validation fails (422)."""
    pass


class RateLimitError(NinaiError):
    """Raised when rate limit is exceeded (429)."""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ServerError(NinaiError):
    """Raised when server returns 5xx error."""
    pass
