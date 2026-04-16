"""Rate limiting for DigitalTwinSim API."""

import os
from typing import Optional

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi import Request
from fastapi.responses import JSONResponse


def _get_tenant_or_ip(request: Request) -> str:
    """Rate-limit key: tenant_id if authenticated, else client IP."""
    tenant = getattr(request.state, "tenant", None)
    if tenant and hasattr(tenant, "tenant_id"):
        return f"tenant:{tenant.tenant_id}"
    return get_remote_address(request)


# Whether rate limiting is active
RATE_LIMIT_ENABLED = os.getenv("DTS_RATE_LIMIT_ENABLED", "false").lower() in ("true", "1", "yes")

limiter = Limiter(
    key_func=_get_tenant_or_ip,
    enabled=RATE_LIMIT_ENABLED,
    default_limits=[],
)

# Limit strings
LIMIT_CREATE_SIM = "10/hour"
LIMIT_SUGGEST_KPIS = "30/hour"
LIMIT_READS = "200/hour"


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom 429 response."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Try again later.",
            "retry_after": str(exc.detail),
        },
    )
