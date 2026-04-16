"""Request logging middleware for DigitalTwinSim."""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from api.logging_config import get_logger

logger = get_logger("middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing, status, and tenant context."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()

        # Extract tenant_id if already set by auth dependency
        tenant_id = getattr(getattr(request.state, "tenant", None), "tenant_id", None)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "request_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 1),
                tenant_id=tenant_id,
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000

        # Skip noisy endpoints
        if request.url.path not in ("/api/health", "/metrics"):
            logger.info(
                "request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 1),
                tenant_id=tenant_id,
            )

        response.headers["X-Request-ID"] = request_id
        return response
