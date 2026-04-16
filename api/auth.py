"""API Key authentication for DigitalTwinSim."""

import json
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Paths exempt from authentication
PUBLIC_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}


@dataclass
class Tenant:
    tenant_id: str
    api_key: str


def _load_keys() -> dict[str, str]:
    """Load API key → tenant_id mapping.

    Supports two env vars:
    - DTS_KEY_MAP: JSON object {"key": "tenant_id", ...}  (takes precedence)
    - DTS_API_KEYS: comma-separated keys (tenant_id = "default" for all)
    """
    key_map_raw = os.getenv("DTS_KEY_MAP", "")
    if key_map_raw:
        try:
            mapping = json.loads(key_map_raw)
            # mapping is {api_key: tenant_id}
            return {str(k): str(v) for k, v in mapping.items()}
        except (json.JSONDecodeError, TypeError):
            pass

    keys_raw = os.getenv("DTS_API_KEYS", "")
    if keys_raw:
        return {k.strip(): "default" for k in keys_raw.split(",") if k.strip()}

    return {}


# Cache key map at import time; reload on each call if empty
_cached_keys: Optional[dict[str, str]] = None


def _get_key_map() -> dict[str, str]:
    global _cached_keys
    if _cached_keys is None:
        _cached_keys = _load_keys()
    return _cached_keys


def auth_enabled() -> bool:
    """Check if authentication is configured."""
    return bool(_get_key_map())


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER),
) -> Optional[Tenant]:
    """FastAPI dependency: verify API key and return Tenant.

    If no keys are configured (DTS_API_KEYS / DTS_KEY_MAP empty),
    auth is disabled and returns None (open access, dev mode).
    """
    key_map = _get_key_map()

    # If no keys configured, auth is disabled (dev mode)
    if not key_map:
        return None

    # Public paths are exempt
    if request.url.path in PUBLIC_PATHS:
        return None

    # Key required
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    tenant_id = key_map.get(api_key)
    if not tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return Tenant(tenant_id=tenant_id, api_key=api_key)


def reload_keys():
    """Force reload of API keys (e.g. after env change)."""
    global _cached_keys
    _cached_keys = None
