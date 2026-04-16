"""Tests for API authentication module — key parsing, tenant isolation, validation."""

import asyncio
import json
import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth import (
    Tenant,
    _load_keys,
    auth_enabled,
    reload_keys,
    verify_api_key,
    PUBLIC_PATHS,
)


def _run(coro):
    """Helper to run async coroutine in sync test."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Tests: _load_keys ───────────────────────────────────────────────────────


class TestLoadKeys:
    def test_dts_api_keys_comma_separated(self):
        """DTS_API_KEYS=a,b,c creates keys with 'default' tenant."""
        with patch.dict(os.environ, {"DTS_API_KEYS": "key1,key2,key3"}, clear=False):
            keys = _load_keys()
            assert "key1" in keys
            assert "key2" in keys
            assert "key3" in keys
            assert keys["key1"] == "default"
            assert keys["key2"] == "default"

    def test_dts_key_map_json(self):
        """DTS_KEY_MAP takes precedence and maps key -> tenant."""
        key_map = {"sk-abc": "acme-corp", "sk-def": "beta-inc"}
        with patch.dict(os.environ, {
            "DTS_KEY_MAP": json.dumps(key_map),
            "DTS_API_KEYS": "should-be-ignored",
        }, clear=False):
            keys = _load_keys()
            assert keys["sk-abc"] == "acme-corp"
            assert keys["sk-def"] == "beta-inc"
            # DTS_API_KEYS should be ignored when DTS_KEY_MAP is set
            assert "should-be-ignored" not in keys

    def test_no_env_returns_empty(self):
        """No DTS_API_KEYS and no DTS_KEY_MAP => empty dict."""
        with patch.dict(os.environ, {}, clear=True):
            keys = _load_keys()
            assert keys == {}

    def test_dts_api_keys_strips_whitespace(self):
        with patch.dict(os.environ, {"DTS_API_KEYS": " key1 , key2 "}, clear=False):
            keys = _load_keys()
            assert "key1" in keys
            assert "key2" in keys

    def test_invalid_json_falls_through(self):
        """Invalid JSON in DTS_KEY_MAP falls back to DTS_API_KEYS."""
        with patch.dict(os.environ, {
            "DTS_KEY_MAP": "not valid json",
            "DTS_API_KEYS": "fallback-key",
        }, clear=False):
            keys = _load_keys()
            assert "fallback-key" in keys

    def test_empty_dts_api_keys(self):
        with patch.dict(os.environ, {"DTS_API_KEYS": ""}, clear=False):
            keys = _load_keys()
            assert keys == {}


# ── Tests: auth_enabled ──────────────────────────────────────────────────────


class TestAuthEnabled:
    def test_enabled_when_keys_present(self):
        with patch.dict(os.environ, {"DTS_API_KEYS": "key1"}, clear=False):
            reload_keys()
            assert auth_enabled() is True
            reload_keys()

    def test_disabled_when_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            reload_keys()
            assert auth_enabled() is False
            reload_keys()


# ── Tests: verify_api_key ────────────────────────────────────────────────────


class TestVerifyApiKey:
    def test_valid_key_returns_tenant(self):
        with patch.dict(os.environ, {"DTS_API_KEYS": "valid-key"}, clear=False):
            reload_keys()
            request = MagicMock()
            request.url.path = "/api/simulations"
            tenant = _run(verify_api_key(request, api_key="valid-key"))
            assert isinstance(tenant, Tenant)
            assert tenant.tenant_id == "default"
            assert tenant.api_key == "valid-key"
            reload_keys()

    def test_invalid_key_raises_403(self):
        with patch.dict(os.environ, {"DTS_API_KEYS": "valid-key"}, clear=False):
            reload_keys()
            request = MagicMock()
            request.url.path = "/api/simulations"
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                _run(verify_api_key(request, api_key="wrong-key"))
            assert exc_info.value.status_code == 403
            reload_keys()

    def test_missing_key_raises_401(self):
        with patch.dict(os.environ, {"DTS_API_KEYS": "valid-key"}, clear=False):
            reload_keys()
            request = MagicMock()
            request.url.path = "/api/simulations"
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                _run(verify_api_key(request, api_key=None))
            assert exc_info.value.status_code == 401
            reload_keys()

    def test_public_path_skips_auth(self):
        """Public paths like /api/health should not require auth."""
        with patch.dict(os.environ, {"DTS_API_KEYS": "valid-key"}, clear=False):
            reload_keys()
            request = MagicMock()
            request.url.path = "/api/health"
            result = _run(verify_api_key(request, api_key=None))
            assert result is None  # No auth required
            reload_keys()

    def test_no_keys_configured_returns_none(self):
        """When auth is disabled (no keys), returns None for any request."""
        with patch.dict(os.environ, {}, clear=True):
            reload_keys()
            request = MagicMock()
            request.url.path = "/api/simulations"
            result = _run(verify_api_key(request, api_key=None))
            assert result is None
            reload_keys()


# ── Tests: Tenant dataclass ──────────────────────────────────────────────────


class TestTenant:
    def test_tenant_creation(self):
        t = Tenant(tenant_id="acme", api_key="sk-123")
        assert t.tenant_id == "acme"
        assert t.api_key == "sk-123"

    def test_tenant_equality(self):
        t1 = Tenant(tenant_id="acme", api_key="sk-123")
        t2 = Tenant(tenant_id="acme", api_key="sk-123")
        assert t1 == t2

    def test_different_tenants_not_equal(self):
        t1 = Tenant(tenant_id="acme", api_key="sk-123")
        t2 = Tenant(tenant_id="beta", api_key="sk-456")
        assert t1 != t2


# ── Tests: tenant isolation via key map ──────────────────────────────────────


class TestTenantIsolation:
    def test_different_keys_different_tenants(self):
        key_map = {"sk-acme": "acme-corp", "sk-beta": "beta-inc"}
        with patch.dict(os.environ, {"DTS_KEY_MAP": json.dumps(key_map)}, clear=False):
            reload_keys()

            request = MagicMock()
            request.url.path = "/api/simulations"

            tenant_a = _run(verify_api_key(request, api_key="sk-acme"))
            tenant_b = _run(verify_api_key(request, api_key="sk-beta"))

            assert tenant_a.tenant_id == "acme-corp"
            assert tenant_b.tenant_id == "beta-inc"
            assert tenant_a.tenant_id != tenant_b.tenant_id
            reload_keys()

    def test_key_map_invalid_key_rejected(self):
        key_map = {"sk-acme": "acme-corp"}
        with patch.dict(os.environ, {"DTS_KEY_MAP": json.dumps(key_map)}, clear=False):
            reload_keys()
            request = MagicMock()
            request.url.path = "/api/simulations"
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                _run(verify_api_key(request, api_key="sk-unknown"))
            assert exc_info.value.status_code == 403
            reload_keys()


# ── Tests: PUBLIC_PATHS ──────────────────────────────────────────────────────


class TestPublicPaths:
    def test_health_is_public(self):
        assert "/api/health" in PUBLIC_PATHS

    def test_docs_is_public(self):
        assert "/docs" in PUBLIC_PATHS

    def test_openapi_is_public(self):
        assert "/openapi.json" in PUBLIC_PATHS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
