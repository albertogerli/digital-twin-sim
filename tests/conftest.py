"""Shared test fixtures for DigitalTwinSim."""

import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tenant():
    """Test tenant dataclass."""
    from api.auth import Tenant
    return Tenant(tenant_id="test-tenant", api_key="test-key-123")


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns canned JSON responses."""
    llm = AsyncMock()
    llm.stats = MagicMock()
    llm.stats.total_cost = 0.05
    llm.stats.total_input_tokens = 1000
    llm.stats.total_output_tokens = 500

    # Default canned responses
    llm.generate_json.return_value = {
        "scenario_name": "Test Scenario",
        "domain": "political",
        "agents": [],
    }
    llm.generate_text.return_value = "Test response text"

    return llm


@pytest.fixture
def tmp_outputs(tmp_path):
    """Temporary outputs directory."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    exports = outputs / "exports"
    exports.mkdir()
    return outputs


@pytest.fixture
def test_client():
    """FastAPI TestClient with valid auth header."""
    # Patch env to enable auth with test key
    with patch.dict(os.environ, {
        "DTS_API_KEYS": "test-key-123",
        "DTS_RATE_LIMIT_ENABLED": "false",
    }):
        # Force reload of auth keys
        from api.auth import reload_keys
        reload_keys()

        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        client.headers["X-API-Key"] = "test-key-123"
        yield client

        # Cleanup
        reload_keys()


@pytest.fixture
def unauth_client():
    """FastAPI TestClient WITHOUT auth header."""
    with patch.dict(os.environ, {
        "DTS_API_KEYS": "test-key-123",
        "DTS_RATE_LIMIT_ENABLED": "false",
    }):
        from api.auth import reload_keys
        reload_keys()

        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        yield client

        reload_keys()
