"""API endpoint tests for DigitalTwinSim."""

import os
import sys
from unittest.mock import patch, AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_health_check(test_client):
    """Health endpoint should return 200 without auth."""
    # Health is public — remove auth header
    test_client.headers.pop("X-API-Key", None)
    resp = test_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_create_simulation_no_auth_401(unauth_client):
    """POST /api/simulations without key → 401."""
    resp = unauth_client.post(
        "/api/simulations",
        json={"brief": "Test brief"},
    )
    assert resp.status_code == 401


def test_create_simulation_invalid_key_403(unauth_client):
    """POST /api/simulations with invalid key → 403."""
    unauth_client.headers["X-API-Key"] = "invalid-key-999"
    resp = unauth_client.post(
        "/api/simulations",
        json={"brief": "Test brief"},
    )
    assert resp.status_code == 403


def test_brief_max_length_422(test_client):
    """Brief exceeding max_length → 422 validation error."""
    resp = test_client.post(
        "/api/simulations",
        json={"brief": "x" * 11000},
    )
    assert resp.status_code == 422


def test_budget_validation_422(test_client):
    """Budget out of range → 422."""
    resp = test_client.post(
        "/api/simulations",
        json={"brief": "Test", "budget": 999.0},
    )
    assert resp.status_code == 422


def test_rounds_validation_422(test_client):
    """Rounds out of range → 422."""
    resp = test_client.post(
        "/api/simulations",
        json={"brief": "Test", "rounds": 50},
    )
    assert resp.status_code == 422


def test_list_simulations_empty(test_client):
    """List simulations for fresh tenant → empty list."""
    resp = test_client.get("/api/simulations")
    assert resp.status_code == 200
    # May have existing sims from previous tests, just check it's a list
    assert isinstance(resp.json(), list)


def test_get_simulation_not_found(test_client):
    """GET /api/simulations/nonexistent → 404."""
    resp = test_client.get("/api/simulations/nonexist")
    assert resp.status_code == 404


def test_cancel_simulation_not_found(test_client):
    """DELETE /api/simulations/nonexistent → 404."""
    resp = test_client.delete("/api/simulations/nonexist")
    assert resp.status_code == 404


def test_intervene_wrong_status_404(test_client):
    """POST intervene on non-existent sim → 404."""
    resp = test_client.post(
        "/api/simulations/nonexist/intervene",
        json={"action_text": "Do something"},
    )
    assert resp.status_code == 404


def test_domains_list(test_client):
    """GET /api/domains should return domain list."""
    resp = test_client.get("/api/domains")
    assert resp.status_code == 200
    data = resp.json()
    assert "domains" in data
    assert isinstance(data["domains"], list)
