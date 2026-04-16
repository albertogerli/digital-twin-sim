"""Integration test: full simulation pipeline with mocked LLM."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Realistic canned LLM responses for scenario building
CANNED_SCENARIO_JSON = {
    "scenario_name": "Integration Test Scenario",
    "description": "Test scenario for integration testing",
    "domain": "political",
    "language": "en",
    "num_rounds": 3,
    "timeline_unit": "week",
    "timeline_labels": ["Week 1", "Week 2", "Week 3"],
    "position_axis": {
        "negative_label": "Against",
        "positive_label": "In favor",
        "neutral_label": "Neutral",
    },
    "channels": ["twitter", "news"],
    "elite_agents": [
        {
            "id": "elite_1",
            "name": "Test Leader",
            "role": "Political leader",
            "position": 0.7,
            "influence": 0.8,
            "rigidity": 0.3,
            "tier": 1,
        }
    ],
    "institutional_agents": [
        {
            "id": "inst_1",
            "name": "Test Institution",
            "role": "Government body",
            "position": 0.2,
            "influence": 0.6,
        }
    ],
    "citizen_clusters": [
        {
            "id": "cluster_1",
            "label": "General public",
            "size": 50,
            "avg_position": 0.0,
            "std_dev": 0.3,
        }
    ],
    "initial_event": "Policy announcement triggers public debate",
    "scenario_context": "A government policy proposal is being debated.",
    "metrics_to_track": ["Public approval", "Media coverage", "Trust index"],
}


@pytest.fixture
def mock_llm_realistic():
    """LLM mock with realistic responses for each pipeline component."""
    llm = AsyncMock()
    llm.stats = MagicMock()
    llm.stats.total_cost = 0.0
    llm.stats.total_input_tokens = 0
    llm.stats.total_output_tokens = 0

    # generate_json returns the scenario config
    llm.generate_json.return_value = CANNED_SCENARIO_JSON

    # generate_text returns event/commentary text
    llm.generate_text.return_value = (
        "Round update: The debate intensifies as new data emerges."
    )

    return llm


@pytest.mark.integration
@pytest.mark.slow
class TestFullSimulationPipeline:
    """Full 3-round simulation with mocked LLM."""

    def test_simulation_request_model_validation(self):
        """SimulationRequest should validate fields correctly."""
        from api.models import SimulationRequest

        # Valid request
        req = SimulationRequest(brief="Test brief", budget=5.0, rounds=3)
        assert req.brief == "Test brief"
        assert req.budget == 5.0
        assert req.rounds == 3

        # Invalid: brief too long
        with pytest.raises(Exception):
            SimulationRequest(brief="x" * 11000)

        # Invalid: budget too high
        with pytest.raises(Exception):
            SimulationRequest(brief="Test", budget=100.0)

        # Invalid: rounds too high
        with pytest.raises(Exception):
            SimulationRequest(brief="Test", rounds=50)

    def test_simulation_status_model(self):
        """SimulationStatus should serialize correctly."""
        from api.models import SimulationStatus

        status = SimulationStatus(
            id="abc12345",
            status="running",
            brief="Test",
            scenario_name="Test Scenario",
            current_round=2,
            total_rounds=5,
            cost=0.03,
            created_at="2024-01-01T00:00:00",
        )
        data = status.model_dump()
        assert data["id"] == "abc12345"
        assert data["status"] == "running"
        assert data["current_round"] == 2

    def test_tenant_isolation_in_manager(self):
        """SimulationManager should filter by tenant_id."""
        from api.simulation_manager import SimulationState, SimulationManager
        from api.models import SimulationRequest

        mgr = SimulationManager.__new__(SimulationManager)
        mgr.simulations = {}
        mgr._semaphore = asyncio.Semaphore(2)

        # Add sims for different tenants
        req = SimulationRequest(brief="Test")
        s1 = SimulationState("sim1", req, tenant_id="tenant_a")
        s1.status = "completed"
        s2 = SimulationState("sim2", req, tenant_id="tenant_b")
        s2.status = "completed"
        s3 = SimulationState("sim3", req, tenant_id="tenant_a")
        s3.status = "completed"

        mgr.simulations = {"sim1": s1, "sim2": s2, "sim3": s3}

        # Tenant A sees only their sims
        results_a = mgr.list_simulations(tenant_id="tenant_a")
        assert len(results_a) == 2
        assert all(r.id in ("sim1", "sim3") for r in results_a)

        # Tenant B sees only their sim
        results_b = mgr.list_simulations(tenant_id="tenant_b")
        assert len(results_b) == 1
        assert results_b[0].id == "sim2"

        # get_status with wrong tenant returns None
        assert mgr.get_status("sim1", tenant_id="tenant_b") is None
        assert mgr.get_status("sim1", tenant_id="tenant_a") is not None

    def test_auth_module(self):
        """Auth module should parse keys correctly."""
        from api.auth import _load_keys, Tenant

        with patch.dict(os.environ, {"DTS_API_KEYS": "key1,key2,key3"}):
            keys = _load_keys()
            assert "key1" in keys
            assert "key2" in keys
            assert keys["key1"] == "default"

        with patch.dict(os.environ, {
            "DTS_KEY_MAP": '{"sk-abc": "acme", "sk-def": "beta"}'
        }):
            keys = _load_keys()
            assert keys["sk-abc"] == "acme"
            assert keys["sk-def"] == "beta"

    def test_wargame_intervention_validation(self):
        """WargameIntervention field constraints."""
        from api.models import WargameIntervention

        # Valid
        wi = WargameIntervention(action_text="Test action")
        assert wi.action_text == "Test action"

        # Too long
        with pytest.raises(Exception):
            WargameIntervention(action_text="x" * 6000)

        # Invalid shock magnitude
        with pytest.raises(Exception):
            WargameIntervention(action_text="Test", shock_magnitude=2.0)
