"""Checkpoint save/load roundtrip tests."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_mock_agent(id, name, role="leader", position=0.5, original_position=0.4,
                     influence=0.7, rigidity=0.3, emotional_state="neutral"):
    """Create a mock agent with to_dict() support."""
    agent = MagicMock()
    agent.id = id
    agent.name = name
    agent.role = role
    agent.position = position
    agent.original_position = original_position
    agent.influence = influence
    agent.rigidity = rigidity
    agent.emotional_state = emotional_state
    agent.key_trait = ""
    agent.category = ""
    agent.to_dict.return_value = {
        "id": id, "name": name, "role": role,
        "position": position, "original_position": original_position,
        "influence": influence, "rigidity": rigidity,
        "emotional_state": emotional_state,
    }
    return agent


def _make_mock_swarm():
    """Create a mock citizen swarm with clusters."""
    cluster = MagicMock()
    cluster.to_dict.return_value = {
        "id": "cluster_1", "label": "Giovani urbani",
        "size": 150, "position": 0.3,
    }
    swarm = MagicMock()
    swarm.clusters = {"cluster_1": cluster}
    return swarm


@pytest.fixture
def mock_agents():
    return [
        _make_mock_agent("elite_1", "Mario Rossi", position=0.6, original_position=0.5),
        _make_mock_agent("elite_2", "Luigi Bianchi", position=-0.4, original_position=-0.3),
    ]


@pytest.fixture
def mock_institutional():
    return [_make_mock_agent("inst_1", "Governo", role="institution", position=0.2, original_position=0.1)]


@pytest.fixture
def mock_swarm():
    return _make_mock_swarm()


@pytest.fixture
def coalition_history():
    return [
        {"round": 1, "coalitions": [{"name": "Pro", "size": 3}]},
        {"round": 2, "coalitions": [{"name": "Pro", "size": 4}]},
        {"round": 3, "coalitions": [{"name": "Pro", "size": 5}]},
    ]


def test_checkpoint_save_load_roundtrip(mock_agents, mock_institutional, mock_swarm, coalition_history, tmp_path):
    """Save and reload checkpoint, data should match."""
    from core.simulation.checkpoint import save_checkpoint, load_checkpoint

    filename = save_checkpoint(
        str(tmp_path), "Test_Scenario", 3,
        elite_agents=mock_agents,
        institutional_agents=mock_institutional,
        citizen_swarm=mock_swarm,
        coalition_history=coalition_history,
        cost=1.23,
    )

    filepath = os.path.join(str(tmp_path), filename)
    assert os.path.exists(filepath)

    loaded = load_checkpoint(filepath)
    assert loaded["round"] == 3
    assert loaded["scenario"] == "Test_Scenario"
    assert len(loaded["elite_agents"]) == 2
    assert loaded["elite_agents"][0]["position"] == 0.6
    assert loaded["cost"] == 1.23


def test_checkpoint_find(mock_agents, mock_institutional, mock_swarm, coalition_history, tmp_path):
    """find_checkpoint should locate the correct round file."""
    from core.simulation.checkpoint import save_checkpoint, find_checkpoint

    for r in [1, 2, 3]:
        save_checkpoint(
            str(tmp_path), "Test_Scenario", r,
            elite_agents=mock_agents,
            institutional_agents=mock_institutional,
            citizen_swarm=mock_swarm,
            coalition_history=coalition_history,
            cost=0.5,
        )

    found = find_checkpoint(str(tmp_path), "Test_Scenario", 2)
    loaded = json.loads(open(found).read())
    assert loaded["round"] == 2


def test_checkpoint_not_found(tmp_path):
    """find_checkpoint for missing round → FileNotFoundError."""
    from core.simulation.checkpoint import find_checkpoint

    with pytest.raises(FileNotFoundError):
        find_checkpoint(str(tmp_path), "Nonexistent", 99)
