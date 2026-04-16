"""Tests for agent classes — BaseAgent, EliteAgent, CitizenSwarm, AgentMemory."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.base_agent import BaseAgent
from core.agents.agent_memory import AgentMemory
from core.agents.elite_agent import EliteAgent
from core.agents.citizen_swarm import CitizenSwarm


# ── Tests: BaseAgent ─────────────────────────────────────────────────────────


class TestBaseAgent:
    def test_creation_with_required_fields(self):
        agent = BaseAgent(
            id="a1", name="Test Agent", role="Leader",
            archetype="populist", position=0.5,
        )
        assert agent.id == "a1"
        assert agent.position == 0.5
        assert agent.tier == 1

    def test_original_position_defaults_to_position(self):
        """If original_position is 0 and position != 0, original_position = position."""
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a", position=0.7,
        )
        assert agent.original_position == 0.7

    def test_explicit_original_position(self):
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a",
            position=0.7, original_position=0.3,
        )
        assert agent.original_position == 0.3

    @pytest.mark.parametrize("position", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_valid_positions(self, position):
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a", position=position,
        )
        assert agent.position == position

    def test_to_dict_contains_all_fields(self):
        agent = BaseAgent(
            id="a1", name="Test", role="Leader", archetype="populist",
            position=0.5, influence=0.8, rigidity=0.3,
        )
        d = agent.to_dict()
        assert d["id"] == "a1"
        assert d["position"] == 0.5
        assert d["influence"] == 0.8
        assert d["rigidity"] == 0.3
        assert "emotional_state" in d

    def test_from_dict_roundtrip(self):
        agent = BaseAgent(
            id="a1", name="Test", role="Leader", archetype="populist",
            position=0.5, influence=0.8, rigidity=0.3,
        )
        d = agent.to_dict()
        restored = BaseAgent.from_dict(d)
        assert restored.id == agent.id
        assert restored.position == agent.position
        assert restored.influence == agent.influence

    def test_tolerance_auto_computed_when_zero(self):
        """When tolerance is falsy, it's auto-computed from rigidity."""
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a",
            position=0.0, rigidity=0.5, tolerance=0,
        )
        # tolerance = 0.3 + (1 - rigidity) * 0.4 = 0.3 + 0.5*0.4 = 0.5
        assert abs(agent.tolerance - 0.5) < 1e-10

    def test_tolerance_preserved_when_set(self):
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a",
            position=0.0, tolerance=0.8,
        )
        assert agent.tolerance == 0.8

    def test_default_emotional_state(self):
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a", position=0.0,
        )
        assert agent.emotional_state == "neutral"


# ── Tests: EliteAgent ────────────────────────────────────────────────────────


class TestEliteAgent:
    def test_elite_tier_is_1(self):
        agent = EliteAgent(
            id="e1", name="Elite", role="PM", archetype="leader",
            position=0.5, original_position=0.5,
        )
        assert agent.tier == 1

    def test_platform_attributes(self):
        agent = EliteAgent(
            id="e1", name="Elite", role="PM", archetype="leader",
            position=0.5, original_position=0.5,
            platform_primary="twitter", platform_secondary="news",
        )
        assert agent.platform_primary == "twitter"
        assert agent.platform_secondary == "news"

    def test_from_spec_dict(self):
        spec = {
            "id": "e1", "name": "Leader", "role": "PM",
            "archetype": "populist", "position": 0.6,
            "influence": 0.9, "rigidity": 0.2,
            "platform_primary": "social", "platform_secondary": "forum",
        }
        agent = EliteAgent.from_spec(spec)
        assert agent.id == "e1"
        assert agent.position == 0.6
        assert agent.tier == 1
        assert agent.platform_primary == "social"

    def test_from_spec_sets_original_position(self):
        spec = {
            "id": "e1", "name": "L", "role": "R",
            "position": 0.4, "influence": 0.5,
        }
        agent = EliteAgent.from_spec(spec)
        assert agent.original_position == 0.4

    def test_generate_round_returns_none_on_error(self):
        """If LLM raises, generate_round should return None, not crash."""
        agent = EliteAgent(
            id="e1", name="E", role="R", archetype="a",
            position=0.5, original_position=0.5,
        )
        llm = AsyncMock()
        llm.generate_json.side_effect = Exception("LLM down")
        llm.stats = MagicMock()
        llm.stats.total_cost = 0.0

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            agent.generate_round(
                llm=llm, round_number=1, timeline_label="Week 1",
                round_event="Test event", viral_posts="",
                polarization=3.0, avg_sentiment="neutral",
                top_narratives="economy", prompt_template="{system_prompt}",
                channel_descriptions={}, channel_max_lengths={"social": 280},
            )
        )
        assert result is None


# ── Tests: AgentMemory ───────────────────────────────────────────────────────


class TestAgentMemory:
    def test_initial_state_empty(self):
        mem = AgentMemory()
        assert len(mem.round_summaries) == 0
        assert len(mem.posts_history) == 0

    def test_add_round(self):
        mem = AgentMemory()
        mem.add_round(
            round_num=1, summary="Agent took position",
            posts=[{"platform": "twitter", "text": "Hello"}],
            engagement={"likes": 10},
        )
        assert len(mem.round_summaries) == 1
        assert "[Round 1]" in mem.round_summaries[0]
        assert len(mem.posts_history) == 1

    def test_multiple_rounds_accumulate(self):
        mem = AgentMemory()
        for i in range(5):
            mem.add_round(round_num=i + 1, summary=f"Round {i+1}",
                          posts=[], engagement={})
        assert len(mem.round_summaries) == 5

    def test_alliances_update(self):
        mem = AgentMemory()
        mem.add_round(1, "r1", [], {}, alliances=["ally1", "ally2"])
        assert mem.alliances == ["ally1", "ally2"]
        mem.add_round(2, "r2", [], {}, alliances=["ally3"])
        assert mem.alliances == ["ally3"]

    def test_targets_update(self):
        mem = AgentMemory()
        mem.add_round(1, "r1", [], {}, targets=["target1"])
        assert mem.targets == ["target1"]

    def test_get_context_no_memory(self):
        mem = AgentMemory()
        ctx = mem.get_context()
        assert "No previous memory" in ctx

    def test_get_context_with_data(self):
        mem = AgentMemory()
        mem.add_round(1, "Summary text", [{"platform": "tw", "text": "Hi"}], {})
        ctx = mem.get_context()
        assert "Summary text" in ctx

    def test_get_context_respects_last_n(self):
        mem = AgentMemory()
        for i in range(10):
            mem.add_round(i + 1, f"UniqueSum_{i+1}_end", [], {})
        ctx = mem.get_context(last_n=2)
        assert "UniqueSum_9_end" in ctx
        assert "UniqueSum_10_end" in ctx
        assert "UniqueSum_1_end" not in ctx
        assert "UniqueSum_7_end" not in ctx

    def test_get_full_history(self):
        mem = AgentMemory()
        mem.add_round(1, "first", [], {})
        mem.add_round(2, "second", [], {})
        history = mem.get_full_history()
        assert "first" in history
        assert "second" in history

    def test_get_full_history_empty(self):
        mem = AgentMemory()
        assert mem.get_full_history() == "No history."


# ── Tests: CitizenSwarm ─────────────────────────────────────────────────────


class TestCitizenSwarm:
    def _make_fake_cluster(self, id, position=0.0, name="Cluster"):
        """Create a minimal mock CitizenCluster."""
        cluster = MagicMock()
        cluster.id = id
        cluster.name = name
        cluster.position = position
        cluster.original_position = position
        cluster.engagement_level = 0.5
        cluster.trust_institutions = 0.5
        cluster.dominant_sentiment = "indifferent"
        cluster.size = 50
        return cluster

    def test_swarm_stores_clusters_by_id(self):
        c1 = self._make_fake_cluster("c1")
        c2 = self._make_fake_cluster("c2")
        swarm = CitizenSwarm([c1, c2])
        assert "c1" in swarm.clusters
        assert "c2" in swarm.clusters

    def test_get_all_positions(self):
        c1 = self._make_fake_cluster("c1", position=0.3)
        c2 = self._make_fake_cluster("c2", position=-0.5)
        swarm = CitizenSwarm([c1, c2])
        positions = swarm.get_all_positions()
        assert positions["c1"] == 0.3
        assert positions["c2"] == -0.5

    def test_get_avg_sentiment(self):
        c1 = self._make_fake_cluster("c1")
        c1.dominant_sentiment = "angry"
        c2 = self._make_fake_cluster("c2")
        c2.dominant_sentiment = "angry"
        c3 = self._make_fake_cluster("c3")
        c3.dominant_sentiment = "hopeful"
        swarm = CitizenSwarm([c1, c2, c3])
        assert swarm.get_avg_sentiment() == "angry"

    def test_get_avg_sentiment_empty(self):
        swarm = CitizenSwarm([])
        assert swarm.get_avg_sentiment() == "indifferent"

    def test_single_cluster_sentiment(self):
        c1 = self._make_fake_cluster("c1")
        c1.dominant_sentiment = "hopeful"
        swarm = CitizenSwarm([c1])
        assert swarm.get_avg_sentiment() == "hopeful"


# ── Tests: position bounds enforcement ───────────────────────────────────────


class TestPositionBounds:
    @pytest.mark.parametrize("pos", [-2.0, -1.5, 1.5, 2.0])
    def test_base_agent_allows_out_of_range(self, pos):
        """BaseAgent itself does not enforce bounds — dynamics models do."""
        agent = BaseAgent(
            id="a1", name="T", role="R", archetype="a", position=pos,
        )
        # BaseAgent stores whatever value is given
        assert agent.position == pos

    def test_position_clamping_is_model_responsibility(self):
        """The opinion dynamics model clamps, not the agent constructor."""
        from core.simulation.opinion_dynamics import OpinionDynamics
        model = OpinionDynamics()
        # Simulate extreme input
        new_pos = model.update_position(
            agent_position=0.99, agent_original_position=0.99,
            agent_rigidity=0.0, agent_tolerance=2.0,
            feed_authors_positions=[(1.0, 1.0, 1.0)],
            event_shock_magnitude=1.0, event_shock_direction=1.0,
        )
        assert -1.0 <= new_pos <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
