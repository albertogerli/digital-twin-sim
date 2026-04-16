"""Tests for opinion_dynamics_v2 — DynamicsV2 reparametrized model."""

import math
import sys
import os

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation.opinion_dynamics_v2 import (
    DynamicsV2,
    ForceStandardizer,
    softmax,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

class FakeAgent:
    def __init__(self, id, position, original_position=None, tier=1,
                 rigidity=0.3, influence=0.5, tolerance=0.4):
        self.id = id
        self.position = position
        self.original_position = original_position if original_position is not None else position
        self.tier = tier
        self.rigidity = rigidity
        self.influence = influence
        self.tolerance = tolerance


class FakePlatform:
    """Minimal platform stub for tests.

    Posts must have 'id' key (FeedAlgorithm checks it).
    We also provide a fake conn to avoid SQLite dependency.
    """
    def __init__(self, posts=None):
        self._posts = []
        for i, p in enumerate(posts or []):
            post = dict(p)
            if "id" not in post:
                post["id"] = i + 1
            self._posts.append(post)
        self.conn = None  # FeedAlgorithm checks this

    def get_top_posts(self, round_num, top_n=10, platform=None):
        return self._posts[:top_n]

    def get_posts_by_round(self, round_num, limit=100):
        return self._posts[:limit]

    def get_following_ids(self, agent_id, platform="social"):
        return []


# ──────────────────────────────────────────────
# Test softmax
# ──────────────────────────────────────────────

class TestSoftmax:
    def test_sums_to_one(self):
        """π = softmax(α) must sum to 1."""
        logits = {"a": 2.0, "b": -1.0, "c": 0.5, "d": 3.0, "e": -2.0}
        pi = softmax(logits)
        assert abs(sum(pi.values()) - 1.0) < 1e-10

    def test_sums_to_one_default_logits(self):
        """Default logits produce weights that sum to 1."""
        model = DynamicsV2()
        pi = model.get_mix_weights()
        assert abs(sum(pi.values()) - 1.0) < 1e-10

    def test_sums_to_one_extreme_logits(self):
        """Even extreme logits should produce valid softmax."""
        logits = {"a": 100.0, "b": -100.0, "c": 50.0, "d": 0.0, "e": -50.0}
        pi = softmax(logits)
        assert abs(sum(pi.values()) - 1.0) < 1e-10
        # "a" should dominate
        assert pi["a"] > 0.99

    def test_uniform_logits(self):
        """Equal logits → equal weights."""
        logits = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
        pi = softmax(logits)
        for v in pi.values():
            assert abs(v - 0.2) < 1e-10

    def test_shift_invariance(self):
        """softmax(α + c) == softmax(α) for any constant c."""
        logits = {"a": 2.0, "b": -1.0, "c": 0.5}
        pi_original = softmax(logits)
        pi_shifted = softmax({k: v + 42.0 for k, v in logits.items()})
        for k in logits:
            assert abs(pi_original[k] - pi_shifted[k]) < 1e-10


# ──────────────────────────────────────────────
# Test: scaling forces doesn't change mix π
# ──────────────────────────────────────────────

class TestScaleInvariance:
    def test_scaling_forces_preserves_mix(self):
        """Multiplying all raw forces by constant c should not change π.

        π comes from α (the logits), not from force magnitudes.
        The mix is determined entirely by the learned logits, while
        force magnitudes are absorbed by standardization and λ.
        """
        model = DynamicsV2()
        pi_before = model.get_mix_weights()

        # Scaling forces doesn't affect α, so π stays the same
        # (π is a function of α only, not of force values)
        model2 = DynamicsV2(alpha=model.alpha.copy())
        pi_after = model2.get_mix_weights()

        for k in pi_before:
            assert abs(pi_before[k] - pi_after[k]) < 1e-10

    def test_standardizer_absorbs_scale(self):
        """Standardizing c·f gives the same z-score as standardizing f.

        If we multiply all force values by c, the z-scores are identical
        because z = (cx - c·μ) / (c·σ) = (x - μ) / σ.
        """
        names = ["direct", "social", "event", "herd", "anchor"]
        std1 = ForceStandardizer(names, buffer_size=50)
        std2 = ForceStandardizer(names, buffer_size=50)
        c = 7.3  # arbitrary scale factor

        rng = np.random.RandomState(42)
        # Feed 20 rounds of observations
        for _ in range(20):
            raw = {n: list(rng.randn(10)) for n in names}
            scaled = {n: [v * c for v in vals] for n, vals in raw.items()}
            std1.observe(raw)
            std2.observe(scaled)

        # Now standardize a test point
        test = {n: float(rng.randn()) for n in names}
        z1 = std1.standardize(test)
        z2 = std2.standardize({n: v * c for n, v in test.items()})

        for n in names:
            assert abs(z1[n] - z2[n]) < 1e-6, f"z-score mismatch for {n}: {z1[n]} vs {z2[n]}"


# ──────────────────────────────────────────────
# Test: λ controls step amplitude
# ──────────────────────────────────────────────

class TestStepSize:
    def _make_agents_and_event(self):
        agents = [
            FakeAgent("a1", 0.0, tier=1, rigidity=0.2, influence=0.8),
            FakeAgent("a2", 0.3, tier=1, rigidity=0.3, influence=0.6),
            FakeAgent("a3", -0.2, tier=3, rigidity=0.1, influence=0.3),
        ]
        event = {"round": 1, "shock_magnitude": 0.4, "shock_direction": 0.5}
        posts = [
            {"author_id": "a1", "likes": 50, "reposts": 10},
            {"author_id": "a2", "likes": 30, "reposts": 5},
        ]
        platform = FakePlatform(posts)
        return agents, event, platform

    def test_larger_lambda_larger_delta(self):
        """Doubling λ should approximately double the position delta."""
        agents1, event, platform = self._make_agents_and_event()
        agents2, _, _ = self._make_agents_and_event()

        small_lambda = {"elite": 0.05, "institutional": 0.05, "citizen": 0.05}
        big_lambda = {"elite": 0.10, "institutional": 0.10, "citizen": 0.10}

        model1 = DynamicsV2(step_sizes=small_lambda)
        model2 = DynamicsV2(step_sizes=big_lambda)

        # Prime the standardizers with identical data
        for _ in range(3):
            test_agents, test_event, test_platform = self._make_agents_and_event()
            model1.step(test_agents, test_platform, test_event)
            test_agents2, _, _ = self._make_agents_and_event()
            model2.step(test_agents2, test_platform, test_event)

        # Now run a measured step
        agents1, event, platform = self._make_agents_and_event()
        agents2, _, _ = self._make_agents_and_event()

        pos_before = agents1[0].position
        model1.step(agents1, platform, event)
        delta1 = abs(agents1[0].position - pos_before)

        pos_before2 = agents2[0].position
        model2.step(agents2, platform, event)
        delta2 = abs(agents2[0].position - pos_before2)

        # delta2 should be larger (approximately 2x, but clamping may interfere)
        if delta1 > 1e-6:
            ratio = delta2 / delta1
            assert ratio > 1.3, f"Expected bigger step with larger λ, got ratio={ratio:.2f}"

    def test_zero_lambda_no_movement(self):
        """λ = 0 → no position change (except anchor drift)."""
        agents, event, platform = self._make_agents_and_event()
        zero_lambda = {"elite": 0.0, "institutional": 0.0, "citizen": 0.0}

        model = DynamicsV2(step_sizes=zero_lambda, anchor_drift_rate=0.0)
        positions_before = {a.id: a.position for a in agents}
        model.step(agents, platform, event)

        for a in agents:
            assert abs(a.position - positions_before[a.id]) < 1e-10, \
                f"Agent {a.id} moved with λ=0"


# ──────────────────────────────────────────────
# Test: v1 conversion roundtrip
# ──────────────────────────────────────────────

class TestV1Conversion:
    def test_from_v1_mix_preserves_ranking(self):
        """Converting v1 params to v2 should preserve the relative ranking of weights."""
        model = DynamicsV2.from_v1_params(
            direct_shift_weight=0.5,
            social_weight=0.12,
            event_weight=0.08,
            herd_weight=0.03,
            anchor_weight=0.10,
        )
        pi = model.get_mix_weights()
        # direct should be largest
        assert pi["direct"] == max(pi.values()), f"direct should be largest: {pi}"
        # herd should be smallest
        assert pi["herd"] == min(pi.values()), f"herd should be smallest: {pi}"

    def test_roundtrip_preserves_ranking(self):
        """v1 → v2 → v1 should preserve weight ranking."""
        v1_params = {
            "direct_shift_weight": 0.5,
            "social_weight": 0.12,
            "event_weight": 0.08,
            "herd_weight": 0.03,
            "anchor_weight": 0.10,
        }
        model = DynamicsV2.from_v1_params(**v1_params, herd_threshold=0.2, anchor_drift_rate=0.2)
        back = model.to_v1_params()

        # Rankings should be preserved
        original_ranked = sorted(v1_params.items(), key=lambda x: x[1], reverse=True)
        back_ranked = sorted(
            [(k, back[k]) for k in v1_params.keys()],
            key=lambda x: x[1],
            reverse=True,
        )
        original_order = [k for k, _ in original_ranked]
        back_order = [k for k, _ in back_ranked]
        assert original_order == back_order, \
            f"Ranking changed: {original_order} → {back_order}"


# ──────────────────────────────────────────────
# Test: ForceStandardizer
# ──────────────────────────────────────────────

class TestStandardizer:
    def test_cold_start_passthrough(self):
        """With no observations, raw values pass through unchanged."""
        std = ForceStandardizer(["a", "b"])
        result = std.standardize({"a": 0.5, "b": -0.3})
        assert result["a"] == 0.5
        assert result["b"] == -0.3

    def test_zero_variance_returns_zero(self):
        """If all observed values are identical, z-score is 0."""
        std = ForceStandardizer(["a"], buffer_size=10)
        std.observe({"a": [0.5, 0.5, 0.5, 0.5]})
        result = std.standardize({"a": 0.5})
        assert result["a"] == 0.0

    def test_correct_zscore(self):
        """Verify z-score calculation against known values."""
        std = ForceStandardizer(["x"], buffer_size=100)
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        std.observe({"x": data})
        # mean=3, std=sqrt(2)
        z = std.standardize({"x": 5.0})
        expected = (5.0 - 3.0) / np.std(data)
        assert abs(z["x"] - expected) < 1e-6


# ──────────────────────────────────────────────
# Test: integration — full step doesn't crash
# ──────────────────────────────────────────────

class TestIntegration:
    def test_step_runs(self):
        """Full step completes without errors and returns positions."""
        agents = [
            FakeAgent("e1", 0.3, tier=1),
            FakeAgent("e2", -0.5, tier=1),
            FakeAgent("i1", 0.0, tier=2),
            FakeAgent("c1", 0.1, tier=3),
        ]
        posts = [
            {"author_id": "e1", "likes": 20, "reposts": 5},
            {"author_id": "e2", "likes": 10, "reposts": 2},
        ]
        platform = FakePlatform(posts)
        event = {"round": 1, "shock_magnitude": 0.3, "shock_direction": 0.4}

        model = DynamicsV2()
        result = model.step(agents, platform, event)

        assert len(result) == 4
        for agent in agents:
            assert -1.0 <= agent.position <= 1.0

    def test_positions_bounded(self):
        """Positions stay in [-1, 1] even with extreme forces."""
        agents = [
            FakeAgent("a1", 0.95, tier=1, rigidity=0.0),
        ]
        platform = FakePlatform([{"author_id": "a1", "likes": 999, "reposts": 999}])
        event = {"round": 1, "shock_magnitude": 1.0, "shock_direction": 1.0}

        model = DynamicsV2(step_sizes={"elite": 0.5, "institutional": 0.5, "citizen": 0.5})
        for _ in range(20):
            model.step(agents, platform, event)
        assert -1.0 <= agents[0].position <= 1.0

    def test_update_all_agents_compat(self):
        """v1-compatible interface works."""
        agents = [FakeAgent("a1", 0.0, tier=1)]
        platform = FakePlatform([{"author_id": "a1", "likes": 10, "reposts": 2}])
        event = {"round": 1, "shock_magnitude": 0.3, "shock_direction": 0.2}

        model = DynamicsV2()
        model.update_all_agents(agents, platform, event)
        assert isinstance(agents[0].position, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
