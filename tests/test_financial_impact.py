"""Financial impact scorer tests."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_scorer():
    """Try to import FinancialImpactScorer."""
    try:
        from core.simulation.financial_impact import FinancialImpactScorer
        return FinancialImpactScorer
    except ImportError:
        pytest.skip("FinancialImpactScorer not available")


def test_crisis_scope_low():
    """Low polarization + positive sentiment → low crisis scope."""
    Scorer = _get_scorer()
    scorer = Scorer()
    result = scorer.score(
        polarization=1.0,
        sentiment={"positive": 0.6, "neutral": 0.3, "negative": 0.1},
        event_shock=0.1,
    )
    assert result["crisis_scope"] < 0.3


def test_crisis_scope_high():
    """High polarization + negative sentiment → high crisis scope."""
    Scorer = _get_scorer()
    scorer = Scorer()
    result = scorer.score(
        polarization=8.0,
        sentiment={"positive": 0.1, "neutral": 0.2, "negative": 0.7},
        event_shock=0.8,
    )
    assert result["crisis_scope"] > 0.5


def test_panic_multiplier():
    """Panic multiplier should amplify with consecutive negative rounds."""
    Scorer = _get_scorer()
    scorer = Scorer()

    # Simulate consecutive negative rounds
    for _ in range(3):
        result = scorer.score(
            polarization=7.0,
            sentiment={"positive": 0.1, "neutral": 0.2, "negative": 0.7},
            event_shock=0.6,
        )

    assert result.get("panic_multiplier", 1.0) >= 1.0


def test_score_returns_expected_keys():
    """Score output should contain standard fields."""
    Scorer = _get_scorer()
    scorer = Scorer()
    result = scorer.score(
        polarization=5.0,
        sentiment={"positive": 0.3, "neutral": 0.4, "negative": 0.3},
        event_shock=0.3,
    )
    # At minimum should have some impact data
    assert isinstance(result, dict)
    assert len(result) > 0


# ── Empirical impulse-response coefficients ────────────────────────────────
# Tests for the new path that replaces hardcoded recovery/escalation factors
# with calibrated coefficients from shared/impulse_response_coefficients.json.

def test_intensity_bin_thresholds():
    """The intensity bucketing must match the calibration script's bins."""
    from core.orchestrator.financial_impact import _intensity_bin
    assert _intensity_bin(0.0) == "low"
    assert _intensity_bin(1.99) == "low"
    assert _intensity_bin(2.0) == "mid"
    assert _intensity_bin(3.99) == "mid"
    assert _intensity_bin(4.0) == "high"
    assert _intensity_bin(8.0) == "high"


def test_empirical_loader_handles_missing_file(tmp_path, monkeypatch):
    """When the JSON is absent, _load_impulse_response returns {} and the
    empirical lookup returns None → caller falls back to heuristic."""
    from core.orchestrator import financial_impact as fi
    monkeypatch.setattr(fi, "_IR_COEFFS_PATH", tmp_path / "nonexistent.json")
    monkeypatch.setattr(fi, "_ir_coeffs_cache", None)
    table = fi._load_impulse_response()
    assert table == {}
    assert fi._empirical_ratios("banking", 4.0) is None


def test_empirical_loader_picks_specific_cell(tmp_path, monkeypatch):
    """When a sector cell with n_obs >= 4 exists, it is preferred over ALL."""
    import json
    from core.orchestrator import financial_impact as fi
    fake = tmp_path / "ir.json"
    fake.write_text(json.dumps({
        "coefficients": {
            "high": {
                "banking": {"t3_over_t1": 1.4, "t7_over_t1": 1.8, "n_obs": 12},
                "ALL":     {"t3_over_t1": 1.0, "t7_over_t1": 1.2, "n_obs": 80},
            }
        }
    }))
    monkeypatch.setattr(fi, "_IR_COEFFS_PATH", fake)
    monkeypatch.setattr(fi, "_ir_coeffs_cache", None)
    r = fi._empirical_ratios("banking", 5.0)
    assert r is not None
    t3, t7, n, label = r
    assert t3 == 1.4 and t7 == 1.8 and n == 12
    assert "banking" in label


def test_empirical_loader_falls_back_to_pooled(tmp_path, monkeypatch):
    """A sparse sector cell (n_obs < 4) defers to the ALL pooled cell."""
    import json
    from core.orchestrator import financial_impact as fi
    fake = tmp_path / "ir.json"
    fake.write_text(json.dumps({
        "coefficients": {
            "mid": {
                "luxury": {"t3_over_t1": 0.3, "t7_over_t1": 0.4, "n_obs": 2},
                "ALL":    {"t3_over_t1": 0.7, "t7_over_t1": 0.9, "n_obs": 60},
            }
        }
    }))
    monkeypatch.setattr(fi, "_IR_COEFFS_PATH", fake)
    monkeypatch.setattr(fi, "_ir_coeffs_cache", None)
    r = fi._empirical_ratios("luxury", 3.0)
    assert r is not None
    t3, t7, n, label = r
    assert t3 == 0.7 and t7 == 0.9 and n == 60
    assert "ALL" in label


# ── Empirical panic-multiplier ─────────────────────────────────────────────

def test_cri_bin_thresholds():
    """The CRI bucketing must match the calibration script's bins."""
    from core.orchestrator.financial_impact import _cri_bin
    assert _cri_bin(0.0) == "low"
    assert _cri_bin(0.39) == "low"
    assert _cri_bin(0.4) == "mid"
    assert _cri_bin(0.69) == "mid"
    assert _cri_bin(0.7) == "high"
    assert _cri_bin(0.84) == "high"
    assert _cri_bin(0.85) == "extreme"
    assert _cri_bin(1.0) == "extreme"


def test_panic_mult_loader_handles_missing(tmp_path, monkeypatch):
    """When the JSON is absent, the loader returns {} and the caller
    falls back to the analytic formula."""
    from core.orchestrator import financial_impact as fi
    monkeypatch.setattr(fi, "_PANIC_MULT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(fi, "_panic_mult_cache", None)
    assert fi._load_panic_multipliers() == {}
    assert fi._empirical_panic_mult(0.95) is None


def test_panic_mult_prefers_median_over_mean(tmp_path, monkeypatch):
    """When both median_ratio and panic_mult (mean) are present, median
    is preferred (robust to outliers like Lehman / COVID)."""
    import json
    from core.orchestrator import financial_impact as fi
    fake = tmp_path / "panic.json"
    fake.write_text(json.dumps({
        "panic_multipliers": {
            "extreme": {
                "panic_mult": 42.07,        # mean (would over-amplify)
                "median_ratio": 10.86,       # median (representative)
                "n_obs": 49,
            }
        }
    }))
    monkeypatch.setattr(fi, "_PANIC_MULT_PATH", fake)
    monkeypatch.setattr(fi, "_panic_mult_cache", None)
    r = fi._empirical_panic_mult(0.95)
    assert r is not None
    mult, label = r
    assert mult == 10.86
    assert "median" in label
    assert "extreme" in label


def test_panic_mult_falls_back_to_mean_when_no_median(tmp_path, monkeypatch):
    """If only the weighted-mean field is present, use it."""
    import json
    from core.orchestrator import financial_impact as fi
    fake = tmp_path / "panic.json"
    fake.write_text(json.dumps({
        "panic_multipliers": {
            "high": {"panic_mult": 5.0, "n_obs": 52}
        }
    }))
    monkeypatch.setattr(fi, "_PANIC_MULT_PATH", fake)
    monkeypatch.setattr(fi, "_panic_mult_cache", None)
    r = fi._empirical_panic_mult(0.78)
    assert r is not None and r[0] == 5.0


def test_panic_mult_skips_sparse_cells(tmp_path, monkeypatch):
    """A cell with n_obs < 4 is treated as unreliable → return None."""
    import json
    from core.orchestrator import financial_impact as fi
    fake = tmp_path / "panic.json"
    fake.write_text(json.dumps({
        "panic_multipliers": {
            "mid": {"median_ratio": 3.0, "panic_mult": 4.0, "n_obs": 2}
        }
    }))
    monkeypatch.setattr(fi, "_PANIC_MULT_PATH", fake)
    monkeypatch.setattr(fi, "_panic_mult_cache", None)
    assert fi._empirical_panic_mult(0.5) is None
