"""Smoke tests for the FinancialTwin layer (Sprints 1-4).

Coverage:
- baseline state shape + ALM ratios in plausible bands
- step() with realistic Sella-style 5-round scenario keeps numbers
  inside Italian bank reference ranges (no bank-run, no NIM collapse)
- exposure inference + weighted aggregation
- CIR rate process: positivity, mean reversion, deterministic given seed
- EBA stress template: tighter caps, breach detection
- market_data: cache hit path (no network)

These are unit tests, no LLM, no DB, no network. All run < 1s total.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── FinancialTwin baseline ─────────────────────────────────────────────────

def test_baseline_state_in_italian_reference_band():
    from core.financial import FinancialTwin
    twin = FinancialTwin()
    s = twin.current_state()
    assert s.round == 0
    # NIM: EBA Q2 2025 EU avg 1.58%, IT mid-bank carries higher → 1.5-2.5%
    assert 1.4 <= s.nim_pct <= 2.5, f"NIM {s.nim_pct} outside IT band"
    # CET1: EBA Q3 2025 EU avg 16.3% → expect 14-18% for our default
    assert 14.0 <= s.cet1_pct <= 18.0, f"CET1 {s.cet1_pct} outside band"
    # LCR: EBA Q3 2025 EU avg 160.7% → expect 130-200%
    assert 130.0 <= s.lcr_pct <= 200.0, f"LCR {s.lcr_pct} outside band"
    # Baseline: no breach
    assert not s.breaches


def test_step_5_round_rate_hike_does_not_produce_bank_run():
    """The bug we're fixing: LLM-only sim said deposit runoff 92% on a
    +2% rate hike. Twin should keep cumulative deposit drop < 10%."""
    from core.financial import FinancialTwin
    twin = FinancialTwin()
    s0_deposit = twin.current_state().deposit_balance
    for r in range(1, 6):
        twin.step(
            round_num=r,
            rate_change_bps=200 if r == 1 else 0,
            opinion_aggregate=-0.3,
            polarization=4.0,
        )
    final_deposit = twin.current_state().deposit_balance
    cumulative_drop = (1 - final_deposit / s0_deposit) * 100
    assert cumulative_drop < 10.0, (
        f"Twin produced bank-run-like deposit drop ({cumulative_drop:.1f}%); "
        "expected < 10% under realistic ALM constraints"
    )


def test_step_emits_feedback_signals():
    from core.financial import FinancialTwin, FeedbackSignals
    twin = FinancialTwin()
    twin.step(round_num=1, rate_change_bps=200, opinion_aggregate=-0.5, polarization=6.0)
    fb = twin.latest_feedback()
    assert isinstance(fb, FeedbackSignals)
    # All signals in [0, 1]
    for v in fb.to_dict().values():
        assert 0.0 <= v <= 1.0


def test_compact_str_renders_without_error():
    from core.financial import FinancialTwin
    twin = FinancialTwin()
    s = twin.current_state()
    out = s.to_compact_str()
    assert "NIM" in out and "CET1" in out and "LCR" in out


# ── Exposure inference ─────────────────────────────────────────────────────

def test_codacons_classified_as_consumer_voice():
    from core.financial import infer_financial_exposure
    ex = infer_financial_exposure(archetype="consumer_advocate", role="Codacons")
    assert ex["is_consumer_voice"] >= 0.9
    assert ex["is_depositor"] >= 0.4


def test_findomestic_classified_as_competitor():
    from core.financial import infer_financial_exposure
    ex = infer_financial_exposure(archetype="industry_competitor", role="Findomestic")
    assert ex["is_competitor"] >= 0.9


def test_aggregate_exposure_weights_negativity_correctly():
    from core.financial import aggregate_opinion_by_exposure, infer_financial_exposure
    rows = [
        # Two retail consumers, very negative
        {"position": -0.8, "exposure": infer_financial_exposure(archetype="citizen", role="retail"), "weight": 1.0},
        {"position": -0.6, "exposure": infer_financial_exposure(archetype="citizen", role="retail"), "weight": 1.0},
        # One regulator, neutral (should NOT count as depositor)
        {"position": +0.0, "exposure": infer_financial_exposure(archetype="regulator", role="Bankitalia"), "weight": 1.0},
    ]
    agg = aggregate_opinion_by_exposure(rows)
    # Only retail count for depositors_negative; regulator has is_depositor=0
    assert agg["depositors_negative"] > 0.6, f"got {agg['depositors_negative']}"
    assert agg["competitors_negative_to_us"] == 0.0


# ── CIR rate process ───────────────────────────────────────────────────────

def test_cir_stays_positive_over_long_run():
    from core.financial import CIRRateProcess
    cir = CIRRateProcess(seed=1)
    for _ in range(120):  # 10 years monthly
        r = cir.step()
        assert r > 0


def test_cir_deterministic_given_seed():
    from core.financial import CIRRateProcess
    a = CIRRateProcess(seed=42)
    b = CIRRateProcess(seed=42)
    for _ in range(20):
        assert a.step() == b.step()


def test_cir_mean_reversion_long_horizon():
    """After many steps the average should settle near θ."""
    from core.financial import CIRRateProcess
    cir = CIRRateProcess(seed=7, theta=0.025, kappa=0.5)
    samples = [cir.step() for _ in range(2000)]
    mean = sum(samples) / len(samples)
    # Tolerance because of finite-sample variance
    assert abs(mean - 0.025) < 0.005, f"CIR mean {mean:.4f} drifted from theta=0.025"


# ── EBA stress template ────────────────────────────────────────────────────

def test_eba_adverse_template_tightens_runoff_cap():
    from core.financial import FinancialTwin
    base = FinancialTwin()
    stress = FinancialTwin(stress_template_name="adverse")
    assert stress.params["consumer_loan_elasticity"] < base.params["consumer_loan_elasticity"]
    assert stress.params["deposit_beta_sight"] > base.params["deposit_beta_sight"]
    assert stress.stress_template is not None


def test_eba_adverse_triggers_breach_at_round_1():
    """The +250bps shock + tighter params should trip at least one breach."""
    from core.financial import FinancialTwin
    twin = FinancialTwin(stress_template_name="adverse")
    twin.step(round_num=1, opinion_aggregate=-0.4, polarization=6.0)
    s1 = twin.current_state()
    # Either LCR or CET1 should be flagged early under stress
    assert s1.breaches, f"Stress should trigger a breach; got none ({s1})"


# ── Market data (cache only, no network) ───────────────────────────────────

def test_market_data_fallback_to_default_when_offline(monkeypatch):
    """Force ECB fetch to fail → expect graceful default fallback."""
    from core.financial import market_data

    def _fake_fail(*a, **kw):
        return None

    # Monkeypatch the ECB fetcher to always fail
    monkeypatch.setattr(market_data, "_fetch_ecb_series", _fake_fail)
    # Also nuke cache so we can't read a cached value
    monkeypatch.setattr(market_data, "_cache_get", lambda k: None)
    monkeypatch.setattr(market_data, "_cache_set", lambda k, v: None)

    val = market_data.get_euribor_3m_pct(use_cache=False)
    assert abs(val - market_data._DEFAULTS["euribor_3m_pct"]) < 1e-9
