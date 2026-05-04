"""Tests for the BYOD sanitizer (core/byod/sanitizer.py).

Covers:
  - Each detector category fires on canonical examples.
  - Non-financial percentages and dates are NOT redacted.
  - Mode semantics (OFF/LOG/STRICT/BLOCK) behave as documented.
  - Audit log rows are well-formed JSONL.
  - The four threshold-aware metric labels (LCR healthy/breaching,
    CET1 healthy/tight, NIM above/below median).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.byod.sanitizer import (
    BYODMode,
    BYODLeakError,
    SanitizeResult,
    audit_summary,
    sanitize_prompt,
)


# ── Mode: OFF ────────────────────────────────────────────────────────────────


def test_mode_off_passthrough_no_audit(tmp_path):
    audit = tmp_path / "audit.jsonl"
    res = sanitize_prompt(
        "LCR is 95% and deposit balance €12,500,000",
        call_site="test_off",
        mode=BYODMode.OFF,
        audit_path=audit,
    )
    assert res.text == "LCR is 95% and deposit balance €12,500,000"
    assert res.detections == []
    assert not audit.exists()


# ── Mode: LOG (detect, don't modify) ─────────────────────────────────────────


def test_mode_log_detects_but_preserves(tmp_path):
    audit = tmp_path / "audit.jsonl"
    res = sanitize_prompt(
        "deposit balance €12,500,000",
        call_site="test_log",
        mode=BYODMode.LOG,
        audit_path=audit,
    )
    assert res.text == "deposit balance €12,500,000"  # unchanged
    assert any(d["category"] == "currency" for d in res.detections)
    assert audit.exists()
    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["mode"] == "LOG"
    assert any(p["category"] == "currency" for p in rows[0]["patterns"])


# ── Mode: STRICT (detect + redact + audit) ───────────────────────────────────


def test_mode_strict_redacts_currency(tmp_path):
    audit = tmp_path / "audit.jsonl"
    res = sanitize_prompt(
        "Apple raised €2.4M in funding and $500K in seed",
        call_site="test_strict_curr",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    assert "€2.4M" not in res.text
    assert "$500K" not in res.text
    assert "[currency-amount]" in res.text
    assert res.modified


def test_mode_strict_redacts_lcr_healthy(tmp_path):
    audit = tmp_path / "audit.jsonl"
    res = sanitize_prompt(
        "LCR is 168%",
        call_site="test_strict_lcr_h",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    assert "168" not in res.text
    assert "[LCR healthy: above 100%]" in res.text


def test_mode_strict_redacts_lcr_breaching(tmp_path):
    audit = tmp_path / "audit.jsonl"
    res = sanitize_prompt(
        "LCR=95",
        call_site="test_strict_lcr_b",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    assert "95" not in res.text
    assert "[LCR breaching: below 100%]" in res.text


def test_mode_strict_redacts_cet1(tmp_path):
    audit = tmp_path / "audit.jsonl"
    res_healthy = sanitize_prompt(
        "CET1 ratio 16%",
        call_site="t1",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    res_tight = sanitize_prompt(
        "CET1 of 11.5%",
        call_site="t2",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    assert "[CET1 healthy: above 12%]" in res_healthy.text
    assert "[CET1 tight: below 12%]" in res_tight.text


def test_mode_strict_redacts_nim(tmp_path):
    res_above = sanitize_prompt(
        "NIM at 1.85%",
        call_site="t1",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    res_below = sanitize_prompt(
        "NIM is 1.40%",
        call_site="t2",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "[NIM above EBA median]" in res_above.text
    assert "[NIM below EBA median]" in res_below.text


def test_mode_strict_redacts_client_id(tmp_path):
    res = sanitize_prompt(
        "Issue affects client-12345 and customer ID 9999",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "12345" not in res.text
    assert "9999" not in res.text
    assert "[client-id]" in res.text


def test_mode_strict_redacts_iban(tmp_path):
    res = sanitize_prompt(
        "Transfer to IT60 X05428 11101 000000123456",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "[IBAN]" in res.text
    assert "IT60" not in res.text


def test_mode_strict_redacts_named_benchmark(tmp_path):
    res_eu = sanitize_prompt(
        "Euribor 3M at 2.4%",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    res_btp = sanitize_prompt(
        "BTP-Bund spread 180bps",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "Euribor" in res_eu.text and "2.4" not in res_eu.text
    assert "BTP" in res_btp.text and "180" not in res_btp.text


def test_mode_strict_redacts_large_amount_in_financial_context(tmp_path):
    res = sanitize_prompt(
        "deposit balance reached 12,500,000 yesterday",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "12,500,000" not in res.text
    assert "[large-amount]" in res.text


# ── Non-financial content must be preserved ──────────────────────────────────


def test_poll_percentage_preserved(tmp_path):
    """A poll opinion percentage with no financial keyword nearby must
    NOT be redacted — that's narrative-safe content."""
    res = sanitize_prompt(
        "The opposition gained 45% of the vote in the latest poll",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "45%" in res.text
    assert res.detections == []


def test_dates_preserved(tmp_path):
    res = sanitize_prompt(
        "On 2024-03-15 the council met to discuss the policy",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "2024-03-15" in res.text


def test_agent_position_preserved(tmp_path):
    """Position scores like -0.42 in [-1, +1] are not financial values."""
    res = sanitize_prompt(
        "Agent X drifted from -0.12 to +0.34 over 5 rounds",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "-0.12" in res.text
    assert "+0.34" in res.text


def test_polarization_score_preserved(tmp_path):
    res = sanitize_prompt(
        "Polarization rose to 7.5 by round 3",
        call_site="t",
        mode=BYODMode.STRICT,
        audit_path=tmp_path / "a.jsonl",
    )
    assert "7.5" in res.text


# ── Mode: BLOCK (raise on detection) ─────────────────────────────────────────


def test_mode_block_raises_on_leak(tmp_path):
    with pytest.raises(BYODLeakError) as exc:
        sanitize_prompt(
            "LCR is 95%",
            call_site="should_block",
            mode=BYODMode.BLOCK,
            audit_path=tmp_path / "a.jsonl",
        )
    assert exc.value.call_site == "should_block"
    assert any(d["category"] == "financial_metric" for d in exc.value.detections)


def test_mode_block_passes_safe_content(tmp_path):
    res = sanitize_prompt(
        "Polarization rose to 7.5 by round 3",
        call_site="t",
        mode=BYODMode.BLOCK,
        audit_path=tmp_path / "a.jsonl",
    )
    assert res.text == "Polarization rose to 7.5 by round 3"
    assert res.detections == []


# ── Env var resolution ──────────────────────────────────────────────────────


def test_get_mode_from_env(monkeypatch):
    from core.byod.sanitizer import get_mode
    monkeypatch.setenv("BYOD_MODE", "STRICT")
    assert get_mode() == BYODMode.STRICT
    monkeypatch.setenv("BYOD_MODE", "log")  # case insensitive
    assert get_mode() == BYODMode.LOG
    monkeypatch.setenv("BYOD_MODE", "garbage")  # unknown → OFF
    assert get_mode() == BYODMode.OFF
    monkeypatch.delenv("BYOD_MODE", raising=False)
    assert get_mode() == BYODMode.OFF


# ── Audit log shape ─────────────────────────────────────────────────────────


def test_audit_log_jsonl_well_formed(tmp_path):
    audit = tmp_path / "a.jsonl"
    sanitize_prompt(
        "LCR 95% deposit €1,200,000 client-9999",
        call_site="multi",
        mode=BYODMode.STRICT,
        audit_path=audit,
        tenant="sella-prod",
    )
    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) >= {"ts", "site", "mode", "raw_chars", "sanitized_chars", "patterns", "tenant"}
    assert row["site"] == "multi"
    assert row["mode"] == "STRICT"
    assert row["tenant"] == "sella-prod"
    assert isinstance(row["patterns"], list)
    assert len(row["patterns"]) >= 2  # currency + financial_metric + client_id


def test_audit_summary_aggregates(tmp_path):
    audit = tmp_path / "a.jsonl"
    sanitize_prompt(
        "deposit €1,200,000",
        call_site="briefing/x.py:10",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    sanitize_prompt(
        "client-12345 has CET1 of 11%",
        call_site="briefing/x.py:10",
        mode=BYODMode.STRICT,
        audit_path=audit,
    )
    summary = audit_summary(audit)
    assert summary["n_rows"] == 2
    assert summary["by_site"]["briefing/x.py:10"] >= 3
    assert summary["by_category"].get("currency", 0) >= 1
    assert summary["by_category"].get("client_id", 0) >= 1
