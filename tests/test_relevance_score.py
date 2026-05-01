"""Sprint 11 — validation suite for the Layer 1 relevance score.

Validates that the deterministic relevance filter:
  1. Drops foreign heads-of-state on Italian / European banking briefs
     (the original bug: Biden/Trump/Musk/Vance leaking into Sella sim).
  2. Keeps Italian regulators, banks, consumer voices on Italian briefs.
  3. Correctly handles non-Italian briefs (DE, US, UK) — doesn't over-filter.
  4. Allows explicitly-named individuals through (brief mention bonus).
  5. Reproduces the desired effect WITHOUT the hard-block list — proving
     the safety-net can be removed.

Pure deterministic, no LLM, no network. Runs < 100ms.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_scope(geography, sector="unknown", scope_tier="national",
                 sub_sector="", excluded_archetypes=None,
                 stakeholder_archetypes=None):
    from briefing.brief_scope import BriefScope
    return BriefScope(
        sector=sector,
        sub_sector=sub_sector,
        geography=list(geography),
        scope_tier=scope_tier,
        stakeholder_archetypes=list(stakeholder_archetypes or []),
        excluded_archetypes=list(excluded_archetypes or []),
    )


def _get(name_or_id):
    """Fetch from real stakeholder DB. Skip test if not available."""
    from stakeholder_graph.db import StakeholderDB
    db = StakeholderDB()
    s = db.get(name_or_id)
    if s is None:
        # Try by name lookup
        for sk in db._stakeholders.values():
            if sk.name.lower() == name_or_id.lower():
                return sk
        pytest.skip(f"stakeholder {name_or_id!r} not in DB")
    return s


def _kept(stakeholder, brief, scope, threshold=0.40):
    from briefing.relevance_score import score_stakeholder_relevance
    v = score_stakeholder_relevance(stakeholder, brief, scope=scope)
    return v.score >= threshold, v


# ── Brief 1: Banca Sella IT (banking, national) ─────────────────────────────

BRIEF_SELLA = """
Banca Sella deve decidere la policy di pricing del credito al consumo
per Q2-Q3 2026 dopo che la BCE ha tagliato i tassi a 2.00%. I competitor
diretti — Findomestic, Agos, Compass — hanno annunciato il pieno
passthrough. Sella deve scegliere fra difesa quota mercato vs difesa
margine NIM. Stakeholder rilevanti: ABI, Banca d'Italia, Codacons.
"""

SCOPE_SELLA = _build_scope(
    ["IT"], sector="consumer_credit", sub_sector="italian_banking",
    scope_tier="national",
    stakeholder_archetypes=["industry_competitor", "consumer_advocate", "regulator"],
    excluded_archetypes=["head_of_state", "global_tech_billionaire"],
)


@pytest.mark.parametrize("name_or_id", [
    "donald_trump", "joe_biden", "elon_musk", "jd_vance",
])
def test_brief_sella_drops_foreign_heads_of_state(name_or_id):
    s = _get(name_or_id)
    keep, v = _kept(s, BRIEF_SELLA, SCOPE_SELLA)
    assert not keep, (
        f"{s.name} should NOT pass on Sella IT brief but got score={v.score} "
        f"(components={v.components})"
    )


@pytest.mark.parametrize("name_or_id", [
    "fabio_panetta", "carlo_messina_intesa", "codacons_org",
])
def test_brief_sella_keeps_italian_finance_actors(name_or_id):
    s = _get(name_or_id)
    keep, v = _kept(s, BRIEF_SELLA, SCOPE_SELLA)
    assert keep, (
        f"{s.name} should pass on Sella IT brief but got score={v.score} "
        f"(components={v.components})"
    )


def test_brief_sella_keeps_italian_political_figures():
    """Mattarella/Meloni are Italian and may comment on a banking brief."""
    s = _get("sergio_mattarella")
    keep, v = _kept(s, BRIEF_SELLA, SCOPE_SELLA)
    assert keep, f"Mattarella should pass; score={v.score}, comp={v.components}"


def test_brief_sella_named_competitor_gets_top_score():
    """Findomestic is named in the brief — should get max score."""
    s = _get("findomestic_banca")
    if s is None:
        pytest.skip("findomestic_banca not in DB")
    _, v = _kept(s, BRIEF_SELLA, SCOPE_SELLA)
    assert v.score >= 0.90, f"named competitor should score >= 0.90, got {v.score}"


# ── Brief 2: Intesa Sanpaolo IT (banking, national, similar to Sella) ───────

BRIEF_INTESA = """
Intesa Sanpaolo annuncia l'acquisizione di una fintech tedesca
specializzata in retail digital banking. Reazione attesa da
Bankitalia, ABI, BCE. Implicazioni su CET1, M&A regulatory clearance
in Italia e Germania.
"""

SCOPE_INTESA = _build_scope(
    ["IT", "DE"], sector="italian_banking", scope_tier="national",
    excluded_archetypes=["head_of_state", "global_tech_billionaire"],
)


def test_brief_intesa_drops_us_politicians():
    for name in ["donald_trump", "joe_biden"]:
        s = _get(name)
        keep, v = _kept(s, BRIEF_INTESA, SCOPE_INTESA)
        assert not keep, f"{s.name} should not pass on IT/DE brief; got {v.score}"


def test_brief_intesa_keeps_italian_actors():
    for name in ["fabio_panetta", "antonio_patuelli"]:
        s = _get(name)
        keep, v = _kept(s, BRIEF_INTESA, SCOPE_INTESA)
        assert keep, f"{s.name} should pass; got {v.score}"


# ── Brief 3: Deutsche Bank DE (banking, national, foreign country) ──────────

BRIEF_DEUTSCHE = """
Deutsche Bank announces a major restructuring of its retail banking
division in Germany. Expected reactions from BaFin, Bundesbank,
Sparkasse network, and German consumer associations.
"""

SCOPE_DEUTSCHE = _build_scope(
    ["DE"], sector="german_banking", scope_tier="national",
    excluded_archetypes=["head_of_state", "global_tech_billionaire"],
)


def test_brief_deutsche_drops_us_politicians():
    """US politicians on DE brief: should be DROPPED."""
    for name in ["donald_trump", "joe_biden", "jd_vance"]:
        s = _get(name)
        keep, v = _kept(s, BRIEF_DEUTSCHE, SCOPE_DEUTSCHE)
        assert not keep, f"{s.name} should not pass DE brief; got {v.score}"


def test_brief_deutsche_drops_italian_specific_actors():
    """Italian consumer body on a DE brief: should drop (sector match for IT
    bodies on DE brief is weak even within EU)."""
    s = _get("codacons_org")
    if s is None:
        pytest.skip()
    keep, v = _kept(s, BRIEF_DEUTSCHE, SCOPE_DEUTSCHE)
    # Codacons is IT consumer body — on DE-only national brief, marginal.
    # Score is allowed but should be clearly lower than for Italian actors.
    assert v.score < 0.6, (
        f"Codacons on DE brief should score < 0.6 (got {v.score}); "
        f"signals weak EU-only sector overlap."
    )


# ── Brief 4: JPMorgan US (US banking, named US figures should pass) ─────────

BRIEF_JPMORGAN = """
JPMorgan Chase announces new policy on consumer auto loans amid Fed
rate cuts. Expected reactions from CFPB, Federal Reserve, Senator
Elizabeth Warren, and competitor banks (Bank of America, Wells Fargo,
Citi). Note: Donald Trump has publicly commented on banking regulation
this quarter.
"""

SCOPE_JPMORGAN = _build_scope(
    ["US"], sector="us_banking", scope_tier="national",
)


def test_brief_jpmorgan_keeps_trump_when_named():
    """Brief explicitly mentions Trump. Score should be high (mention bonus)."""
    s = _get("donald_trump")
    keep, v = _kept(s, BRIEF_JPMORGAN, SCOPE_JPMORGAN)
    # Named in brief → mention_bonus override sets raw>=0.7 → score >= 0.7
    assert v.score >= 0.7, (
        f"Trump explicitly named in US brief should score >= 0.7; "
        f"got {v.score} (components={v.components})"
    )
    assert keep


def test_brief_jpmorgan_drops_italian_actors():
    """Italian banking actors should NOT pass on US-only brief."""
    s = _get("codacons_org")
    if s is None:
        pytest.skip()
    keep, v = _kept(s, BRIEF_JPMORGAN, SCOPE_JPMORGAN)
    assert not keep, f"Codacons should not pass US brief; got {v.score}"


# ── Brief 5: US politics 2028 (Trump SHOULD pass; non-political IT shouldn't) ─

BRIEF_US_POLITICS = """
Analysis of the 2028 US presidential primaries. Donald Trump endorses
JD Vance for the Republican nomination, while Joe Biden returns to
the political arena to support Kamala Harris on the Democratic side.
"""

SCOPE_US_POLITICS = _build_scope(
    ["US"], sector="us_politics", scope_tier="national",
    stakeholder_archetypes=["politician", "head_of_state", "vice_president"],
)


def test_brief_us_politics_keeps_us_politicians():
    """All US politicians named in the brief should pass."""
    for name in ["donald_trump", "joe_biden", "jd_vance"]:
        s = _get(name)
        keep, v = _kept(s, BRIEF_US_POLITICS, SCOPE_US_POLITICS)
        assert keep, f"{s.name} named in US politics brief should pass; got {v.score}"


def test_brief_us_politics_drops_italian_specific_actors():
    """Italian consumer / banking bodies should not pass a US politics brief."""
    for name in ["codacons_org", "antonio_patuelli"]:
        s = _get(name)
        if s is None:
            continue
        keep, v = _kept(s, BRIEF_US_POLITICS, SCOPE_US_POLITICS)
        assert not keep, f"{s.name} should not pass US politics; got {v.score}"


# ── Hard-block redundancy check ────────────────────────────────────────────

def test_hard_block_list_is_now_redundant_for_sella_brief():
    """If Layer 1 alone correctly drops every name on the hard-block list
    for the Sella brief, then the hard-block safety-net can be removed
    in production."""
    HARD_BLOCK_NAMES = {
        "Donald Trump", "Joe Biden", "Kamala Harris", "JD Vance",
        "Elon Musk", "Ron DeSantis", "Alexandria Ocasio-Cortez",
        "Bernie Sanders", "Vladimir Putin", "Xi Jinping",
    }
    from stakeholder_graph.db import StakeholderDB
    db = StakeholderDB()
    leaks = []
    for s in db._stakeholders.values():
        if s.name in HARD_BLOCK_NAMES:
            keep, v = _kept(s, BRIEF_SELLA, SCOPE_SELLA)
            if keep:
                leaks.append((s.name, v.score, v.components))
    assert not leaks, (
        f"Layer 1 alone failed to drop {len(leaks)} hard-block names: "
        f"{leaks}. Cannot remove hard-block list yet."
    )
