"""Heuristic mapping: agent archetype / role → financial exposure flags.

Used by the closed-loop coupling (Sprint 2) to weight opinion aggregates.
A citizen cluster of "young homeowners" is far more sensitive to mortgage-
rate news than an academic economist, even if both have the same opinion
position. The financial twin should reflect that.

This is *deterministic* exposure inference from public role; it doesn't
make assumptions about specific people's personal finances. For citizen
clusters, exposure is inferred from the cluster's demographic description.

Tested via core.financial.tests.test_exposure (pending).
"""

from __future__ import annotations


# ── Exposure profile ────────────────────────────────────────────────────────

def default_exposure() -> dict:
    """Zero-exposure profile. Returned for archetypes we can't map."""
    return {
        "is_depositor": 0.0,        # weight 0..1: how much this agent's opinion
                                     # should drive the depositor-runoff signal
        "is_borrower": 0.0,         # weight on borrower-demand signal
        "is_competitor": 0.0,       # 1.0 = a peer bank that benefits if we lose
        "is_regulator": 0.0,        # 1.0 = central bank / supervisor
        "is_industry_lobby": 0.0,   # 1.0 = ABI / industry association
        "is_consumer_voice": 0.0,   # 1.0 = Codacons / Altroconsumo / consumer mag
        "is_journalist": 0.0,       # 1.0 = financial press
        "is_government": 0.0,       # 1.0 = MEF / political officeholder on finance
    }


# ── Mapping rules (Italian commercial banking domain) ──────────────────────

_ARCHETYPE_RULES: dict[str, dict] = {
    "consumer_advocate":      {"is_consumer_voice": 1.0, "is_depositor": 0.4, "is_borrower": 0.4},
    "industry_competitor":    {"is_competitor": 1.0, "is_depositor": 0.6},
    "industry_lobby":         {"is_industry_lobby": 1.0, "is_depositor": 0.5},
    "regulator":              {"is_regulator": 1.0},
    "journalist":             {"is_journalist": 1.0, "is_depositor": 0.5, "is_borrower": 0.3},
    "academic":               {"is_journalist": 0.4, "is_depositor": 0.4},
    "executive":              {"is_competitor": 0.3, "is_depositor": 0.6},  # peer-bank CEOs
    "government":             {"is_government": 1.0, "is_depositor": 0.3},
    "central_bank":           {"is_regulator": 1.0},
    # Citizen tiers — full retail exposure
    "citizen":                {"is_depositor": 1.0, "is_borrower": 0.7},
    "retail":                 {"is_depositor": 1.0, "is_borrower": 0.7},
    "sme":                    {"is_borrower": 1.0, "is_depositor": 0.5},
}


# Role substring fallback when archetype is unknown / generic.
_ROLE_KEYWORDS: list[tuple[str, dict]] = [
    ("Codacons",                {"is_consumer_voice": 1.0, "is_depositor": 0.5}),
    ("Altroconsumo",            {"is_consumer_voice": 1.0, "is_depositor": 0.5}),
    ("Federconsumatori",        {"is_consumer_voice": 1.0, "is_depositor": 0.5}),
    ("ABI",                     {"is_industry_lobby": 1.0}),
    ("Banca d'Italia",          {"is_regulator": 1.0}),
    ("Bankitalia",              {"is_regulator": 1.0}),
    ("Consob",                  {"is_regulator": 1.0}),
    ("IVASS",                   {"is_regulator": 1.0}),
    ("Findomestic",             {"is_competitor": 1.0}),
    ("Agos",                    {"is_competitor": 1.0}),
    ("Compass",                 {"is_competitor": 1.0}),
    ("Intesa",                  {"is_competitor": 1.0}),
    ("UniCredit",               {"is_competitor": 1.0}),
    ("Mediobanca",              {"is_competitor": 0.7}),
    ("BPM",                     {"is_competitor": 1.0}),
    ("BPER",                    {"is_competitor": 1.0}),
    ("Credit Agricole",         {"is_competitor": 1.0}),
    ("BNP",                     {"is_competitor": 0.7}),
    ("Ministero",               {"is_government": 1.0}),
    ("MEF",                     {"is_government": 1.0}),
    ("Corriere",                {"is_journalist": 1.0}),
    ("Sole 24",                 {"is_journalist": 1.0}),
    ("Repubblica",              {"is_journalist": 0.7}),
    ("PMI",                     {"is_borrower": 1.0, "is_depositor": 0.5}),
    ("Imprenditore",            {"is_borrower": 0.7, "is_depositor": 0.5}),
]


def infer_financial_exposure(
    archetype: str = "",
    role: str = "",
    party_or_org: str = "",
) -> dict:
    """Return exposure-flag dict for an agent based on public-role hints.

    Resolution order:
      1. archetype direct match (canonical taxonomy)
      2. keyword match on role + party_or_org (free text)
      3. default_exposure() (all zero)

    Multiple matches are merged via max() per key (most-conservative high
    weight wins).
    """
    out = default_exposure()

    # Layer 1: archetype
    arch_clean = (archetype or "").strip().lower()
    rule = _ARCHETYPE_RULES.get(arch_clean)
    if rule:
        for k, v in rule.items():
            out[k] = max(out.get(k, 0.0), v)

    # Layer 2: keyword match on role + party
    haystack = f"{role or ''} {party_or_org or ''}".lower()
    for kw, partial in _ROLE_KEYWORDS:
        if kw.lower() in haystack:
            for k, v in partial.items():
                out[k] = max(out.get(k, 0.0), v)

    return out


def aggregate_opinion_by_exposure(
    agents_with_positions: list[dict],
) -> dict:
    """Aggregate opinion across agents, weighted by exposure flags.

    Each `agents_with_positions` entry must have keys:
      - position: float ∈ [-1, +1]
      - exposure: dict from infer_financial_exposure()
    Optional: `weight` (defaults to 1.0) for tier weighting.

    Returns dict consumable by FinancialTwin.step(opinion_by_exposure=...):
      - depositors_negative: weighted mean of max(0, -position) over depositors
      - borrowers_negative: same, over borrowers
      - competitors_negative_to_us: weighted mean of max(0, -position)
        over competitor agents (positive value = competitors expressing
        anti-our-bank sentiment, which translates into market-share grab)

    All outputs in [0, 1]. Returns zeros if no agent has the relevant flag.
    """
    if not agents_with_positions:
        return {"depositors_negative": 0.0, "borrowers_negative": 0.0,
                "competitors_negative_to_us": 0.0}

    def _weighted(flag_key: str) -> float:
        num, den = 0.0, 0.0
        for a in agents_with_positions:
            ex = a.get("exposure") or {}
            w = float(ex.get(flag_key, 0.0)) * float(a.get("weight", 1.0))
            if w <= 0:
                continue
            neg = max(0.0, -float(a.get("position", 0.0)))
            num += w * neg
            den += w
        return round(num / den, 4) if den > 0 else 0.0

    return {
        "depositors_negative": _weighted("is_depositor"),
        "borrowers_negative": _weighted("is_borrower"),
        "competitors_negative_to_us": _weighted("is_competitor"),
    }
