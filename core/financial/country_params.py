"""Per-country default parameters for FinancialTwin (euro area).

Each country function returns a parameter dict that overrides the
literature-based generic defaults in twin.default_italian_bank_params().
Differences capture cross-country heterogeneity in:

- deposit β (passthrough): driven by banking-sector concentration,
  retail mobility, regulated savings products (Livret A in FR), branch
  density (Sparkassen in DE)
- CET1 baseline: structural capital intensity by jurisdiction
- LCR baseline: liquidity profile (DE/NL hold more HQLA per €deposit)
- consumer loan elasticity: not strongly country-specific in EU; we keep
  the IT median (-1.7) as a conservative anchor for all
- mortgage var/fix mix: historical preference (e.g. NL ~95% fixed,
  ES ~70% variable, IT ~50/50 stock)

Numbers are illustrative defaults grounded in published 2024-2025
ECB / EBA Risk Dashboard / national supervisor data. Override per
specific institution at scenario-config time.

Country dispatch is automatic via select_country_params(geography_codes):
the first non-supranational ISO code in scope.geography wins.
"""

from __future__ import annotations

from typing import Optional

from .twin import default_italian_bank_params


# ── IT (canonical reference) ───────────────────────────────────────────────

def default_italian_bank_params_v2() -> dict:
    """Italy — re-export of the canonical defaults (already in twin.py).
    Kept here so country dispatch is symmetric: every country has a
    function in this module."""
    return default_italian_bank_params()


# ── DE — Sparkassen / Landesbanken / commercial mix ────────────────────────

def default_german_bank_params() -> dict:
    """Germany — characterised by Sparkasse + Volksbanken networks holding
    ~40% of retail deposits with very low passthrough (deposit-rate stickiness).
    Mortgage market dominantly fixed-rate (Bauspar tradition). Higher CET1.
    """
    p = default_italian_bank_params()
    p.update({
        "deposit_beta_sight": 0.30,    # Sparkassen pull avg β down
        "deposit_beta_term": 0.65,
        "sight_share": 0.70,           # higher share of demand deposits
        "consumer_loan_elasticity": -1.5,  # Germany less rate-sensitive (savings culture)
        "mortgage_var_share_stock": 0.20,  # mostly fixed
        "mortgage_var_share_new": 0.10,
        "tier1_capital": 0.090,        # baseline CET1 ~16.5% (>IT)
        "rwa_density": 0.50,
        "btp_bund_spread_bps": 0,      # DE = Bund itself
        "policy_rate_pct": 2.40,       # ECB DFR
        # Stress responses tighter (BaFin / Bundesbank tradition)
        "deposit_runoff_max_per_round": 0.035,
        "cet1_min_pct": 11.0,
        "cet1_alarm_pct": 12.5,
    })
    return p


# ── FR — concentrated, regulated savings, large mortgage book ──────────────

def default_french_bank_params() -> dict:
    """France — concentrated market (BNP / Credit Agricole / Société Générale /
    BPCE), Livret A passthrough is regulated by the state (decoupled from ECB),
    mortgage market dominantly fixed-rate. Slightly lower CET1 than DE/IT due
    to dense mortgage book on balance sheet."""
    p = default_italian_bank_params()
    p.update({
        "deposit_beta_sight": 0.40,    # Livret A regulation dampens
        "deposit_beta_term": 0.70,
        "sight_share": 0.60,
        "consumer_loan_elasticity": -1.6,
        "mortgage_var_share_stock": 0.15,  # France ~85% fixed
        "mortgage_var_share_new": 0.05,
        "tier1_capital": 0.084,        # baseline CET1 ~15.8%
        "rwa_density": 0.52,
        "btp_bund_spread_bps": 25,     # OAT-Bund spread, narrow
        "policy_rate_pct": 2.40,
        "loan_repricing_speed": 0.30,  # slower repricing because fixed-rate stock
    })
    return p


# ── ES — competitive retail, variable-rate mortgages, post-2012 deleveraging ──

def default_spanish_bank_params() -> dict:
    """Spain — characterised by high retail deposit competition, mortgage book
    dominantly variable-rate (referenced to Euribor), and structurally lower
    CET1 (legacy of post-2012 cleanup). Higher LCR (Cajas legacy)."""
    p = default_italian_bank_params()
    p.update({
        "deposit_beta_sight": 0.55,    # competitive, fast passthrough
        "deposit_beta_term": 0.80,
        "sight_share": 0.65,
        "consumer_loan_elasticity": -2.0,  # Spain very rate-sensitive
        "mortgage_var_share_stock": 0.70,  # historically variable
        "mortgage_var_share_new": 0.50,    # more fixed post-2022
        "tier1_capital": 0.072,        # baseline CET1 ~13.5%
        "rwa_density": 0.55,
        "hqla_balance": 0.095,         # higher liquidity buffer
        "btp_bund_spread_bps": 80,     # Bono-Bund spread
        "policy_rate_pct": 2.40,
        "loan_repricing_speed": 0.55,  # fast repricing due to variable mortgage
    })
    return p


# ── NL — concentrated, fixed mortgages, conservative ──────────────────────

def default_dutch_bank_params() -> dict:
    """Netherlands — highly concentrated (ING / Rabobank / ABN AMRO),
    near-100% fixed-rate mortgages (10y/20y common), structurally high CET1.
    Conservative liquidity profile."""
    p = default_italian_bank_params()
    p.update({
        "deposit_beta_sight": 0.50,
        "deposit_beta_term": 0.75,
        "sight_share": 0.65,
        "consumer_loan_elasticity": -1.5,
        "mortgage_var_share_stock": 0.05,  # NL is essentially all fixed
        "mortgage_var_share_new": 0.02,
        "tier1_capital": 0.092,        # baseline CET1 ~17%
        "rwa_density": 0.50,
        "btp_bund_spread_bps": 15,     # DSL-Bund, very narrow
        "policy_rate_pct": 2.40,
        "loan_repricing_speed": 0.20,  # very slow due to fixed-rate stock
        "duration_gap_yrs": 2.2,       # longer duration mortgage assets
    })
    return p


# ── US — diversified, fee-driven, mortgage securitised, FDIC insured ──────

def default_us_bank_params() -> dict:
    """United States — characterised by:
    - mortgages largely securitised (off balance sheet via Fannie/Freddie),
      so on-bs duration gap is shorter than EU
    - more fee/non-interest income → NIM less sensitive
    - deposit beta historically higher (more competitive market for term)
    - LCR similar (~120-140% post-Basel III)
    - CET1 lower than EU (~13.5% avg)
    """
    p = default_italian_bank_params()
    p.update({
        "deposit_beta_sight": 0.40,    # NIB / sweep accounts hold β down
        "deposit_beta_term": 0.85,     # CDs and brokered deposits very competitive
        "sight_share": 0.55,           # smaller share of pure sight in US mix
        "consumer_loan_elasticity": -1.4,  # less elastic, credit-card-driven
        "mortgage_var_share_stock": 0.10,  # 30y fixed is dominant
        "mortgage_var_share_new": 0.05,
        "tier1_capital": 0.068,        # baseline CET1 ~13.5%
        "rwa_density": 0.50,
        "hqla_balance": 0.075,         # baseline LCR ~135-150%
        "btp_bund_spread_bps": 0,      # n/a (overwritten by FRED feed)
        "policy_rate_pct": 4.00,       # Fed funds, will refresh from FRED
        "duration_gap_yrs": 1.2,       # shorter than EU due to securitisation
        "loan_repricing_speed": 0.50,  # variable-rate consumer loans common
        "cet1_min_pct": 10.5,          # US SREP equivalent
        "cet1_alarm_pct": 11.5,
        "lcr_min_pct": 100.0,
    })
    return p


# ── UK — concentrated retail + structured savings, ring-fencing rules ─────

def default_uk_bank_params() -> dict:
    """United Kingdom — characterised by:
    - Big-four concentration (HSBC / Barclays / NatWest / Lloyds)
    - ring-fencing post-2019: retail subsidiaries hold higher capital
    - mortgage market mostly variable / 2-5y fixed reset (faster repricing
      than EU continent)
    - PRA stress tests align with EBA but more conservative
    """
    p = default_italian_bank_params()
    p.update({
        "deposit_beta_sight": 0.35,    # Big-four oligopoly dampens passthrough
        "deposit_beta_term": 0.75,
        "sight_share": 0.65,
        "consumer_loan_elasticity": -1.6,
        "mortgage_var_share_stock": 0.40,  # 2-5y fixed common but resets fast
        "mortgage_var_share_new": 0.30,
        "tier1_capital": 0.085,        # baseline CET1 ~16% (PRA buffer)
        "rwa_density": 0.50,
        "hqla_balance": 0.080,
        "btp_bund_spread_bps": 0,      # n/a
        "policy_rate_pct": 4.25,       # BoE Bank Rate, will refresh from BoE
        "duration_gap_yrs": 1.4,
        "loan_repricing_speed": 0.45,
        "cet1_min_pct": 11.0,
        "cet1_alarm_pct": 12.5,
    })
    return p


# ── Country dispatch ───────────────────────────────────────────────────────

_COUNTRY_DISPATCH = {
    "IT": default_italian_bank_params_v2,
    "DE": default_german_bank_params,
    "FR": default_french_bank_params,
    "ES": default_spanish_bank_params,
    "NL": default_dutch_bank_params,
    "US": default_us_bank_params,
    "GB": default_uk_bank_params,
    "UK": default_uk_bank_params,  # alias
}

_SUPRANATIONAL = {"EU", "EUROZONE", "GLOBAL", "WORLD"}


def supported_countries() -> list[str]:
    """ISO codes of supported per-country default sets."""
    return sorted(_COUNTRY_DISPATCH.keys())


def select_country_params(
    geography_codes: list[str],
    fallback: str = "IT",
) -> tuple[str, dict]:
    """Select per-country params from a list of ISO geography codes.

    Args:
        geography_codes: e.g. ["DE"], ["IT", "EU"], ["FR", "DE"], ["EU"]
            Supranational codes are skipped; first matching ISO wins.
        fallback: ISO code to use if no match (default "IT" — keeps the
            current behaviour of the codebase).

    Returns:
        (chosen_country_code, params_dict)
    """
    for code in (geography_codes or []):
        c = (code or "").strip().upper()
        if c in _SUPRANATIONAL:
            continue
        if c in _COUNTRY_DISPATCH:
            return c, _COUNTRY_DISPATCH[c]()
    # Fallback
    return fallback, _COUNTRY_DISPATCH.get(fallback, default_italian_bank_params)()
