"""DORA major-incident classification helpers.

Translates DigitalTwinSim's quantitative crisis metrics into the
qualitative levels the DORA RTS Annex I requires for the 7 classification
criteria. The translation is *deterministic and explicit* so a CRO can
audit how a given simulator output mapped to "high" vs "critical" on
each axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema import ClassificationCriteria


def _band_clients(impacted: int) -> str:
    """RTS Annex I — clients-affected criterion."""
    if impacted < 100:
        return "low"
    if impacted < 10_000:
        return "medium"
    if impacted < 1_000_000:
        return "high"
    return "critical"


def _band_economic(eur: float) -> tuple[str, str]:
    """Returns (qualitative_level, eur_band) per DORA RTS thresholds."""
    if eur < 100_000:
        return "low", "<100k"
    if eur < 1_000_000:
        return "medium", "100k-1m"
    if eur < 10_000_000:
        return "high", "1m-10m"
    return "critical", ">10m"


def _band_geographic(countries: int) -> str:
    if countries <= 1:
        return "local"
    if countries <= 3:
        return "national"
    if countries <= 10:
        return "cross_border"
    return "eu_wide"


def _band_reputational(polarization: float, viral_count: int) -> str:
    """Heuristic on simulator outputs:
    polarization >= 7.0 OR viral_posts >= 10 → high; >= 8.5 OR >= 25 → critical.
    """
    if polarization >= 8.5 or viral_count >= 25:
        return "critical"
    if polarization >= 7.0 or viral_count >= 10:
        return "high"
    if polarization >= 5.0 or viral_count >= 3:
        return "medium"
    return "low"


def _band_data(records_lost: int) -> str:
    if records_lost < 100:
        return "low"
    if records_lost < 10_000:
        return "medium"
    if records_lost < 1_000_000:
        return "high"
    return "critical"


def _band_criticality(affected_core_functions: int) -> str:
    if affected_core_functions == 0:
        return "non_critical"
    if affected_core_functions <= 2:
        return "critical"
    return "vital"


def classify_from_simulation(
    *,
    customers_affected: int = 0,
    economic_impact_eur: float = 0.0,
    countries_affected: int = 1,
    polarization_peak: float = 0.0,
    viral_posts_count: int = 0,
    data_records_lost: int = 0,
    affected_core_functions: int = 0,
    downtime_hours: float = 0.0,
) -> ClassificationCriteria:
    """Map raw simulator metrics → DORA 7-criterion classification.

    All arguments are kwargs so the call site can supply only the
    metrics it has — missing fields default to "no impact".
    """
    _, econ_band = _band_economic(economic_impact_eur)
    return ClassificationCriteria(
        clients_affected=_band_clients(customers_affected),
        data_losses=_band_data(data_records_lost),
        reputational_impact=_band_reputational(polarization_peak, viral_posts_count),
        duration_downtime_hours=float(downtime_hours),
        geographical_spread=_band_geographic(countries_affected),
        economic_impact_eur_band=econ_band,
        criticality_of_services_affected=_band_criticality(affected_core_functions),
    )
