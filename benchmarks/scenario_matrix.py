"""Scenario diversity matrix for stress testing.

Enumerates scenarios across independent axes (domain × geography × tension)
so regressions on a single axis don't get hidden by an averaged metric. The
matrix is used by the benchmark runner to guarantee every axis gets at least
one sample, and by the stress tests to fail loudly when one bucket empties.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


DOMAINS = [
    "financial", "commercial", "corporate", "political",
    "public_health", "environmental", "labor",
]
REGIONS = ["EU", "US", "APAC", "LATAM", "GLOBAL"]
TENSION_LEVELS = ["low", "moderate", "high", "critical"]


@dataclass(frozen=True)
class ScenarioCell:
    domain: str
    region: str
    tension: str

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.region}/{self.tension}"


def full_matrix() -> list[ScenarioCell]:
    return [
        ScenarioCell(d, r, t)
        for d, r, t in itertools.product(DOMAINS, REGIONS, TENSION_LEVELS)
    ]


def coverage_report(cells: list[ScenarioCell]) -> dict:
    """Given a sample of cells (e.g. from evaluated scenarios), report
    which axis values are missing."""
    observed = {
        "domain": {c.domain for c in cells},
        "region": {c.region for c in cells},
        "tension": {c.tension for c in cells},
    }
    expected = {
        "domain": set(DOMAINS),
        "region": set(REGIONS),
        "tension": set(TENSION_LEVELS),
    }
    missing = {k: sorted(expected[k] - observed[k]) for k in expected}
    return {
        "cells": len(cells),
        "unique_cells": len({c.key for c in cells}),
        "missing": missing,
        "complete": all(not m for m in missing.values()),
    }
