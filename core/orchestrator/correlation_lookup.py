"""Empirical cross-market correlation lookup.

Loads the precomputed Pearson correlation matrix at
``shared/correlation_matrix.json`` (built by
``scripts/compute_correlation_matrix.py`` over ~190 tickers, daily log
returns, 2018→present) and exposes two query functions used by the
financial impact scorer:

  ``top_correlated_globally(seed_tickers, k, ...)``
      Returns the k tickers with the highest absolute correlation to the
      *average* of the seed basket — used to expand a sector-impacted
      basket to globally connected names.

  ``derive_pair_trade(seed_tickers, k_short, k_long, ...)``
      Splits the top-correlated list into a short leg (positive
      correlation = will move in the same direction as the impacted
      basket → short to hedge) and a long leg (negative correlation =
      defensive offset). Replaces the hardcoded ``CRISIS_PAIR_TRADES``
      dict in ``financial_impact.py`` with empirical, data-driven legs.

The matrix is loaded once on first use and cached in module scope. If
the JSON is missing or corrupt, ``MATRIX_AVAILABLE`` is False and the
calling code is expected to fall back to its legacy heuristic and log
a warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MATRIX_PATH = REPO_ROOT / "shared" / "correlation_matrix.json"


@dataclass(frozen=True)
class TickerNeighbour:
    ticker: str
    sector: str
    country: str
    correlation: float

    @property
    def is_positive(self) -> bool:
        return self.correlation > 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "country": self.country,
            "correlation": round(self.correlation, 3),
        }


_matrix_cache: Optional[dict] = None
_meta_cache: Optional[dict[str, dict]] = None


def _load() -> tuple[Optional[dict], dict[str, dict]]:
    """Load the matrix + ticker metadata. Cached after the first call."""
    global _matrix_cache, _meta_cache
    if _matrix_cache is not None and _meta_cache is not None:
        return _matrix_cache, _meta_cache
    try:
        payload = json.loads(MATRIX_PATH.read_text())
    except FileNotFoundError:
        logger.warning(
            "correlation_matrix.json missing at %s — empirical lookup disabled",
            MATRIX_PATH,
        )
        _matrix_cache = {}
        _meta_cache = {}
        return _matrix_cache, _meta_cache
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load correlation matrix: %s", e)
        _matrix_cache = {}
        _meta_cache = {}
        return _matrix_cache, _meta_cache

    _matrix_cache = payload.get("matrix", {})
    # Pull in ticker metadata (sector, country) from the contagion graph
    # payload, which lives in frontend/public/data and is the canonical
    # node-level metadata source used by the UI as well.
    graph_path = REPO_ROOT / "frontend" / "public" / "data" / "contagion_graph.json"
    meta: dict[str, dict] = {}
    try:
        graph = json.loads(graph_path.read_text())
        for node in graph.get("nodes", []):
            meta[node["id"]] = {
                "sector": node.get("sector", "other"),
                "country": node.get("country", "NA"),
            }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    _meta_cache = meta
    logger.info(
        "Loaded correlation matrix: %d tickers, %d with metadata",
        len(_matrix_cache),
        len(_meta_cache),
    )
    return _matrix_cache, _meta_cache


def matrix_available() -> bool:
    """True if the precomputed matrix is loadable and non-empty."""
    matrix, _ = _load()
    return bool(matrix)


def average_row(tickers: list[str], matrix: dict) -> dict[str, float]:
    """Return the mean correlation row across the seed basket. Tickers
    not in the matrix are silently skipped; if none are present, returns
    an empty dict."""
    rows = [matrix[t] for t in tickers if t in matrix]
    if not rows:
        return {}
    out: dict[str, float] = {}
    other_tickers = set(rows[0].keys())
    for r in rows[1:]:
        other_tickers &= set(r.keys())
    seed_set = set(tickers)
    for t in other_tickers:
        if t in seed_set:
            continue
        vals = [r[t] for r in rows]
        out[t] = sum(vals) / len(vals)
    return out


def top_correlated_globally(
    seed_tickers: list[str],
    k: int = 10,
    min_corr: float = 0.30,
    exclude_countries: Optional[set[str]] = None,
    exclude_sectors: Optional[set[str]] = None,
) -> list[TickerNeighbour]:
    """Return up to k tickers with the highest absolute correlation to
    the average of the seed basket.

    Filters:
      ``min_corr``: drop entries with |r| below this floor.
      ``exclude_countries`` / ``exclude_sectors``: drop matches in these
      sets — useful to force geographic / sectoral diversification.
    """
    matrix, meta = _load()
    if not matrix or not seed_tickers:
        return []
    row = average_row(seed_tickers, matrix)
    if not row:
        return []
    excluded_c = exclude_countries or set()
    excluded_s = exclude_sectors or set()
    candidates: list[TickerNeighbour] = []
    for t, r in row.items():
        if abs(r) < min_corr:
            continue
        m = meta.get(t, {})
        if m.get("country") in excluded_c:
            continue
        if m.get("sector") in excluded_s:
            continue
        candidates.append(
            TickerNeighbour(
                ticker=t,
                sector=m.get("sector", "other"),
                country=m.get("country", "NA"),
                correlation=float(r),
            )
        )
    candidates.sort(key=lambda x: abs(x.correlation), reverse=True)
    return candidates[:k]


def derive_pair_trade(
    seed_tickers: list[str],
    k_short: int = 4,
    k_long: int = 4,
    min_corr: float = 0.30,
) -> dict[str, list[TickerNeighbour]]:
    """Split global neighbours into short (positively correlated, will
    move with the impacted basket → short to hedge / capture downside)
    and long (negatively correlated, defensive offset).

    Returns ``{"short": [...], "long": [...], "source": "empirical"}``.
    Empty legs are returned when the matrix is unavailable or yields no
    matches.
    """
    matrix, meta = _load()
    if not matrix or not seed_tickers:
        return {"short": [], "long": [], "source": "unavailable"}
    row = average_row(seed_tickers, matrix)
    if not row:
        return {"short": [], "long": [], "source": "unavailable"}

    pos: list[TickerNeighbour] = []
    neg: list[TickerNeighbour] = []
    for t, r in row.items():
        if abs(r) < min_corr:
            continue
        m = meta.get(t, {})
        nb = TickerNeighbour(
            ticker=t,
            sector=m.get("sector", "other"),
            country=m.get("country", "NA"),
            correlation=float(r),
        )
        (pos if r > 0 else neg).append(nb)
    pos.sort(key=lambda x: x.correlation, reverse=True)
    neg.sort(key=lambda x: x.correlation)  # most negative first

    return {
        "short": pos[:k_short],
        "long": neg[:k_long],
        "source": "empirical",
    }
