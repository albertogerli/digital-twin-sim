"""Market data providers — pluggable sources for the MarketContext.

The legacy `UniverseLoader` was a module-level singleton that read
`shared/stock_universe.json` on first access. That coupled every scenario to
the same Italian calibration and made live-data providers impossible to swap
in.

This module introduces `MarketDataProvider` as the minimal protocol every
data source must satisfy. The initial implementation, `StaticUniverseProvider`,
preserves today's behaviour by reading the same JSON file — but multiple
scenarios can now instantiate their own provider (or share one per process)
without going through a singleton. A future `YFinanceProvider` /
`BloombergProvider` can implement the same protocol without touching scorer
code.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_UNIVERSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "shared",
    "stock_universe.json",
)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Minimal interface every market data source must satisfy.

    The scorer only reads — providers never mutate across method calls.
    """

    def sectors(self) -> dict[str, dict]:
        """Return the sector definitions (key → {label, gics_code, betas}).

        `betas` must be a mapping of regime name → coefficient dict with keys
        `political_beta`, `spread_beta`, `crisis_alpha`, `volatility_multiplier`.
        Missing regimes must fall back to `default` at lookup time.
        """
        ...

    def stocks(self) -> list[dict]:
        """Return the stock universe as a list of stock records."""
        ...

    def indices(self) -> list[dict]:
        """Return the benchmark index entries."""
        ...

    def org_aliases(self) -> dict[str, list[str]]:
        """Return the org-name → ticker-list alias map."""
        ...

    def macro(self) -> dict[str, dict]:
        """Return the per-geography macro config (local index + sovereign).

        Keys are geography codes (e.g. "IT", "US", "EU", "default"). Each entry
        must contain `beta_regime`, `local_index`, and `sovereign` sub-blocks.
        """
        ...


class StaticUniverseProvider:
    """Reads the canonical `shared/stock_universe.json` once into memory.

    Not a singleton: tests and scenarios may construct their own instances
    pointing at different snapshots. The provider is safe to share across
    `MarketContext` instances within one process because it exposes only
    read-only accessors (the dict is never mutated post-init).
    """

    def __init__(self, path: str | None = None):
        self._path = path or _DEFAULT_UNIVERSE_PATH
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, "r") as f:
                self._data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Stock universe not found at {self._path}")
            self._data = {}
        except json.JSONDecodeError as e:
            logger.error(f"Stock universe at {self._path} is not valid JSON: {e}")
            self._data = {}

    # ── MarketDataProvider interface ──────────────────────────────────────

    def sectors(self) -> dict[str, dict]:
        return self._data.get("sectors", {})

    def stocks(self) -> list[dict]:
        return self._data.get("stocks", [])

    def indices(self) -> list[dict]:
        return self._data.get("indices", [])

    def org_aliases(self) -> dict[str, list[str]]:
        return self._data.get("org_aliases", {})

    def macro(self) -> dict[str, dict]:
        return self._data.get("macro", {})


_DEFAULT_PROVIDER: StaticUniverseProvider | None = None


def get_default_provider() -> StaticUniverseProvider:
    """Process-wide default provider reading the canonical JSON.

    This is an explicit, nameable shared instance — not a hidden singleton
    behind `__new__`. Callers who need a bespoke source (tests, live feeds)
    construct their own `StaticUniverseProvider` or implement the protocol.
    """
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = StaticUniverseProvider()
    return _DEFAULT_PROVIDER
