"""BYOD prompt sanitizer.

Detects and (optionally) redacts customer-sensitive financial information
in any string about to be sent to an external LLM provider. The contract
the BYOD enclave guarantees:

  - Currency amounts (€2.4M, $500K, EUR 1,200, "2.5 milioni") are
    redacted to ``[currency-amount]`` (or to a magnitude bucket).
  - Financial-metric values (LCR/CET1/NIM/RWA/deposit-β with a number)
    are redacted to a categorical descriptor ("LCR healthy",
    "LCR breaching") preserving the threshold-relevant information
    without leaking the exact internal value.
  - Customer / account IDs are redacted to ``[client-id]``.
  - Bare large numbers (≥10,000) within ±20 chars of a financial
    keyword are redacted to ``[large-amount]``.
  - Named benchmarks with values (Euribor / BTP-Bund spread) are
    redacted to ``[Euribor at level]`` etc.

What we DO NOT touch:

  - Non-financial percentages (poll opinion, market share, win
    probability) — only redacted when adjacent to a financial keyword.
  - Agent positions in [-1, +1].
  - Polarization scores in [0, 10].
  - Dates, ISO years, generic numbers without financial context.
  - Public news text, agent personas of public figures, generic
    crisis context — the categorical FeedbackSignals injected by
    the FinancialTwin.

The four modes (``BYODMode``):

  ``OFF``     — passthrough, no checking. The default; the engine works
                exactly as before. Use in single-tenant trusted setups.
  ``LOG``     — detect patterns, write to audit log, do NOT modify the
                prompt. Use to measure the leak surface of a new
                deployment before flipping STRICT.
  ``STRICT``  — detect, redact, write to audit log. Production BYOD.
  ``BLOCK``   — detect, raise ``BYODLeakError`` instead of redacting.
                Fail-closed mode for paranoid deployments.

The mode is read from the ``BYOD_MODE`` environment variable
(case-insensitive, default ``OFF``).

Usage::

    from core.byod.sanitizer import sanitize_prompt, BYODMode

    result = sanitize_prompt(
        prompt="LCR is 95%, deposit balance €12,500,000, customer-12345",
        call_site="briefing/agent_generator.py:456",
    )
    # result.text == "LCR is [LCR breaching: below 100%], deposit balance
    #                 [currency-amount], [client-id]"
    # result.detections == [{"category": "financial_metric", ...}, ...]

Test coverage: see ``tests/test_byod_sanitizer.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUDIT_PATH = REPO_ROOT / "outputs" / "byod_audit.jsonl"


class BYODMode(str, Enum):
    OFF = "OFF"
    LOG = "LOG"
    STRICT = "STRICT"
    BLOCK = "BLOCK"


class BYODLeakError(RuntimeError):
    """Raised when BYOD_MODE=BLOCK and a sensitive pattern is detected."""

    def __init__(self, call_site: str, detections: list[dict]):
        cats = sorted({d["category"] for d in detections})
        super().__init__(
            f"BYOD leak detected at {call_site}: categories={cats} "
            f"(BYOD_MODE=BLOCK refuses to send the prompt). "
            f"Switch to STRICT to redact instead, or LOG to audit-only."
        )
        self.call_site = call_site
        self.detections = detections


@dataclass
class SanitizeResult:
    text: str
    detections: list[dict] = field(default_factory=list)
    modified: bool = False
    mode: str = BYODMode.OFF.value


# ── Detector patterns ────────────────────────────────────────────────────────
# Each detector is a (name, regex, replacement_fn) triple. The replacement
# function receives the regex match and returns the redacted string;
# the categorical replacement preserves the kind of information without
# leaking the precise value.

_FINANCIAL_KEYWORDS = (
    r"(?:LCR|CET1|NIM|RWA|liquidity\s*coverage|capital\s*ratio|"
    r"deposit[s]?\s*(?:β|beta|balance|outflow|run-?off)?|"
    r"net\s*interest\s*margin|"
    r"BTP[-/]Bund|spread|Euribor|tasso\s*BCE|"
    r"balance\s*sheet|loan[s]?|mortgage[s]?|bond[s]?|hedging|"
    r"depositi|raccolta|impieghi|prestiti|mutui)"
)


def _redact_currency(m: re.Match) -> str:
    return "[currency-amount]"


def _redact_financial_metric(m: re.Match) -> str:
    """Preserve the metric name + a healthy/breaching descriptor when
    we can compute a threshold; redact the exact number otherwise."""
    metric = m.group("metric").upper().replace(" ", "")
    raw = m.group("value")
    try:
        val = float(raw.replace(",", "."))
    except (ValueError, AttributeError):
        return f"[{metric} at level]"
    # Known thresholds (regulatory floors / EBA medians)
    if metric == "LCR":
        return f"[{metric} {'healthy: above 100%' if val >= 100 else 'breaching: below 100%'}]"
    if metric == "CET1":
        return f"[{metric} {'healthy: above 12%' if val >= 12 else 'tight: below 12%'}]"
    if metric == "NIM":
        return f"[{metric} {'above EBA median' if val >= 1.6 else 'below EBA median'}]"
    return f"[{metric} at level]"


def _redact_client_id(m: re.Match) -> str:
    return "[client-id]"


def _redact_iban(m: re.Match) -> str:
    return "[IBAN]"


def _redact_large_amount(m: re.Match) -> str:
    return "[large-amount]"


def _redact_bench_value(m: re.Match) -> str:
    name = m.group("bench")
    return f"[{name} at level]"


_DETECTORS: list[tuple[str, re.Pattern, callable]] = [
    # IBAN — IT60 X05428 11101 000000123456 (very specific, must come before generic)
    (
        "iban",
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9 ]{15,32}\b"),
        _redact_iban,
    ),
    # Currency amounts: €2.4M, $500K, EUR 1,200, "2.5 milioni di euro"
    (
        "currency",
        re.compile(
            r"(?:€|\$|£|EUR|USD|GBP|CHF)\s?\d[\d.,]*\s?(?:[kKmMbB]|mln|milioni|miliardi|thousand|million|billion)?"
            r"|"
            r"\b\d[\d.,]*\s?(?:euro|euros|dollari|dollars)\b"
            r"|"
            r"\b\d+([.,]\d+)?\s?(?:milioni|miliardi|million|billion)\s?(?:di\s)?(?:euro|dollari|USD|EUR)?",
            re.IGNORECASE,
        ),
        _redact_currency,
    ),
    # Financial metric with explicit value: "LCR 168%" / "CET1 14.2%" /
    # "NIM 1.85%" / "CET1 ratio of 14%" / "LCR is 95" / "LCR=95"
    (
        "financial_metric",
        re.compile(
            r"(?P<metric>LCR|CET1|NIM|RWA|deposit[s]?\s*β|deposit\s*beta)"
            r"[\s\w=:.\-]{0,15}?"  # short connectors: " ratio of ", "=", " is ", ": "
            r"(?P<value>\d+(?:[.,]\d+)?)\s*%?",
            re.IGNORECASE,
        ),
        _redact_financial_metric,
    ),
    # Named benchmark with bps/% level: "Euribor 3M at 2.4%", "BTP-Bund 180bps"
    (
        "benchmark_value",
        re.compile(
            r"(?P<bench>Euribor(?:\s*\d+M)?|BTP[-/]Bund\s*spread|BTP[-/]Bund|tasso\s*BCE)\s*"
            r"(?:at|=|:|of|al)?\s*\d+(?:[.,]\d+)?\s*(?:%|bps|punti)",
            re.IGNORECASE,
        ),
        _redact_bench_value,
    ),
    # Customer / account IDs: client-12345, customer ID xxx, account 9999
    (
        "client_id",
        re.compile(
            r"\b(?:client|customer|account|cliente|conto)[- _]?(?:id|ID|n°|num)?[- _:]?\s*[A-Z0-9-]{3,}",
            re.IGNORECASE,
        ),
        _redact_client_id,
    ),
    # Bare large numbers near financial keywords (within ±20 chars)
    # Captures e.g. "deposit balance 12,500,000", "RWA of 4.5 billion"
    # Placed last to avoid stealing matches from more-specific detectors.
    (
        "large_amount_in_context",
        re.compile(
            rf"(?:{_FINANCIAL_KEYWORDS})[\s\w,.:-]{{0,30}}?"
            rf"\b\d{{2,}}(?:[.,]\d{{3}})+(?:[.,]\d+)?\b",
            re.IGNORECASE,
        ),
        _redact_large_amount,
    ),
]


def get_mode() -> BYODMode:
    """Read BYOD_MODE from env; default OFF."""
    raw = os.environ.get("BYOD_MODE", "OFF").upper()
    try:
        return BYODMode(raw)
    except ValueError:
        return BYODMode.OFF


def sanitize_prompt(
    prompt: str,
    call_site: str,
    mode: Optional[BYODMode] = None,
    audit_path: Optional[Path] = None,
    tenant: Optional[str] = None,
) -> SanitizeResult:
    """Sanitize ``prompt`` according to ``mode`` (defaults to env var).

    Returns ``SanitizeResult`` with the (possibly redacted) text and a
    list of detections. Writes a JSONL audit row whenever mode != OFF
    and at least one pattern was detected.

    Raises ``BYODLeakError`` if mode==BLOCK and any pattern was detected.
    """
    if mode is None:
        mode = get_mode()
    if mode == BYODMode.OFF:
        return SanitizeResult(text=prompt, mode=mode.value)

    detections: list[dict] = []
    redacted = prompt
    # Apply detectors in declaration order. Each replacement is recorded.
    for category, pattern, repl in _DETECTORS:
        matches = list(pattern.finditer(redacted))
        if not matches:
            continue
        detections.append({
            "category": category,
            "count": len(matches),
            "samples": [m.group(0)[:60] for m in matches[:3]],  # truncate samples
        })
        if mode in (BYODMode.STRICT, BYODMode.BLOCK):
            redacted = pattern.sub(repl, redacted)

    if mode == BYODMode.BLOCK and detections:
        raise BYODLeakError(call_site, detections)

    # Audit log
    if detections:
        _append_audit(
            audit_path or DEFAULT_AUDIT_PATH,
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "site": call_site,
                "mode": mode.value,
                "raw_chars": len(prompt),
                "sanitized_chars": len(redacted),
                "patterns": [{"category": d["category"], "count": d["count"]} for d in detections],
                "tenant": tenant or os.environ.get("BYOD_TENANT", "default"),
            },
        )

    return SanitizeResult(
        text=redacted if mode in (BYODMode.STRICT, BYODMode.BLOCK) else prompt,
        detections=detections,
        modified=(redacted != prompt) and mode in (BYODMode.STRICT, BYODMode.BLOCK),
        mode=mode.value,
    )


def _append_audit(path: Path, row: dict) -> None:
    """Append one JSONL row. Best-effort: silently swallows IO errors so
    the simulation never blocks on audit-log issues."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


# Convenience for downstream tools / dashboards.
def audit_summary(path: Optional[Path] = None) -> dict:
    """Return aggregate counts per (site, category) from the audit log."""
    p = path or DEFAULT_AUDIT_PATH
    if not p.exists():
        return {"n_rows": 0, "by_site": {}, "by_category": {}}
    by_site: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            site = row.get("site", "?")
            by_site[site] = by_site.get(site, 0) + sum(p.get("count", 0) for p in row.get("patterns", []))
            for p in row.get("patterns", []):
                cat = p.get("category", "?")
                by_cat[cat] = by_cat.get(cat, 0) + p.get("count", 0)
    return {"n_rows": n, "by_site": by_site, "by_category": by_cat}
