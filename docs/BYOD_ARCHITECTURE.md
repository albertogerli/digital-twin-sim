# BYOD enclave architecture

**Status:** Phase 1-4 implemented (Sprints 84-87, May 2026).
**Mode:** opt-in via `BYOD_MODE` environment variable, default `OFF` (passthrough).

The "Bring Your Own Data" enclave is the architectural guarantee that
no customer-sensitive financial data — internal LCR / NIM / CET1 /
deposit-β / proprietary balances / customer IDs / IBANs — leaves the
customer's perimeter via the LLM provider's API. This document is the
contract.

## Data-flow boundaries

```
┌──────────────────────────────────────────────────────────────────────┐
│ Customer perimeter (Sella on-prem container OR private cloud tenant)│
│                                                                      │
│  ┌──────────────────────┐      ┌─────────────────────────────────┐  │
│  │ Customer parameters  │      │ FinancialTwin (ALM engine)      │  │
│  │ (LCR, NIM, deposit β,│ ───▶ │ — runs entirely IN-PROCESS      │  │
│  │  customer cohorts,   │      │ — no LLM call                   │  │
│  │  proprietary metrics)│      │ — emits CATEGORICAL signals:    │  │
│  └──────────────────────┘      │   {nim_anxiety, cet1_alarm,    │  │
│                                 │    runoff_panic, ...}           │  │
│                                 └────────┬────────────────────────┘  │
│                                          │ (categorical signals)    │
│                                          ▼                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Multi-agent layer + briefing pipeline                           ││
│  │   - constructs prompts for each agent / scope detector          ││
│  │   - financial-sensitive content NEVER inlined as numbers        ││
│  └────────┬────────────────────────────────────────────────────────┘│
│           │                                                          │
│           │ prompt (text)                                            │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ ◆ BYOD SANITIZER (core/byod/sanitizer.py) ◆                     ││
│  │   - regex-detects: currency, financial metrics, client IDs,     ││
│  │     IBANs, large amounts in financial context, named            ││
│  │     benchmarks (Euribor, BTP-Bund spread)                       ││
│  │   - mode=STRICT: redacts to categorical descriptors             ││
│  │   - mode=BLOCK: raises BYODLeakError                            ││
│  │   - mode=LOG: audit-only                                        ││
│  │   - audit log: outputs/byod_audit.jsonl (append-only JSONL)     ││
│  └────────┬────────────────────────────────────────────────────────┘│
│           │ sanitized prompt                                         │
└───────────│──────────────────────────────────────────────────────────┘
            │
            ▼
   ┌───────────────────────┐
   │ External LLM provider │
   │ (Gemini/OpenAI/Claude)│
   │  Receives ONLY the    │
   │  sanitized prompt.    │
   └───────────────────────┘
```

## What the LLM sees

After sanitization, a prompt that originally read:

> The bank's LCR is 95%, deposit balance reached €12,500,000, with
> client-12345 issuing a complaint via IBAN IT60 X05428 11101 00...

becomes:

> The bank's [LCR breaching: below 100%], deposit balance reached
> [large-amount], with [client-id] issuing a complaint via [IBAN]

The narrative meaning is preserved (the LLM still understands
"LCR is breaching"), but the exact internal value never leaves the
process. The same applies to currency amounts (`€2.4M` →
`[currency-amount]`), client IDs, IBANs, and named benchmark values.

## What is NEVER sent in the first place

The architecture itself isolates these without needing the sanitizer:

- **Specific financial KPI values** (NIM, CET1, LCR, deposit β as numbers).
  The `FinancialTwin` (`core/financial/twin.py`) computes them locally
  and emits categorical `FeedbackSignals` that the agent prompts consume
  ("CET1 vicino al limite regolatorio (alarm 0.74)", not "CET1 = 14.2%").

- **Customer cohort / segmentation tables**. Loaded from local JSON,
  consumed by the ALM engine, never serialised into prompts.

- **Internal exposure breakdowns** (`core/financial/exposure.py`).
  Pure Python computation, no LLM round-trip.

The sanitizer is the *defense-in-depth* — it catches anything that
slips through despite the architectural isolation.

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `BYOD_MODE` | `OFF` | Passthrough (no checking, no audit). Use only in trusted single-tenant setups. |
| `BYOD_MODE=LOG` | — | Detect patterns, write to audit log, do NOT modify the prompt. Use to measure leak surface of a new deployment before flipping STRICT. |
| `BYOD_MODE=STRICT` | — | Detect, redact, write to audit log. **Production BYOD.** |
| `BYOD_MODE=BLOCK` | — | Detect, raise `BYODLeakError` instead of redacting. Fail-closed mode for paranoid deployments. |
| `BYOD_TENANT` | `default` | Tenant label written into every audit row, for filtering/reporting. |

The mode is read at every `generate()` call (no module-level cache),
so changing it between calls (per-tenant overrides, test fixtures)
takes effect immediately.

## Audit log format

`outputs/byod_audit.jsonl` (append-only). One JSON object per line:

```json
{
  "ts": "2026-05-04T11:23:45Z",
  "site": "llm:agent_round.elite",
  "mode": "STRICT",
  "raw_chars": 4823,
  "sanitized_chars": 4791,
  "patterns": [
    {"category": "currency", "count": 2},
    {"category": "financial_metric", "count": 1}
  ],
  "tenant": "sella-prod"
}
```

**Compliance review query:** any row with non-empty `patterns` is
proof that the sanitizer caught a sensitive pattern.

```bash
jq 'select(.patterns | length > 0)' outputs/byod_audit.jsonl
```

For an aggregate summary across the log:

```python
from core.byod.sanitizer import audit_summary
summary = audit_summary()
# {"n_rows": 12345, "by_site": {...}, "by_category": {...}}
```

## Detector categories

| Category | Example matches | Replacement |
|---|---|---|
| `currency` | `€2.4M`, `$500K`, `EUR 1,200`, `2.5 milioni di euro` | `[currency-amount]` |
| `financial_metric` | `LCR 168%`, `CET1 14.2%`, `NIM 1.85%`, `LCR=95`, `CET1 ratio of 16%` | `[LCR healthy]` / `[LCR breaching]` / `[CET1 healthy]` / `[CET1 tight]` / `[NIM above EBA median]` (threshold-aware) |
| `client_id` | `client-12345`, `customer ID xxx`, `account 9999` | `[client-id]` |
| `iban` | `IT60 X05428 11101 000000123456` | `[IBAN]` |
| `benchmark_value` | `Euribor 3M at 2.4%`, `BTP-Bund spread 180bps` | `[Euribor at level]` / `[BTP-Bund spread at level]` |
| `large_amount_in_context` | `deposit balance reached 12,500,000` | `[large-amount]` |

## What is NOT redacted

- **Non-financial percentages** (`45% of the vote`, `12% market share`
  in non-financial context).
- **Polarization scores** (range `[0, 10]`).
- **Agent positions** (range `[-1, +1]`).
- **Dates, ISO years, generic numbers without financial keyword adjacency.**

## Tests

`tests/test_byod_sanitizer.py` — 20 unit tests:
- Each detector category fires on canonical examples
- Non-financial percentages preserved
- Dates / agent positions / polarization preserved
- Mode semantics (OFF / LOG / STRICT / BLOCK)
- Audit log JSONL well-formed
- Threshold-aware metric labels (LCR healthy/breaching, CET1, NIM)

Integration: enable `BYOD_MODE=STRICT` and run any simulation; verify
`outputs/byod_audit.jsonl` has rows for every LLM call site that
processed customer-sensitive content, with `patterns` populated.

## Known limitations (honesty box)

1. The sanitizer is **regex-based**, not semantic. It will NOT catch
   sensitive numbers that are spelled out in narrative form
   ("twelve million euros") or split across sentence boundaries.
   For paranoid deployments use `BYOD_MODE=BLOCK` plus an additional
   semantic-classifier pass (e.g. local NER on the prompt before
   calling `generate()`).

2. The `large_amount_in_context` detector requires the financial
   keyword to be within ±30 chars of the number. Long narrative
   sentences may evade it.

3. The current deployment ships the sanitizer as **defense-in-depth**
   against the architectural isolation — most financial values are
   already kept out of prompts by `FinancialTwin` design. The
   sanitizer's job is to catch leaks if and when the architectural
   invariant is violated by a future code change. Complete coverage
   would require type-level enforcement (e.g. a `SanitizedStr`
   newtype passed to `llm.generate()`), which is a v1.0 candidate.

4. Audit log writes are best-effort (silent IO error swallowing).
   In production deploy with `outputs/` on a persistent volume
   that pre-allocates space; otherwise audit rows can be silently
   lost if the disk fills mid-simulation.
