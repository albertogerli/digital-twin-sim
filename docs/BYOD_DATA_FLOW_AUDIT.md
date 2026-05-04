# BYOD data-flow audit — what crosses the LLM boundary

**Generated:** May 2026 · Sprint 84
**Scope:** every call site in `core/`, `briefing/`, `domains/`, `api/` that invokes an external LLM (Gemini, OpenAI, Claude).

The BYOD enclave architecture (Sella relazione §13.3) requires that no
customer-sensitive financial data — internal LCR / NIM / CET1 / deposit-β /
proprietary balances / customer IDs — leaks to the LLM provider's API.
This audit is the prerequisite: identify every call site, classify the
data it sends, and decide which calls need the BYOD sanitizer hook.

## Architecture insight

**The financial numbers were never the leak risk.** The
`core/financial/twin.py` ALM engine runs entirely locally. What it injects
into agent prompts is the `FeedbackSignals` dataclass (`nim_anxiety`,
`cet1_alarm`, `runoff_panic`, `competitor_pressure`, `rate_pressure`) —
**categorical labels**, not numbers. The LLM sees "CET1 vicino al limite
regolatorio (alarm 0.74)", not "CET1 = 14.2%".

**The real leak risk is the customer's brief text** (which flows through
9 of the 17 LLM call sites in the briefing pipeline) and the **named
stakeholders** in seed/agent rosters (which can encode customer scope,
geography, sector specialization, etc.).

So the sanitizer's primary job is: protect the brief + agent metadata,
not the financial numbers (which are already isolated by architecture).

## Call-site classification

### Narrative-only (6 sites — safe to send to external LLM)

| # | Site | Function | Data |
|---|---|---|---|
| 1 | `core/simulation/event_injector.py:147` | `EventInjector.generate_event()` | scenario context, agent positions, history |
| 2 | `core/simulation/event_injector.py:178` | retry path | same as #1 |
| 3 | `core/simulation/validators.py:91` | `check_event_plausibility()` | event text + scenario context (truncated) |
| 4 | `core/simulation/reporting.py:97` | `ReportingService.generate_report()` | round summaries, agent names, positions |
| 5 | `evaluation/realism_scorer.py:140` | `check_event_coherence()` | scenario name + event descriptions (truncated) |
| 6 | `briefing/brief_scope.py:171` | `analyze_scope()` | brief + web context, NO named people |

### Potentially-sensitive (9 sites — REQUIRE sanitizer)

| # | Site | Function | Risk |
|---|---|---|---|
| 7 | `briefing/brief_analyzer.py:132` | `analyze_brief()` | full customer brief + verified stakeholder names |
| 8 | `briefing/agent_generator.py:255` | `generate_agents_multistep()` 3a | brief + scope + archetype guidance |
| 9 | `briefing/agent_generator.py:456` | step 3b (Elite Agents) | brief + scope + seed stakeholders + entity research |
| 10 | `briefing/agent_generator.py:496` | step 3c (Institutional) | scope + brief + elite agent summary |
| 11 | `briefing/agent_generator.py:530` | step 3d (Citizen Clusters) | scope + verified seed demographics |
| 12 | `briefing/realism_gate.py:288` | `run_realism_gate()` | brief excerpt + full agent rosters |
| 13 | `briefing/realism_gate.py:627` | `regenerate_rejected_elite_agents()` | scope + brief + rejected IDs + seeds |
| 14 | `briefing/realism_gate.py:669` | `regenerate_rejected_institutional_agents()` | scope + brief + rejected IDs |
| 15 | `api/main.py:305` | `suggest_kpis()` | customer brief + optional domain hint |

### Mixed (2 sites — depends on customer-authored content)

| # | Site | Function | Risk |
|---|---|---|---|
| 16 | `core/agents/elite_agent.py:64` | `EliteAgent.generate_round()` | system_prompt may be customer-authored |
| 17 | `core/agents/citizen_swarm.py:86` | `CitizenSwarm._simulate_cluster()` | cluster_description may be customer-targeted |

## Sanitizer hook priority

**Critical (CUSTOMER PERIMETER LEAK — implement first):** #7 → #15.
Every call site that sees the raw `brief` text gets the sanitizer
applied at the prompt boundary.

**High (context inference risk):** #16, #17. Apply sanitizer to
`system_prompt` and `cluster_description` BEFORE they enter the agent
constructor, so the prompt the LLM sees has already been sanitized.

**Skip (safe by design):** #1-6. Already pass only categorical /
truncated / public-facing content.

## Patterns the sanitizer must detect

These are the regex / heuristic categories implemented in
`core/byod/sanitizer.py`:

1. **Currency amounts** — `€2.4M`, `$500K`, `EUR 1,200`, `2.5 milioni`,
   `1,200,000 euro`, etc. → `[currency-amount]`.
2. **Financial-metric values** — `LCR 168%`, `CET1 14.2%`, `NIM 1.85%`,
   `deposit β 0.45`. Healthy/breaching label preserved when threshold
   known: `LCR 95%` → `[LCR breaching: below 100%]`.
3. **Customer / account IDs** — `client-12345`, `customer ID xxx`,
   `account 9999`, `IBAN IT60...`. → `[client-id]`.
4. **Bare large numbers in financial context** — number ≥ 10,000 within
   ±20 chars of a financial keyword (`deposit`, `balance`, `capital`,
   `RWA`, `loan`, `mortgage`, `bond`, `BTP`, `spread`). → `[large-amount]`.
5. **Named financial benchmarks with values** — `Euribor 3M = 2.4%`,
   `BTP-Bund 180bps`. → `[Euribor at level]`, `[spread at level]`.

Non-financial percentages (poll opinion `45%`, market share `12%`) are
**preserved** — only redacted when adjacent to financial keywords.
Dates, agent positions in `[-1, +1]`, polarization scores in `[0, 10]`
are also preserved.

## Modes

- `OFF` — passthrough, no checking. Default.
- `LOG` — detect patterns, write to audit log, do NOT modify the prompt.
  Useful for first-time deployment to measure leak surface before
  enabling redaction.
- `STRICT` — detect, redact, write to audit log. Production BYOD mode.
- `BLOCK` — detect, raise `BYODLeakError` instead of redacting. Paranoid
  fail-closed mode for high-stakes deployments.

## Audit log format

Every call site emits one JSONL row per LLM invocation when mode ≠ OFF:

```jsonl
{"ts": "2026-05-04T11:23:45Z", "site": "briefing/agent_generator.py:456",
 "mode": "STRICT", "raw_chars": 4823, "sanitized_chars": 4791,
 "patterns": [{"category": "currency", "count": 2}, {"category": "financial_metric", "count": 1}],
 "tenant": "sella-prod"}
```

Compliance review: `jq 'select(.patterns | length > 0)' outputs/byod_audit.jsonl`.

If the file shows zero rows with non-empty `patterns` for a tenant,
the BYOD contract is verifiably enforced.
