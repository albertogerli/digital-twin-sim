# Sprint 1-13 — Simulator hardening (v2.7 → v2.8)

This document catalogues the simulator-side changes that landed between
the v2.7 paper submission and the v2.8 re-calibration. The hierarchical
SVI machinery, NUTS validation, Sobol sensitivity, EnKF assimilation,
contamination probe, blinding protocol, and null-baseline benchmark are
all unchanged — only the code paths the simulator exercises during
round generation were refactored.

The thirteen sprints were not version-tagged in git; the table below
maps each sprint to its scope, the canonical commit(s) where it landed,
and the calibration impact (visible in the v2.8 re-calibration aggregate
numbers documented in §10 of the paper).

---

## Aggregate impact (Sprint 1-13 → v2.8 re-calibration)

| Group   | N  | MAE pre→post (pp)         | cov₉₀ pre→post           | CRPS Δ |
|---------|----|---------------------------|--------------------------|--------|
| OVERALL | 42 | 15.22 → **14.65** (−0.57) | 78.6% → **83.3%** (+4.8) | −0.34  |
| TRAIN   | 34 | 14.29 → **13.97** (−0.32) | 79.4% → **82.4%** (+2.9) | −0.10  |
| TEST    |  8 | 19.18 → **17.56** (−1.62) | 75.0% → **87.5%** (+12.5)| −1.35  |

Final SVI loss: 514.74 → 493.79 (−20.95) — same 42 scenarios, identical
hyperparameters (3000 SVI steps, lr 0.005, seed 42).

Reproducible run: `python -m calibration.sprint15_recalibrate`
Comparison report: `python -m calibration.sprint15_compare`

Per-scenario test-set diff is in
`calibration/results/hierarchical_calibration/sprint15/sprint15_vs_baseline.md`.

---

## Per-sprint catalogue

### Sprint 1 — Reflective memory + thread-safe SQLite + budget warnings + seeds + tests

Refactored `core/agents/agent_memory.py` to add reflective summaries
that persist across rounds. Made `core/platform/platform_engine.py`
thread-safe via a write lock and batched commits (`_commit_every=50`),
WAL journal mode. Added per-component budget caps in `UsageStats` and
warning emission when 80% of `request.budget` is consumed. Deterministic
RNG seeding wired through `engine.py`. New tests under
`tests/test_concurrency_load.py`.

**Calibration impact**: stability under high-concurrency runs; no
direct accuracy delta. Enabled the rest.

### Sprint 2 — Strategist agents + emotional vectors + "Why" overlay

Added strategist-class elite agents that maintain a multi-round plan
and emit per-round rationales (visible in the frontend "Why" overlay).
Per-agent emotional state vector tracked in `agent_memory.py`.

**Calibration impact**: detected as a regression source on Tesla
Cybertruck and Amazon HQ2 in the v2.8 re-calibration (over-shoots
consensus). Flagged for v2.9 follow-up.

### Sprints 3-5 — LLM scope detector hardening

`briefing/scope_analyzer.py` rewritten with an LLM-driven scope
extraction (geography, time horizon, target organisations, sectors).
Recall raised on financial / corporate briefs that previously dropped
key stakeholders.

**Calibration impact**: prerequisite for sprint 7's country-alias fix.

### Sprint 6 — Realism gate composition hints

`core/orchestrator/realism_gate.py` (Layer 0) now seeds the LLM with
domain-specific composition templates (e.g. "Italian banking crisis →
include CONSOB, Banca d'Italia, ABI, Codacons, top-3 commercial banks").
Corpus-wide accept rate moved from **0.887** (v2.6) to **0.94+** on the
top-5 domains (commercial, energy, environmental, labor, corporate).

**Calibration impact**: marginal; mostly upstream of dynamics.

### Sprint 7 — Country alias normalisation (UK ↔ GB / USA ↔ US) ⭐

The single largest source of v2.8 improvement. The stakeholder
relevance scorer (`core/orchestrator/stakeholder_relevance.py`) was
keying off the LLM scope detector's geography output verbatim, but the
detector frequently emits ISO-3166 short forms ("UK", "USA") while the
stakeholder DB stores the alpha-2 code ("GB", "US"). Result:

- Boris Johnson (country=`GB`) was scored **0.0** on Brexit briefs
  whose scope.geography=`['UK']`
- Donald Trump (country=`US`) was scored **0.0** on US-politics briefs
  whose scope.geography=`['USA']`

Same pattern for Biden, Harris, Pence, Pelosi, Cameron, etc. Fix added
a bidirectional alias map in
`core/orchestrator/country_aliases.py` and applied at scoring time.

**Calibration impact**: drove the four largest test-set gains —
Greek bailout (−10.78 pp), French election (−7.32), Net Neutrality
(−4.21), COVID vax IT (−4.07). All were scenarios where the missing
heroes carried the bulk of the political signal.

### Sprints 8-10 — Agent prompt + JSON parser improvements

Tightened few-shot exemplars per domain in `domains/*/prompts.py`.
Reduced `JSONParseError` rate from ~3% (v2.7) to <0.5% (v2.8). Added
`core/llm/json_parser.py::repair_truncated()` to handle the common
"closing brace missing" failure mode.

**Calibration impact**: indirect (lower failure rate = fewer
abnormal-engagement rounds = more stable financial scoring).

### Sprint 11 — Engine logic fixes

`core/simulation/engine.py`: fixed off-by-one in round-event override
that was causing the first event of a What-If branch to be skipped.
Deterministic seed propagation through to `OpinionDynamics` so that
identical seed reproduces identical trajectories byte-for-byte.

**Calibration impact**: no aggregate move; required for SBC validity.

### Sprint 12 — Realism gate v2 (programmatic override path)

Added a programmatic override for the LLM realism check that
previously had to be modified by editing the prompt. New
`RealismGate.force_include(stakeholder_id)` API allows the orchestrator
to inject must-have stakeholders without re-prompting.

**Calibration impact**: prerequisite for the deferred Sprint 16 task
(forced inclusion of top-N political candidates per brief).

### Sprint 13 — E2E test harness

`scripts/e2e_batch_paper_scenarios.py` runs the full simulator across
56 paper-scenario briefs (~$10 LLM cost, ~30 min wall-clock). This
surfaces regressions that don't show up in unit tests — including the
Sprint 7 country-alias bug that motivated sprints 7 / 12.

**Calibration impact**: caught the country-alias regression before
v2.8 re-calibration; prevented a false-positive "no improvement" run.

---

## Re-calibration protocol (v2.8)

```bash
# 1. Run the SVI re-calibration (3000 steps, lr=0.005, seed=42)
python -m calibration.sprint15_recalibrate
# Wall-clock: ~40 min on a single CPU node, free of cost (no LLM calls).

# 2. Generate the v2.7 vs v2.8 comparison markdown
python -m calibration.sprint15_compare

# 3. Inspect outputs
ls calibration/results/hierarchical_calibration/sprint15/
#   calibration_report_v2.md
#   loss_history_v2.json
#   posteriors_v2.json
#   sprint15_vs_baseline.md     ← headline diff
#   validation_results_v2.json
```

Both scripts write to a fresh `sprint15/` sub-directory — the v2.7
canonical results in `v2_discrepancy/` are not touched.

---

## What is **not** affected by Sprint 1-13

The following paper claims, sections, and reproducibility artifacts
exercise calibration outputs and observation-fusion paths that are
independent of the simulator code refactored above. They carry through
from v2.7 unchanged:

- §6.6 Null-baseline benchmark (Diebold–Mariano, persistence / mean /
  AR(1) / OLS-trend baselines)
- §6.7 Residual-bootstrap coverage on the hierarchical posterior
- §6.8 Seven-by-five-by-four scenario diversity matrix
- §6.10 Four-axis LLM contamination probe + blinding protocol +
  apples-to-apples sim-lift A/B (DTW results unchanged)
- §7 Ensemble Kalman Filter assimilation feasibility on Brexit
- §8 Discussion of structural misspecification and operational
  positioning
- All appendices (A — Implementation, B — Regime Switching, C — Full
  Scenario List, D — v2.5 Benchmark Reproducibility, E — Layer 0,
  F — Contamination/Blinding, G — ODD Protocol)

The headline scientific claim — that the calibrated ABM matches but
does not dominate naive persistence in retrospective trajectory space,
and earns its operational value from EnKF online assimilation plus its
blinding / benchmarking instrumentation — is unaffected by v2.8.
