# DORA Economic Impact — Calibration Pipeline

**Status:** Sprints A · A.2 · B (partial) · C · D.1 · D.2 · D.3 · D.5 · **E.1 · E.2 · E.3 · E.4 · E.5 · E.6** LIVE.  D.4 infrastructure shipped, replay backfill is the operator's call (cost ~$25, 3h).

**Sprint E (CI tightening, 2026-05-05):**

NOTE: earlier internal copy attributed each layer to a Nobel laureate as a rhetorical framing ("Shiller-King bootstrap", "Andrew Lo HMM", etc.). The framing was creative naming, not citation-accurate. The actual literature references for each technique are below.

- E.1 — Empirical pairs-bootstrap of α (B=5000 replicates) replaces the ±1.645σ Gaussian band. Surfaced as "Epistemic range" with N=40 footnote. Reference: Efron (1979); Davison & Hinkley (1997, *Bootstrap Methods and Their Application*).
- E.2 — 2-state Gaussian HMM on log(VIX) monthly 1997-2025, fit by Baum-Welch / forward-backward. Each incident inherits posterior P(high_vol_regime|date). α fitted as smooth mixture α_low·(1-p)·s + α_high·p·s. Replaces brittle hand-coded calm/stressed/crisis label. Empirical regime amplification ≈20× (high vs low). Reference: Hamilton (1989, *Econometrica*); Baum et al. (1970).
- E.3 — HC3 (Eicker–Huber–White–MacKinnon) sandwich SE on α. Reported alongside the homoscedastic σ. References: Eicker (1967), Huber (1967), White (1980), MacKinnon & White (1985).
- E.4 — Hill estimator for the Pareto tail index on |residuals| above the 90° percentile. Flags infinite-variance regimes (α̂<2). Reference: Hill (1975, *Annals of Statistics*).
- E.5 — 2SLS-IV using HMM regime posterior as instrument for shock_units. Reports first-stage F as endogeneity diagnostic. Currently F<10 ⇒ instrument too weak for headline use, published as a sanity check. References: Theil (1953); Basmann (1957); Stock & Yogo (2005, weak-instrument thresholds).
- E.6 — Log-log slope γ as fragility exponent (cost = β·s^γ, OLS in log space). Overall γ=3.36 (R²=0.72) ⇒ strong convex/fragile signal. Promoted to primary headline in commit f3ee55b. The fragility/convexity *concept* is from Taleb (2012, *Antifragile*); the log-log power-law fit itself is standard regression analysis dating to Pareto (1896).

Files: `core/dora/regime_hmm.py` (new), `core/dora/economic_impact.py` (extended), `shared/vix_monthly_cache.json` (new), `shared/dora_reference_incidents.json` (schema v1.1, +incident_date), `frontend/app/compliance/page.tsx` (Hero updated with epistemic-range relabel + 4 new diagnostic chips + 2 new methodology blocks).
**Owner:** Alberto Gerli
**Last refit:** see `outputs/dora_calibration.json` `generated_at`

---

## Why this matters

`economic_impact_eur` is the headline number on `/compliance` → DORA. On it the entire compliance pitch hinges. We need it to be:

1. **Defensible** — every euro can be traced to a methodology + reference incident
2. **Live** — refit nightly so α tracks fresh historical-cost annotations
3. **Auditable** — CRO sees formula, inputs, similar incidents, residuals
4. **Honest** — confidence band reported, outliers flagged

---

## Three-method estimator

The number you see in the UI is `combine(A, B, C)` where:

| Method | What it does | Status | Source of truth |
|---|---|---|---|
| **A · Anchor** | OLS-fitted α × Σ shock units of the sim | ✅ Sprint A live (40 incidents, R²=0.246 overall, R²=0.83 sovereign, R²=0.88 banking_it) | `shared/dora_reference_incidents.json` |
| **B · Ticker** | Σ \|cum_pct\| × market_cap × γ_contagion (1.6×) | ✅ Sprint A live for newer sims (checkpoint patches `ticker_prices`) | `shared/ticker_market_caps.json` |
| **C · LLM judge** | gemini-3.1-pro-preview reasons over brief + trajectory + RAG-retrieved similar incidents | ⚠️ Scaffold only — Sprint C TODO | `core/dora/llm_judge.py` |

Combined = **max(A, B)** today. Will become **weighted_avg(A, B, C)** with weights calibrated on holdout once C ships.

---

## Sprint A — Reference incident calibration (DONE)

### Adding a new incident

1. Edit `shared/dora_reference_incidents.json` and append:
   ```json
   { "id": "snake_case_id_year",
     "shock_units": 1.X,
     "cost_eur_m": NNNN,
     "category": "banking_it|banking_eu|banking_us|sovereign|cyber|telco|energy",
     "label": "Human-readable name (year)",
     "sources": ["regulator filing or 10-K", "news"] }
   ```
2. Run refit:
   ```bash
   python -m scripts.calibrate_dora_alpha --by-category --print
   ```
3. Inspect `outputs/dora_calibration.json` → check that R² didn't degrade > 5pp and your new entry isn't a 2σ outlier (would suggest the shock_units estimate is off).
4. Commit + push. The next API request picks up the new α automatically (loader is process-cached but refreshes on import).

### Estimating shock_units for a new incident

Heuristic tiers (until we re-simulate each event):
- **0.5 – 1.2** — single-firm contained (Tercas, Popolare Bari, ENI Gabon)
- **1.2 – 2.0** — protracted single-firm or sector-wide moderate (Wirecard, Carige, MPS bail-in)
- **2.0 – 3.0** — major sector or sovereign-adjacent (SVB, Cyprus 2013, Uniper)
- **3.0 +** — systemic / sovereign-class (Brexit Wave-1, Argentina default, Lehman)

For higher fidelity: write a brief that replays the incident, run the sim, observe Σ |shock_magnitude × shock_direction|, and use that as the empirical shock_units value.

### Cron / refit cadence

Hooked into the GitHub Actions nightly cron at `04:30 UTC` after the calibration-forecast job finishes:
```yaml
- name: Refit DORA α
  run: python -m scripts.calibrate_dora_alpha --by-category
- name: Commit refresh
  uses: stefanzweifel/git-auto-commit-action@v5
  with:
    commit_message: "DORA: nightly α refit (auto)"
    file_pattern: outputs/dora_calibration.json
```
(Wire in `.github/workflows/nightly-jobs.yml` as the new step.)

Plus: admin job `dora-recalibrate` lets the operator trigger a manual refit from `/admin/jobs` after editing the incident table.

### Sprint A.2 — known follow-ups

- [ ] Per-category α exposed in the API + UI (currently only computed in `--by-category` mode)
- [ ] Auto-pick the per-category α based on the brief's domain detection
- [ ] OLS + intercept once N > 50 (Tercas anchor at zero is suppressing the constant)
- [ ] Heteroscedastic-robust SE (HC3) on the residual band (Lehman dominates the sigma now)
- [ ] Bootstrap 1000-sample CI on α (replace the ±1.65σ approximation)

---

## Sprint B — Live data ingestion (SCAFFOLDED)

`core/dora/live_data.py` exposes the final interface but every function returns a stub. Implementation plan:

1. **`refresh_market_caps(force=False)`**
   - Loop tickers from `shared/stock_universe.json`
   - `yf.Ticker(t).fast_info` → `shares_outstanding × close`
   - Convert to EUR via ECB SDW reference rate snapshot
   - Overwrite `shared/ticker_market_caps.json`, rounded to nearest 10M EUR
   - 24h disk cache; `force=True` bypasses
2. **`fetch_recent_incidents_from_news(since_days=30)`**
   - Query Reuters/FT/Bloomberg for resolution / bail-in / ransomware / fine / downgrade keywords
   - LLM-extract `{entity, date, cost_eur_m, sources, category}` per article using gemini-3.1-pro-preview (Sprint C model)
   - Dedup against existing reference table
   - Return candidate list to `/admin/jobs` UI for human review → approved entries are committed to JSON
3. **`sovereign_spread_snapshot()`**
   - Already partially live via `core.financial.market_data` (ECB SDW + FRED)
   - Adds regime tag `"calm" | "stressed" | "crisis"` so α can be sliced

New cron job `dora-refresh-live` (added to `.github/workflows/nightly-jobs.yml` and `_ADMIN_JOBS` registry) — runs all three steps nightly before the calibration refit.

---

## Sprint C — LLM judge layer (SCAFFOLDED)

`core/dora/llm_judge.py:estimate_via_llm_judge(...)` returns None today. When implemented:

1. Embed brief + final trajectory summary
2. Retrieve top-K (K=5) most-similar incidents from `shared/dora_reference_incidents.json` via the existing RAG store (`core.rag.rag_store`)
3. Build system prompt — "senior risk officer analysing this simulated incident relative to K closest historical analogues; estimate realistic euro cost"
4. Call **gemini-3.1-pro-preview** (note: pro variant, not flash-lite — authorised explicitly for analytical-reasoning use cases per user 2026-05-05)
5. Structured JSON response:
   ```json
   { "point_eur": ..., "low_eur": ..., "high_eur": ...,
     "reasoning": "...",
     "per_analog_relevance": [{"id": "mps_bailin_2017", "similarity": 0.71, "weight": 0.4}, ...],
     "confidence_score": 0-1 }
   ```
6. Validate (point within 0.1× and 10× of (A+B)/2) and return
7. Combined → `weighted_avg(A, B, C)` with W tuned on holdout

### Why pro vs flash-lite for this one

flash-lite is great for narrative agents (10s of LLM calls per round) where speed and cost dominate. The DORA judge call is **once per sim, post-completion, high-stakes**. Pro's deeper reasoning + longer context is worth the extra cost ($1.25/M input vs $0.25/M for flash-lite, and we use ~3-4k tokens per call → +$0.004 per sim).

---

## Files modified by this pipeline

```
shared/
  dora_reference_incidents.json     ← source of truth (40 entries, schema v1)
  ticker_market_caps.json           ← Method B reference (45 tickers)
core/dora/
  economic_impact.py                ← combine(A, B, C) entry-point
  live_data.py                      ← Sprint B stubs
  llm_judge.py                      ← Sprint C stubs
scripts/
  calibrate_dora_alpha.py           ← refit script (re-runnable)
outputs/
  dora_calibration.json             ← refit snapshot (auto-generated)
.github/workflows/
  nightly-jobs.yml                  ← TODO: add refit step + dora-refresh-live
api/main.py
  /api/admin/jobs registry          ← TODO: add "dora-recalibrate" job
  /api/compliance/dora/preview      ← consumes economic_impact_breakdown
docs/
  DORA_ECONOMIC_IMPACT_PIPELINE.md  ← this doc
```

---

## Quick-reference numbers (today)

```
N reference incidents:    40
α (overall):              €24.3B / shock-unit
R² (overall):             0.246    ← low because Lehman is a 5σ outlier
R² (banking_it):          0.883
R² (banking_eu):          0.547
R² (banking_us):          0.463
R² (sovereign):           0.827
R² (cyber):               0.285
R² (telco):               0.574
R² (energy):              0.798
γ contagion (Method B):   1.6×
Calibration script:       python -m scripts.calibrate_dora_alpha --by-category
```
