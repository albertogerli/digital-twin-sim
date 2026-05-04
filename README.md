# DigitalTwinSim

> **Hybrid physics + narrative digital twin for the banking sector.**
> Multi-agent opinion dynamics (LLM-driven) coupled to a deterministic
> ALM financial twin, with empirical calibration on 95 historical events
> across 20 countries, BYOD compliance enclave, DORA Major Incident
> Report XML export, and a continuous self-calibration loop.

[![Tests](https://img.shields.io/badge/tests-494%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-42%25-yellow)](tests/)
[![Paper](https://img.shields.io/badge/paper-v2.8%20JASSS-blue)](paper/digitaltwin_calibration_v2.8.md)
[![License](https://img.shields.io/badge/license-proprietary-lightgrey)](LICENSE)
[![Stack](https://img.shields.io/badge/stack-Python%203.11%20%2B%20Next.js%2014-blue)]()

**Live demo:** [digital-twin-sim.vercel.app](https://digital-twin-sim.vercel.app)
**Paper (working):** `paper/digitaltwin_calibration_v2.8.md` (1988 lines, JASSS submission)
**Technical dossier:** [`docs/TECHNICAL_DOSSIER.md`](docs/TECHNICAL_DOSSIER.md)

---

## What it does

Given a brief in natural language (e.g. *"Sella reduces the BCE rate
pass-through by 25 bps"*), DigitalTwinSim produces in 8–15 minutes:

1. **Stakeholder trajectories** — N real political/regulatory/CEO/citizen
   actors evolving on the topic axis [-1, +1] over 5–9 simulation rounds.
2. **Agent-generated posts** — what each stakeholder would say on TV /
   press / social / forum, with engagement metrics.
3. **Bank KPI trajectories** — NIM, CET1, LCR, deposit balance, loan
   demand, BTP–Bund spread, FTSE MIB, all coherent with EBA / ECB /
   Banca d'Italia regulatory parameters.
4. **Equity impact per ticker** — T+1 / T+3 / T+7 returns on ~190 global
   tickers, direction (long/short), pair trade derived from empirical
   correlation matrix.
5. **Print-ready HTML report** + interactive dashboard + auditable JSON +
   on-request DORA Major Incident Report XML for regulatory submission.

## What it is NOT

- **Not a forecasting model.** Paper benchmarks against four naive
  forecasters (persistence / running-mean / OLS / AR(1)) and finds the
  ABM matches but does not dominate persistence in retrospective
  trajectory space.
- **Not a substitute for a bank's ALM core.** It sits next to it as a
  what-if narrative layer.
- **Not a pure generative-AI tool.** Hybrid by design: LLM for
  narrative, deterministic Python for KPIs.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Briefing Layer    (briefing/)                               │
│    Brief → scope → 3-layer stakeholder filtering with audit    │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Multi-Agent Layer (core/agents/)                            │
│    Tier 1 Elite (8-14) · Tier 2 Institutional (6-10) ·         │
│    Tier 3 Citizen swarms (5-8). Stakeholder graph: 744 actors. │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Opinion Engine    (core/simulation/opinion_dynamics_v2.py)  │
│    5-force softmax mixture, gauge-fixed.                       │
│    Bayesian-calibrated coefficients (NumPyro SVI + NUTS).      │
└────────────────────────────────────────────────────────────────┘
        ↓ (lockstep)
┌────────────────────────────────────────────────────────────────┐
│ 4. FinancialTwin ALM (core/financial/)                         │
│    7 countries (IT/DE/FR/ES/NL/US/GB) with country-aware       │
│    parameters. CIR rate model. EBA stress templates.           │
│    Empirical sector betas (86 cells, 20 countries).            │
│    Cross-market correlation matrix (180 tickers × 8 years).    │
│    VAR(1) sector spillover network (39 directional edges).     │
└────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Reporting & Compliance (core/dora/, core/byod/)             │
│    HTML print-to-PDF · Dashboard · JSON · DORA XML export ·    │
│    BYOD audit log · Continuous self-calibration loop.          │
└────────────────────────────────────────────────────────────────┘
```

## Key features

### Empirical calibration (v0.8 — May 2026)

Four heuristic constants in the financial impact scorer were replaced
with coefficients derived from event studies on a 95-scenario corpus
(60 IT + 35 global, 310 valid event×ticker observations):

| Constant (legacy) | Value (heuristic) | Replacement (empirical) |
|---|---|---|
| `panic_mult = exp(cri × 1.5)` | 2.3x → 4.0x | **median per CRI bin** (mid 2.7x · high 5.5x · extreme 10.9x) |
| `recovery_factor = 0.5 + 0.1×i` | 0.5–0.7 | **T+3/T+1 per (intensity bin × sector)**, 23 cells |
| `escalation_factor = 1.0 + 0.1×(i−2)` | 1.0–1.6 | as above |
| `t7 = t3 × 1.3 + ...` | 1.3 persistence | **T+7/T+1 per (intensity bin × sector)** |
| `CRISIS_PAIR_TRADES[topic]` | hardcoded dict | **derived from correlation matrix** |

See [`docs/EMPIRICAL_VALIDATION_REPORT.md`](docs/EMPIRICAL_VALIDATION_REPORT.md)
for the PRE/POST A-B validation: empirical wiring improves direction
accuracy by **+3.8 pp** and MAE T+3 by **−0.50 pp**, regresses MAE T+7
by **+2.83 pp** (slow-burn events — fix in v0.9 roadmap).

### Compliance enclave

| Track | Status | Files | Tests |
|---|---|---|---|
| **BYOD** — sanitizer + audit, redacts financial-sensitive content from LLM prompts | ✅ implemented | `core/byod/sanitizer.py`, [`docs/BYOD_ARCHITECTURE.md`](docs/BYOD_ARCHITECTURE.md) | 20 |
| **DORA** — Major Incident Report XML per EBA/EIOPA/ESMA JC 2024-43 (Art. 19-20) | ✅ MVP | `core/dora/{schema,exporter,classification}.py`, [`docs/DORA_EXPORT_SCOPE.md`](docs/DORA_EXPORT_SCOPE.md) | 15 |
| **Self-calibration** — nightly shadow forecasts scored against realised yfinance returns at T+1/T+3/T+7 | ✅ MVP | `core/calibration/continuous.py`, `scripts/continuous_calibration.py` | 14 |

All three live in `/compliance` page with three tabs.

### Frontend

- `/` — Dashboard (KPIs, scenario list)
- `/new` — 4-step simulation wizard (brief → KB → engine → review)
- `/sim/[id]` — Live monitor (SSE streaming, replay)
- `/scenario/[id]` — Detailed report (per-round, per-agent, financial impact)
- `/wargame` — Interactive crisis simulator
- `/backtest` — 64-event backtest dashboard with Cross-Market Contagion
  Graph (D3 force layout, 180 tickers, Louvain communities) +
  Sector Spillover VAR network (39 directed edges)
- `/compliance` — CISO/CRO console with BYOD + DORA + Self-calibration tabs
- `/paper` — Working paper UI
- `/admin/invites` — Generate invite links

## Quick start (Docker)

```bash
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY=your_key

docker-compose up --build
```

- Frontend: http://localhost (via nginx) or http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

## Quick start (development)

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add GOOGLE_API_KEY
python run_api.py

# Frontend (in another terminal)
cd frontend && npm install && npm run dev

# Tests
pytest tests/ -v --tb=short
cd frontend && npm run test:ci
```

## Continuous self-calibration (cron-friendly)

```bash
# Nightly cron job (recommended for production):
0 18 * * * cd /opt/digital-twin-sim && \
    python scripts/continuous_calibration.py forecast && \
    python scripts/continuous_calibration.py evaluate --horizon 1 && \
    python scripts/continuous_calibration.py evaluate --horizon 3 && \
    python scripts/continuous_calibration.py evaluate --horizon 7

# Manual report any time:
python scripts/continuous_calibration.py report
```

Default watchlist: UCG.MI, ISP.MI, ENI.MI, ENEL.MI, STLAM.MI, G.MI.
Configurable via `--tickers`. Cost: $0/month (no LLM call in the
shadow forecast).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google Gemini API key (required) |
| `OPENAI_API_KEY` | — | OpenAI key (optional, alternative provider) |
| `DTS_API_KEYS` | — | Comma-separated API keys for auth |
| `DTS_KEY_MAP` | — | JSON `{"key": "tenant_id"}` for multi-tenant |
| `DTS_CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `DTS_RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `DTS_MAX_CONCURRENT` | `2` | Max parallel simulations |
| `DTS_ENV` | `development` | `production` or `development` |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | — | Redis connection string |
| `DTS_SENTRY_DSN_BACKEND` | — | Sentry DSN for backend |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for frontend |
| **`BYOD_MODE`** | `OFF` | BYOD sanitizer mode: `OFF` / `LOG` / `STRICT` / `BLOCK` |
| **`BYOD_TENANT`** | `default` | Tenant label written to BYOD audit log |
| **`DORA_ENTITY_NAME`** | placeholder | Entity legal name for DORA reports |
| **`DORA_ENTITY_LEI`** | placeholder | 20-char LEI code |
| **`DORA_COMPETENT_AUTHORITY`** | `Banca d'Italia` | Competent authority |

## Project structure

```
├── api/                       # FastAPI application
│   ├── main.py                # 32 endpoints incl. /compliance/{byod,dora,calibration}
│   ├── auth.py                # HMAC-SHA256 cookie + API key auth
│   └── ...
├── core/                      # Simulation engine
│   ├── agents/                # Elite, Institutional, CitizenSwarm
│   ├── simulation/            # Engine, opinion dynamics v1+v2
│   ├── financial/             # ALM twin, CIR rates, country params
│   ├── orchestrator/          # Financial impact, correlation lookup, EnKF
│   ├── byod/                  # Sanitizer + audit log
│   ├── dora/                  # XML exporter + Pydantic schema
│   ├── calibration/           # Continuous self-calibration loop
│   ├── llm/                   # Provider abstraction (Gemini / OpenAI)
│   └── platform/              # Social media engine
├── briefing/                  # Brief → scope → 3-layer filtering
├── domains/                   # Pluggable domain modules (6)
├── stakeholder_graph/         # 744 verified real actors
├── frontend/                  # Next.js 14 + TypeScript + D3
├── shared/                    # Calibration JSONs (correlation matrix,
│                              # sector betas, impulse response, panic mult,
│                              # VAR contagion, stock universe)
├── scripts/                   # Calibration + maintenance scripts
├── tests/                     # 502 pytest tests
├── paper/                     # Academic paper v2.8 (JASSS)
└── docs/                      # Architecture, BYOD, DORA, Sella relazione,
                               # technical dossier, validation report
```

## API highlights

| Endpoint | Purpose |
|---|---|
| `POST /api/simulations` | Launch simulation from brief |
| `GET /api/simulations/{id}/stream` | SSE real-time progress |
| `POST /api/simulations/{id}/intervene` | Wargame player action |
| `POST /api/scenarios/{id}/branch` | What-if branching |
| `POST /api/simulations/{id}/observe` | EnKF data assimilation |
| `GET /api/compliance/byod/status` | BYOD mode + audit summary |
| `POST /api/compliance/byod/test` | Interactive sanitizer playground |
| `GET /api/compliance/dora/preview/{sim_id}` | 7-criterion DORA classification |
| `GET /api/compliance/dora/export/{sim_id}` | Download DORA XML |
| `GET /api/compliance/calibration/summary` | Self-calibration loop status |
| `GET /api/health` | Health (Postgres + Redis) |
| `GET /api/usage` | LLM cost tracking per tenant |
| `GET /metrics` | Prometheus metrics |

## Documentation

| Document | Purpose |
|---|---|
| [`docs/TECHNICAL_DOSSIER.md`](docs/TECHNICAL_DOSSIER.md) | Full ~5000-word technical reference (12 sections + numerical fingerprint) |
| [`docs/RELAZIONE_BANCA_SELLA.md`](docs/RELAZIONE_BANCA_SELLA.md) | Commercial + technical pitch (Italian, v0.8) |
| [`docs/BYOD_ARCHITECTURE.md`](docs/BYOD_ARCHITECTURE.md) | BYOD enclave architecture + data-flow diagram |
| [`docs/BYOD_DATA_FLOW_AUDIT.md`](docs/BYOD_DATA_FLOW_AUDIT.md) | Audit of 17 LLM call sites |
| [`docs/DORA_EXPORT_SCOPE.md`](docs/DORA_EXPORT_SCOPE.md) | DORA MVP scope decision rationale |
| [`docs/EMPIRICAL_VALIDATION_REPORT.md`](docs/EMPIRICAL_VALIDATION_REPORT.md) | Empirical wiring A/B (PRE vs POST) |
| [`paper/digitaltwin_calibration_v2.8.md`](paper/digitaltwin_calibration_v2.8.md) | Working paper (JASSS, 1988 lines) |
| [`paper/REPRODUCIBILITY.md`](paper/REPRODUCIBILITY.md) | Pinned-version reproducibility README |
| [`docs/SPRINT_1-13_CHANGELOG.md`](docs/SPRINT_1-13_CHANGELOG.md) | Sprint 1-13 changelog |
| [`CITATION.cff`](CITATION.cff) | Citation metadata (Zenodo-ready) |

## Honest limits (compliance honesty box)

1. **Not a forecasting model.** Validates as scenario exploration, not
   open-loop predictor (paper §6.6).
2. **Financial domain RMSE 0.063 vs 0.012 political domain.** Explicitly
   declared in paper.
3. **LLM is non-differentiable.** Bayesian calibration uses a JAX shadow
   simulator for gradient flow.
4. **Stakeholder DB curated manually** — 744 actors, ~5h/month maintenance.
5. **Live data depends on free public endpoints** (ECB SDW, FRED, BoE).
   Graceful fallback to literature-based defaults.
6. **No formal regulatory certification yet** (ESMA / Bankitalia).
   System is a *pre-deliberation* tool, not a *regulatory submission*
   tool — except DORA XML export which is format-conformant but not
   yet XSD-validated against the official EBA spec.
7. **BYOD sanitizer is regex-based**, not semantic. For paranoid
   deployments add a semantic NER pass.
8. **Self-calibration data moat requires 12+ months** of continuous
   operation. Today the infrastructure is ready; the history is not.

## Citation

If you use DigitalTwinSim in academic work, please cite via the
`CITATION.cff` file at the repository root, or the working paper:

```bibtex
@misc{gerli2026digitaltwinsim,
  author = {Gerli, Alberto Giovanni},
  title  = {A Calibrated LLM-Conditioned Agent-Based Model of Public
            Opinion Dynamics with Null-Baseline Benchmarking,
            Data-Contamination Auditing, and Online Assimilation},
  year   = {2026},
  note   = {Working paper v2.8, JASSS submission},
  url    = {https://github.com/albertogerli/digital-twin-sim}
}
```

## License

Proprietary. Contact `alberto@albertogerli.it` for commercial licensing.
