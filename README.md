# DigitalTwinSim

Universal Digital Twin Simulation Platform — agent-based opinion dynamics with LLM-powered scenario generation, wargame mode, what-if branching, and Monte Carlo analysis.

## Architecture

```
Client → Nginx → Frontend (Next.js) / Backend (FastAPI)
                                        ↓
                              PostgreSQL / Redis / LLM (Gemini/OpenAI)
```

**Backend** (`/`): Python 3.11, FastAPI, async simulation engine with 6 pluggable domain modules (political, corporate, financial, commercial, marketing, public health). Agent types: Elite (named stakeholders), Institutional (organizations), Citizen clusters.

**Frontend** (`frontend/`): Next.js 14, TypeScript, Tailwind CSS, D3.js visualizations, real-time SSE streaming, replay system with post→graph causality.

## Quick Start — Docker

```bash
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

docker-compose up --build
```

- Frontend: http://localhost (via nginx) or http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

## Quick Start — Development

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add GOOGLE_API_KEY
python run_api.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
# Backend
pip install pytest pytest-asyncio pytest-cov
pytest tests/ -v

# Frontend
cd frontend
npm run test:ci
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Google Gemini API key (required) |
| `OPENAI_API_KEY` | — | OpenAI key (optional, for OpenAI provider) |
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

## Project Structure

```
├── api/                    # FastAPI application
│   ├── main.py             # Endpoints
│   ├── auth.py             # API key authentication
│   ├── rate_limiter.py     # Rate limiting (slowapi)
│   ├── simulation_manager.py # Simulation lifecycle
│   ├── models.py           # Pydantic schemas
│   ├── db.py               # PostgreSQL layer
│   ├── job_queue.py        # Redis job queue
│   ├── middleware.py        # Request logging
│   ├── metrics.py          # Prometheus metrics
│   └── logging_config.py   # Structured logging
├── core/                   # Simulation engine
│   ├── agents/             # Agent types
│   ├── simulation/         # Engine, dynamics, checkpoints
│   ├── llm/                # LLM client abstraction
│   └── platform/           # Social media simulation
├── domains/                # Pluggable domain modules
├── briefing/               # Scenario builder (brief → config)
├── frontend/               # Next.js 14 dashboard
│   ├── app/                # App Router pages
│   ├── components/         # React components (57+)
│   ├── lib/                # Types, utils, replay engine
│   └── __tests__/          # Vitest tests
├── tests/                  # Backend pytest tests
├── docker-compose.yml      # Full stack deployment
├── Dockerfile              # Backend container
└── nginx.conf              # Reverse proxy config
```

## API Highlights

- `POST /api/simulations` — Launch simulation from brief
- `GET /api/simulations/{id}/stream` — SSE real-time progress
- `POST /api/simulations/{id}/intervene` — Wargame player action
- `POST /api/scenarios/{id}/branch` — What-If branching
- `POST /api/simulations/{id}/observe` — EnKF data assimilation
- `GET /api/health` — Health check with Postgres/Redis status
- `GET /api/usage` — LLM cost tracking per tenant
- `GET /metrics` — Prometheus metrics
