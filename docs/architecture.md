# Architecture

## System Overview

```mermaid
graph TB
    Client[Browser/API Client]

    subgraph Infrastructure
        Nginx[Nginx :80]
        Frontend[Next.js :3000]
        Backend[FastAPI :8000]
        Postgres[(PostgreSQL :5432)]
        Redis[(Redis :6379)]
    end

    subgraph External
        Gemini[Google Gemini API]
        OpenAI[OpenAI API]
    end

    Client --> Nginx
    Nginx -->|/api/*| Backend
    Nginx -->|/*| Frontend
    Backend --> Postgres
    Backend --> Redis
    Backend --> Gemini
    Backend --> OpenAI
    Frontend -->|SSE| Backend
```

## Request Flow

### Simulation Launch

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API (main.py)
    participant M as SimulationManager
    participant Q as JobQueue (Redis)
    participant E as SimulationEngine
    participant L as LLM (Gemini)

    C->>A: POST /api/simulations {brief}
    A->>A: verify_api_key()
    A->>M: launch(request, tenant_id)
    M->>Q: acquire(sim_id)
    M->>L: ScenarioBuilder.build_from_brief()
    L-->>M: ScenarioConfig
    M->>E: engine.run()

    loop Each Round
        E->>L: generate events, agent reactions
        L-->>E: responses
        E->>M: on_progress(round_data)
        M-->>C: SSE: round_complete
    end

    M->>M: export_scenario()
    M-->>C: SSE: completed
    Q-->>M: release slot
```

## Internal Components

### Backend Stack

| Component | Module | Purpose |
|-----------|--------|---------|
| API Server | `api/main.py` | FastAPI endpoints, middleware |
| Auth | `api/auth.py` | API key → tenant mapping |
| Rate Limiter | `api/rate_limiter.py` | Per-tenant request limits |
| Simulation Manager | `api/simulation_manager.py` | Lifecycle, streaming, wargame |
| Database | `api/db.py` | PostgreSQL async layer |
| Job Queue | `api/job_queue.py` | Redis distributed semaphore |
| Middleware | `api/middleware.py` | Request logging, timing |
| Metrics | `api/metrics.py` | Prometheus instrumentation |
| Scenario Builder | `briefing/scenario_builder.py` | Brief → ScenarioConfig via LLM |
| Engine | `core/simulation/engine.py` | Round execution, agent orchestration |
| Opinion Dynamics | `core/simulation/opinion_dynamics_v2.py` | Calibrated force model |
| Agents | `core/agents/` | Elite, Institutional, Citizen |
| Platform | `core/platform/` | Social media simulation |
| Domains | `domains/` | 6 pluggable domain modules |

### Frontend Stack

| Component | Path | Purpose |
|-----------|------|---------|
| Dashboard | `app/page.tsx` | Scenario list, launch new |
| Scenario View | `app/scenario/[id]/` | Results dashboard |
| Replay | `app/scenario/[id]/replay/` | Real-time simulation replay |
| Wargame | `app/wargame/` | Interactive wargame UI |
| Backtest | `app/backtest/` | Financial backtest results |
| Replay Engine | `lib/replay/` | Playback, timeline, animations |
| Schemas | `lib/schemas.ts` | Zod runtime validation |
| API Client | `lib/api.ts` | Typed fetch with validation |

## Data Flow

### Persistence

- **Simulations**: PostgreSQL `simulations` table (or `simulations.json` fallback)
- **LLM Usage**: PostgreSQL `llm_usage` table
- **Checkpoints**: JSON files in `outputs/{scenario}/`
- **Exports**: JSON + Markdown in `outputs/exports/scenario_{id}/`
- **Job Queue**: Redis sets + keys with TTL

### Multi-Tenancy

Each tenant's data is isolated:
- Simulations are tagged with `tenant_id`
- List/get operations filter by tenant
- Export directories can be tenant-scoped
- Usage tracking is per-tenant
