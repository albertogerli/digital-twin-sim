"""PostgreSQL database layer for DigitalTwinSim.

Falls back to JSON file persistence when DATABASE_URL is not set.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Schema DDL
# Wargame fields are added via ALTER TABLE so existing prod databases get
# upgraded in-place on next deploy without manual migration.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS simulations (
    id VARCHAR(8) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    brief TEXT,
    scenario_name TEXT,
    scenario_id TEXT,
    domain TEXT,
    current_round INT DEFAULT 0,
    total_rounds INT DEFAULT 0,
    cost REAL DEFAULT 0,
    agents_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error TEXT,
    wargame_mode BOOLEAN DEFAULT FALSE,
    player_role TEXT,
    wargame_sitrep JSONB
);

ALTER TABLE simulations ADD COLUMN IF NOT EXISTS wargame_mode BOOLEAN DEFAULT FALSE;
ALTER TABLE simulations ADD COLUMN IF NOT EXISTS player_role TEXT;
ALTER TABLE simulations ADD COLUMN IF NOT EXISTS wargame_sitrep JSONB;

CREATE INDEX IF NOT EXISTS idx_sims_tenant ON simulations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sims_status ON simulations(status);
CREATE INDEX IF NOT EXISTS idx_sims_wargame ON simulations(wargame_mode) WHERE wargame_mode = TRUE;

CREATE TABLE IF NOT EXISTS llm_usage (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(64),
    sim_id VARCHAR(8),
    component VARCHAR(64),
    model VARCHAR(64),
    input_tokens INT,
    output_tokens INT,
    cost_usd REAL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_tenant ON llm_usage(tenant_id);
CREATE INDEX IF NOT EXISTS idx_usage_sim ON llm_usage(sim_id);
"""

_pool = None


async def get_pool():
    """Get or create asyncpg connection pool."""
    global _pool
    if _pool is None and DATABASE_URL:
        try:
            import asyncpg
            _pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            async with _pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            logger.info("PostgreSQL connected and schema initialized")
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable, using JSON fallback: {e}")
            _pool = None
    return _pool


def is_available() -> bool:
    """Check if PostgreSQL is configured."""
    return bool(DATABASE_URL)


async def check_health() -> bool:
    """Check PostgreSQL connectivity."""
    pool = await get_pool()
    if not pool:
        return False
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False


async def upsert_simulation(data: dict):
    """Insert or update a simulation record."""
    pool = await get_pool()
    if not pool:
        return
    sitrep = data.get("wargame_sitrep")
    sitrep_json = json.dumps(sitrep) if sitrep is not None else None
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO simulations (id, tenant_id, status, brief, scenario_name, scenario_id,
                domain, current_round, total_rounds, cost, agents_count, created_at, completed_at, error,
                wargame_mode, player_role, wargame_sitrep)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                scenario_name = EXCLUDED.scenario_name,
                scenario_id = EXCLUDED.scenario_id,
                domain = EXCLUDED.domain,
                current_round = EXCLUDED.current_round,
                total_rounds = EXCLUDED.total_rounds,
                cost = EXCLUDED.cost,
                agents_count = EXCLUDED.agents_count,
                completed_at = EXCLUDED.completed_at,
                error = EXCLUDED.error,
                wargame_sitrep = COALESCE(EXCLUDED.wargame_sitrep, simulations.wargame_sitrep)
            """,
            data.get("id"),
            data.get("tenant_id", "default"),
            data.get("status", "queued"),
            data.get("brief"),
            data.get("scenario_name"),
            data.get("scenario_id"),
            data.get("domain"),
            data.get("current_round", 0),
            data.get("total_rounds", 0),
            data.get("cost", 0),
            data.get("agents_count", 0),
            data.get("created_at"),
            data.get("completed_at"),
            data.get("error"),
            bool(data.get("wargame_mode", False)),
            data.get("player_role"),
            sitrep_json,
        )


async def list_simulations(tenant_id: Optional[str] = None) -> list[dict]:
    """List simulations, optionally filtered by tenant."""
    pool = await get_pool()
    if not pool:
        return []
    async with pool.acquire() as conn:
        if tenant_id and tenant_id != "default":
            rows = await conn.fetch(
                "SELECT * FROM simulations WHERE tenant_id = $1 ORDER BY created_at DESC",
                tenant_id,
            )
        else:
            rows = await conn.fetch("SELECT * FROM simulations ORDER BY created_at DESC")
        return [dict(row) for row in rows]


async def mark_running_as_failed():
    """Mark non-wargame running simulations as failed (called on startup).

    Wargame sims that were paused at awaiting_player are NOT touched: their
    SITREP is preserved so the UI can render the last state and tell the
    player to start a new sim (the live RoundManager / LLM client cannot be
    rebuilt across restarts in C-light)."""
    pool = await get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE simulations
            SET status = 'failed',
                error = 'Server restarted during simulation'
            WHERE status IN ('running', 'analyzing', 'configuring', 'exporting')
              AND wargame_mode = FALSE
            """
        )


async def get_simulation(sim_id: str) -> Optional[dict]:
    """Fetch a single simulation by id."""
    pool = await get_pool()
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM simulations WHERE id = $1", sim_id)
        return dict(row) if row else None


async def delete_simulation(sim_id: str):
    """Remove a simulation row."""
    pool = await get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM simulations WHERE id = $1", sim_id)


async def record_llm_usage(
    tenant_id: str, sim_id: str, component: str, model: str,
    input_tokens: int, output_tokens: int, cost_usd: float,
):
    """Record an LLM API call for cost tracking."""
    pool = await get_pool()
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage (tenant_id, sim_id, component, model, input_tokens, output_tokens, cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            tenant_id, sim_id, component, model, input_tokens, output_tokens, cost_usd,
        )


async def get_usage(tenant_id: Optional[str] = None, since: Optional[str] = None) -> list[dict]:
    """Get LLM usage records."""
    pool = await get_pool()
    if not pool:
        return []
    query = "SELECT * FROM llm_usage WHERE 1=1"
    params = []
    idx = 1
    if tenant_id and tenant_id != "default":
        query += f" AND tenant_id = ${idx}"
        params.append(tenant_id)
        idx += 1
    if since:
        query += f" AND recorded_at >= ${idx}"
        params.append(since)
        idx += 1
    query += " ORDER BY recorded_at DESC LIMIT 1000"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]
