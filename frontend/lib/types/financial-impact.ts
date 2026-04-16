/**
 * Financial Impact schema — mirrors core/orchestrator/financial_impact.py
 *
 * This file is the TypeScript side of the Python↔TS bridge. When the Python
 * schema changes, bump FIN_SCHEMA_VERSION here AND in financial_impact.py.
 *
 * Backend-simulated payloads ship with `provenance: "backend-simulated"`.
 * Client-computed fallbacks ship with `provenance: "client-fallback"` and
 * should be surfaced in the UI via a visible badge (see F1.3).
 */

export const FIN_SCHEMA_VERSION = "2.0.0";

export type Provenance = "backend-simulated" | "client-fallback";

export interface TickerImpact {
  ticker: string;
  name: string;                  // enriched from stock universe
  sector: string;                // canonical key (e.g. "banking")
  sectorLabel: string;           // human-readable (e.g. "Banking")
  direction: "short" | "long";
  t1_pct: number;
  t3_pct: number;
  t7_pct: number;
  beta: number;
  confidence: number;            // 0-1
  source: string;                // "agent:X", "pair_trade:Y", "universe:Z"
}

export interface PairTrade {
  topic: string;
  rationale: string;
  short_leg: TickerImpact[];
  long_leg: TickerImpact[];
}

export type VolatilityWarning = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";

export interface RoundFinancial {
  round: number;
  timeline_label?: string;

  // Schema / provenance
  schema_version: string;
  provenance: Provenance;

  // Core signals
  volatility_warning: VolatilityWarning;
  headline: string;

  // Crisis scope
  crisis_scope: "macro_systematic" | "micro_idiosyncratic";
  scope_confidence: number;
  scope_disclaimer: string;

  // Tickers + pair trades
  tickers: TickerImpact[];
  pair_trades: PairTrade[];

  // Aggregate market
  ftse_mib_impact_pct: number;
  btp_spread_bps: number;

  // Crisis metadata
  crisis_wave: number;
  contagion_risk: number;
  engagement_score: number;
  institutional_actors_count: number;
}

export interface FinancialImpactPayload {
  schema_version: string;
  scenario: string;
  domain: string;
  num_rounds: number;
  provenance: Provenance;
  rounds: RoundFinancial[];
}

/** Type guard: verify a loaded JSON matches the current schema. */
export function isCompatible(payload: unknown): payload is FinancialImpactPayload {
  if (!payload || typeof payload !== "object") return false;
  const p = payload as Record<string, unknown>;
  return (
    typeof p.schema_version === "string" &&
    p.schema_version.split(".")[0] === FIN_SCHEMA_VERSION.split(".")[0] &&
    Array.isArray(p.rounds)
  );
}
