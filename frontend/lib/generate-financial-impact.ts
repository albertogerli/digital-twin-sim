/**
 * REMOVED — client-side financial impact generator.
 *
 * The previous fallback in this module returned hardcoded sector betas and
 * synthetic T+1/T+3/T+7 ratios with `provenance: "client-fallback"`. That
 * was replaced by an explicit empty state in the UI to avoid silently
 * shipping fake numbers to users when the backend scorer is offline.
 *
 * Financial impact data is now strictly backend-only (FinancialImpactScorer
 * → *_financial_impact.json). Callers should check provenance and render an
 * "unavailable" state when the backend has not emitted data.
 *
 * The legacy export below preserves the function name as a no-op so any
 * stale import does not break the build, but it returns an empty array.
 * Remove the function call at the call site, do not call this.
 */

import type { RoundFinancial } from "./types/financial-impact";

/** @deprecated No-op. Use backend-simulated payload only. */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function generateFinancialImpact(...args: unknown[]): RoundFinancial[] {
  return [];
}

export type { RoundFinancial };
