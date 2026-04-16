/**
 * Client-side financial impact generator (FALLBACK ONLY).
 *
 * This module is used when the backend has not emitted a `*_financial_impact.json`
 * (e.g. legacy scenarios, offline demos). Outputs ship with
 * `provenance: "client-fallback"` so the UI can flag them.
 *
 * Prefer backend-simulated data from FinancialImpactScorer whenever available.
 */

import {
  FIN_SCHEMA_VERSION,
  type RoundFinancial,
  type TickerImpact,
  type VolatilityWarning,
} from "./types/financial-impact";

interface SectorBeta {
  label: string;
  political_beta: number;
  crisis_alpha: number;
}

const SECTOR_BETAS: Record<string, SectorBeta> = {
  banking: { label: "Banking", political_beta: 1.85, crisis_alpha: -3.2 },
  insurance: { label: "Insurance", political_beta: 1.45, crisis_alpha: -1.8 },
  automotive: { label: "Automotive", political_beta: 1.30, crisis_alpha: -1.5 },
  energy_fossil: { label: "Energy (O&G)", political_beta: 1.10, crisis_alpha: -0.8 },
  energy_renewable: { label: "Renewables", political_beta: 0.75, crisis_alpha: 0.3 },
  utilities: { label: "Utilities", political_beta: 0.60, crisis_alpha: 0.1 },
  defense: { label: "Defence", political_beta: 0.90, crisis_alpha: 0.5 },
  telecom: { label: "Telecom", political_beta: 1.15, crisis_alpha: -1.2 },
  tech: { label: "Technology", political_beta: 0.85, crisis_alpha: -0.3 },
  healthcare: { label: "Healthcare", political_beta: 0.55, crisis_alpha: 0.2 },
  luxury: { label: "Luxury", political_beta: 0.80, crisis_alpha: -0.5 },
  real_estate: { label: "Real Estate", political_beta: 1.40, crisis_alpha: -2.0 },
  infrastructure: { label: "Infrastructure", political_beta: 0.70, crisis_alpha: 0.0 },
  food_consumer: { label: "Food & Consumer", political_beta: 0.50, crisis_alpha: 0.1 },
};

interface TickerDef { ticker: string; sector: string; name: string }

const DOMAIN_TICKERS: Record<string, TickerDef[]> = {
  financial: [
    { ticker: "UCG.MI", sector: "banking", name: "UniCredit" },
    { ticker: "ISP.MI", sector: "banking", name: "Intesa Sanpaolo" },
    { ticker: "BMPS.MI", sector: "banking", name: "Monte dei Paschi" },
    { ticker: "G.MI", sector: "insurance", name: "Generali" },
    { ticker: "FBK.MI", sector: "banking", name: "FinecoBank" },
    { ticker: "MB.MI", sector: "banking", name: "Mediobanca" },
  ],
  commercial: [
    { ticker: "AAPL", sector: "tech", name: "Apple" },
    { ticker: "STM.MI", sector: "tech", name: "STMicro" },
    { ticker: "RACE.MI", sector: "automotive", name: "Ferrari" },
    { ticker: "MONC.MI", sector: "luxury", name: "Moncler" },
    { ticker: "CPR.MI", sector: "food_consumer", name: "Campari" },
    { ticker: "AMP.MI", sector: "healthcare", name: "Amplifon" },
  ],
  corporate: [
    { ticker: "STLAM.MI", sector: "automotive", name: "Stellantis" },
    { ticker: "ENI.MI", sector: "energy_fossil", name: "ENI" },
    { ticker: "ENEL.MI", sector: "energy_renewable", name: "Enel" },
    { ticker: "TLIT.MI", sector: "telecom", name: "TIM" },
    { ticker: "LDO.MI", sector: "defense", name: "Leonardo" },
    { ticker: "REC.MI", sector: "healthcare", name: "Recordati" },
  ],
  political: [
    { ticker: "UCG.MI", sector: "banking", name: "UniCredit" },
    { ticker: "ISP.MI", sector: "banking", name: "Intesa Sanpaolo" },
    { ticker: "ENI.MI", sector: "energy_fossil", name: "ENI" },
    { ticker: "ENEL.MI", sector: "energy_renewable", name: "Enel" },
    { ticker: "LDO.MI", sector: "defense", name: "Leonardo" },
    { ticker: "SNAM.MI", sector: "utilities", name: "Snam" },
  ],
  public_health: [
    { ticker: "REC.MI", sector: "healthcare", name: "Recordati" },
    { ticker: "DIA.MI", sector: "healthcare", name: "DiaSorin" },
    { ticker: "AMP.MI", sector: "healthcare", name: "Amplifon" },
    { ticker: "G.MI", sector: "insurance", name: "Generali" },
    { ticker: "UNI.MI", sector: "insurance", name: "Unipol" },
    { ticker: "CPR.MI", sector: "food_consumer", name: "Campari" },
  ],
  sport: [
    { ticker: "RACE.MI", sector: "automotive", name: "Ferrari" },
    { ticker: "MONC.MI", sector: "luxury", name: "Moncler" },
    { ticker: "MFEB.MI", sector: "tech", name: "MFE/Mediaset" },
    { ticker: "CPR.MI", sector: "food_consumer", name: "Campari" },
    { ticker: "BC.MI", sector: "luxury", name: "Brunello Cucinelli" },
    { ticker: "SFER.MI", sector: "luxury", name: "Ferragamo" },
  ],
};

const DEFAULT_TICKERS: TickerDef[] = [
  { ticker: "UCG.MI", sector: "banking", name: "UniCredit" },
  { ticker: "ENI.MI", sector: "energy_fossil", name: "ENI" },
  { ticker: "ENEL.MI", sector: "energy_renewable", name: "Enel" },
  { ticker: "STM.MI", sector: "tech", name: "STMicro" },
  { ticker: "SNAM.MI", sector: "utilities", name: "Snam" },
  { ticker: "LDO.MI", sector: "defense", name: "Leonardo" },
];

/**
 * Generate financial impact data from replay rounds + polarization.
 * Returns payload marked with provenance: "client-fallback".
 */
export function generateFinancialImpact(
  domain: string,
  rounds: any[],
  polarizationData: any[],
): RoundFinancial[] {
  const tickers = DOMAIN_TICKERS[domain] ?? DEFAULT_TICKERS;
  const results: RoundFinancial[] = [];

  for (const round of rounds) {
    const r = round.round;
    const rwe = round.realWorldEffects ?? {};
    const overview = rwe.overview ?? {};
    const event = round.event ?? {};

    const tension = (overview.tension_index ?? 30) / 100;
    const stability = (overview.stability_index ?? 70) / 100;
    const shockMag = event.shock_magnitude ?? 0.3;
    const shockDir = event.shock_direction ?? -1;

    const polPoint = polarizationData.find((p: any) => p.round === r);
    const prevPol = polarizationData.find((p: any) => p.round === r - 1);
    const polDelta = polPoint && prevPol ? polPoint.polarization - prevPol.polarization : 0;

    const contagionRisk = Math.min(1, tension * 0.5 + shockMag * 0.3 + Math.abs(polDelta) * 0.2);

    let warning: VolatilityWarning = "LOW";
    if (contagionRisk > 0.7) warning = "CRITICAL";
    else if (contagionRisk > 0.5) warning = "HIGH";
    else if (contagionRisk > 0.3) warning = "MODERATE";

    const baseMove = shockMag * shockDir * (1 - stability) * 3.0;
    const ftseMib = baseMove * (1 + tension * 0.5);
    const btpBps = Math.round(Math.abs(baseMove) * 15 * (1 + contagionRisk));

    const tickerImpacts: TickerImpact[] = tickers.map((t) => {
      const beta = SECTOR_BETAS[t.sector]?.political_beta ?? 1.0;
      const crisisAlpha = SECTOR_BETAS[t.sector]?.crisis_alpha ?? 0;
      const label = SECTOR_BETAS[t.sector]?.label ?? t.sector;

      const direction: "short" | "long" = baseMove < 0
        ? (beta > 1.0 ? "short" : "long")
        : (beta > 1.0 ? "long" : "short");

      const t1 = +(baseMove * beta + crisisAlpha * contagionRisk * 0.3).toFixed(2);
      const t3 = +(t1 * 0.6).toFixed(2);
      const t7 = +(t1 * 0.4 + polDelta * beta * 2).toFixed(2);

      return {
        ticker: t.ticker,
        name: t.name,
        sector: t.sector,
        sectorLabel: label,
        direction,
        beta,
        t1_pct: t1,
        t3_pct: t3,
        t7_pct: t7,
        confidence: 0.5,
        source: `fallback:${domain}`,
      };
    });

    tickerImpacts.sort((a, b) => {
      if (a.direction !== b.direction) return a.direction === "short" ? -1 : 1;
      return a.direction === "short" ? a.t1_pct - b.t1_pct : b.t1_pct - a.t1_pct;
    });

    const worstTicker = tickerImpacts.find(t => t.direction === "short");
    const bestTicker = tickerImpacts.find(t => t.direction === "long");
    let headline = "";
    if (warning === "CRITICAL") {
      headline = `FLASH: ${worstTicker?.sectorLabel ?? "High-beta"} sotto pressione (β=${worstTicker?.beta.toFixed(2)}), contagion risk ${(contagionRisk * 100).toFixed(0)}%. Pair: SHORT ${worstTicker?.name ?? "—"} / LONG ${bestTicker?.name ?? "—"}.`;
    } else if (warning === "HIGH") {
      headline = `Volatilità elevata su ${worstTicker?.sectorLabel ?? "settori ciclici"}. FTSE MIB stimato ${ftseMib > 0 ? "+" : ""}${ftseMib.toFixed(1)}%. Monitor BTP spread +${btpBps}bps.`;
    } else if (warning === "MODERATE") {
      headline = `Movimenti contenuti. ${worstTicker?.sectorLabel ?? "Settori esposti"} in lieve tensione. Beta-weighted pair trade attivo.`;
    } else {
      headline = `Mercato stabile. Impatto limitato sui settori monitorati. Nessun segnale di contagio.`;
    }

    results.push({
      round: r,
      schema_version: FIN_SCHEMA_VERSION,
      provenance: "client-fallback",
      volatility_warning: warning,
      headline,
      crisis_scope: contagionRisk > 0.5 ? "macro_systematic" : "micro_idiosyncratic",
      scope_confidence: 0.5,
      scope_disclaimer: "Client-side heuristic fallback — scope inferred from contagion risk only.",
      tickers: tickerImpacts,
      pair_trades: [],
      ftse_mib_impact_pct: +ftseMib.toFixed(2),
      btp_spread_bps: btpBps,
      crisis_wave: Math.max(1, Math.round(shockMag * 3)),
      contagion_risk: +contagionRisk.toFixed(3),
      engagement_score: 0,
      institutional_actors_count: 0,
    });
  }

  return results;
}

export type { TickerImpact, RoundFinancial };
