#!/usr/bin/env npx tsx
/**
 * Generate financial_impact.json for each scenario based on existing replay data.
 *
 * Uses the scenario domain, agent positions, polarization shifts, and event shocks
 * to estimate market impacts using the stock universe sector beta model.
 */

import fs from "fs";
import path from "path";
import { selectRelevantTickers, getBeta } from "../lib/ticker-relevance";
import type { StockUniverse, SelectedTicker } from "../lib/ticker-relevance";
import {
  FIN_SCHEMA_VERSION,
  type FinancialImpactPayload,
  type RoundFinancial,
  type TickerImpact,
  type VolatilityWarning,
} from "../lib/types/financial-impact";

const DATA_DIR = path.join(__dirname, "..", "public", "data");

// Hardcoded fallback when no universe file exists (6 Italian blue-chips, IT regime betas inline)
const DEFAULT_TICKERS_FALLBACK: Array<{ ticker: string; sector: string; name: string; beta: number; crisis_alpha: number; label: string }> = [
  { ticker: "UCG.MI", sector: "banking", name: "UniCredit", beta: 1.85, crisis_alpha: -3.2, label: "Banking" },
  { ticker: "ENI.MI", sector: "energy_fossil", name: "ENI", beta: 1.10, crisis_alpha: -0.8, label: "Energy (O&G)" },
  { ticker: "ENEL.MI", sector: "energy_renewable", name: "Enel", beta: 0.75, crisis_alpha: 0.3, label: "Renewables" },
  { ticker: "STM.MI", sector: "tech", name: "STMicro", beta: 0.85, crisis_alpha: -0.3, label: "Technology" },
  { ticker: "SNAM.MI", sector: "utilities", name: "Snam", beta: 0.60, crisis_alpha: 0.1, label: "Utilities" },
  { ticker: "LDO.MI", sector: "defense", name: "Leonardo", beta: 0.90, crisis_alpha: 0.5, label: "Defence" },
];

function computeFinancialImpact(
  domain: string,
  rounds: any[],
  polarizationData: any[],
  universe?: StockUniverse | null,
): RoundFinancial[] {
  // Resolve tickers and beta lookup based on whether universe is available
  let tickerDefs: Array<{ ticker: string; sector: string; name: string }>;
  let betaRegime: string;
  let lookupBeta: (sector: string) => { political_beta: number; crisis_alpha: number; label: string };

  if (universe) {
    const relevantUniv = selectRelevantTickers(universe, domain, [], "", []);
    betaRegime = relevantUniv.betaRegime;

    tickerDefs = relevantUniv.tickers.map((stock: SelectedTicker) => ({
      ticker: stock.ticker,
      sector: stock.sector,
      name: stock.name,
    }));

    lookupBeta = (sector: string) => {
      const b = getBeta(universe, sector, betaRegime);
      const sectorEntry = universe.sectors[sector];
      return {
        political_beta: b.political_beta,
        crisis_alpha: b.crisis_alpha,
        label: sectorEntry?.label ?? sector,
      };
    };
  } else {
    // Fallback: use hardcoded Italian tickers
    betaRegime = "IT";
    tickerDefs = DEFAULT_TICKERS_FALLBACK.map((t) => ({
      ticker: t.ticker,
      sector: t.sector,
      name: t.name,
    }));

    const fallbackMap = new Map(DEFAULT_TICKERS_FALLBACK.map((t) => [t.sector, t]));
    lookupBeta = (sector: string) => {
      const fb = fallbackMap.get(sector);
      return {
        political_beta: fb?.beta ?? 1.0,
        crisis_alpha: fb?.crisis_alpha ?? 0,
        label: fb?.label ?? sector,
      };
    };
  }

  const results: RoundFinancial[] = [];

  for (const round of rounds) {
    const r = round.round;
    const rwe = round.realWorldEffects ?? {};
    const overview = rwe.overview ?? {};
    const opinion = rwe.opinion ?? {};
    const event = round.event ?? {};

    const tension = (overview.tension_index ?? 30) / 100;
    const stability = (overview.stability_index ?? 70) / 100;
    const shockMag = event.shock_magnitude ?? 0.3;
    const shockDir = event.shock_direction ?? -1;

    // Polarization delta
    const polPoint = polarizationData.find((p: any) => p.round === r);
    const prevPol = polarizationData.find((p: any) => p.round === r - 1);
    const polDelta = polPoint && prevPol
      ? polPoint.polarization - prevPol.polarization
      : 0;

    // Contagion risk from tension + shock
    const contagionRisk = Math.min(1, tension * 0.5 + shockMag * 0.3 + Math.abs(polDelta) * 0.2);

    // Volatility warning
    let warning: VolatilityWarning = "LOW";
    if (contagionRisk > 0.7) warning = "CRITICAL";
    else if (contagionRisk > 0.5) warning = "HIGH";
    else if (contagionRisk > 0.3) warning = "MODERATE";

    // Base market move (%)
    const baseMove = shockMag * shockDir * (1 - stability) * 3.0;

    // FTSE MIB estimate
    const ftseMib = baseMove * (1 + tension * 0.5);

    // BTP spread
    const btpBps = Math.round(Math.abs(baseMove) * 15 * (1 + contagionRisk));

    // Compute per-ticker impacts
    const tickerImpacts: TickerImpact[] = tickerDefs.map((t) => {
      const betaData = lookupBeta(t.sector);
      const beta = betaData.political_beta;
      const crisisAlpha = betaData.crisis_alpha;
      const label = betaData.label;

      // Direction: high-beta sectors go short in negative shocks, low-beta go long
      const direction: "short" | "long" = baseMove < 0
        ? (beta > 1.0 ? "short" : "long")
        : (beta > 1.0 ? "long" : "short");

      // T+1: immediate panic/euphoria scaled by beta
      const t1 = +(baseMove * beta + crisisAlpha * contagionRisk * 0.3).toFixed(2);
      // T+3: partial mean-reversion (60% of T+1)
      const t3 = +(t1 * 0.6).toFixed(2);
      // T+7: structural outcome (40% of T+1 + polarization effect)
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
        source: `script:${betaRegime}`,
      };
    });

    // Sort: shorts first (worst t1), then longs (best t1)
    tickerImpacts.sort((a, b) => {
      if (a.direction !== b.direction) return a.direction === "short" ? -1 : 1;
      return a.direction === "short" ? a.t1_pct - b.t1_pct : b.t1_pct - a.t1_pct;
    });

    // Generate headline
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
      scope_disclaimer: "Script-side heuristic — scope inferred from contagion risk only.",
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

// ── Main ──────────────────────────────────────────────────────────────────

function main() {
  // Load stock universe if available
  const universePath = path.join(DATA_DIR, "stock_universe.json");
  let universe: StockUniverse | null = null;
  if (fs.existsSync(universePath)) {
    universe = JSON.parse(fs.readFileSync(universePath, "utf8"));
  }

  const scenarioDirs = fs.readdirSync(DATA_DIR).filter(d =>
    d.startsWith("scenario_") && fs.statSync(path.join(DATA_DIR, d)).isDirectory()
  );

  console.log(`Found ${scenarioDirs.length} scenarios`);
  if (universe) {
    console.log(`  Using stock universe v${universe.version} (${universe.stocks.length} stocks)`);
  } else {
    console.log(`  No stock_universe.json found, using fallback Italian tickers`);
  }

  for (const dir of scenarioDirs) {
    const base = path.join(DATA_DIR, dir);
    const metaPath = path.join(base, "metadata.json");
    const polPath = path.join(base, "polarization.json");

    if (!fs.existsSync(metaPath)) continue;

    const metadata = JSON.parse(fs.readFileSync(metaPath, "utf8"));
    const polarization = fs.existsSync(polPath)
      ? JSON.parse(fs.readFileSync(polPath, "utf8"))
      : [];

    // Load all replay rounds
    const rounds: any[] = [];
    for (let r = 1; r <= metadata.num_rounds; r++) {
      const roundPath = path.join(base, `replay_round_${r}.json`);
      if (fs.existsSync(roundPath)) {
        rounds.push(JSON.parse(fs.readFileSync(roundPath, "utf8")));
      }
    }

    if (rounds.length === 0) continue;

    const financial = computeFinancialImpact(
      metadata.domain,
      rounds,
      polarization,
      universe,
    );

    const payload: FinancialImpactPayload = {
      schema_version: FIN_SCHEMA_VERSION,
      scenario: metadata.scenario_id ?? dir,
      domain: metadata.domain,
      num_rounds: financial.length,
      provenance: "client-fallback",
      rounds: financial,
    };

    const outPath = path.join(base, "financial_impact.json");
    fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
    console.log(`  ${dir}: ${financial.length} rounds, domain=${metadata.domain} [client-fallback]`);
  }

  console.log("Done.");
}

main();
