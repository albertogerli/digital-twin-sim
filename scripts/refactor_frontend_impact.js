const fs = require('fs');

let content = fs.readFileSync('frontend/lib/generate-financial-impact.ts', 'utf8');

const importStatement = `import { TickerRelevanceScorer, RelevantUniverse } from "./ticker-relevance";\nimport universeData from "../../shared/stock_universe.json";\n\n`;

// Remove DOMAIN_TICKERS
content = content.replace(/const DOMAIN_TICKERS: Record<string, TickerDef\[\]> = \{[\s\S]*?\};\n\n/g, '');

// Remove DEFAULT_TICKERS
content = content.replace(/const DEFAULT_TICKERS: TickerDef\[\] = \[[\s\S]*?\];\n\n/g, '');

// Update SECTOR_BETAS fetching (or remove it and use universe)
content = content.replace(/const SECTOR_BETAS: Record<string, SectorBeta> = \{[\s\S]*?\};\n\n/g, '');

// Need to update generateFinancialImpact
const newFunction = `export function generateFinancialImpact(
  domain: string,
  rounds: any[],
  polarizationData: any[],
  geography: string = "IT",
  entities: string[] = [],
  tags: string[] = []
): RoundFinancial[] {
  const scorer = new TickerRelevanceScorer();
  const relevantUniv = scorer.select(domain, geography, entities, tags);
  const tickerDefs = relevantUniv.tickers.map(t => {
    const stock = (universeData.stocks as any[]).find(s => s.ticker === t);
    return { ticker: t, sector: stock?.sector || "unknown", name: stock?.name || t };
  });
  
  const getBetaData = (sector: string, regime: string) => {
    const sData = (universeData.sectors as any)[sector] || {};
    const betas = sData.betas || {};
    const target = betas[regime] || betas["default"] || {};
    return {
      political_beta: target.political_beta || 1.0,
      crisis_alpha: target.crisis_alpha || 0.0,
      label: sData.label || sector
    };
  };

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

    // Polarization delta
    const polPoint = polarizationData.find((p: any) => p.round === r);
    const prevPol = polarizationData.find((p: any) => p.round === r - 1);
    const polDelta = polPoint && prevPol ? polPoint.polarization - prevPol.polarization : 0;

    const contagionRisk = Math.min(1, tension * 0.5 + shockMag * 0.3 + Math.abs(polDelta) * 0.2);

    let warning: RoundFinancial["volatility_warning"] = "LOW";
    if (contagionRisk > 0.7) warning = "CRITICAL";
    else if (contagionRisk > 0.5) warning = "HIGH";
    else if (contagionRisk > 0.3) warning = "MODERATE";

    const baseMove = shockMag * shockDir * (1 - stability) * 3.0;
    const ftseMib = baseMove * (1 + tension * 0.5);
    const btpBps = Math.round(Math.abs(baseMove) * 15 * (1 + contagionRisk));

    const tickerImpacts: TickerImpact[] = tickerDefs.map((t) => {
      const betaData = getBetaData(t.sector, relevantUniv.beta_regime);
      const beta = betaData.political_beta;
      const crisisAlpha = betaData.crisis_alpha;
      const label = betaData.label;

      const direction: "short" | "long" = baseMove < 0
        ? (beta > 1.0 ? "short" : "long")
        : (beta > 1.0 ? "long" : "short");

      const t1 = +(baseMove * beta + crisisAlpha * contagionRisk * 0.3).toFixed(2);
      const t3 = +(t1 * 0.6).toFixed(2);
      const t7 = +(t1 * 0.4 + polDelta * beta * 2).toFixed(2);

      return { ticker: t.ticker, name: t.name, sector: t.sector, sectorLabel: label, direction, beta, t1_pct: t1, t3_pct: t3, t7_pct: t7 };
    });

    tickerImpacts.sort((a, b) => {
      if (a.direction !== b.direction) return a.direction === "short" ? -1 : 1;
      return a.direction === "short" ? a.t1_pct - b.t1_pct : b.t1_pct - a.t1_pct;
    });

    const worstTicker = tickerImpacts.find(t => t.direction === "short");
    const bestTicker = tickerImpacts.find(t => t.direction === "long");
    let headline = "";
    if (warning === "CRITICAL") {
      headline = \`FLASH: \${worstTicker?.sectorLabel ?? "High-beta"} sotto pressione (β=\${worstTicker?.beta.toFixed(2)}), contagion risk \${(contagionRisk * 100).toFixed(0)}%. Pair: SHORT \${worstTicker?.name ?? "—"} / LONG \${bestTicker?.name ?? "—"}.\`;
    } else if (warning === "HIGH") {
      headline = \`Volatilità elevata su \${worstTicker?.sectorLabel ?? "settori ciclici"}. Base est. \${ftseMib > 0 ? "+" : ""}\${ftseMib.toFixed(1)}%. Monitor spread +\${btpBps}bps.\`;
    } else if (warning === "MODERATE") {
      headline = \`Movimenti contenuti. \${worstTicker?.sectorLabel ?? "Settori esposti"} in lieve tensione. Beta-weighted pair trade attivo.\`;
    } else {
      headline = \`Mercato stabile. Impatto limitato sui settori monitorati. Nessun segnale di contagio.\`;
    }

    results.push({
      round: r,
      volatility_warning: warning,
      ftse_mib_impact_pct: +ftseMib.toFixed(2),
      btp_spread_bps: btpBps,
      contagion_risk: +contagionRisk.toFixed(3),
      tickers: tickerImpacts,
      headline,
    });
  }

  return results;
}
`;

content = content.replace(/export function generateFinancialImpact\([\s\S]*?\];\n\n}/, newFunction);

fs.writeFileSync('frontend/lib/generate-financial-impact.ts', importStatement + content);
