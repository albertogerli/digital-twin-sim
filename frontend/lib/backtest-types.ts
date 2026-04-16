/** Backtest data types — matches backtest_results.json schema */

export interface TickerResult {
  predicted_t1: number;
  actual_t1: number;
}

export interface BacktestScenario {
  scenario: string;
  category: string;
  crisis_scope: string;
  wave: number;
  warning: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  tickers: Record<string, TickerResult>;
  mae_t1: number | null;
  mae_t3: number | null;
  mae_t7: number | null;
  direction_accuracy: string; // "X/Y"
  ftse_impact: number;
  btp_spread: number;
}

export interface AggregateStats {
  totalScenarios: number;
  directionAccuracy: number; // 0-1
  directionCorrect: number;
  directionTotal: number;
  maeT1: number;
  maeT3: number;
  maeT7: number;
  macroCount: number;
  idioCount: number;
  totalTickers: number;
}

export interface CategoryStats {
  category: string;
  count: number;
  dirCorrect: number;
  dirTotal: number;
  maeT1: number;
  avgBtp: number;
  avgFtse: number;
}

export interface TickerStats {
  ticker: string;
  appearances: number;
  dirCorrect: number;
  dirTotal: number;
  maeT1: number;
  avgPredicted: number;
  avgActual: number;
}

export function computeAggregates(data: BacktestScenario[]): AggregateStats {
  let dirCorrect = 0, dirTotal = 0;
  let sumMaeT1 = 0, countT1 = 0;
  let sumMaeT3 = 0, countT3 = 0;
  let sumMaeT7 = 0, countT7 = 0;
  let macroCount = 0, idioCount = 0;
  let totalTickers = 0;

  for (const s of data) {
    const [c, t] = s.direction_accuracy.split("/").map(Number);
    dirCorrect += c;
    dirTotal += t;
    if (s.mae_t1 != null) { sumMaeT1 += s.mae_t1; countT1++; }
    if (s.mae_t3 != null) { sumMaeT3 += s.mae_t3; countT3++; }
    if (s.mae_t7 != null) { sumMaeT7 += s.mae_t7; countT7++; }
    if (s.crisis_scope === "macro_systematic") macroCount++; else idioCount++;
    totalTickers += Object.keys(s.tickers).length;
  }

  return {
    totalScenarios: data.length,
    directionAccuracy: dirTotal > 0 ? dirCorrect / dirTotal : 0,
    directionCorrect: dirCorrect,
    directionTotal: dirTotal,
    maeT1: countT1 > 0 ? sumMaeT1 / countT1 : 0,
    maeT3: countT3 > 0 ? sumMaeT3 / countT3 : 0,
    maeT7: countT7 > 0 ? sumMaeT7 / countT7 : 0,
    macroCount,
    idioCount,
    totalTickers,
  };
}

export function computeCategoryStats(data: BacktestScenario[]): CategoryStats[] {
  const map: Record<string, { count: number; dc: number; dt: number; maes: number[]; btps: number[]; ftses: number[] }> = {};
  for (const s of data) {
    if (!map[s.category]) map[s.category] = { count: 0, dc: 0, dt: 0, maes: [], btps: [], ftses: [] };
    const c = map[s.category];
    c.count++;
    const [dc, dt] = s.direction_accuracy.split("/").map(Number);
    c.dc += dc;
    c.dt += dt;
    if (s.mae_t1 != null) c.maes.push(s.mae_t1);
    c.btps.push(s.btp_spread);
    c.ftses.push(s.ftse_impact);
  }
  return Object.entries(map)
    .map(([category, c]) => ({
      category,
      count: c.count,
      dirCorrect: c.dc,
      dirTotal: c.dt,
      maeT1: c.maes.length > 0 ? c.maes.reduce((a, b) => a + b, 0) / c.maes.length : 0,
      avgBtp: c.btps.reduce((a, b) => a + b, 0) / c.btps.length,
      avgFtse: c.ftses.reduce((a, b) => a + b, 0) / c.ftses.length,
    }))
    .sort((a, b) => b.count - a.count);
}

export function computeTickerStats(data: BacktestScenario[]): TickerStats[] {
  const map: Record<string, { preds: number[]; actuals: number[]; dc: number; dt: number }> = {};
  for (const s of data) {
    for (const [ticker, tr] of Object.entries(s.tickers)) {
      if (!map[ticker]) map[ticker] = { preds: [], actuals: [], dc: 0, dt: 0 };
      map[ticker].preds.push(tr.predicted_t1);
      map[ticker].actuals.push(tr.actual_t1);
      // direction: if predicted < 0 and actual < -0.5 → correct short
      const predictedDown = tr.predicted_t1 < 0;
      const actualDown = tr.actual_t1 < -0.5;
      const actualUp = tr.actual_t1 > 0.5;
      map[ticker].dt++;
      if ((predictedDown && actualDown) || (!predictedDown && actualUp)) {
        map[ticker].dc++;
      }
    }
  }
  return Object.entries(map)
    .map(([ticker, t]) => ({
      ticker,
      appearances: t.preds.length,
      dirCorrect: t.dc,
      dirTotal: t.dt,
      maeT1: t.preds.reduce((sum, p, i) => sum + Math.abs(p - t.actuals[i]), 0) / t.preds.length,
      avgPredicted: t.preds.reduce((a, b) => a + b, 0) / t.preds.length,
      avgActual: t.actuals.reduce((a, b) => a + b, 0) / t.actuals.length,
    }))
    .sort((a, b) => b.appearances - a.appearances);
}

export const WARNING_COLORS: Record<string, string> = {
  LOW: "#22c55e",
  MODERATE: "#f59e0b",
  HIGH: "#f97316",
  CRITICAL: "#ef4444",
};

export const CATEGORY_COLORS: Record<string, string> = {
  "Political Crisis": "#ef4444",
  "Banking / Financial": "#f59e0b",
  "Labor / Industrial": "#3b82f6",
  "Fiscal / EU Standoff": "#8b5cf6",
  "Energy / Environment": "#22c55e",
  "Defense / Geopolitical": "#6366f1",
  "Corporate / Other": "#64748b",
  "Healthcare / COVID": "#ec4899",
  "Immigration / Social": "#14b8a6",
  "Media / Tech": "#06b6d4",
  "Constitutional Reform": "#a855f7",
};
