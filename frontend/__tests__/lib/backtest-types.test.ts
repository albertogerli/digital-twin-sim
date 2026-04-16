import { describe, it, expect } from "vitest";
import {
  computeAggregates,
  computeCategoryStats,
  type BacktestScenario,
} from "@/lib/backtest-types";

const SAMPLE_DATA: BacktestScenario[] = [
  {
    scenario: "Scenario A",
    category: "corporate",
    crisis_scope: "macro_systematic",
    wave: 2,
    warning: "HIGH",
    tickers: {
      AAPL: { predicted_t1: -2.5, actual_t1: -3.0 },
      MSFT: { predicted_t1: -1.0, actual_t1: 0.5 },
    },
    mae_t1: 1.5,
    mae_t3: 2.0,
    mae_t7: 3.0,
    direction_accuracy: "1/2",
    ftse_impact: -1.2,
    btp_spread: 5.0,
  },
  {
    scenario: "Scenario B",
    category: "political",
    crisis_scope: "idiosyncratic",
    wave: 1,
    warning: "LOW",
    tickers: {
      ENI: { predicted_t1: 1.0, actual_t1: 1.5 },
    },
    mae_t1: 0.5,
    mae_t3: null,
    mae_t7: null,
    direction_accuracy: "1/1",
    ftse_impact: 0.3,
    btp_spread: -2.0,
  },
];

describe("computeAggregates", () => {
  it("computes correct totals", () => {
    const agg = computeAggregates(SAMPLE_DATA);
    expect(agg.totalScenarios).toBe(2);
    expect(agg.directionCorrect).toBe(2);
    expect(agg.directionTotal).toBe(3);
    expect(agg.directionAccuracy).toBeCloseTo(2 / 3);
    expect(agg.macroCount).toBe(1);
    expect(agg.idioCount).toBe(1);
    expect(agg.totalTickers).toBe(3);
  });

  it("averages MAE correctly", () => {
    const agg = computeAggregates(SAMPLE_DATA);
    expect(agg.maeT1).toBeCloseTo(1.0); // (1.5 + 0.5) / 2
    expect(agg.maeT3).toBeCloseTo(2.0); // only one non-null
  });

  it("handles empty data", () => {
    const agg = computeAggregates([]);
    expect(agg.totalScenarios).toBe(0);
    expect(agg.directionAccuracy).toBe(0);
  });
});

describe("computeCategoryStats", () => {
  it("groups by category", () => {
    const stats = computeCategoryStats(SAMPLE_DATA);
    expect(stats.length).toBe(2);
    const corporate = stats.find((s) => s.category === "corporate");
    expect(corporate).toBeDefined();
    expect(corporate!.count).toBe(1);
  });
});
