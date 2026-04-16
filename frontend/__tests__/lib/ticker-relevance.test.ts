import { describe, it, expect, beforeAll } from "vitest";
import fs from "fs";
import path from "path";
import {
  getBeta,
  resolveOrg,
  selectRelevantTickers,
  type StockUniverse,
} from "../../lib/ticker-relevance";

let universe: StockUniverse;

beforeAll(() => {
  const raw = fs.readFileSync(
    path.join(__dirname, "../../public/data/stock_universe.json"),
    "utf8"
  );
  universe = JSON.parse(raw);
});

describe("getBeta", () => {
  it("returns IT regime for banking", () => {
    const beta = getBeta(universe, "banking", "IT");
    expect(beta.political_beta).toBe(1.85);
    expect(beta.crisis_alpha).toBe(-3.2);
  });

  it("returns US regime for tech", () => {
    const beta = getBeta(universe, "tech", "US");
    expect(beta.political_beta).toBe(1.0);
  });

  it("falls back to default for unknown regime", () => {
    const beta = getBeta(universe, "banking", "NONEXISTENT");
    expect(beta.political_beta).toBe(1.2);
  });
});

describe("resolveOrg", () => {
  it("resolves Apple to AAPL", () => {
    expect(resolveOrg(universe, "Apple")).toContain("AAPL");
  });

  it("resolves case-insensitively", () => {
    expect(resolveOrg(universe, "APPLE")).toContain("AAPL");
  });

  it("resolves UniCredit to UCG.MI", () => {
    expect(resolveOrg(universe, "unicredit")).toContain("UCG.MI");
  });

  it("returns empty for unknown org", () => {
    expect(resolveOrg(universe, "nonexistentcorp")).toEqual([]);
  });
});

describe("selectRelevantTickers", () => {
  it("selects AAPL and Samsung for commercial domain with entities", () => {
    const result = selectRelevantTickers(
      universe, "commercial", ["Apple", "Samsung"], "US", ["smartphone"]
    );
    const tickers = result.tickers.map((t) => t.ticker);
    expect(tickers).toContain("AAPL");
    expect(tickers).toContain("005930.KS");
    expect(result.betaRegime).toBe("US");
  });

  it("caps at 30 tickers", () => {
    const result = selectRelevantTickers(
      universe, "commercial", [], "US", ["ai", "cloud"]
    );
    expect(result.tickers.length).toBeLessThanOrEqual(30);
  });

  it("ensures min 3 sectors", () => {
    const result = selectRelevantTickers(
      universe, "political", [], "", []
    );
    const sectors = new Set(result.tickers.map((t) => t.sector));
    expect(sectors.size).toBeGreaterThanOrEqual(3);
  });

  it("includes 1-3 indices", () => {
    const result = selectRelevantTickers(
      universe, "financial", [], "US", []
    );
    expect(result.indices.length).toBeGreaterThanOrEqual(1);
    expect(result.indices.length).toBeLessThanOrEqual(3);
  });

  it("detects EM beta regime for Brazil", () => {
    const result = selectRelevantTickers(
      universe, "financial", [], "Brazil", []
    );
    expect(result.betaRegime).toBe("EM");
  });

  it("detects IT beta regime for Italy", () => {
    const result = selectRelevantTickers(
      universe, "financial", [], "Italy", []
    );
    expect(result.betaRegime).toBe("IT");
  });

  it("defaults to 'default' regime for empty geography", () => {
    const result = selectRelevantTickers(
      universe, "financial", [], "", []
    );
    expect(result.betaRegime).toBe("default");
  });

  it("includes direct entity matches with priority", () => {
    const result = selectRelevantTickers(
      universe, "corporate", ["Tesla", "Boeing"], "", []
    );
    const tickers = result.tickers.map((t) => t.ticker);
    expect(tickers).toContain("TSLA");
    expect(tickers).toContain("BA");
  });
});
