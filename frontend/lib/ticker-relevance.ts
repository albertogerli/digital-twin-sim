/**
 * Ticker Relevance System
 *
 * Selects relevant stock tickers for a simulation scenario based on
 * domain, geography, mentioned entities, and keyword tags.
 */

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface BetaCoefficients {
  political_beta: number;
  spread_beta: number;
  crisis_alpha: number;
  volatility_multiplier: number;
}

export interface SectorDef {
  label: string;
  gics_code: string;
  betas: Record<string, BetaCoefficients>;
}

export interface StockEntry {
  ticker: string;
  name: string;
  country: string;
  region: string;
  sector: string;
  industry_group: string;
  market_cap_tier: string;
  index_membership: string[];
  tags: string[];
}

export interface IndexEntry {
  ticker: string;
  name: string;
  country: string;
  region: string;
}

export interface StockUniverse {
  version: string;
  updated: string;
  indices: IndexEntry[];
  stocks: StockEntry[];
  sectors: Record<string, SectorDef>;
  org_aliases: Record<string, string[]>;
}

export interface SelectedTicker {
  ticker: string;
  name: string;
  sector: string;
  reason: string;
}

export interface TickerSelection {
  tickers: SelectedTicker[];
  indices: IndexEntry[];
  betaRegime: string;
}

/* ------------------------------------------------------------------ */
/*  Geography -> Beta Regime                                           */
/* ------------------------------------------------------------------ */

const COUNTRY_TO_REGIME: Record<string, string> = {
  Italy: "IT",
  IT: "IT",
  US: "US",
  "United States": "US",
  Brazil: "EM",
  BR: "EM",
  India: "EM",
  IN: "EM",
  China: "EM",
  CN: "EM",
  Mexico: "EM",
  MX: "EM",
  Argentina: "EM",
  AR: "EM",
  Chile: "EM",
  CL: "EM",
  "South Korea": "EM",
  KR: "EM",
  Taiwan: "EM",
  TW: "EM",
  Germany: "US",
  DE: "US",
  France: "US",
  FR: "US",
  UK: "US",
  GB: "US",
  Japan: "US",
  JP: "US",
};

/** Map a geography string (country name or ISO code) to a beta regime key. */
function resolveRegime(geography: string): string {
  if (!geography) return "default";
  return COUNTRY_TO_REGIME[geography] ?? "default";
}

/* ------------------------------------------------------------------ */
/*  Domain -> preferred sectors                                        */
/* ------------------------------------------------------------------ */

const DOMAIN_SECTORS: Record<string, string[]> = {
  political: [
    "banking",
    "defense",
    "energy_fossil",
    "sovereign_debt",
    "media",
  ],
  financial: ["banking", "insurance", "real_estate", "sovereign_debt", "energy_fossil", "utilities", "tech", "automotive"],
  commercial: ["tech", "automotive", "food_consumer", "luxury", "telecom"],
  corporate: ["tech", "automotive", "healthcare", "infrastructure", "defense"],
  energy: [
    "energy_fossil",
    "energy_renewable",
    "utilities",
    "infrastructure",
  ],
  health: ["healthcare", "food_consumer", "insurance"],
  military: ["defense", "infrastructure", "tech"],
};

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Look up beta coefficients for a sector + regime.
 * Falls back to "default" regime if the requested one is missing.
 */
export function getBeta(
  universe: StockUniverse,
  sector: string,
  regime: string,
): BetaCoefficients {
  const sectorDef = universe.sectors[sector];
  if (!sectorDef) {
    return {
      political_beta: 1.0,
      spread_beta: 0,
      crisis_alpha: 0,
      volatility_multiplier: 1.0,
    };
  }
  return sectorDef.betas[regime] ?? sectorDef.betas["default"];
}

/**
 * Resolve an organisation name to ticker(s) via the alias map.
 * Case-insensitive lookup.
 */
export function resolveOrg(universe: StockUniverse, org: string): string[] {
  const key = org.toLowerCase();
  return universe.org_aliases[key] ?? [];
}

/**
 * Select the most relevant tickers for a scenario.
 *
 * Strategy:
 *  1. Direct entity matches (highest priority)
 *  2. Tag-matched stocks (keywords matched against stock tags)
 *  3. Domain-sector fill (sector preferences per domain)
 *  4. Diversity guarantee (min 3 sectors)
 *  5. Cap at 30 tickers
 *  6. Pick 1-3 indices by geography
 */
export function selectRelevantTickers(
  universe: StockUniverse,
  domain: string,
  entities: string[],
  geography: string,
  keywords: string[],
): TickerSelection {
  const betaRegime = resolveRegime(geography);
  const selected = new Map<string, SelectedTicker>();

  // 1. Direct entity matches
  for (const entity of entities) {
    const tickers = resolveOrg(universe, entity);
    for (const t of tickers) {
      const stock = universe.stocks.find((s) => s.ticker === t);
      if (stock && !selected.has(t)) {
        selected.set(t, {
          ticker: t,
          name: stock.name,
          sector: stock.sector,
          reason: `entity:${entity}`,
        });
      }
    }
  }

  // 2. Tag-matched stocks
  if (keywords.length > 0) {
    const lowerKw = keywords.map((k) => k.toLowerCase());
    for (const stock of universe.stocks) {
      if (selected.size >= 30) break;
      if (selected.has(stock.ticker)) continue;
      const stockTags = stock.tags.map((t) => t.toLowerCase());
      const match = lowerKw.some((kw) => stockTags.includes(kw));
      if (match) {
        selected.set(stock.ticker, {
          ticker: stock.ticker,
          name: stock.name,
          sector: stock.sector,
          reason: `tag:${keywords.find((k) => stockTags.includes(k.toLowerCase()))}`,
        });
      }
    }
  }

  // 3. Domain-sector fill
  const preferredSectors = DOMAIN_SECTORS[domain] ?? Object.keys(universe.sectors).slice(0, 5);
  for (const sector of preferredSectors) {
    if (selected.size >= 30) break;
    const sectorStocks = universe.stocks
      .filter((s) => s.sector === sector && !selected.has(s.ticker))
      .sort((a, b) => {
        const tierOrder: Record<string, number> = {
          mega: 0,
          large: 1,
          mid: 2,
          small: 3,
        };
        return (tierOrder[a.market_cap_tier] ?? 4) - (tierOrder[b.market_cap_tier] ?? 4);
      });
    // Take top 2-3 per sector
    const take = Math.min(3, sectorStocks.length, 30 - selected.size);
    for (let i = 0; i < take; i++) {
      const stock = sectorStocks[i];
      selected.set(stock.ticker, {
        ticker: stock.ticker,
        name: stock.name,
        sector: stock.sector,
        reason: `domain:${domain}/${sector}`,
      });
    }
  }

  // 4. Diversity guarantee: ensure at least 3 distinct sectors
  const currentSectors = new Set(
    Array.from(selected.values()).map((t) => t.sector),
  );
  if (currentSectors.size < 3) {
    const allSectors = Object.keys(universe.sectors);
    for (const sector of allSectors) {
      if (currentSectors.size >= 3) break;
      if (currentSectors.has(sector)) continue;
      if (selected.size >= 30) break;
      const stock = universe.stocks.find(
        (s) => s.sector === sector && !selected.has(s.ticker),
      );
      if (stock) {
        selected.set(stock.ticker, {
          ticker: stock.ticker,
          name: stock.name,
          sector: stock.sector,
          reason: `diversity:${sector}`,
        });
        currentSectors.add(sector);
      }
    }
  }

  // 5. Cap at 30
  const tickers = Array.from(selected.values()).slice(0, 30);

  // 6. Select 1-3 indices by geography
  const indices = selectIndices(universe, geography, betaRegime);

  return { tickers, indices, betaRegime };
}

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                   */
/* ------------------------------------------------------------------ */

function selectIndices(
  universe: StockUniverse,
  geography: string,
  regime: string,
): IndexEntry[] {
  const result: IndexEntry[] = [];

  // Try to find a local index
  const countryCode = geography.length === 2 ? geography : undefined;
  if (countryCode) {
    const local = universe.indices.find((i) => i.country === countryCode);
    if (local) result.push(local);
  }

  // Map regime to primary index
  if (regime === "IT") {
    const mib = universe.indices.find((i) => i.country === "IT");
    if (mib && !result.some((r) => r.ticker === mib.ticker)) result.push(mib);
    const stoxx = universe.indices.find((i) => i.country === "EU");
    if (stoxx && !result.some((r) => r.ticker === stoxx.ticker))
      result.push(stoxx);
  } else if (regime === "US") {
    const sp = universe.indices.find((i) => i.ticker === "^GSPC");
    if (sp && !result.some((r) => r.ticker === sp.ticker)) result.push(sp);
  } else if (regime === "EM") {
    // Find matching EM country index
    const countryMap: Record<string, string> = {
      Brazil: "BR",
      BR: "BR",
      India: "IN",
      IN: "IN",
      China: "CN",
      CN: "CN",
      Mexico: "MX",
      MX: "MX",
      "South Korea": "KR",
      KR: "KR",
    };
    const cc = countryMap[geography];
    if (cc) {
      const idx = universe.indices.find((i) => i.country === cc);
      if (idx && !result.some((r) => r.ticker === idx.ticker))
        result.push(idx);
    }
  }

  // Always include S&P 500 as global reference if not yet present
  if (!result.some((r) => r.ticker === "^GSPC")) {
    const sp = universe.indices.find((i) => i.ticker === "^GSPC");
    if (sp) result.push(sp);
  }

  return result.slice(0, 3);
}
