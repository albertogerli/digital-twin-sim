"use client";

import { useState, useMemo } from "react";
import type {
  RoundFinancial,
  TickerImpact,
  Provenance,
} from "../../lib/types/financial-impact";

interface Props {
  data: RoundFinancial[];
  provenance?: Provenance;
  schemaVersion?: string;
}

type RegionKey = "all" | "US" | "Europe" | "Asia" | "LatAm" | "Indices";
type SortKey = "t1_pct" | "beta" | "sector";

const REGIONS: RegionKey[] = ["all", "US", "Europe", "Asia", "LatAm", "Indices"];

const WARNING_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  LOW: { bg: "bg-green-50", text: "text-green-700", dot: "bg-green-500" },
  MODERATE: { bg: "bg-amber-50", text: "text-amber-700", dot: "bg-amber-500" },
  HIGH: { bg: "bg-orange-50", text: "text-orange-700", dot: "bg-orange-500" },
  CRITICAL: { bg: "bg-red-50", text: "text-red-700", dot: "bg-red-500" },
};

function inferRegion(ticker: string): RegionKey {
  if (ticker.startsWith("^")) return "Indices";
  if (ticker.endsWith(".MI") || ticker.endsWith(".PA") || ticker.endsWith(".DE") || ticker.endsWith(".AS") || ticker.endsWith(".MC") || ticker.endsWith(".L")) return "Europe";
  if (ticker.endsWith(".T") || ticker.endsWith(".KS") || ticker.endsWith(".HK") || ticker.endsWith(".NS") || ticker.endsWith(".BO") || ticker.endsWith(".SS") || ticker.endsWith(".SZ")) return "Asia";
  if (ticker.endsWith(".SA") || ticker.endsWith(".MX")) return "LatAm";
  // No suffix = US
  return "US";
}

function PctCell({ value, muted }: { value: number; muted?: boolean }) {
  const color = value < 0 ? "text-red-600" : value > 0 ? "text-emerald-600" : "text-gray-400";
  return (
    <span className={`font-mono text-xs tabular-nums ${color} ${muted ? "opacity-50" : ""}`}>
      {value > 0 ? "+" : ""}{value.toFixed(1)}%
    </span>
  );
}

function TickerRow({ t }: { t: TickerImpact }) {
  return (
    <div className="grid grid-cols-[1fr_56px_56px_56px] gap-1 items-center py-1.5 px-3 border-b border-gray-100 last:border-b-0 hover:bg-gray-50/50 transition-colors">
      <div className="flex items-center gap-2 min-w-0">
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${t.direction === "short" ? "bg-red-500" : "bg-emerald-500"}`} />
        <div className="min-w-0">
          <div className="text-xs font-semibold text-gray-900 truncate">{t.name}</div>
          <div className="text-[10px] text-gray-400 font-mono">{t.ticker} &middot; &beta;{t.beta.toFixed(2)}</div>
        </div>
      </div>
      <div className="text-right"><PctCell value={t.t1_pct} /></div>
      <div className="text-right"><PctCell value={t.t3_pct} muted /></div>
      <div className="text-right"><PctCell value={t.t7_pct} muted /></div>
    </div>
  );
}

function SectorHeatmap({ tickers }: { tickers: TickerImpact[] }) {
  const sectors = useMemo(() => {
    const map = new Map<string, { label: string; avg: number; count: number }>();
    tickers.forEach((t) => {
      const entry = map.get(t.sector);
      if (entry) {
        entry.avg += t.t1_pct;
        entry.count += 1;
      } else {
        map.set(t.sector, { label: t.sectorLabel, avg: t.t1_pct, count: 1 });
      }
    });
    const result: { sector: string; label: string; avg: number }[] = [];
    map.forEach((v, k) => result.push({ sector: k, label: v.label, avg: v.avg / v.count }));
    return result;
  }, [tickers]);

  return (
    <div className="flex flex-wrap gap-1 px-3 py-2">
      {sectors.map((s) => {
        const bg = s.avg < -1 ? "bg-red-200" : s.avg < 0 ? "bg-red-100" : s.avg > 1 ? "bg-emerald-200" : s.avg > 0 ? "bg-emerald-100" : "bg-gray-100";
        const text = s.avg < 0 ? "text-red-700" : s.avg > 0 ? "text-emerald-700" : "text-gray-500";
        return (
          <span key={s.sector} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono ${bg} ${text}`}>
            {s.label} {s.avg > 0 ? "+" : ""}{s.avg.toFixed(1)}%
          </span>
        );
      })}
    </div>
  );
}

function sortTickers(tickers: TickerImpact[], sortBy: SortKey): TickerImpact[] {
  return [...tickers].sort((a, b) => {
    if (sortBy === "beta") return b.beta - a.beta;
    if (sortBy === "sector") return a.sectorLabel.localeCompare(b.sectorLabel);
    return Math.abs(b.t1_pct) - Math.abs(a.t1_pct);
  });
}

function TickerLeg({
  label,
  tickers,
  sortBy,
  isShort,
  collapsed,
  maxVisible,
}: {
  label: string;
  tickers: TickerImpact[];
  sortBy: SortKey;
  isShort: boolean;
  collapsed: boolean;
  maxVisible: number;
}) {
  const sorted = useMemo(() => sortTickers(tickers, sortBy), [tickers, sortBy]);
  const visible = collapsed ? sorted.slice(0, maxVisible) : sorted;

  if (tickers.length === 0) return null;

  return (
    <>
      <div className={`px-3 py-1 border-b ${isShort ? "bg-red-50/50 border-red-100" : "bg-emerald-50/50 border-emerald-100"}`}>
        <span className={`text-[9px] font-mono font-bold uppercase tracking-wider ${isShort ? "text-red-500" : "text-emerald-600"}`}>
          {label} ({tickers.length})
        </span>
      </div>
      {visible.map((t) => (
        <TickerRow key={t.ticker} t={t} />
      ))}
      {collapsed && sorted.length > maxVisible && (
        <div className="px-3 py-1 text-[9px] font-mono text-gray-400 text-center">
          +{sorted.length - maxVisible} more
        </div>
      )}
    </>
  );
}

function RoundCard({
  round,
  regionFilter,
  sortBy,
  isExpanded,
  onToggleExpand,
}: {
  round: RoundFinancial;
  regionFilter: RegionKey;
  sortBy: SortKey;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const wc = WARNING_COLORS[round.volatility_warning] ?? WARNING_COLORS.LOW;

  const filteredTickers = useMemo(() => {
    if (regionFilter === "all") return round.tickers;
    return round.tickers.filter((t) => inferRegion(t.ticker) === regionFilter);
  }, [round.tickers, regionFilter]);

  const shorts = filteredTickers.filter((t) => t.direction === "short");
  const longs = filteredTickers.filter((t) => t.direction === "long");

  const maxVisible = 5;
  const totalTickers = filteredTickers.length;
  const needsExpand = shorts.length > maxVisible || longs.length > maxVisible;

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-gray-900">Round {round.round}</span>
          <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${wc.bg} ${wc.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${wc.dot}`} />
            {round.volatility_warning}
          </span>
          <span className="text-[10px] text-gray-400 font-mono">{totalTickers} tickers</span>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div>
            <span className="text-gray-400">FTSE MIB </span>
            <span className={round.ftse_mib_impact_pct < 0 ? "text-red-600 font-semibold" : "text-emerald-600 font-semibold"}>
              {round.ftse_mib_impact_pct > 0 ? "+" : ""}{round.ftse_mib_impact_pct.toFixed(2)}%
            </span>
          </div>
          <div>
            <span className="text-gray-400">BTP </span>
            <span className="text-amber-600 font-semibold">+{round.btp_spread_bps}bps</span>
          </div>
        </div>
      </div>

      {/* Headline */}
      <div className="px-4 py-2.5 bg-gray-50 border-b border-gray-100">
        <p className="text-xs text-gray-600 leading-relaxed italic">{round.headline}</p>
      </div>

      {/* Sector heatmap strip */}
      {filteredTickers.length > 0 && (
        <div className="border-b border-gray-100">
          <SectorHeatmap tickers={filteredTickers} />
        </div>
      )}

      {/* Ticker grid header */}
      <div className="grid grid-cols-[1fr_56px_56px_56px] gap-1 px-3 py-1.5 border-b border-gray-200 bg-gray-50">
        <span className="text-[9px] text-gray-400 font-mono uppercase tracking-wider">Ticker</span>
        <span className="text-[9px] text-gray-400 font-mono uppercase tracking-wider text-right">T+1</span>
        <span className="text-[9px] text-gray-400 font-mono uppercase tracking-wider text-right">T+3</span>
        <span className="text-[9px] text-gray-400 font-mono uppercase tracking-wider text-right">T+7</span>
      </div>

      {/* Ticker list with optional scroll */}
      <div className={isExpanded ? "max-h-[400px] overflow-y-auto" : ""}>
        <TickerLeg label="Short Leg" tickers={shorts} sortBy={sortBy} isShort collapsed={!isExpanded} maxVisible={maxVisible} />
        <TickerLeg label="Long Leg" tickers={longs} sortBy={sortBy} isShort={false} collapsed={!isExpanded} maxVisible={maxVisible} />
      </div>

      {/* Expand / collapse button */}
      {needsExpand && (
        <button
          onClick={onToggleExpand}
          className="w-full px-4 py-1.5 text-[10px] font-mono text-gray-500 hover:text-gray-700 hover:bg-gray-50 border-t border-gray-100 transition-colors text-center"
        >
          {isExpanded ? "Show less" : `Show all ${totalTickers} tickers`}
        </button>
      )}

      {/* Contagion bar */}
      <div className="px-4 py-2.5 border-t border-gray-100 flex items-center gap-3">
        <span className="text-[9px] text-gray-400 font-mono uppercase tracking-wider whitespace-nowrap">Contagion Risk</span>
        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${round.contagion_risk * 100}%`,
              backgroundColor: round.contagion_risk > 0.7 ? "#ef4444" : round.contagion_risk > 0.4 ? "#f59e0b" : "#22c55e",
            }}
          />
        </div>
        <span className="text-[10px] font-mono font-semibold text-gray-600 tabular-nums">
          {(round.contagion_risk * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function ProvenanceBadge({ provenance, schemaVersion }: { provenance?: Provenance; schemaVersion?: string }) {
  const isBackend = provenance === "backend-simulated";
  const isUnavailable = provenance === "unavailable";
  const label = isBackend
    ? "Backend-simulated"
    : isUnavailable
    ? "Unavailable"
    : "Client fallback (deprecated)";
  const tooltip = isBackend
    ? "Computed by Python FinancialImpactScorer with full crisis model"
    : isUnavailable
    ? "Backend scorer did not emit financial data for this scenario"
    : "Legacy client-side heuristic — do not use for decisions";
  const tone = isBackend
    ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : isUnavailable
    ? "bg-gray-50 text-gray-700 border-gray-200"
    : "bg-amber-50 text-amber-700 border-amber-200";
  const dot = isBackend ? "bg-emerald-500" : isUnavailable ? "bg-gray-400" : "bg-amber-500";
  return (
    <span
      title={tooltip}
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-mono font-semibold uppercase tracking-wider border ${tone}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {label}
      {schemaVersion && <span className="text-gray-400 font-normal ml-1">v{schemaVersion}</span>}
    </span>
  );
}

export default function FinancialImpactPanel({ data, provenance, schemaVersion }: Props) {
  const [expanded, setExpanded] = useState(true);
  const [regionFilter, setRegionFilter] = useState<RegionKey>("all");
  const [sortBy, setSortBy] = useState<SortKey>("t1_pct");
  const [expandedRounds, setExpandedRounds] = useState<Set<number>>(new Set());

  // Explicit empty state when backend has not emitted financial data.
  // We do NOT fabricate numbers client-side anymore.
  if (!data || data.length === 0) {
    return (
      <section className="border border-ki-border rounded p-5 bg-ki-surface-sunken">
        <div className="flex items-center gap-3 mb-2">
          <h2 className="text-xl font-bold text-gray-900">Financial Impact</h2>
          <ProvenanceBadge provenance={provenance ?? "unavailable"} schemaVersion={schemaVersion} />
        </div>
        <p className="text-sm text-gray-600 leading-relaxed">
          The backend financial scorer did not emit data for this scenario.
          We deliberately do not fabricate sector betas, pair trades, or
          T+1/T+3/T+7 returns client-side — those numbers must come from the
          calibrated <code>FinancialImpactScorer</code> on the backend, or
          they are not shown.
        </p>
        <p className="text-xs text-gray-500 mt-2">
          Cause: scorer offline, scenario predates the scorer, or the
          provider rejected the brief. Re-run the simulation against a
          healthy backend to populate this panel.
        </p>
      </section>
    );
  }

  const filteredTickerCount = data.reduce((sum, r) => {
    const count = regionFilter === "all"
      ? r.tickers.length
      : r.tickers.filter((t) => inferRegion(t.ticker) === regionFilter).length;
    return sum + count;
  }, 0);
  const avgTickersPerRound = data.length > 0 ? Math.round(filteredTickerCount / data.length) : 0;

  const worstRound = data.reduce((worst, r) =>
    r.contagion_risk > worst.contagion_risk ? r : worst, data[0]);

  const toggleRound = (round: number) => {
    setExpandedRounds((prev) => {
      const next = new Set(prev);
      if (next.has(round)) next.delete(round);
      else next.add(round);
      return next;
    });
  };

  const sortOptions: { key: SortKey; label: string }[] = [
    { key: "t1_pct", label: "Impact" },
    { key: "beta", label: "Beta" },
    { key: "sector", label: "Sector" },
  ];

  return (
    <section>
      {/* Section header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-gray-900">
                <path d="M2 20h20M5 20V8l3-3 4 5 4-7 4 4v13" />
              </svg>
              <h2 className="text-xl font-bold text-gray-900">Financial Impact</h2>
            </div>
            <span className="text-xs font-mono text-gray-400">
              {data.length} rounds &middot; {avgTickersPerRound} tickers{regionFilter !== "all" ? ` (${regionFilter})` : ""}
            </span>
            <ProvenanceBadge provenance={provenance} schemaVersion={schemaVersion} />
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Estimated market impact using historical sector beta coefficients (&beta;) and pair trade logic.
            T+1/T+3/T+7 indicate 1-day, 3-day, and 7-day projected moves.
          </p>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-gray-400 hover:text-gray-600 transition-colors font-mono"
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
      </div>

      {/* Region filter tabs + Sort controls */}
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <div className="flex items-center gap-1">
          {REGIONS.map((r) => (
            <button
              key={r}
              onClick={() => setRegionFilter(r)}
              className={`px-2.5 py-1 text-[10px] font-mono font-semibold rounded-full border transition-colors ${regionFilter === r
                  ? "bg-gray-900 text-white border-gray-900"
                  : "bg-white text-gray-500 border-gray-200 hover:border-gray-300 hover:text-gray-700"
                }`}
            >
              {r === "all" ? "All" : r}
            </button>
          ))}
        </div>

        <div className="w-px h-4 bg-gray-200 mx-1" />

        <div className="flex items-center gap-1">
          <span className="text-[9px] text-gray-400 font-mono uppercase tracking-wider mr-1">Sort:</span>
          {sortOptions.map((opt) => (
            <button
              key={opt.key}
              onClick={() => setSortBy(opt.key)}
              className={`px-2 py-0.5 text-[10px] font-mono rounded border transition-colors ${sortBy === opt.key
                  ? "bg-gray-100 text-gray-800 border-gray-300 font-semibold"
                  : "bg-white text-gray-400 border-gray-200 hover:border-gray-300 hover:text-gray-600"
                }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Summary strip */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <div className="text-[10px] text-gray-400 font-mono uppercase">Peak Contagion</div>
          <div className={`text-lg font-bold font-mono ${worstRound.contagion_risk > 0.5 ? "text-red-600" : "text-amber-600"}`}>
            {(worstRound.contagion_risk * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-400">Round {worstRound.round}</div>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <div className="text-[10px] text-gray-400 font-mono uppercase">FTSE MIB Range</div>
          <div className="text-lg font-bold font-mono text-gray-900">
            {Math.min(...data.map((d) => d.ftse_mib_impact_pct)).toFixed(1)}% / {Math.max(...data.map((d) => d.ftse_mib_impact_pct)) > 0 ? "+" : ""}{Math.max(...data.map((d) => d.ftse_mib_impact_pct)).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <div className="text-[10px] text-gray-400 font-mono uppercase">BTP Spread Peak</div>
          <div className="text-lg font-bold font-mono text-amber-600">
            +{Math.max(...data.map((d) => d.btp_spread_bps))}bps
          </div>
        </div>
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <div className="text-[10px] text-gray-400 font-mono uppercase">Worst Alert</div>
          <div className={`text-lg font-bold font-mono ${WARNING_COLORS[worstRound.volatility_warning]?.text ?? "text-gray-600"}`}>
            {worstRound.volatility_warning}
          </div>
        </div>
      </div>

      {/* Per-round cards */}
      {expanded && (
        <div className="space-y-4">
          {data.map((round) => (
            <RoundCard
              key={round.round}
              round={round}
              regionFilter={regionFilter}
              sortBy={sortBy}
              isExpanded={expandedRounds.has(round.round)}
              onToggleExpand={() => toggleRound(round.round)}
            />
          ))}
        </div>
      )}

      {/* Disclaimer */}
      <div className="mt-4 px-4 py-2.5 bg-gray-50 rounded-lg border border-gray-200">
        <p className="text-[10px] text-gray-400 leading-relaxed">
          <span className="font-semibold">Disclaimer:</span> Stime parametriche basate su beta settoriali calibrati su crisi italiane (2011-2023).
          Non costituiscono consulenza finanziaria. I pair trade sono indicativi della direzione del rischio, non raccomandazioni operative.
          Fonte beta: Bloomberg event studies, BTP-Bund spread regressions.
        </p>
      </div>
    </section>
  );
}
