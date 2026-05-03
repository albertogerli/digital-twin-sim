"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";

// --- Dynamic imports (ssr: false for D3/recharts/markdown components) ---

const NetworkGraph = dynamic(() => import("../../../components/NetworkGraph"), {
  ssr: false,
  loading: () => <SectionSkeleton />,
});

const ReportSection = dynamic(
  () => import("../../../components/report/ReportSection"),
  {
    ssr: false,
    loading: () => <SectionSkeleton />,
  }
);

// --- Static imports for non-D3 editorial sections ---

import ScenarioHero from "../../../components/scenario/ScenarioHero";
import AgentsSection from "../../../components/agents/AgentsSection";
import PolarizationSection from "../../../components/polarization/PolarizationSection";
import Timeline from "../../../components/timeline/Timeline";
import type { RoundData } from "../../../components/timeline/Timeline";
import ViralShowcase from "../../../components/viral-posts/ViralShowcase";
import FinancialImpactPanel from "../../../components/scenario/FinancialImpactPanel";
import {
  FIN_SCHEMA_VERSION,
  isCompatible,
  type Provenance,
  type RoundFinancial,
} from "../../../lib/types/financial-impact";

// ============================================================
// Types
// ============================================================

interface Metadata {
  scenario_id: string;
  scenario_name: string;
  num_rounds: number;
  domain: string;
  description: string;
}

interface Agent {
  id: string;
  name: string;
  role: string;
  archetype: string;
  tier: number;
  initial_position: number;
  final_position: number;
  position_delta: number;
  influence: number;
  emotional_state: string;
}

interface PolarizationPoint {
  round: number;
  polarization: number;
  avg_position: number;
  num_agents?: number;
}

interface ViralPost {
  id: string;
  author_name: string;
  author_avatar?: string;
  platform: string;
  round: number;
  text: string;
  total_engagement: number;
  likes?: number;
  reposts?: number;
  replies?: number;
}

interface GraphSnapshot {
  round: number;
  month: string;
  event_label: string;
  nodes: any[];
  edges: any[];
  stats: {
    total_nodes: number;
    total_edges: number;
    avg_position: number;
  };
}

interface ReplayRoundRaw {
  round: number;
  month?: string;
  event: { event: string; shock_magnitude?: number; shock_direction?: number };
  posts: any[];
  graphSnapshot?: any;
  indicators?: {
    polarization?: number;
    engagement?: number;
    sentiment?: any;
    trendingHashtags?: any;
  };
  coalitions?: { coalitions?: any[] };
  key_insight?: string;
  postImpacts?: any[];
  realWorldEffects?: any[];
  financial_twin?: {
    round: number;
    nim_pct: number;
    cet1_pct: number;
    lcr_pct: number;
    deposit_balance: number;
    loan_demand_index: number;
    deposit_runoff_round_pct: number;
    policy_rate_pct: number;
    btp_bund_spread_bps: number;
    breaches?: string[];
  };
  financial_feedback?: {
    nim_anxiety: number;
    cet1_alarm: number;
    runoff_panic: number;
    competitor_pressure: number;
    rate_pressure: number;
  };
}

// ============================================================
// Helpers
// ============================================================

/**
 * Try API endpoint first, fall back to static /data/ path.
 * For JSON files, parse automatically. For text (e.g. .md), return raw text.
 */
async function fetchWithFallback(
  id: string,
  file: string,
  asText = false
): Promise<any> {
  const apiUrl = `/api/scenarios/${id}/${file}`;
  const staticUrl = `/data/scenario_${id}/${file}`;

  const parse = (r: Response) => (asText ? r.text() : r.json());

  // Try API first
  try {
    const r = await fetch(apiUrl);
    if (r.ok) return parse(r);
    console.warn(`[fetchWithFallback] API ${r.status} for ${apiUrl}`);
  } catch (err) {
    console.warn(`[fetchWithFallback] API error for ${apiUrl}:`, err);
  }

  // Fall back to static
  try {
    const r2 = await fetch(staticUrl);
    if (r2.ok) return parse(r2);
    console.warn(`[fetchWithFallback] Static ${r2.status} for ${staticUrl}`);
    throw new Error(`Failed to fetch ${file} (API and static both failed for scenario "${id}")`);
  } catch (err) {
    throw err instanceof Error ? err : new Error(`Failed to fetch ${file} for scenario "${id}"`);
  }
}

/**
 * Convert raw replay round JSON into the RoundData shape expected by Timeline.
 */
function replayToRoundData(raw: ReplayRoundRaw): RoundData {
  // Pick top 5 posts by engagement for the round chapter
  const topPosts = [...(raw.posts || [])]
    .sort(
      (a, b) =>
        (b.total_engagement ?? b.engagement_score ?? 0) -
        (a.total_engagement ?? a.engagement_score ?? 0)
    )
    .slice(0, 5)
    .map((p: any) => ({
      id: p.id ?? `${raw.round}_${p.author_id}`,
      author_name: p.author_name ?? "Unknown",
      platform: p.platform ?? "social",
      text: p.text ?? "",
      total_engagement:
        p.total_engagement ??
        (p.likes ?? 0) + (p.reposts ?? 0) + (p.replies ?? 0),
    }));

  // Build coalition snapshot as Record<string, number>
  const coalitionSnapshot: Record<string, number> = {};
  const coalitions = raw.coalitions?.coalitions ?? [];
  for (const c of coalitions) {
    if (c.label && c.size != null) {
      coalitionSnapshot[c.label] = c.size;
    }
  }

  return {
    round: raw.round,
    label: raw.month,
    event: raw.event,
    top_posts: topPosts,
    polarization: raw.indicators?.polarization ?? 0,
    avg_position: undefined,
    coalition_snapshot:
      Object.keys(coalitionSnapshot).length > 0 ? coalitionSnapshot : undefined,
    key_insight: raw.key_insight,
  };
}

// ============================================================
// Skeleton component
// ============================================================

function SectionSkeleton() {
  return (
    <div className="animate-pulse space-y-4 py-12 px-4">
      <div className="h-6 w-48 bg-gray-200 rounded" />
      <div className="h-4 w-96 bg-gray-200 rounded" />
      <div className="grid grid-cols-3 gap-4 mt-6">
        <div className="h-32 bg-gray-100 rounded-lg" />
        <div className="h-32 bg-gray-100 rounded-lg" />
        <div className="h-32 bg-gray-100 rounded-lg" />
      </div>
    </div>
  );
}

function FullPageSkeleton() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
        {/* Hero skeleton */}
        <div className="animate-pulse space-y-4">
          <div className="h-4 w-24 bg-gray-200 rounded-full" />
          <div className="h-12 w-2/3 bg-gray-200 rounded" />
          <div className="h-5 w-1/2 bg-gray-200 rounded" />
          <div className="flex gap-4 mt-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-16 w-28 bg-gray-100 rounded-lg" />
            ))}
          </div>
        </div>
        {/* Section skeletons */}
        {[1, 2, 3, 4].map((i) => (
          <SectionSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

// ============================================================
// Main component
// ============================================================

export default function ScenarioDashboard({
  params,
}: {
  params: { id: string };
}) {
  const { id } = params;

  // --- State ---
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [polarization, setPolarization] = useState<PolarizationPoint[]>([]);
  const [topPosts, setTopPosts] = useState<ViralPost[]>([]);
  const [graphSnapshots, setGraphSnapshots] = useState<GraphSnapshot[]>([]);
  const [reportMarkdown, setReportMarkdown] = useState<string>("");
  const [roundsData, setRoundsData] = useState<RoundData[]>([]);
  const [roundGraphSnapshots, setRoundGraphSnapshots] = useState<any[]>([]);
  const [almRounds, setAlmRounds] = useState<Array<{ round: number; financial_twin?: any; financial_feedback?: any; }>>([]);
  const [financialImpact, setFinancialImpact] = useState<RoundFinancial[]>([]);
  const [finProvenance, setFinProvenance] = useState<Provenance>("unavailable");
  const [finSchemaVersion, setFinSchemaVersion] = useState<string>(FIN_SCHEMA_VERSION);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // --- Data loading ---
  useEffect(() => {
    let cancelled = false;

    async function loadData() {
      try {
        // Phase 1: Load metadata first (we need num_rounds)
        const meta: Metadata = await fetchWithFallback(id, "metadata.json");
        if (cancelled) return;
        setMetadata(meta);

        // Phase 2: Load everything else in parallel (including all replay rounds)
        const roundIndices = Array.from(
          { length: meta.num_rounds },
          (_, i) => i + 1
        );

        const [ag, pol, posts, graph, report, finImpact, ...replayRounds] =
          await Promise.all([
            fetchWithFallback(id, "agents.json").catch(() => []),
            fetchWithFallback(id, "polarization.json").catch(() => []),
            fetchWithFallback(id, "top_posts.json").catch(() => []),
            fetchWithFallback(id, "evolving_graph.json").catch(() => []),
            fetchWithFallback(id, "report.md", true).catch(() => ""),
            fetchWithFallback(id, "financial_impact.json").catch(() => []),
            ...roundIndices.map((n) =>
              fetchWithFallback(id, `replay_round_${n}.json`).catch(() => null)
            ),
          ]);

        if (cancelled) return;

        setAgents(ag);
        setPolarization(pol);
        setTopPosts(posts);
        setGraphSnapshots(Array.isArray(graph) ? graph : []);
        setReportMarkdown(typeof report === "string" ? report : "");
        // Process replay rounds into Timeline-compatible data
        const validRounds = replayRounds.filter(
          (r): r is ReplayRoundRaw => r != null
        );
        const converted = validRounds
          .sort((a, b) => a.round - b.round)
          .map(replayToRoundData);
        setRoundsData(converted);

        // Extract financial_twin / financial_feedback (banking domain)
        const almPayload = validRounds
          .sort((a, b) => a.round - b.round)
          .filter((r) => r.financial_twin)
          .map((r) => ({
            round: r.round,
            financial_twin: r.financial_twin,
            financial_feedback: r.financial_feedback,
          }));
        setAlmRounds(almPayload);

        // Financial impact: backend-simulated payload only. No client-side
        // heuristic fallback — if the backend hasn't emitted financial data,
        // the panel renders an explicit empty state instead of fake numbers.
        let finRounds: RoundFinancial[] = [];
        let provenance: Provenance = "unavailable";
        let schemaVersion = FIN_SCHEMA_VERSION;

        if (isCompatible(finImpact)) {
          finRounds = finImpact.rounds;
          provenance = finImpact.provenance ?? "backend-simulated";
          schemaVersion = finImpact.schema_version;
        } else if (Array.isArray(finImpact) && finImpact.length > 0) {
          // Legacy: bare round array — accept but mark as unknown schema
          finRounds = finImpact as RoundFinancial[];
          provenance = (finImpact[0] as any)?.provenance ?? "backend-simulated";
          schemaVersion = (finImpact[0] as any)?.schema_version ?? "legacy";
        }
        // No fallback: leave finRounds empty + provenance="unavailable"
        setFinancialImpact(finRounds);
        setFinProvenance(provenance);
        setFinSchemaVersion(schemaVersion);

        // Extract graph snapshots from replay rounds for Timeline mini-networks
        const snapshots = validRounds
          .sort((a, b) => a.round - b.round)
          .map((r) => r.graphSnapshot)
          .filter(Boolean);
        setRoundGraphSnapshots(snapshots);

        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to load scenario data:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load scenario data."
        );
        setLoading(false);
      }
    }

    loadData();
    return () => {
      cancelled = true;
    };
  }, [id]);

  // --- Loading state ---
  if (loading) {
    return <FullPageSkeleton />;
  }

  // --- Error state ---
  if (error || !metadata) {
    return (
      <div className="min-h-screen bg-ki-surface flex flex-col items-center justify-center gap-6 px-4">
        <div className="bg-ki-surface-raised border border-ki-error/30 rounded p-8 max-w-md text-center">
          <div className="w-1.5 h-1.5 rounded-full bg-ki-error mx-auto mb-4" />
          <p className="text-ki-on-surface text-[15px] font-medium mb-2">
            Failed to load scenario
          </p>
          <p className="text-ki-on-surface-secondary text-[12px] mb-6">
            {error || "Scenario not found."}
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-ki-primary hover:text-ki-primary-muted transition-colors text-[12px] font-medium"
          >
            ← Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  // --- Derived values ---
  const finalPolarization =
    polarization.length > 0
      ? polarization[polarization.length - 1].polarization
      : 0;

  // --- Render ---
  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface">
      {/* Sub-toolbar: breadcrumb + actions (this page is fullscreen, no AppShell) */}
      <div className="sticky top-0 z-30 bg-ki-surface-raised/95 backdrop-blur border-b border-ki-border h-11 flex items-center px-4 gap-3">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors"
        >
          ← Dashboard
        </Link>
        <span className="text-ki-border-strong">/</span>
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">
          Scenario
        </span>
        <span className="text-ki-border-strong">/</span>
        <span className="text-[13px] font-medium text-ki-on-surface truncate flex-1 min-w-0">
          {metadata.scenario_name}
        </span>
        <Link
          href={`/scenario/${id}/replay`}
          className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm bg-ki-on-surface text-ki-surface text-[11px] font-medium hover:bg-ki-on-surface-secondary transition-colors"
        >
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 500" }}>
            play_arrow
          </span>
          Live command
        </Link>
        <Link
          href={`/scenario/${id}/branches`}
          className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors"
        >
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>
            account_tree
          </span>
          Branches
        </Link>
        {reportMarkdown && reportMarkdown.trim().length > 0 && (
          <a
            href={`/api/scenarios/${id}/report.html`}
            target="_blank"
            rel="noopener"
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors"
          >
            <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>
              download
            </span>
            Export PDF
          </a>
        )}
      </div>

      {/* ScenarioHero */}
      <ScenarioHero
        metadata={metadata}
        scenarioId={id}
        agentsCount={agents.length}
        finalPolarization={finalPolarization}
      />

      {/* Agents Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AgentsSection agents={agents} />
      </div>

      {/* Polarization Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <PolarizationSection polarization={polarization} />
      </div>

      {/* Financial Impact Section */}
      {financialImpact.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <FinancialImpactPanel
            data={financialImpact}
            provenance={finProvenance}
            schemaVersion={finSchemaVersion}
          />
        </div>
      )}

      {/* Timeline Section (with replay round data) */}
      {roundsData.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Timeline
            rounds={roundsData}
            graphSnapshots={
              roundGraphSnapshots.length > 0 ? roundGraphSnapshots : undefined
            }
          />
        </div>
      )}

      {/* Viral Showcase */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ViralShowcase posts={topPosts} />
      </div>

      {/* Network Graph (D3 — dynamic import) */}
      {graphSnapshots.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <NetworkGraph snapshots={graphSnapshots} />
        </div>
      )}

      {/* ALM section (banking domain only) */}
      {almRounds.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="eyebrow mb-1">ALM · Asset–Liability Management</div>
          <h2 className="text-[18px] font-medium tracking-tight2 text-ki-on-surface mb-2">Stato bancario per round</h2>
          <p className="text-[12px] text-ki-on-surface-secondary mb-4 max-w-3xl leading-relaxed">
            Modello deterministico parametrato su benchmark italiani 2025
            (deposit β EBA, elasticità credito al consumo Bonaccorsi/Magri,
            NIM/CET1/LCR EBA Risk Dashboard). I numeri stanno entro vincoli
            ALM realistici per banca commerciale italiana media.
          </p>
          <div className="overflow-x-auto bg-ki-surface-raised border border-ki-border rounded-sm">
            <table className="w-full text-xs">
              <thead className="bg-ki-surface-sunken">
                <tr className="border-b border-ki-border">
                  <th className="px-2 py-2 text-left font-semibold">Round</th>
                  <th className="px-2 py-2 text-right font-semibold">NIM</th>
                  <th className="px-2 py-2 text-right font-semibold">CET1</th>
                  <th className="px-2 py-2 text-right font-semibold">LCR</th>
                  <th className="px-2 py-2 text-right font-semibold">Depositi</th>
                  <th className="px-2 py-2 text-right font-semibold">Domanda credito</th>
                  <th className="px-2 py-2 text-right font-semibold">Runoff/round</th>
                  <th className="px-2 py-2 text-right font-semibold">Policy rate</th>
                  <th className="px-2 py-2 text-left font-semibold">Breach</th>
                </tr>
              </thead>
              <tbody>
                {almRounds.map((r) => {
                  const t = r.financial_twin || {};
                  const breaches = (t.breaches as string[]) || [];
                  return (
                    <tr key={r.round} className="border-b border-ki-border/50">
                      <td className="px-2 py-1.5 font-data">R{r.round}</td>
                      <td className="px-2 py-1.5 text-right font-data">{(t.nim_pct ?? 0).toFixed(2)}%</td>
                      <td className="px-2 py-1.5 text-right font-data">{(t.cet1_pct ?? 0).toFixed(1)}%</td>
                      <td className="px-2 py-1.5 text-right font-data">{(t.lcr_pct ?? 0).toFixed(0)}%</td>
                      <td className="px-2 py-1.5 text-right font-data">{(t.deposit_balance ?? 0).toFixed(3)}</td>
                      <td className="px-2 py-1.5 text-right font-data">{(t.loan_demand_index ?? 0).toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right font-data">{((t.deposit_runoff_round_pct ?? 0) * 100).toFixed(2)}%</td>
                      <td className="px-2 py-1.5 text-right font-data">{(t.policy_rate_pct ?? 0).toFixed(2)}%</td>
                      <td className="px-2 py-1.5">
                        {breaches.length > 0 ? (
                          <span className="text-ki-error font-semibold">{breaches.join(", ")}</span>
                        ) : (
                          <span className="text-ki-on-surface-muted">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Report Section (Markdown — dynamic import) */}
      {reportMarkdown && reportMarkdown.trim().length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-end justify-between mb-4">
            <div>
              <div className="eyebrow mb-1">Editorial</div>
              <h2 className="text-[18px] font-medium tracking-tight2 text-ki-on-surface">Report di simulazione</h2>
            </div>
            <a
              href={`/api/scenarios/${id}/report.html`}
              target="_blank"
              rel="noopener"
              className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm bg-ki-on-surface text-ki-surface text-[11px] font-medium hover:bg-ki-on-surface-secondary transition-colors"
              title="Apri il report stampabile in una nuova scheda. Premi ⌘P / Ctrl+P per esportare in PDF."
            >
              <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 500" }}>
                download
              </span>
              Export PDF
            </a>
          </div>
          <ReportSection markdown={reportMarkdown} />
        </div>
      )}
    </div>
  );
}
