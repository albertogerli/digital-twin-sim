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
import { generateFinancialImpact } from "../../../lib/generate-financial-impact";
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
  const [financialImpact, setFinancialImpact] = useState<RoundFinancial[]>([]);
  const [finProvenance, setFinProvenance] = useState<Provenance>("client-fallback");
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

        // Financial impact: prefer backend-simulated payload (schema-validated),
        // fall back to client-side heuristic generator with provenance marker.
        let finRounds: RoundFinancial[] = [];
        let provenance: Provenance = "client-fallback";
        let schemaVersion = FIN_SCHEMA_VERSION;

        if (isCompatible(finImpact)) {
          finRounds = finImpact.rounds;
          provenance = finImpact.provenance ?? "backend-simulated";
          schemaVersion = finImpact.schema_version;
        } else if (Array.isArray(finImpact) && finImpact.length > 0) {
          // Legacy: bare round array — accept but mark as unknown schema
          finRounds = finImpact as RoundFinancial[];
          provenance = (finImpact[0] as any)?.provenance ?? "client-fallback";
          schemaVersion = (finImpact[0] as any)?.schema_version ?? "legacy";
        } else {
          finRounds = generateFinancialImpact(
            meta.domain,
            validRounds.sort((a, b) => a.round - b.round),
            pol,
          );
          provenance = "client-fallback";
        }
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
      <div className="min-h-screen bg-white flex flex-col items-center justify-center gap-6 px-4">
        <div className="bg-red-50 border border-red-200 rounded-xl p-8 max-w-md text-center">
          <svg
            className="w-12 h-12 text-red-400 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
            />
          </svg>
          <p className="text-red-700 text-lg font-medium mb-2">
            Failed to Load Scenario
          </p>
          <p className="text-red-600 text-sm mb-6">
            {error || "Scenario not found."}
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-cyan-600 hover:text-cyan-600 transition-colors text-sm font-medium"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
            Back to home
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
    <div className="min-h-screen bg-white text-gray-900">
      {/* Back link */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-gray-500 hover:text-cyan-600 transition-colors text-sm"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
          Back to scenarios
        </Link>
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

      {/* Report Section (Markdown — dynamic import) */}
      {reportMarkdown && reportMarkdown.trim().length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-headline font-bold">Report di simulazione</h2>
            <a
              href={`/api/scenarios/${id}/report.html`}
              target="_blank"
              rel="noopener"
              className="px-3 py-1.5 rounded-sm bg-ki-primary hover:bg-ki-primary-muted text-white font-semibold transition-colors text-xs flex items-center gap-1.5"
              title="Apri il report stampabile in una nuova scheda. Premi ⌘P / Ctrl+P per esportare in PDF."
            >
              <svg width="12" height="12" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M6 2v6m0 0L3 5m3 3l3-3M3 13v3a1 1 0 001 1h12a1 1 0 001-1v-3" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Esporta Report (PDF)
            </a>
          </div>
          <ReportSection markdown={reportMarkdown} />
        </div>
      )}
    </div>
  );
}
