"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import BriefingProgress from "@/components/sim/BriefingProgress";
import RoundPhaseIndicator from "@/components/sim/RoundPhaseIndicator";
import PolarizationChart from "@/components/sim/PolarizationChart";
import SentimentEvolution from "@/components/sim/SentimentEvolution";
import AgentPositionStrip from "@/components/sim/AgentPositionStrip";
import CoalitionEvolution from "@/components/sim/CoalitionEvolution";
import ShockBadge from "@/components/sim/ShockBadge";
import MonteCarloPanel from "@/components/sim/MonteCarloPanel";
import ConfidenceBand from "@/components/sim/ConfidenceBand";
import RegimeIndicator from "@/components/sim/RegimeIndicator";
import CalibrationBadge from "@/components/sim/CalibrationBadge";
import ChartErrorBoundary from "@/components/sim/ChartErrorBoundary";

// --- Types ---

interface SimStatus {
  id: string;
  status: string;
  brief: string;
  scenario_name?: string;
  scenario_id?: string;
  domain?: string;
  current_round: number;
  total_rounds: number;
  cost: number;
  created_at: string;
  completed_at?: string;
  error?: string;
  agents_count: number;
}

interface ProgressEvent {
  type: string;
  message: string;
  round?: number;
  phase?: string;
  data: Record<string, any>;
}

interface LiveRound {
  round: number;
  timeline_label: string;
  event: string;
  polarization: number;
  posts_count: number;
  reactions_count: number;
  top_posts: any[];
  agents: any[];
  sentiment: { positive: number; neutral: number; negative: number };
  coalitions: any[];
  custom_metrics: Record<string, number>;
  cost: number;
  shock_magnitude: number;
  shock_direction: number;
  confidence_interval?: {
    pro_pct_mean: number;
    pro_pct_ci95_lo: number;
    pro_pct_ci95_hi: number;
    sigma_pp: number;
  } | null;
  regime_info?: { regime_prob: number; regime_label: string } | null;
  calibration_source?: string;
}

interface BriefingStep {
  phase: string;
  message: string;
  done: boolean;
}

interface CurrentPhase {
  index: number;
  total: number;
  message: string;
}

interface PositionAxis {
  negative_label: string;
  positive_label: string;
}

// --- Helpers ---

function formatEngagement(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

// --- Component ---

export default function SimulationLiveDashboard({ params }: { params: { id: string } }) {
  const { id } = params;
  const [status, setStatus] = useState<SimStatus | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [rounds, setRounds] = useState<LiveRound[]>([]);
  const [connected, setConnected] = useState(false);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [showLog, setShowLog] = useState(false);
  const [warning, setWarning] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // New state for enriched dashboard
  const [briefingSteps, setBriefingSteps] = useState<BriefingStep[]>([]);
  const [currentPhase, setCurrentPhase] = useState<CurrentPhase>({ index: 0, total: 7, message: "" });
  const [positionAxis, setPositionAxis] = useState<PositionAxis | null>(null);
  const [monteCarloData, setMonteCarloData] = useState<any>(null);

  // Load initial status — if already completed, hydrate rounds from exports
  useEffect(() => {
    fetch(`/api/simulations/${id}`)
      .then((r) => r.json())
      .then(async (data: SimStatus) => {
        setStatus(data);
        if (data.scenario_id) setScenarioId(data.scenario_id);

        // If sim is already completed and we have no rounds, load from exports
        if (data.status === "completed" && data.scenario_id && rounds.length === 0) {
          try {
            const totalRounds = data.total_rounds || 9;
            const loadedRounds: LiveRound[] = [];
            for (let r = 1; r <= totalRounds; r++) {
              const res = await fetch(`/api/scenarios/${data.scenario_id}/replay_round_${r}.json`);
              if (!res.ok) break;
              const raw = await res.json();
              const ind = raw.indicators || {};
              const sent = ind.sentiment || { positive: 0, neutral: 1, negative: 0 };
              const coalitions = raw.coalitions?.coalitions || [];
              const topPosts = [...(raw.posts || [])]
                .sort((a: any, b: any) => (b.engagement_score ?? 0) - (a.engagement_score ?? 0))
                .slice(0, 10)
                .map((p: any) => ({
                  id: p.id,
                  author_id: p.author_id,
                  author_name: p.author_name || p.author_id,
                  platform: p.platform,
                  text: p.text,
                  likes: p.likes || 0,
                  reposts: p.reposts || 0,
                  replies: p.replies || 0,
                  total_engagement: (p.likes || 0) + (p.reposts || 0) * 2 + (p.replies || 0) * 3,
                }));
              const agents = (raw.graphSnapshot?.nodes || []).map((n: any) => ({
                id: n.id,
                name: n.label || n.name || n.id,
                role: n.role || "",
                position: n.position ?? 0,
                emotional_state: n.emotional_state || "neutral",
                tier: n.tier ?? 1,
                cluster_size: n.cluster_size,
              }));
              loadedRounds.push({
                round: raw.round || r,
                timeline_label: raw.month || `Round ${r}`,
                event: raw.event?.event || "",
                polarization: ind.polarization ?? 0,
                posts_count: (raw.posts || []).length,
                reactions_count: 0,
                top_posts: topPosts,
                agents,
                sentiment: { positive: sent.positive ?? 0, neutral: sent.neutral ?? 0, negative: sent.negative ?? 0 },
                coalitions,
                custom_metrics: {},
                cost: data.cost || 0,
                shock_magnitude: raw.event?.shock_magnitude ?? 0,
                shock_direction: raw.event?.shock_direction ?? 0,
                confidence_interval: raw.confidence_interval || null,
                regime_info: raw.regime_info || null,
              });
            }
            if (loadedRounds.length > 0) {
              setRounds(loadedRounds);
            }
          } catch (err) {
            console.warn("Could not load replay rounds for completed sim:", err);
          }
        }
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // SSE connection
  useEffect(() => {
    const es = new EventSource(`/api/simulations/${id}/stream`);

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const handleEvent = (_type: string) => (e: MessageEvent) => {
      try {
        const data: ProgressEvent = JSON.parse(e.data);
        setEvents((prev) => [...prev, data]);

        if (data.type === "brief_analyzed" && data.data) {
          setStatus((prev) => prev ? {
            ...prev,
            status: "configuring",
            scenario_name: data.data.scenario_name,
            domain: data.data.domain,
            total_rounds: data.data.num_rounds,
            agents_count: (data.data.elite_agents || 0) + (data.data.institutional_agents || 0) + (data.data.citizen_clusters || 0),
          } : prev);
          // Store position axis
          if (data.data.position_axis) {
            setPositionAxis({
              negative_label: data.data.position_axis.negative_label,
              positive_label: data.data.position_axis.positive_label,
            });
          }
          // Mark all briefing steps as done
          setBriefingSteps((prev) => prev.map((s) => ({ ...s, done: true })));
        }

        // Briefing phase events (pre-round: web_research, entity_research, agent_generation)
        if (data.type === "round_phase" && data.phase && !data.round) {
          const briefingPhases = ["web_research", "documents", "entity_research", "agent_generation", "brief_analysis"];
          if (briefingPhases.includes(data.phase)) {
            setBriefingSteps((prev) => {
              // Mark previous steps as done, add new one
              const updated = prev.map((s) => ({ ...s, done: true }));
              // Only add if different phase or new message
              const existing = updated.find((s) => s.phase === data.phase && !s.done);
              if (!existing) {
                updated.push({ phase: data.phase!, message: data.message, done: false });
              }
              return updated;
            });
          }
        }

        // Round phase events (during round execution)
        if (data.type === "round_phase" && data.data?.phase_index) {
          setCurrentPhase({
            index: data.data.phase_index,
            total: data.data.total_phases || 7,
            message: data.message,
          });
        }

        // Warning phase (e.g. all agents failed)
        if (data.type === "round_phase" && data.data?.phase === "warning") {
          setWarning(data.message || "Warning durante la simulazione");
        }

        if (data.type === "round_start") {
          setStatus((prev) => prev ? {
            ...prev,
            status: "running",
            current_round: data.round || prev.current_round,
          } : prev);
          // Reset phase indicator for new round
          setCurrentPhase({ index: 0, total: 7, message: "" });
        }

        // Accumulate round data for live dashboard
        if (data.type === "round_complete" && data.data) {
          const d = data.data;
          const liveRound: LiveRound = {
            round: d.round || data.round || 0,
            timeline_label: d.timeline_label || `Round ${d.round}`,
            event: d.event || "",
            polarization: d.polarization || 0,
            posts_count: d.posts_count || 0,
            reactions_count: d.reactions_count || 0,
            top_posts: d.top_posts || [],
            agents: d.agents || [],
            sentiment: d.sentiment || { positive: 0, neutral: 0, negative: 0 },
            coalitions: d.coalitions || [],
            custom_metrics: d.custom_metrics || {},
            cost: d.cost || 0,
            shock_magnitude: d.shock_magnitude ?? 0,
            shock_direction: d.shock_direction ?? 0,
            confidence_interval: d.confidence_interval || null,
            regime_info: d.regime_info || null,
            calibration_source: d.calibration_source || "",
          };
          setRounds((prev) => [...prev, liveRound]);
          setStatus((prev) => prev ? {
            ...prev,
            status: "running",
            current_round: liveRound.round,
            cost: liveRound.cost,
          } : prev);
          // Reset phase indicator
          setCurrentPhase({ index: 7, total: 7, message: "Round completato" });
        }

        if (data.type === "completed") {
          setStatus((prev) => prev ? { ...prev, status: "completed", cost: data.data?.cost || prev.cost } : prev);
          setScenarioId(data.data?.scenario_id || null);
          if (data.data?.monte_carlo) {
            setMonteCarloData(data.data.monte_carlo);
          }
          es.close();
        }

        if (data.type === "error") {
          setStatus((prev) => prev ? { ...prev, status: "failed", error: data.message } : prev);
          es.close();
        }

        if (data.type === "cancelled") {
          setStatus((prev) => prev ? { ...prev, status: "cancelled" } : prev);
          es.close();
        }
      } catch {}
    };

    for (const type of [
      "status", "brief_analyzed", "round_start", "round_phase",
      "round_complete", "completed", "error", "cancelled", "heartbeat"
    ]) {
      es.addEventListener(type, handleEvent(type));
    }

    return () => es.close();
  }, [id]);

  // Auto-scroll to new rounds
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [rounds]);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [events]);

  async function handleCancel() {
    await fetch(`/api/simulations/${id}`, { method: "DELETE" });
  }

  const isActive = status && ["queued", "analyzing", "configuring", "running", "exporting"].includes(status.status);
  const isBriefing = status && ["queued", "analyzing", "configuring"].includes(status.status);
  const progressPct = status && status.total_rounds > 0
    ? (status.current_round / status.total_rounds) * 100
    : 0;

  // Collect all custom metric keys across rounds
  const allMetricKeys = [...new Set(rounds.flatMap((r) => Object.keys(r.custom_metrics)))];

  // Chart data
  const polarizationData = rounds.map((r) => ({ round: r.round, polarization: r.polarization }));
  const sentimentData = rounds.map((r) => ({ round: r.round, ...r.sentiment }));
  const coalitionData = rounds.map((r) => ({ round: r.round, coalitions: r.coalitions }));
  const latestRound = rounds.length > 0 ? rounds[rounds.length - 1] : null;

  return (
    <main className="text-ki-on-surface">
      <div className="max-w-7xl mx-auto px-4 py-4">
        {/* Header */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-0.5">
            <h1 className="text-base font-headline font-extrabold">
              {status?.scenario_name || "Simulazione in avvio..."}
            </h1>
            {status?.domain && (
              <span className="px-2 py-0.5 rounded-sm text-[10px] font-semibold uppercase tracking-wide bg-ki-surface-sunken border border-ki-border text-ki-on-surface-muted">
                {status.domain.replace("_", " ")}
              </span>
            )}
            {connected && isActive && (
              <span className="flex items-center gap-1 text-[10px] text-ki-success font-medium">
                <span className="w-1.5 h-1.5 bg-ki-success rounded-full animate-pulse" />
                Live
              </span>
            )}
            {/* Calibration badge */}
            {latestRound?.calibration_source && (
              <CalibrationBadge source={latestRound.calibration_source} showDetails />
            )}
            {/* Regime indicator */}
            {latestRound?.regime_info && latestRound.regime_info.regime_prob > 0.05 && (
              <RegimeIndicator regimeInfo={latestRound.regime_info} />
            )}
          </div>
          <p className="text-ki-on-surface-muted text-xs">{status?.brief}</p>
        </div>

        {/* Progress bar + RoundPhaseIndicator */}
        <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-2 mb-4">
          <div className="flex items-center justify-between mb-1.5 text-xs">
            <span className="font-semibold">
              {status?.status === "completed" ? "Completata!" :
               status?.status === "failed" ? "Errore" :
               status?.status === "running" ? `Round ${status.current_round} di ${status.total_rounds}` :
               status?.status?.replace("_", " ") || "..."}
            </span>
            <div className="flex items-center gap-3 text-ki-on-surface-muted font-data text-[10px]">
              <span>{status?.agents_count || "..."} agenti</span>
              <span>${(status?.cost || 0).toFixed(3)}</span>
            </div>
          </div>
          <div className="w-full bg-ki-surface-sunken rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${
                status?.status === "completed" ? "bg-ki-success" :
                status?.status === "failed" ? "bg-ki-error" : "bg-ki-primary"
              }`}
              style={{ width: `${status?.status === "completed" ? 100 : progressPct}%` }}
            />
          </div>
          {/* Round Phase Indicator (only during round execution) */}
          {status?.status === "running" && currentPhase.index > 0 && (
            <RoundPhaseIndicator
              phaseIndex={currentPhase.index}
              totalPhases={currentPhase.total}
              message={currentPhase.message}
            />
          )}
        </div>

        {/* Briefing Progress (pre-round phase) — visible anche prima del primo evento */}
        <BriefingProgress steps={briefingSteps} visible={!!isBriefing} />

        {/* Main content: 2-column layout on desktop */}
        {rounds.length > 0 && (
          <div className="flex flex-col lg:flex-row gap-4 mb-4">
            {/* LEFT: Round cards + Custom metrics */}
            <div className="flex-1 min-w-0 lg:w-[60%]">
              {/* Custom Metrics Chart (if any) */}
              {allMetricKeys.length > 0 && (
                <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-3 mb-3">
                  <h2 className="text-xs font-semibold text-ki-on-surface-secondary mb-3">Metriche Scenario</h2>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {allMetricKeys.map((key) => {
                      const latest = rounds[rounds.length - 1]?.custom_metrics[key] ?? 0;
                      const prev = rounds.length > 1 ? rounds[rounds.length - 2]?.custom_metrics[key] ?? 0 : null;
                      const delta = prev !== null ? latest - prev : null;
                      return (
                        <div key={key} className="bg-ki-surface-sunken rounded-sm p-2">
                          <div className="text-[10px] font-data uppercase text-ki-on-surface-muted mb-1 truncate" title={key}>
                            {key}
                          </div>
                          <div className="flex items-end gap-1.5">
                            <span className="text-xl font-bold text-ki-on-surface">{latest}</span>
                            <span className="text-[10px] text-ki-on-surface-muted">/100</span>
                            {delta !== null && delta !== 0 && (
                              <span className={`text-[10px] font-semibold ${delta > 0 ? "text-ki-success" : "text-ki-error"}`}>
                                {delta > 0 ? "+" : ""}{delta}
                              </span>
                            )}
                          </div>
                          {rounds.length > 1 && (
                            <div className="flex items-end gap-px mt-1.5 h-5">
                              {rounds.map((r, i) => {
                                const val = r.custom_metrics[key] ?? 0;
                                return (
                                  <div
                                    key={i}
                                    className="flex-1 bg-ki-primary rounded-t-sm min-w-[3px] transition-all duration-300"
                                    style={{ height: `${Math.max(val, 2)}%` }}
                                  />
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Round Cards */}
              <div className="space-y-3">
                {rounds.map((round) => (
                  <LiveRoundCard
                    key={round.round}
                    round={round}
                    positionAxis={positionAxis}
                  />
                ))}
              </div>
            </div>

            {/* RIGHT: Sidebar charts */}
            <div className="lg:w-[40%] space-y-3">
              <ChartErrorBoundary fallbackLabel="Polarizzazione non disponibile">
                <PolarizationChart rounds={polarizationData} />
              </ChartErrorBoundary>
              <ChartErrorBoundary fallbackLabel="Sentiment non disponibile">
                <SentimentEvolution rounds={sentimentData} />
              </ChartErrorBoundary>
              {latestRound && (
                <ChartErrorBoundary fallbackLabel="Posizioni agenti non disponibili">
                  <AgentPositionStrip
                    agents={latestRound.agents}
                    negativeLabel={positionAxis?.negative_label}
                    positiveLabel={positionAxis?.positive_label}
                  />
                </ChartErrorBoundary>
              )}
              <ChartErrorBoundary fallbackLabel="Coalizioni non disponibili">
                <CoalitionEvolution rounds={coalitionData} />
              </ChartErrorBoundary>
              {/* Confidence Band (v2 calibration) */}
              {rounds.some((r) => r.confidence_interval) && (
                <ChartErrorBoundary fallbackLabel="Intervallo di confidenza non disponibile">
                  <ConfidenceBand
                    rounds={rounds.map((r) => ({ round: r.round, confidence_interval: r.confidence_interval }))}
                    positionAxis={positionAxis}
                  />
                </ChartErrorBoundary>
              )}
            </div>
          </div>
        )}

        {/* Waiting animation when no rounds yet */}
        {rounds.length === 0 && isActive && !isBriefing && (
          <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-6 mb-4 text-center">
            <div className="animate-pulse space-y-2">
              <div className="h-3 w-48 bg-ki-surface-sunken rounded mx-auto" />
              <div className="h-2 w-64 bg-ki-surface-sunken rounded mx-auto" />
              <p className="text-xs text-ki-on-surface-muted mt-3">
                In attesa del primo round...
              </p>
            </div>
          </div>
        )}

        {/* Monte Carlo Results */}
        {monteCarloData && (
          <div className="mb-4">
            <MonteCarloPanel
              data={monteCarloData}
              positiveLabel={positionAxis?.positive_label}
              negativeLabel={positionAxis?.negative_label}
            />
          </div>
        )}

        {/* Error */}
        {status?.status === "failed" && status.error && (
          <div className="bg-ki-error/10 border border-ki-error/25 rounded-sm p-3 mb-4">
            <h3 className="text-xs font-semibold text-ki-error mb-1">Errore</h3>
            <p className="text-ki-error text-xs font-data">{status.error}</p>
          </div>
        )}

        {/* Warning banner */}
        {warning && (
          <div className="mb-3 bg-ki-warning/10 border border-ki-warning/25 rounded-sm p-3 flex items-start gap-2">
            <svg className="w-4 h-4 text-ki-warning flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <div className="flex-1">
              <p className="text-xs font-medium text-ki-warning">{warning}</p>
              <p className="text-[10px] text-ki-warning/80 mt-0.5">Il parser JSON potrebbe non aver gestito la risposta LLM. I risultati di questo round potrebbero essere incompleti.</p>
            </div>
            <button onClick={() => setWarning(null)} className="text-ki-warning/50 hover:text-ki-warning">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}

        {/* Event Log (collapsible) */}
        <div className="mb-4">
          <button
            onClick={() => setShowLog(!showLog)}
            className="flex items-center gap-2 text-[10px] text-ki-on-surface-muted hover:text-ki-on-surface-secondary transition-colors mb-1.5"
          >
            <svg className={`w-3 h-3 transition-transform ${showLog ? "rotate-90" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Log eventi ({events.filter(e => e.type !== "heartbeat").length})
          </button>
          {showLog && (
            <div className="bg-ki-surface-raised border border-ki-border rounded-sm overflow-hidden">
              <div
                ref={logRef}
                className="h-40 overflow-y-auto p-2 space-y-0.5 font-data text-[11px]"
              >
                {events.filter(e => e.type !== "heartbeat").map((ev, i) => (
                  <div
                    key={i}
                    className={`${
                      ev.type === "error" ? "text-ki-error" :
                      ev.type === "completed" ? "text-ki-success" :
                      ev.type === "round_start" ? "text-ki-primary" :
                      ev.type === "round_complete" ? "text-ki-primary-muted" :
                      ev.type === "brief_analyzed" ? "text-ki-warning" :
                      (ev.phase === "warning" ? "text-ki-warning" :
                      "text-ki-on-surface-muted")
                    }`}
                  >
                    <span className="text-ki-border select-none">[{ev.type}]</span> {ev.message}
                  </div>
                ))}
                {events.length === 0 && (
                  <div className="text-ki-border animate-pulse">In attesa...</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          {isActive && (
            <button
              onClick={handleCancel}
              className="px-4 py-1.5 rounded-sm bg-ki-error/10 hover:bg-ki-error/20 text-ki-error font-medium transition-colors text-xs"
            >
              Annulla
            </button>
          )}
          {status?.status === "completed" && scenarioId && (
            <>
              <Link
                href={`/scenario/${scenarioId}`}
                className="px-4 py-1.5 rounded-sm bg-ki-primary hover:bg-ki-primary-muted text-white font-semibold transition-colors text-xs"
              >
                Vedi Dashboard Completa
              </Link>
              <a
                href={`/api/scenarios/${scenarioId}/report.html`}
                target="_blank"
                rel="noopener"
                className="px-4 py-1.5 rounded-sm bg-ki-surface-raised border border-ki-border hover:bg-ki-surface-hover text-ki-on-surface font-semibold transition-colors text-xs flex items-center gap-1.5"
                title="Apri il report in una nuova scheda. Premi ⌘P / Ctrl+P per esportare in PDF."
              >
                <svg width="12" height="12" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M6 2v6m0 0L3 5m3 3l3-3M3 13v3a1 1 0 001 1h12a1 1 0 001-1v-3" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Esporta Report (PDF)
              </a>
            </>
          )}
          <Link
            href="/new"
            className="px-4 py-1.5 rounded-sm bg-ki-surface-sunken hover:bg-ki-surface-hover text-ki-on-surface-secondary font-medium transition-colors text-xs"
          >
            Nuova Simulazione
          </Link>
        </div>

        <div ref={bottomRef} />
      </div>
    </main>
  );
}

// --- Live Round Card ---

function LiveRoundCard({ round, positionAxis }: { round: LiveRound; positionAxis: PositionAxis | null }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
      {/* Round header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2.5 flex items-center gap-3 hover:bg-ki-surface-hover transition-colors cursor-pointer"
      >
        <div className="w-8 h-8 rounded-sm bg-ki-primary/10 border border-ki-primary/25 flex items-center justify-center flex-shrink-0">
          <span className="font-data text-xs font-bold text-ki-primary">{round.round}</span>
        </div>
        <div className="flex-1 min-w-0 text-left">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-headline font-semibold text-sm text-ki-on-surface">{round.timeline_label}</span>
            <span className="px-1.5 py-0.5 rounded-sm bg-ki-surface-sunken font-data text-[10px] text-ki-on-surface-muted">
              {round.posts_count} posts
            </span>
            {round.reactions_count > 0 && (
              <span className="px-1.5 py-0.5 rounded-sm bg-ki-surface-sunken font-data text-[10px] text-ki-on-surface-muted">
                {round.reactions_count} reactions
              </span>
            )}
            <span className="px-1.5 py-0.5 rounded-sm bg-ki-surface-sunken font-data text-[10px] text-ki-on-surface-muted">
              pol. {round.polarization.toFixed(1)}
            </span>
            {round.shock_magnitude > 0 && (
              <ShockBadge magnitude={round.shock_magnitude} direction={round.shock_direction} />
            )}
            {round.posts_count === 0 && (
              <span className="px-1.5 py-0.5 rounded-sm bg-ki-warning/10 border border-ki-warning/25 font-data text-[10px] text-ki-warning">
                no content
              </span>
            )}
          </div>
          <p className="text-xs text-ki-on-surface-muted mt-0.5">{round.event}</p>
        </div>

        {/* Sentiment mini-bar */}
        <div className="flex h-5 w-14 rounded-sm overflow-hidden flex-shrink-0">
          <div className="bg-ki-success" style={{ width: `${round.sentiment.positive * 100}%` }} />
          <div className="bg-ki-border" style={{ width: `${round.sentiment.neutral * 100}%` }} />
          <div className="bg-ki-error" style={{ width: `${round.sentiment.negative * 100}%` }} />
        </div>

        {/* Custom metrics badges */}
        {Object.entries(round.custom_metrics).slice(0, 2).map(([k, v]) => (
          <span key={k} className="hidden md:inline-block px-1.5 py-0.5 rounded-sm bg-ki-primary/10 font-data text-[10px] text-ki-primary flex-shrink-0">
            {k.slice(0, 20)}: {v}
          </span>
        ))}

        <svg className={`w-3.5 h-3.5 text-ki-on-surface-muted transition-transform flex-shrink-0 ${expanded ? "rotate-180" : ""}`} fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-3 pb-3 border-t border-ki-border pt-3">
          {/* Agent Position Strip in expanded view */}
          {round.agents.length > 0 && (
            <div className="mb-3">
              <AgentPositionStrip
                agents={round.agents}
                negativeLabel={positionAxis?.negative_label}
                positiveLabel={positionAxis?.positive_label}
              />
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            {/* Top posts */}
            <div className="lg:col-span-2 space-y-1.5">
              <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-1.5">Top Posts</p>
              {round.top_posts.slice(0, 5).map((post, i) => (
                <div key={post.id || i} className="bg-ki-surface-sunken border border-ki-border rounded-sm p-2">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <div className="w-4 h-4 rounded-sm bg-ki-surface-hover flex items-center justify-center text-ki-on-surface-muted font-data text-[8px] font-bold">
                      {(post.author_name || "?").charAt(0).toUpperCase()}
                    </div>
                    <span className="text-[11px] font-semibold text-ki-on-surface-secondary">{post.author_name}</span>
                    <span className="px-1 py-0.5 rounded-sm bg-ki-surface-sunken font-data text-[9px] text-ki-on-surface-muted uppercase">
                      {post.platform}
                    </span>
                    <span className="ml-auto font-data text-[10px] text-ki-primary">
                      {formatEngagement(post.total_engagement || 0)}
                    </span>
                  </div>
                  <p className="text-[11px] text-ki-on-surface-secondary leading-relaxed line-clamp-3">{post.text}</p>
                </div>
              ))}
              {round.top_posts.length === 0 && (
                <p className="text-[11px] text-ki-on-surface-muted italic">Nessun post in questo round</p>
              )}
            </div>

            {/* Sidebar: coalitions + metrics + agents */}
            <div className="space-y-3">
              {/* Custom metrics */}
              {Object.keys(round.custom_metrics).length > 0 && (
                <div>
                  <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-1.5">Metriche</p>
                  <div className="space-y-1.5">
                    {Object.entries(round.custom_metrics).map(([k, v]) => (
                      <div key={k}>
                        <div className="flex justify-between text-[11px] mb-0.5">
                          <span className="text-ki-on-surface-secondary truncate">{k}</span>
                          <span className="font-data text-ki-on-surface font-semibold">{v}/100</span>
                        </div>
                        <div className="h-1 bg-ki-surface-sunken rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-ki-primary transition-all duration-500"
                            style={{ width: `${v}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Coalitions */}
              {round.coalitions.length > 0 && (
                <div>
                  <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-1.5">Coalizioni</p>
                  <div className="space-y-0.5">
                    {round.coalitions.map((c: any, i: number) => (
                      <div key={i} className="flex items-center gap-2 text-[11px]">
                        <span className="text-ki-on-surface-secondary truncate flex-1">{c.label || `Coalition ${i + 1}`}</span>
                        <span className="font-data text-ki-on-surface">{c.size || c.members?.length || "?"}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Agent mood */}
              <div>
                <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-1.5">Sentiment</p>
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="text-ki-success">{(round.sentiment.positive * 100).toFixed(0)}% pos</span>
                  <span className="text-ki-on-surface-muted">{(round.sentiment.neutral * 100).toFixed(0)}% neu</span>
                  <span className="text-ki-error">{(round.sentiment.negative * 100).toFixed(0)}% neg</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
