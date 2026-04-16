/**
 * Wargame SSE Client — connects to FastAPI backend and streams round data.
 *
 * Flow:
 *   1. POST /api/simulations  → {id, status}
 *   2. GET  /api/simulations/{id}/stream  → SSE events
 *   3. POST /api/simulations/{id}/intervene → submit player action
 */

import type {
  WgAgent,
  WgPost,
  WgTicker,
  WgRoundState,
  WgSitrep,
  WargameConfig,
  BriefingStep,
  LogEntry,
  FlashColor,
} from "./wargame-types";

// ── SSE Event Types ──────────────────────────────────────────────

interface SSEEvent {
  type: string;
  message: string;
  round: number | null;
  phase: string | null;
  data: Record<string, unknown>;
  confidence_interval?: Record<string, unknown>;
  regime_info?: Record<string, unknown>;
}

// ── Callbacks ────────────────────────────────────────────────────

export interface WargameCallbacks {
  onPhaseChange: (phase: string) => void;
  onBriefingStep: (step: BriefingStep) => void;
  onRoundStart: (round: number, message: string) => void;
  onRoundPhase: (round: number, phase: string, message: string) => void;
  onRoundComplete: (
    state: WgRoundState,
    posts: WgPost[],
    tickers: WgTicker[],
    agents: WgAgent[],
  ) => void;
  onAwaitingIntervention: (sitrep: WgSitrep) => void;
  onCompleted: (message: string) => void;
  onError: (message: string) => void;
  onLog: (entry: LogEntry) => void;
  onAgentCount: (count: number) => void;
}

// ── Helpers: Backend → Frontend type mapping ─────────────────────

function mapPosts(rawPosts: Record<string, unknown>[], round: number): WgPost[] {
  return rawPosts.map((p, i) => {
    const authorId = (p.author_id as string) || `unk_${i}`;
    const engagement = (p.total_engagement as number) || (p.engagement as number) || 0;
    const tone = ((p.emotional_tone as string) || "neutral").toLowerCase();
    let sentiment: WgPost["sentiment"] = "neutral";
    if (tone.includes("anger") || tone.includes("fear") || tone.includes("negativ") || tone.includes("outrage") || tone.includes("alarm") || tone.includes("frustrat")) {
      sentiment = "negative";
    } else if (tone.includes("hope") || tone.includes("positiv") || tone.includes("relief") || tone.includes("optim") || tone.includes("confiden") || tone.includes("reassur")) {
      sentiment = "positive";
    }
    return {
      id: `${round}_${i}`,
      authorId,
      authorName: (p.author_name as string) || "Unknown",
      authorTier: authorId.startsWith("cit") ? 3 : authorId.startsWith("inst") ? 2 : 1,
      text: (p.text as string) || "",
      platform: (p.platform as string) || "X",
      engagement,
      sentiment,
      round,
      ts: i * 400,
    };
  });
}

function mapTickers(financial: Record<string, unknown>): WgTicker[] {
  const tickerImpacts = (financial.ticker_impacts as Record<string, unknown>[]) || [];
  return tickerImpacts.map((t) => {
    const t1 = (t.t1_pct as number) || 0;
    let flash: FlashColor = "none";
    if (t1 < -1) flash = "red";
    else if (t1 > 0.5) flash = "green";
    return {
      ticker: (t.ticker as string) || "???",
      sector: (t.sector as string) || "Unknown",
      t1,
      t3: (t.t3_pct as number) || 0,
      t7: (t.t7_pct as number) || 0,
      direction: (t.direction as WgTicker["direction"]) || "short",
      beta: (t.beta as number) || 1.0,
      confidence: (t.confidence as number) || 0.5,
      flash,
    };
  });
}

function mapAgents(agentsData: Record<string, unknown>): WgAgent[] {
  const agents: WgAgent[] = [];

  // Elite agents
  const elites = (agentsData.elite as Record<string, unknown>[]) || [];
  for (const a of elites) {
    agents.push({
      id: (a.id as string) || `elite_${agents.length}`,
      name: (a.name as string) || "Unknown Elite",
      role: (a.role as string) || "",
      tier: 1,
      position: (a.position as number) || 0,
      sentiment: classifyAgentSentiment(a),
      cluster: (a.cluster as string) || (a.coalition as string) || "Elite",
    });
  }

  // Institutional agents
  const inst = (agentsData.institutional as Record<string, unknown>[]) || [];
  for (const a of inst) {
    agents.push({
      id: (a.id as string) || `inst_${agents.length}`,
      name: (a.name as string) || "Unknown Institution",
      role: (a.role as string) || (a.category as string) || "",
      tier: 2,
      position: (a.position as number) || 0,
      sentiment: classifyAgentSentiment(a),
      cluster: (a.cluster as string) || (a.coalition as string) || "Institutional",
    });
  }

  // Citizen clusters
  const citizens = (agentsData.citizen_clusters as Record<string, unknown>[]) || [];
  for (const c of citizens) {
    const clusterName = (c.name as string) || (c.label as string) || "Citizens";
    const size = (c.size as number) || (c.cluster_size as number) || 10;
    const pos = (c.avg_position as number) || (c.position as number) || 0;
    // Create representative agents for citizen cluster
    for (let i = 0; i < Math.min(size, 50); i++) {
      const jitter = (Math.random() - 0.5) * 0.3;
      agents.push({
        id: `cit_${clusterName}_${i}`,
        name: `${clusterName} #${i + 1}`,
        role: "citizen",
        tier: 3,
        position: Math.max(-1, Math.min(1, pos + jitter)),
        sentiment: pos + jitter > 0.2 ? "positive" : pos + jitter < -0.2 ? "negative" : "neutral",
        cluster: clusterName,
      });
    }
  }

  return agents;
}

function classifyAgentSentiment(a: Record<string, unknown>): WgAgent["sentiment"] {
  const emotion = ((a.emotional_state as string) || "").toLowerCase();
  if (emotion.includes("anger") || emotion.includes("fear") || emotion.includes("alarm") || emotion.includes("frustrat") || emotion.includes("outrage")) return "negative";
  if (emotion.includes("hope") || emotion.includes("relief") || emotion.includes("confid") || emotion.includes("optim")) return "positive";
  const pos = (a.position as number) || 0;
  if (pos < -0.4) return "negative";
  if (pos > 0.4) return "positive";
  return "neutral";
}

function deriveWarning(cri: number, wave: number, pol: number): string {
  if (cri > 0.8 || (wave >= 3 && pol > 7)) return "CRITICAL";
  if (cri > 0.6 || wave >= 3 || pol > 7) return "HIGH";
  if (cri > 0.3 || wave >= 2 || pol > 5) return "MODERATE";
  return "LOW";
}

// ── API Functions ────────────────────────────────────────────────

export async function createSimulation(config: WargameConfig): Promise<string> {
  const body = {
    brief: config.brief,
    provider: config.provider || "gemini",
    model: config.model || undefined,
    domain: config.domain || undefined,
    rounds: config.rounds || undefined,
    budget: config.budget || 5.0,
    wargame_mode: true,
    player_role: config.playerRole || "",
    metrics_to_track: config.metricsToTrack || [],
  };

  const res = await fetch("/api/simulations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Failed to create simulation: ${res.status} — ${err}`);
  }

  const data = await res.json();
  return data.id as string;
}

export async function submitIntervention(
  simId: string,
  actionText: string,
  actionType: string = "press_release",
  targetAudience: string = "",
  skip: boolean = false,
): Promise<void> {
  const body = {
    action_text: actionText,
    action_type: actionType,
    target_audience: targetAudience,
    skip,
  };

  const res = await fetch(`/api/simulations/${simId}/intervene`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Intervention failed: ${res.status} — ${err}`);
  }
}

export async function rollbackSimulation(simId: string, targetRound: number): Promise<void> {
  const res = await fetch(`/api/simulations/${simId}/rollback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_round: targetRound }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Rollback failed: ${res.status} — ${err}`);
  }
}

// ── SSE Stream ───────────────────────────────────────────────────

export function connectSSE(simId: string, cb: WargameCallbacks): () => void {
  const url = `/api/simulations/${simId}/stream`;
  const evtSource = new EventSource(url);
  let totalRounds = 9;

  const log = (type: string, message: string) => {
    cb.onLog({ ts: Date.now(), type, message });
  };

  // Generic handler for known event types
  const handleEvent = (eventType: string, raw: string) => {
    let evt: SSEEvent;
    try {
      evt = JSON.parse(raw);
    } catch {
      log("parse_error", `Failed to parse ${eventType}: ${raw.slice(0, 100)}`);
      return;
    }

    switch (eventType) {
      case "status": {
        log("status", evt.message);
        if (evt.data?.total_rounds) totalRounds = evt.data.total_rounds as number;
        break;
      }

      case "brief_analyzed": {
        cb.onPhaseChange("briefing");
        cb.onBriefingStep({ phase: "brief_analyzed", message: evt.message, done: true });
        if (evt.data?.total_rounds) totalRounds = evt.data.total_rounds as number;
        if (evt.data?.agent_count) cb.onAgentCount(evt.data.agent_count as number);
        log("brief", evt.message);
        break;
      }

      case "round_start": {
        cb.onPhaseChange("running");
        cb.onRoundStart(evt.round || 0, evt.message);
        log("round_start", `R${evt.round}: ${evt.message}`);
        break;
      }

      case "round_phase": {
        cb.onRoundPhase(evt.round || 0, evt.phase || "", evt.message);
        log("phase", `R${evt.round} [${evt.phase}]: ${evt.message}`);
        break;
      }

      case "round_complete": {
        const d = evt.data;
        const orch = (d.orchestrator as Record<string, unknown>) || {};
        const financial = (orch.financial_impact as Record<string, unknown>) || {};
        const sentimentRaw = (d.sentiment as Record<string, number>) || { positive: 0.33, neutral: 0.34, negative: 0.33 };
        const cri = (orch.contagion_risk_index as number) || 0;
        const wave = (orch.active_wave as number) || 1;
        const pol = (d.polarization as number) || 0;

        const roundState: WgRoundState = {
          round: (d.round as number) || evt.round || 0,
          totalRounds,
          polarization: pol,
          contagionRisk: cri,
          engagementScore: (orch.engagement_score as number) || 0,
          wave,
          warning: (financial.market_volatility_warning as string) || deriveWarning(cri, wave, pol),
          event: (d.event as string) || "",
          sentiment: {
            positive: Math.round((sentimentRaw.positive || 0) * 100),
            neutral: Math.round((sentimentRaw.neutral || 0) * 100),
            negative: Math.round((sentimentRaw.negative || 0) * 100),
          },
          coalitions: ((d.coalitions as Record<string, unknown>[]) || []).map((c) => ({
            label: (c.label as string) || "",
            size: (c.size as number) || 0,
            position: (c.avg_position as number) || 0,
          })),
        };

        const posts = mapPosts((d.top_posts as Record<string, unknown>[]) || [], roundState.round);
        const tickers = mapTickers(financial);
        const agents = mapAgents((d.agents as Record<string, unknown>) || {});

        cb.onRoundComplete(roundState, posts, tickers, agents);
        log("round_complete", `R${roundState.round} complete — POL ${pol.toFixed(1)} CRI ${(cri * 100).toFixed(0)}%`);
        break;
      }

      case "awaiting_intervention": {
        cb.onPhaseChange("awaiting");
        const sitrep = evt.data as unknown as WgSitrep;
        cb.onAwaitingIntervention(sitrep);
        log("awaiting", `SITREP R${sitrep.round_completed}: ${sitrep.threats?.join("; ") || "—"}`);
        break;
      }

      case "player_action": {
        cb.onPhaseChange("running");
        log("player_action", evt.message);
        break;
      }

      case "rollback": {
        log("rollback", evt.message);
        break;
      }

      case "completed": {
        cb.onPhaseChange("completed");
        cb.onCompleted(evt.message);
        log("completed", evt.message);
        evtSource.close();
        break;
      }

      case "error": {
        cb.onPhaseChange("error");
        cb.onError(evt.message);
        log("error", evt.message);
        evtSource.close();
        break;
      }

      case "cancelled": {
        cb.onPhaseChange("error");
        cb.onError("Simulation cancelled");
        log("cancelled", evt.message);
        evtSource.close();
        break;
      }

      case "heartbeat":
        break;

      default:
        log("unknown", `Unknown event: ${eventType}`);
    }
  };

  // Register handlers for each known event type
  const eventTypes = [
    "status", "brief_analyzed", "round_start", "round_phase",
    "round_complete", "awaiting_intervention", "player_action",
    "rollback", "completed", "error", "cancelled", "heartbeat",
  ];

  for (const type of eventTypes) {
    evtSource.addEventListener(type, (e: MessageEvent) => handleEvent(type, e.data));
  }

  // Fallback for unnamed events
  evtSource.onmessage = (e) => handleEvent("unknown", e.data);

  evtSource.onerror = () => {
    // EventSource auto-reconnects; only log once
    log("sse_error", "SSE connection interrupted — reconnecting...");
  };

  // Return cleanup function
  return () => {
    evtSource.close();
  };
}
