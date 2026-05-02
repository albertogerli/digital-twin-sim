"use client";

import { useState, useRef, useCallback } from "react";
import type {
  WgPost,
  WgTicker,
  WgAgent,
  WgRoundState,
  WgSitrep,
  WargamePhase,
  BriefingStep,
  LogEntry,
} from "@/lib/wargame-types";
import {
  createSimulation,
  connectSSE,
  submitIntervention,
} from "@/lib/wargame-client";
import { AgentFeed } from "./AgentFeed";
import { ContagionGraph } from "./ContagionGraph";
import { TickerPanel } from "./TickerPanel";
import { CommandBar } from "./CommandBar";
import { StatusBar } from "./StatusBar";
import { SetupScreen } from "./SetupScreen";
import { SitrepOverlay } from "./SitrepOverlay";
import { InterventionModal } from "./InterventionModal";

const EMPTY_STATE: WgRoundState = {
  round: 0,
  totalRounds: 9,
  polarization: 0,
  contagionRisk: 0,
  engagementScore: 0,
  wave: 1,
  warning: "LOW",
  event: "",
  sentiment: { positive: 33, neutral: 34, negative: 33 },
  coalitions: [],
};

export default function WargameTerminal() {
  // ── Core state ───────────────────────────────────────────────
  const [phase, setPhase] = useState<WargamePhase>("idle");
  const [simId, setSimId] = useState<string | null>(null);
  const [roundState, setRoundState] = useState<WgRoundState>(EMPTY_STATE);
  const [posts, setPosts] = useState<WgPost[]>([]);
  const [tickers, setTickers] = useState<WgTicker[]>([]);
  const [agents, setAgents] = useState<WgAgent[]>([]);
  const [sitrep, setSitrep] = useState<WgSitrep | null>(null);
  const [lastIntervention, setLastIntervention] = useState("");
  const [agentCount, setAgentCount] = useState(0);
  const [briefingSteps, setBriefingSteps] = useState<BriefingStep[]>([]);
  const [, setLogs] = useState<LogEntry[]>([]);
  const [phaseMessage, setPhaseMessage] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [interveneOpen, setInterveneOpen] = useState(false);

  const cleanupRef = useRef<(() => void) | null>(null);
  const postQueueRef = useRef<WgPost[]>([]);
  const streamTimerRef = useRef<NodeJS.Timeout | null>(null);

  // ── Post streaming: drip posts into feed one-by-one ────────
  const startPostDrip = useCallback((newPosts: WgPost[]) => {
    // Add to queue
    postQueueRef.current = [...postQueueRef.current, ...newPosts];
    // If already dripping, the timer will pick them up
    if (streamTimerRef.current) return;
    streamTimerRef.current = setInterval(() => {
      const next = postQueueRef.current.shift();
      if (next) {
        setPosts((prev) => [...prev, next]);
      } else {
        if (streamTimerRef.current) clearInterval(streamTimerRef.current);
        streamTimerRef.current = null;
      }
    }, 1200);
  }, []);

  // ── Start simulation ─────────────────────────────────────────
  const handleStart = useCallback(async (brief: string, playerRole: string, provider: string, rounds?: number) => {
    setPhase("configuring");
    setErrorMsg("");
    setLogs([]);
    setBriefingSteps([]);
    setPosts([]);
    setTickers([]);
    setAgents([]);
    setRoundState(EMPTY_STATE);
    postQueueRef.current = [];

    try {
      const id = await createSimulation({
        brief,
        playerRole,
        provider,
        rounds,
      });
      setSimId(id);

      // Connect SSE
      const cleanup = connectSSE(id, {
        onPhaseChange: (p) => setPhase(p as WargamePhase),
        onBriefingStep: (step) => {
          setBriefingSteps((prev) => [...prev, step]);
          setPhaseMessage(step.message);
        },
        onRoundStart: (round, msg) => {
          setPhaseMessage(`Round ${round}: ${msg}`);
        },
        onRoundPhase: (_round, _phase, msg) => {
          setPhaseMessage(msg);
        },
        onRoundComplete: (state, newPosts, newTickers, newAgents) => {
          setRoundState(state);
          setTickers(newTickers);
          setAgents(newAgents);
          startPostDrip(newPosts);
          setPhaseMessage(`Round ${state.round} complete`);
        },
        onAwaitingIntervention: (s) => {
          setSitrep(s);
        },
        onCompleted: (msg) => {
          setPhaseMessage(msg);
        },
        onError: (msg) => {
          setErrorMsg(msg);
        },
        onLog: (entry) => {
          setLogs((prev) => [...prev.slice(-200), entry]);
        },
        onAgentCount: (count) => {
          setAgentCount(count);
        },
      });

      cleanupRef.current = cleanup;
    } catch (err) {
      setPhase("error");
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  }, [startPostDrip]);

  // ── Submit intervention ──────────────────────────────────────
  const handleIntervention = useCallback(async (text: string) => {
    if (!simId) return;
    setLastIntervention(text);
    setSitrep(null);
    setPhaseMessage("PROCESSING INTERVENTION...");
    try {
      await submitIntervention(simId, text);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  }, [simId]);

  // ── Skip intervention ────────────────────────────────────────
  const handleSkip = useCallback(async () => {
    if (!simId) return;
    setSitrep(null);
    setPhaseMessage("AUTO-GENERATING NEXT ROUND...");
    try {
      await submitIntervention(simId, "", "press_release", "", true);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    }
  }, [simId]);

  // ── Idle: show setup screen ──────────────────────────────────
  if (phase === "idle") {
    return <SetupScreen onStart={handleStart} />;
  }

  const warnColor =
    roundState.warning === "CRITICAL" ? "var(--neg)" :
    roundState.warning === "HIGH"     ? "var(--warn)" :
    roundState.warning === "MODERATE" ? "var(--warn)" :
    "var(--pos)";
  const isProcessing = phase === "configuring" || phase === "briefing" || (phase === "running" && roundState.round === 0);
  const displayAgentCount = agentCount || agents.length || 0;

  return (
    <div className="h-screen w-screen bg-ki-surface text-ki-on-surface flex flex-col overflow-hidden">
      {/* Top Status Bar */}
      <StatusBar
        state={roundState}
        processing={isProcessing}
        agentCount={displayAgentCount}
        onIntervene={phase === "running" || phase === "awaiting" ? () => setInterveneOpen(true) : undefined}
      />

      <InterventionModal
        open={interveneOpen}
        round={roundState.round}
        onClose={() => setInterveneOpen(false)}
        onSubmit={async ({ actor, action, message, kbDoc }) => {
          if (action === "inject_kb" && kbDoc && simId) {
            // Direct call: bypass `handleIntervention` (which is text-only) so we can
            // ship the KB doc payload to the backend for live RAG ingestion.
            try {
              await submitIntervention(
                simId,
                `[${actor}/${action}] ${message}`,
                "inject_kb",
                "",
                false,
                { title: kbDoc.name, text: kbDoc.text, source: kbDoc.sourceType },
              );
              setLastIntervention(message);
              setSitrep(null);
              setPhaseMessage("Document injected — resuming with live KB…");
            } catch (err) {
              setErrorMsg(err instanceof Error ? err.message : String(err));
            }
          } else {
            handleIntervention(`[${actor}/${action}] ${message}`);
          }
          setInterveneOpen(false);
        }}
        submitting={false}
      />

      {/* Error banner */}
      {errorMsg && (
        <div className="h-7 flex items-center px-3 gap-3 bg-ki-error-soft border-b border-ki-error/30 shrink-0">
          <span className="font-data text-[11px] text-ki-error">ERROR — {errorMsg}</span>
          <button onClick={() => setErrorMsg("")} className="ml-auto font-data text-[11px] text-ki-error/70 hover:text-ki-error transition-colors">Dismiss</button>
        </div>
      )}

      {/* Briefing overlay */}
      {(phase === "configuring" || phase === "briefing") && (
        <div className="flex-1 flex items-center justify-center min-h-0">
          <div className="max-w-md w-full px-4">
            <div className="eyebrow mb-3">
              {phase === "configuring" ? "Initializing simulation" : "Analyzing brief"}
            </div>
            {briefingSteps.map((step, i) => (
              <div key={i} className="flex items-center gap-2 mb-1">
                <span className="w-1.5 h-1.5 rounded-full bg-ki-success shrink-0" />
                <span className="text-[12px] text-ki-on-surface-secondary">{step.message}</span>
              </div>
            ))}
            <div className="mt-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-ki-warning animate-pulse shrink-0" />
              <span className="text-[12px] text-ki-warning animate-pulse">{phaseMessage || "Connecting…"}</span>
            </div>
            {agentCount > 0 && (
              <div className="mt-4 font-data tabular text-[11px] text-ki-on-surface-muted">
                {agentCount} agents generated
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main 3-column layout (only when rounds are running) */}
      {phase !== "configuring" && phase !== "briefing" && (
        <div className="flex-1 flex min-h-0">
          {/* LEFT: Agent Feed */}
          <div className="w-[340px] border-r border-ki-border flex flex-col min-h-0">
            <div className="h-7 flex items-center px-3 border-b border-ki-border bg-ki-surface-sunken shrink-0">
              <span className="eyebrow">
                Agent feed{displayAgentCount > 0 ? ` · ${displayAgentCount} active` : ""}
              </span>
              <span className="ml-auto font-data tabular text-[11px] text-ki-on-surface-muted">
                R{roundState.round}/{roundState.totalRounds}
              </span>
            </div>
            <AgentFeed posts={posts} />
          </div>

          {/* CENTER: Contagion Graph */}
          <div className="flex-1 flex flex-col min-h-0 border-r border-ki-border">
            <div className="h-7 flex items-center px-3 border-b border-ki-border bg-ki-surface-sunken shrink-0 justify-between">
              <span className="eyebrow">
                Social graph · contagion overlay
              </span>
              <div className="flex gap-3 font-data tabular text-[11px] text-ki-on-surface-secondary">
                <span>CRI <span style={{ color: roundState.contagionRisk > 0.7 ? "var(--neg)" : "var(--warn)" }}>{roundState.contagionRisk.toFixed(2)}</span></span>
                <span>POL <span style={{ color: roundState.polarization > 7 ? "var(--neg)" : "var(--warn)" }}>{roundState.polarization.toFixed(1)}</span></span>
                <span>W<span style={{ color: warnColor }}>{roundState.wave}</span></span>
              </div>
            </div>
            <ContagionGraph agents={agents} state={roundState} />
          </div>

          {/* RIGHT: Ticker Panel */}
          <div className="w-[280px] flex flex-col min-h-0">
            <div className="h-7 flex items-center px-3 border-b border-ki-border bg-ki-surface-sunken shrink-0">
              <span className="eyebrow">
                Estimated impact
              </span>
              <span className="ml-auto font-data tabular text-[11px]" style={{ color: warnColor }}>
                {roundState.warning}
              </span>
            </div>
            <TickerPanel tickers={tickers} state={roundState} />
          </div>
        </div>
      )}

      {/* SITREP Overlay when awaiting intervention */}
      {phase === "awaiting" && sitrep && (
        <SitrepOverlay sitrep={sitrep} onSkip={handleSkip} />
      )}

      {/* BOTTOM: Command Bar */}
      {phase !== "configuring" && phase !== "briefing" && (
        <CommandBar
          onSubmit={handleIntervention}
          processing={phase === "running"}
          lastIntervention={lastIntervention}
          round={roundState.round}
          disabled={phase === "completed" || phase === "error"}
          awaiting={phase === "awaiting"}
        />
      )}

      {/* Completed overlay */}
      {phase === "completed" && (
        <div className="absolute inset-0 bg-ki-surface/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="text-center bg-ki-surface-raised border border-ki-border rounded p-8 max-w-md">
            <div className="w-1.5 h-1.5 rounded-full bg-ki-success mx-auto mb-3" />
            <div className="text-[16px] font-medium text-ki-on-surface mb-1">Simulation complete</div>
            <div className="font-data tabular text-[12px] text-ki-on-surface-muted mb-5">
              {roundState.totalRounds} rounds executed
            </div>
            <button
              onClick={() => { cleanupRef.current?.(); setPhase("idle"); }}
              className="inline-flex items-center justify-center h-8 px-3 rounded-sm bg-ki-on-surface text-ki-surface text-[12px] font-medium hover:bg-ki-on-surface-secondary transition-colors"
            >
              New simulation
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
