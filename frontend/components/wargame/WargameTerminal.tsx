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

  const warnColor = roundState.warning === "CRITICAL" ? "#ff3b3b" : roundState.warning === "HIGH" ? "#ff7700" : roundState.warning === "MODERATE" ? "#ffaa00" : "#00d26a";
  const isProcessing = phase === "configuring" || phase === "briefing" || (phase === "running" && roundState.round === 0);
  const displayAgentCount = agentCount || agents.length || 0;

  return (
    <div className="h-screen w-screen bg-ki-surface text-ki-on-surface flex flex-col overflow-hidden">
      {/* Top Status Bar */}
      <StatusBar state={roundState} processing={isProcessing} agentCount={displayAgentCount} />

      {/* Error banner */}
      {errorMsg && (
        <div className="h-6 flex items-center px-2 bg-[#1a0a0a] border-b border-[#ff3b3b20] shrink-0">
          <span className="font-data text-[9px] text-[#ff3b3b]">ERROR: {errorMsg}</span>
          <button onClick={() => setErrorMsg("")} className="ml-auto font-data text-[8px] text-ki-on-surface-muted hover:text-ki-on-surface">DISMISS</button>
        </div>
      )}

      {/* Briefing overlay */}
      {(phase === "configuring" || phase === "briefing") && (
        <div className="flex-1 flex items-center justify-center min-h-0">
          <div className="max-w-md w-full px-4">
            <div className="font-data text-[11px] text-ki-on-surface-muted mb-4 uppercase tracking-wider">
              {phase === "configuring" ? "INITIALIZING SIMULATION..." : "ANALYZING BRIEF..."}
            </div>
            {briefingSteps.map((step, i) => (
              <div key={i} className="flex items-center gap-2 mb-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00d26a] shrink-0" />
                <span className="font-data text-[10px] text-ki-on-surface-muted">{step.message}</span>
              </div>
            ))}
            <div className="mt-3 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#ffaa00] animate-pulse shrink-0" />
              <span className="font-data text-[10px] text-[#ffaa00] animate-pulse">{phaseMessage || "CONNECTING..."}</span>
            </div>
            {agentCount > 0 && (
              <div className="mt-4 font-data text-[10px] text-ki-on-surface-muted">
                {agentCount} AGENTS GENERATED
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
            <div className="h-6 flex items-center px-2 border-b border-ki-border-strong bg-ki-surface-sunken shrink-0">
              <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
                Agent Feed{displayAgentCount > 0 ? ` — ${displayAgentCount} active` : ""}
              </span>
              <span className="ml-auto font-data text-[9px] text-ki-on-surface-muted">
                R{roundState.round}/{roundState.totalRounds}
              </span>
            </div>
            <AgentFeed posts={posts} />
          </div>

          {/* CENTER: Contagion Graph */}
          <div className="flex-1 flex flex-col min-h-0 border-r border-ki-border">
            <div className="h-6 flex items-center px-2 border-b border-ki-border-strong bg-ki-surface-sunken shrink-0 justify-between">
              <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
                Social Graph — Contagion Overlay
              </span>
              <div className="flex gap-3 font-data text-[9px]">
                <span>CRI: <span style={{ color: roundState.contagionRisk > 0.7 ? "#ff3b3b" : "#ffaa00" }}>{roundState.contagionRisk.toFixed(2)}</span></span>
                <span>POL: <span style={{ color: roundState.polarization > 7 ? "#ff3b3b" : "#ffaa00" }}>{roundState.polarization.toFixed(1)}</span></span>
                <span>W<span style={{ color: warnColor }}>{roundState.wave}</span></span>
              </div>
            </div>
            <ContagionGraph agents={agents} state={roundState} />
          </div>

          {/* RIGHT: Ticker Panel */}
          <div className="w-[280px] flex flex-col min-h-0">
            <div className="h-6 flex items-center px-2 border-b border-ki-border-strong bg-ki-surface-sunken shrink-0">
              <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
                Estimated Impact
              </span>
              <span className="ml-auto font-data text-[9px]" style={{ color: warnColor }}>
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
        <div className="absolute inset-0 bg-ki-surface/80 flex items-center justify-center z-50">
          <div className="text-center">
            <div className="font-data text-[14px] text-[#00d26a] font-bold mb-2">SIMULATION COMPLETE</div>
            <div className="font-data text-[10px] text-ki-on-surface-muted">{roundState.totalRounds} rounds executed</div>
            <button
              onClick={() => { cleanupRef.current?.(); setPhase("idle"); }}
              className="mt-4 font-data text-[10px] px-3 py-1.5 border border-ki-border text-ki-on-surface-muted hover:text-ki-on-surface hover:border-ki-border-strong transition-colors"
            >
              NEW SIMULATION
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
