"use client";

import { useEffect, useState, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import type { ReplayMeta, RoundData } from "@/lib/types";
import type { ReplayRoundData } from "@/lib/replay/types";
import { useSimulationReplay } from "@/lib/replay/useSimulationReplay";
import { SPEEDS } from "@/lib/replay/constants";
import ErrorBoundary from "@/components/ui/ErrorBoundary";
import LiveFeed from "./feed/LiveFeed";
import LiveNetworkCanvas from "./network/LiveNetworkCanvas";
import IndicatorPanel from "./indicators/IndicatorPanel";
import BottomBar from "./BottomBar";

/* ───────────────────────────────────────────────────────────
   CommandCenter — Quiet Intelligence "Live Command" screen
   matched 1:1 against screen-live.jsx in the design package:

     ┌───────────────────────────────────────────────┐
     │ <DomainCap> Title  [LIVE]  sim_id   [Pause][Intervene][Fork][Snapshot]
     ├───────────────────────────────────────────────┤
     │  Tick │ Polarization │ Net sent │ Posts │ Conf │ LLM spend  │
     ├──────────┬─────────────────────┬──────────────┤
     │ INDIC.   │   NETWORK CENTER    │  LIVE FEED   │
     │ rail     │   (Canvas + d3)     │  (stream)    │
     ├──────────┴─────────────────────┴──────────────┤
     │  Agent strip / scrubber / transport bar       │
     └───────────────────────────────────────────────┘
   ─────────────────────────────────────────────────────────── */

interface Props {
  scenarioId: string;
  meta: ReplayMeta | null;
  rounds: RoundData[];
}

export default function CommandCenter({ scenarioId, meta, rounds }: Props) {
  const router = useRouter();
  const replayRounds = rounds as unknown as ReplayRoundData[];

  const {
    state,
    visiblePosts,
    graphSnapshot,
    currentEvent,
    keyInsight,
    indicators,
    coalitions,
    activeImpact,
    selectedPostId,
    realWorldEffects,
    controls,
  } = useSimulationReplay(replayRounds.length > 0 ? replayRounds : null);

  const selectedPostImpact = selectedPostId
    ? visiblePosts.find((p) => p.id === selectedPostId)?.impact ?? null
    : null;

  const totalRounds = meta?.totalRounds ?? rounds.length;

  // Keyboard shortcuts (unchanged)
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement) return;
      switch (e.code) {
        case "Space":
          e.preventDefault();
          controls.toggle();
          break;
        case "ArrowRight":
          if (state.currentRound < totalRounds) controls.seekToRound(state.currentRound + 1);
          break;
        case "ArrowLeft":
          if (state.currentRound > 1) controls.seekToRound(state.currentRound - 1);
          break;
        case "Digit1": controls.setSpeed(1); break;
        case "Digit2": controls.setSpeed(2); break;
        case "Digit3": controls.setSpeed(4); break;
        case "Digit4": controls.setSpeed(8); break;
        case "KeyR": controls.restart(); break;
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [controls, state.currentRound, totalRounds]);

  // ── Derived KPIs ─────────────────────────────────────────
  const kpi = useMemo(() => {
    const totalSent =
      indicators.sentimentDistribution.positive +
      indicators.sentimentDistribution.neutral +
      indicators.sentimentDistribution.negative;
    const netSent = totalSent > 0
      ? (indicators.sentimentDistribution.positive - indicators.sentimentDistribution.negative) / totalSent
      : 0;
    return {
      tick: `${state.currentRound}/${totalRounds || "?"}`,
      polarization: indicators.polarization.toFixed(2),
      netSent,
      posts: indicators.postCount,
      confidence: 0.84, // placeholder until calibration band threads through
    };
  }, [indicators, state.currentRound, totalRounds]);

  const isPlaying = state.status === "playing";

  // ── Snapshot / Fork / Intervene placeholders ─────────────
  const [snapshotFlash, setSnapshotFlash] = useState(false);
  const onSnapshot = () => {
    setSnapshotFlash(true);
    setTimeout(() => setSnapshotFlash(false), 800);
  };
  const onFork = () => router.push(`/scenario/${scenarioId}/branches`);
  const onIntervene = () => {
    // Wargame is a separate code path; surface the existing route
    router.push(`/wargame`);
  };

  return (
    <div className="h-screen flex flex-col bg-ki-surface text-ki-on-surface overflow-hidden">
      {/* ── Sub-toolbar: identity + transport + actions ─────── */}
      <header className="flex items-center gap-3 h-11 px-4 border-b border-ki-border bg-ki-surface-raised flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          {/* Domain cap (defaults to financial indigo if unknown) */}
          <span
            className="w-[2px] h-4 rounded-sm shrink-0"
            style={{ background: "var(--accent)" }}
            aria-hidden
          />
          <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted shrink-0">
            DigitalTwinSim
          </span>
          <span className="text-ki-border-strong shrink-0">/</span>
          <span className="text-[14px] font-medium text-ki-on-surface tracking-[-0.005em] truncate">
            {meta?.title ?? "Simulation"}
          </span>
          <span className={`inline-flex items-center gap-1.5 px-1.5 h-5 rounded-sm font-data text-[10px] uppercase tracking-[0.06em] shrink-0 ${
            isPlaying ? "bg-ki-error-soft text-ki-error" : "bg-ki-surface-sunken text-ki-on-surface-muted border border-ki-border"
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isPlaying ? "bg-ki-error animate-live-pulse" : "bg-ki-on-surface-muted"}`} />
            {isPlaying ? "Live" : state.status === "finished" ? "Completed" : "Paused"}
          </span>
          <span className="font-data text-[11px] text-ki-on-surface-muted shrink-0 truncate">
            sim_{scenarioId.slice(0, 12)}
          </span>
          {currentEvent && (
            <>
              <span className="text-ki-border-strong shrink-0">·</span>
              <span className="font-data text-[11px] text-ki-on-surface-muted truncate">
                {currentEvent.month}
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={controls.toggle}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border bg-ki-surface-raised text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors"
          >
            <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 500" }}>
              {isPlaying ? "pause" : "play_arrow"}
            </span>
            {isPlaying ? "Pause" : "Resume"}
          </button>
          <button
            onClick={onIntervene}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm bg-ki-primary text-white text-[11px] font-medium hover:bg-ki-primary-muted transition-colors"
          >
            <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 500" }}>bolt</span>
            Intervene
          </button>
          <button
            onClick={onFork}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border bg-ki-surface-raised text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors"
          >
            <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>account_tree</span>
            Fork
          </button>
          <button
            onClick={onSnapshot}
            className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border text-[11px] transition-colors ${
              snapshotFlash
                ? "bg-ki-success-soft border-ki-success text-ki-success"
                : "border-ki-border bg-ki-surface-raised text-ki-on-surface hover:bg-ki-surface-hover"
            }`}
          >
            <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>
              {snapshotFlash ? "check" : "save"}
            </span>
            {snapshotFlash ? "Saved" : "Snapshot"}
          </button>
          <Link
            href={`/scenario/${scenarioId}`}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm text-[11px] text-ki-on-surface-muted hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors"
          >
            ← Analysis
          </Link>
        </div>
      </header>

      {/* ── KPI strip — 6 cells ─────────────────────────────── */}
      <div className="flex border-b border-ki-border bg-ki-surface-raised flex-shrink-0">
        <KPICell label="Tick" value={kpi.tick} sub={`round ${state.currentRound}`} />
        <KPICell
          label="Polarization"
          value={kpi.polarization}
          sub={`σ²(opinion) · range 0–10`}
        />
        <KPICell
          label="Net sentiment"
          value={`${kpi.netSent > 0 ? "+" : ""}${kpi.netSent.toFixed(2)}`}
          sub={`${(indicators.sentimentDistribution.positive * 100).toFixed(0)}% pos · ${(indicators.sentimentDistribution.negative * 100).toFixed(0)}% neg`}
          tone={kpi.netSent > 0 ? "pos" : kpi.netSent < 0 ? "neg" : "neutral"}
        />
        <KPICell
          label="Posts"
          value={kpi.posts.toLocaleString()}
          sub={`${indicators.reactionCount.toLocaleString()} reactions`}
        />
        <KPICell
          label="Confidence"
          value={kpi.confidence.toFixed(2)}
          sub="EnKF assimilation"
        />
        <KPICell
          label="Speed"
          value={`${state.speed}×`}
          sub={
            <span className="inline-flex gap-0.5">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  onClick={() => controls.setSpeed(s)}
                  className={`px-1 rounded-sm transition-colors font-data tabular text-[10px] ${
                    state.speed === s ? "bg-ki-primary-soft text-ki-primary" : "text-ki-on-surface-muted hover:bg-ki-surface-hover"
                  }`}
                >
                  {s}×
                </button>
              ))}
            </span>
          }
          last
        />
      </div>

      {/* ── 3-pane working area: indicators · network · feed ── */}
      <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[320px_1fr_380px] overflow-auto md:overflow-hidden">
        {/* LEFT — Indicators rail */}
        <aside className="overflow-y-auto scrollbar-thin bg-ki-surface-sunken border-b md:border-b-0 md:border-r border-ki-border">
          <ErrorBoundary>
            <IndicatorPanel
              indicators={indicators}
              coalitions={coalitions}
              realWorldEffects={realWorldEffects}
            />
          </ErrorBoundary>
        </aside>

        {/* CENTER — Network graph (the marquee) */}
        <section className="border-b md:border-b-0 md:border-r border-ki-border overflow-hidden bg-ki-surface-sunken min-h-[300px] md:min-h-0 relative">
          <ErrorBoundary>
            <LiveNetworkCanvas
              snapshot={graphSnapshot}
              activeAgentIds={indicators.activeAgents.map((a) => a.id)}
              activeImpact={activeImpact}
              selectedPostId={selectedPostId}
              selectedPostImpact={selectedPostImpact}
            />
          </ErrorBoundary>
        </section>

        {/* RIGHT — Live feed */}
        <aside className="overflow-hidden bg-ki-surface min-h-[400px] md:min-h-0">
          <ErrorBoundary>
            <LiveFeed
              posts={visiblePosts}
              keyInsight={keyInsight}
              status={state.status}
              onPlay={controls.play}
              selectedPostId={selectedPostId}
              onSelectPost={controls.selectPost}
            />
          </ErrorBoundary>
        </aside>
      </div>

      {/* ── Bottom: agent strip + scrubber ──────────────────── */}
      <div className="flex-shrink-0 border-t border-ki-border bg-ki-surface-raised hidden md:flex flex-col">
        {/* Scrubber row */}
        <div className="flex items-center gap-3 px-4 h-9 border-b border-ki-border-faint">
          <span className="font-data tabular text-[11px] text-ki-on-surface-muted w-20">
            R{state.currentRound} / {totalRounds}
          </span>
          <div className="flex-1 flex items-center gap-1">
            {Array.from({ length: totalRounds }, (_, i) => (
              <button
                key={i}
                onClick={() => controls.seekToRound(i + 1)}
                className={`flex-1 h-1 rounded-full transition-colors cursor-pointer ${
                  i + 1 < state.currentRound
                    ? "bg-ki-primary"
                    : i + 1 === state.currentRound
                    ? "bg-ki-primary/60"
                    : "bg-ki-surface-sunken hover:bg-ki-border"
                }`}
                aria-label={`Seek to round ${i + 1}`}
              />
            ))}
          </div>
          <span className="font-data tabular text-[11px] text-ki-on-surface-muted w-20 text-right">
            {state.elapsedDisplay}
          </span>
        </div>
        <BottomBar
          agents={meta?.agents || []}
          activeAgentIds={indicators.activeAgents.map((a) => a.id)}
          currentRound={state.currentRound}
          totalRounds={totalRounds}
        />
      </div>
    </div>
  );
}

/* ── KPI cell ─────────────────────────────────────────────── */
function KPICell({
  label,
  value,
  sub,
  tone = "neutral",
  last = false,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "pos" | "neg" | "neutral";
  last?: boolean;
}) {
  const valueColor =
    tone === "pos" ? "text-ki-success" :
    tone === "neg" ? "text-ki-error" :
    "text-ki-on-surface";
  return (
    <div className={`flex-1 px-4 py-3 min-w-0 ${last ? "" : "border-r border-ki-border"}`}>
      <div className="eyebrow">{label}</div>
      <div className="mt-1">
        <span className={`font-data tabular text-[20px] font-medium tracking-tight2 ${valueColor}`}>
          {value}
        </span>
      </div>
      {sub && <div className="text-[11px] text-ki-on-surface-muted mt-0.5">{sub}</div>}
    </div>
  );
}
