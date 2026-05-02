"use client";

import Link from "next/link";
import type { ReplayState, ReplayControls } from "@/lib/replay/types";
import { SPEEDS } from "@/lib/replay/constants";

interface Props {
  state: ReplayState;
  currentEvent: { month: string; event: { event: string; shock_magnitude: number; shock_direction: number } } | null;
  controls: ReplayControls;
  scenarioId: string;
  scenarioTitle: string;
  totalRounds: number;
}

export default function TopBar({ state, currentEvent, controls, scenarioId, scenarioTitle, totalRounds }: Props) {
  return (
    <div className="h-11 flex items-center gap-3 px-4">
      {/* Brand + scenario */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">
          DigitalTwinSim
        </span>
        <span className="text-ki-border-strong">/</span>
        <span className="text-[14px] font-medium text-ki-on-surface tracking-[-0.005em] truncate max-w-[280px]">
          {scenarioTitle}
        </span>
      </div>

      <div className="w-px h-5 bg-ki-border" />

      {/* Play / Pause */}
      <button
        onClick={controls.toggle}
        aria-label={state.status === "playing" ? "Pause" : "Play"}
        className="w-7 h-7 rounded-sm border border-ki-border bg-ki-surface-raised flex items-center justify-center text-ki-on-surface-secondary hover:text-ki-on-surface hover:border-ki-border-strong transition-colors flex-shrink-0"
      >
        {state.status === "playing" ? (
          <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor">
            <rect x="1" y="1" width="4" height="10" rx="1" />
            <rect x="7" y="1" width="4" height="10" rx="1" />
          </svg>
        ) : (
          <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor">
            <polygon points="2,1 11,6 2,11" />
          </svg>
        )}
      </button>

      {/* Speed */}
      <div className="flex gap-0.5 flex-shrink-0">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => controls.setSpeed(s)}
            className={`px-1.5 h-6 rounded-sm font-data tabular text-[11px] transition-colors ${
              state.speed === s
                ? "bg-ki-primary-soft text-ki-primary"
                : "text-ki-on-surface-muted hover:text-ki-on-surface-secondary hover:bg-ki-surface-hover"
            }`}
          >
            {s}×
          </button>
        ))}
      </div>

      {/* Round Progress */}
      <div className="flex-1 flex items-center gap-1 min-w-0 mx-2">
        {Array.from({ length: totalRounds }, (_, i) => (
          <button
            key={i}
            onClick={() => controls.seekToRound(i + 1)}
            aria-label={`Seek to round ${i + 1}`}
            className={`flex-1 h-1 rounded-full transition-colors cursor-pointer ${
              i + 1 < state.currentRound
                ? "bg-ki-primary"
                : i + 1 === state.currentRound
                ? "bg-ki-primary/60"
                : "bg-ki-surface-sunken hover:bg-ki-border"
            }`}
          />
        ))}
      </div>

      {/* Event banner */}
      <div className="flex-shrink min-w-0 max-w-[420px]">
        {currentEvent ? (
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-ki-error animate-live-pulse flex-shrink-0" />
            <p className="font-data tabular text-[11px] text-ki-on-surface-muted truncate">
              <span className="text-ki-on-surface font-medium">R{state.currentRound}</span>
              {" "}· {currentEvent.month} — {currentEvent.event.event.substring(0, 80)}…
            </p>
          </div>
        ) : (
          <p className="font-data tabular text-[11px] text-ki-on-surface-muted">
            Press <span className="kbd">Space</span> to start
          </p>
        )}
      </div>

      {/* Back to scenario */}
      <Link
        href={`/scenario/${scenarioId}`}
        className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border bg-ki-surface-raised text-[11px] text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors flex-shrink-0"
      >
        ← Analysis
      </Link>

      {/* Elapsed + status */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="font-data tabular text-[11px] text-ki-on-surface-muted">{state.elapsedDisplay}</span>
        <div
          className={`w-1.5 h-1.5 rounded-full ${
            state.status === "playing"
              ? "bg-ki-success animate-live-pulse"
              : state.status === "finished"
              ? "bg-ki-warning"
              : "bg-ki-on-surface-muted"
          }`}
        />
      </div>
    </div>
  );
}
