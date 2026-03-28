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
    <div className="h-full flex items-center gap-3 px-4 py-2">
      {/* Logo + Scenario Title */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="font-mono text-sm font-bold text-gray-900">DigitalTwinSim</span>
        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-blue-50 text-blue-600 border border-blue-700/50">
          {scenarioTitle}
        </span>
      </div>

      {/* Divider */}
      <div className="w-px h-7 bg-gray-200" />

      {/* Play / Pause */}
      <button
        onClick={controls.toggle}
        className="w-8 h-8 rounded-full bg-gray-100 border border-gray-300 flex items-center justify-center text-gray-500 hover:text-gray-900 hover:border-blue-500 transition-all flex-shrink-0"
      >
        {state.status === "playing" ? (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
            <rect x="1" y="1" width="4" height="10" rx="1" />
            <rect x="7" y="1" width="4" height="10" rx="1" />
          </svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
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
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-all ${
              state.speed === s
                ? "bg-blue-900/60 text-blue-600 font-bold"
                : "text-gray-400 hover:text-gray-700"
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* Round Progress */}
      <div className="flex-1 flex items-center gap-1 min-w-0 mx-2">
        {Array.from({ length: totalRounds }, (_, i) => (
          <button
            key={i}
            onClick={() => controls.seekToRound(i + 1)}
            className={`flex-1 h-1.5 rounded-full transition-all cursor-pointer ${
              i + 1 < state.currentRound
                ? "bg-blue-500"
                : i + 1 === state.currentRound
                ? "bg-blue-400"
                : "bg-gray-200"
            }`}
          />
        ))}
      </div>

      {/* Event Banner */}
      <div className="flex-shrink min-w-0 max-w-[400px]">
        {currentEvent ? (
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse flex-shrink-0" />
            <p className="font-mono text-[10px] text-gray-500 truncate">
              <span className="text-gray-900 font-semibold">R{state.currentRound}</span>
              {" "}{currentEvent.month} — {currentEvent.event.event.substring(0, 80)}...
            </p>
          </div>
        ) : (
          <p className="font-mono text-[10px] text-gray-400">
            Press Space to start simulation
          </p>
        )}
      </div>

      {/* Back to Scenario Link */}
      <Link
        href={`/scenario/${scenarioId}`}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-100 border border-gray-300 text-xs font-mono font-semibold text-gray-700 hover:bg-gray-200 hover:text-gray-900 transition-all flex-shrink-0"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M15 19l-7-7 7-7" />
        </svg>
        Analysis
      </Link>

      {/* Elapsed + Status */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="font-mono text-xs text-gray-500">{state.elapsedDisplay}</span>
        <div
          className={`w-2 h-2 rounded-full ${
            state.status === "playing"
              ? "bg-green-500 animate-pulse"
              : state.status === "finished"
              ? "bg-amber-500"
              : "bg-gray-300"
          }`}
        />
      </div>
    </div>
  );
}
