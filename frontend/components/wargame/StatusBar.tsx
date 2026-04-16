"use client";

import { type WgRoundState } from "@/lib/wargame-types";

export function StatusBar({ state, processing, agentCount }: { state: WgRoundState; processing: boolean; agentCount?: number }) {
  const warnColor = { CRITICAL: "#ff3b3b", HIGH: "#ff7700", MODERATE: "#ffaa00", LOW: "#00d26a" }[state.warning] || "#6b7280";

  return (
    <div className="h-7 flex items-center px-2 border-b border-ki-border bg-ki-surface-sunken shrink-0 gap-0">
      {/* Logo / Mode */}
      <div className="flex items-center gap-2 pr-3 border-r border-ki-border">
        <span className="font-data text-[10px] text-ki-on-surface font-bold tracking-wider">WARGAME</span>
        {processing ? (
          <span className="font-data text-[9px] text-[#ffaa00] animate-pulse">RECALCULATING</span>
        ) : (
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00d26a] animate-pulse" />
            <span className="font-data text-[9px] text-[#00d26a]">LIVE</span>
          </span>
        )}
      </div>

      {/* Round */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[9px] text-ki-on-surface-muted">ROUND</span>
        <span className="font-data text-[11px] text-ki-on-surface font-bold">{state.round}</span>
        <span className="font-data text-[9px] text-ki-on-surface-muted">/{state.totalRounds}</span>
      </div>

      {/* Warning */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[9px]" style={{ color: warnColor }}>{state.warning}</span>
      </div>

      {/* Wave */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[9px] text-ki-on-surface-muted">WAVE</span>
        <span className="font-data text-[10px] font-bold" style={{
          color: state.wave === 3 ? "#ff3b3b" : state.wave === 2 ? "#ffaa00" : "#00d26a"
        }}>{state.wave}</span>
      </div>

      {/* CRI */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[9px] text-ki-on-surface-muted">CRI</span>
        <span className="font-data text-[10px]" style={{
          color: state.contagionRisk > 0.8 ? "#ff3b3b" : state.contagionRisk > 0.5 ? "#ffaa00" : "#00d26a"
        }}>{(state.contagionRisk * 100).toFixed(0)}%</span>
      </div>

      {/* Sentiment */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[9px] text-[#00d26a]">+{state.sentiment.positive}%</span>
        <span className="font-data text-[9px] text-ki-on-surface-muted">{state.sentiment.neutral}%</span>
        <span className="font-data text-[9px] text-[#ff3b3b]">-{state.sentiment.negative}%</span>
      </div>

      {/* Polarization */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[9px] text-ki-on-surface-muted">POL</span>
        <span className="font-data text-[10px]" style={{
          color: state.polarization > 7 ? "#ff3b3b" : state.polarization > 5 ? "#ffaa00" : "#00d26a"
        }}>{state.polarization.toFixed(1)}</span>
      </div>

      {/* Event */}
      <div className="flex-1 px-3 overflow-hidden">
        <span className="font-data text-[9px] text-ki-on-surface-muted truncate block">
          {state.event}
        </span>
      </div>

      {/* Agents count */}
      <div className="flex items-center gap-1 px-2">
        <span className="font-data text-[9px] text-ki-on-surface-muted">{agentCount || 0} AGENTS</span>
      </div>
    </div>
  );
}
