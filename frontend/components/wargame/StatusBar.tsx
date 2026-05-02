"use client";

import { type WgRoundState } from "@/lib/wargame-types";

export function StatusBar({
  state,
  processing,
  agentCount,
  onIntervene,
}: {
  state: WgRoundState;
  processing: boolean;
  agentCount?: number;
  onIntervene?: () => void;
}) {
  const warnColor =
    state.warning === "CRITICAL" ? "var(--neg)" :
    state.warning === "HIGH"     ? "var(--neg)" :
    state.warning === "MODERATE" ? "var(--warn)" :
    "var(--pos)";

  return (
    <div className="h-9 flex items-center px-3 border-b border-ki-border bg-ki-surface-raised shrink-0 gap-0">
      {/* Brand / mode */}
      <div className="flex items-center gap-2 pr-3 border-r border-ki-border">
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">DigitalTwinSim</span>
        <span className="text-ki-border-strong">/</span>
        <span className="text-[12px] font-medium text-ki-on-surface">Wargame</span>
        {processing ? (
          <span className="font-data text-[10px] text-ki-warning animate-pulse">recalculating…</span>
        ) : (
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-ki-success animate-live-pulse" />
            <span className="font-data text-[10px] text-ki-success">live</span>
          </span>
        )}
      </div>

      {/* Round */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="eyebrow">Round</span>
        <span className="font-data tabular text-[12px] text-ki-on-surface font-medium">{state.round}</span>
        <span className="font-data tabular text-[10px] text-ki-on-surface-muted">/{state.totalRounds}</span>
      </div>

      {/* Warning */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="font-data text-[10px] uppercase tracking-[0.04em]" style={{ color: warnColor }}>{state.warning}</span>
      </div>

      {/* Wave */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="eyebrow">Wave</span>
        <span className="font-data tabular text-[11px]" style={{
          color: state.wave === 3 ? "var(--neg)" : state.wave === 2 ? "var(--warn)" : "var(--pos)"
        }}>{state.wave}</span>
      </div>

      {/* CRI */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="eyebrow">CRI</span>
        <span className="font-data tabular text-[11px]" style={{
          color: state.contagionRisk > 0.8 ? "var(--neg)" : state.contagionRisk > 0.5 ? "var(--warn)" : "var(--pos)"
        }}>{(state.contagionRisk * 100).toFixed(0)}%</span>
      </div>

      {/* Sentiment triplet */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border font-data tabular text-[10px]">
        <span className="text-ki-success">+{state.sentiment.positive}%</span>
        <span className="text-ki-on-surface-muted">{state.sentiment.neutral}%</span>
        <span className="text-ki-error">-{state.sentiment.negative}%</span>
      </div>

      {/* Polarization */}
      <div className="flex items-center gap-1 px-3 border-r border-ki-border">
        <span className="eyebrow">Pol</span>
        <span className="font-data tabular text-[11px]" style={{
          color: state.polarization > 7 ? "var(--neg)" : state.polarization > 5 ? "var(--warn)" : "var(--pos)"
        }}>{state.polarization.toFixed(1)}</span>
      </div>

      {/* Event */}
      <div className="flex-1 px-3 overflow-hidden">
        <span className="font-data text-[11px] text-ki-on-surface-muted truncate block">
          {state.event}
        </span>
      </div>

      {/* Agents count */}
      <div className="flex items-center gap-2 px-3 border-l border-ki-border">
        <span className="font-data tabular text-[11px] text-ki-on-surface-secondary">{agentCount || 0} agents</span>
      </div>

      {/* Intervene button */}
      {onIntervene && (
        <button
          onClick={onIntervene}
          className="inline-flex items-center gap-1.5 h-7 px-2.5 ml-2 rounded-sm bg-ki-primary text-white text-[11px] font-medium hover:bg-ki-primary-muted transition-colors"
        >
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 500" }}>bolt</span>
          Intervene
        </button>
      )}
    </div>
  );
}
