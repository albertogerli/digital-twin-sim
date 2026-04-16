"use client";

import type { WgSitrep } from "@/lib/wargame-types";

interface SitrepOverlayProps {
  sitrep: WgSitrep;
  onSkip: () => void;
}

export function SitrepOverlay({ sitrep, onSkip }: SitrepOverlayProps) {
  const cri = sitrep.contagion_risk || 0;
  const criColor = cri > 0.8 ? "#ff3b3b" : cri > 0.5 ? "#ffaa00" : "#00d26a";

  return (
    <div className="absolute top-7 right-0 w-[280px] bottom-0 bg-ki-surface border-l border-ki-border z-40 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-8 flex items-center px-2 border-b border-[#ff3b3b20] bg-[#1a0a0a] shrink-0">
        <span className="font-data text-[10px] text-[#ff3b3b] font-bold tracking-wider animate-pulse">
          ⚠ SITREP — YOUR MOVE
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {/* Round info */}
        <div className="flex items-center justify-between">
          <span className="font-data text-[9px] text-ki-on-surface-muted">ROUND COMPLETED</span>
          <span className="font-data text-[11px] text-ki-on-surface font-bold">{sitrep.round_completed}</span>
        </div>

        {/* Player role */}
        {sitrep.player_role && (
          <div className="p-1.5 border border-ki-border bg-ki-surface-sunken">
            <span className="font-data text-[8px] text-ki-on-surface-muted uppercase block mb-0.5">YOUR ROLE</span>
            <span className="font-data text-[10px] text-[#ffaa00]">{sitrep.player_role}</span>
          </div>
        )}

        {/* Threats */}
        {sitrep.threats && sitrep.threats.length > 0 && (
          <div>
            <span className="font-data text-[8px] text-[#ff3b3b] uppercase tracking-wider">THREATS</span>
            {sitrep.threats.map((t, i) => (
              <div key={i} className="flex items-start gap-1.5 mt-1">
                <span className="w-1 h-1 rounded-full bg-[#ff3b3b] mt-1 shrink-0" />
                <span className="font-data text-[9px] text-ki-on-surface-muted">{t}</span>
              </div>
            ))}
          </div>
        )}

        {/* Key metrics */}
        <div className="grid grid-cols-2 gap-1.5">
          <div className="p-1.5 border border-ki-border-strong">
            <span className="font-data text-[7px] text-ki-on-surface-muted uppercase block">POL</span>
            <span className="font-data text-[12px] font-bold" style={{
              color: sitrep.polarization > 7 ? "#ff3b3b" : sitrep.polarization > 5 ? "#ffaa00" : "#00d26a"
            }}>{sitrep.polarization?.toFixed(1) || "0"}</span>
          </div>
          <div className="p-1.5 border border-ki-border-strong">
            <span className="font-data text-[7px] text-ki-on-surface-muted uppercase block">CRI</span>
            <span className="font-data text-[12px] font-bold" style={{ color: criColor }}>
              {(cri * 100).toFixed(0)}%
            </span>
          </div>
          <div className="p-1.5 border border-ki-border-strong">
            <span className="font-data text-[7px] text-ki-on-surface-muted uppercase block">WAVE</span>
            <span className="font-data text-[12px] font-bold text-ki-on-surface">{sitrep.active_wave}</span>
          </div>
          <div className="p-1.5 border border-ki-border-strong">
            <span className="font-data text-[7px] text-ki-on-surface-muted uppercase block">SENT NEG</span>
            <span className="font-data text-[12px] font-bold text-[#ff3b3b]">
              {((sitrep.sentiment?.negative || 0) * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Top narratives */}
        {sitrep.top_narratives && sitrep.top_narratives.length > 0 && (
          <div>
            <span className="font-data text-[8px] text-ki-on-surface-muted uppercase tracking-wider">DOMINANT NARRATIVES</span>
            {sitrep.top_narratives.slice(0, 3).map((n, i) => (
              <div key={i} className="mt-1 p-1.5 border border-ki-border-strong bg-ki-surface-sunken">
                <div className="flex items-center justify-between mb-0.5">
                  <span className="font-data text-[8px] text-ki-on-surface font-medium">{n.author}</span>
                  <span className="font-data text-[7px] text-ki-on-surface-muted">{n.engagement > 1000 ? `${(n.engagement / 1000).toFixed(1)}K` : n.engagement}</span>
                </div>
                <span className="font-data text-[8px] text-ki-on-surface-muted leading-tight block">{n.text.slice(0, 120)}</span>
              </div>
            ))}
          </div>
        )}

        {/* Coalitions */}
        {sitrep.coalitions && sitrep.coalitions.length > 0 && (
          <div>
            <span className="font-data text-[8px] text-ki-on-surface-muted uppercase tracking-wider">COALITIONS</span>
            {sitrep.coalitions.map((c, i) => (
              <div key={i} className="flex items-center justify-between mt-1">
                <span className="font-data text-[8px] text-ki-on-surface-muted">{c.label}</span>
                <div className="flex items-center gap-2">
                  <span className="font-data text-[8px]" style={{
                    color: c.avg_position > 0.3 ? "#00d26a" : c.avg_position < -0.3 ? "#ff3b3b" : "#6b7280"
                  }}>{c.size}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Suggested actions */}
        {sitrep.suggested_actions && sitrep.suggested_actions.length > 0 && (
          <div>
            <span className="font-data text-[8px] text-ki-on-surface-muted uppercase tracking-wider">SUGGESTED MOVES</span>
            {sitrep.suggested_actions.map((a, i) => (
              <div key={i} className="mt-1 flex items-center gap-1.5">
                <span className="font-data text-[8px] text-ki-on-surface-muted">{i + 1}.</span>
                <span className="font-data text-[8px] text-ki-on-surface-muted">{a}</span>
              </div>
            ))}
          </div>
        )}

        {/* Financial impact headline */}
        {typeof sitrep.financial_impact?.headline === "string" && (
          <div className="p-1.5 border border-[#ffaa0020] bg-[#1a1508]">
            <span className="font-data text-[7px] text-[#ffaa00] uppercase block mb-0.5">MARKET IMPACT</span>
            <span className="font-data text-[8px] text-ki-on-surface-muted">{String(sitrep.financial_impact.headline)}</span>
          </div>
        )}
      </div>

      {/* Skip button */}
      <div className="p-2 border-t border-ki-border shrink-0">
        <button
          onClick={onSkip}
          className="w-full h-7 font-data text-[9px] border border-ki-border text-ki-on-surface-muted hover:text-ki-on-surface hover:border-ki-border-strong transition-colors"
        >
          SKIP — AUTO-GENERATE NEXT ROUND
        </button>
      </div>
    </div>
  );
}
