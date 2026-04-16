"use client";

interface RegimeInfo {
  regime_prob: number;
  regime_label: string;
}

interface RegimeIndicatorProps {
  regimeInfo: RegimeInfo | null;
}

export default function RegimeIndicator({ regimeInfo }: RegimeIndicatorProps) {
  if (!regimeInfo) return null;

  const prob = regimeInfo.regime_prob;
  const isCrisis = regimeInfo.regime_label === "crisis";
  const pct = Math.round(prob * 100);

  // Color gradient: green (0) -> yellow (0.3) -> orange (0.5) -> red (0.8+)
  const getColor = (p: number) => {
    if (p < 0.2) return "#16a34a";
    if (p < 0.4) return "#d97706";
    if (p < 0.6) return "#d97706";
    return "#dc2626";
  };

  const color = getColor(prob);

  return (
    <div className={`flex items-center gap-2 px-2 py-1 rounded-sm border text-xs ${
      isCrisis
        ? "bg-ki-error/10 border-ki-error/25 text-ki-error"
        : "bg-ki-surface-sunken border-ki-border text-ki-on-surface-secondary"
    }`}>
      {/* Pulsing dot for crisis */}
      <span className="relative flex h-2.5 w-2.5">
        {isCrisis && (
          <span
            className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
            style={{ backgroundColor: color }}
          />
        )}
        <span
          className="relative inline-flex rounded-full h-2.5 w-2.5"
          style={{ backgroundColor: color }}
        />
      </span>

      <span className="font-medium capitalize">{regimeInfo.regime_label}</span>

      {/* Mini bar */}
      <div className="w-12 h-1.5 bg-ki-surface-sunken rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>

      <span className="text-[10px] text-ki-on-surface-muted">{pct}%</span>
    </div>
  );
}
