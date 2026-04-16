"use client";

interface Agent {
  id: string;
  name: string;
  role: string;
  position: number;
  tier: number;
  cluster_size?: number;
}

interface AgentPositionStripProps {
  agents: Agent[];
  negativeLabel?: string;
  positiveLabel?: string;
}

const TIER_COLORS: Record<number, string> = {
  1: "#1a6dff", // ki-primary — elite
  2: "#7c3aed", // purple — institutional
  3: "#d97706", // ki-warning — citizen
};

function truncate(s: string, max: number) {
  return s.length > max ? s.slice(0, max - 1) + "\u2026" : s;
}

export default function AgentPositionStrip({
  agents,
  negativeLabel = "-1",
  positiveLabel = "+1",
}: AgentPositionStripProps) {
  if (!agents || agents.length === 0) return null;

  // Deterministic jitter based on agent id
  const jitter = (id: string) => {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
    return ((h % 100) / 100 - 0.5) * 14;
  };

  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-2" role="img" aria-label={`Posizioni di ${agents.length} agenti sull'asse`}>
      <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-2">
        Posizioni Agenti
      </p>
      {/* Axis labels above the strip */}
      <div className="flex justify-between text-[10px] text-ki-on-surface-muted mb-1 px-1">
        <span className="truncate max-w-[45%]" title={negativeLabel}>{truncate(negativeLabel, 20)}</span>
        <span className="truncate max-w-[45%] text-right" title={positiveLabel}>{truncate(positiveLabel, 20)}</span>
      </div>
      {/* Strip */}
      <div className="relative h-10 bg-gradient-to-r from-ki-error/10 via-ki-surface-sunken to-ki-success/10 rounded-sm border border-ki-border overflow-hidden">
        {/* Center line */}
        <div className="absolute left-1/2 top-1 bottom-1 w-px bg-ki-border-strong" />
        {/* Dots */}
        {agents.map((a) => {
          const leftPct = ((a.position + 1) / 2) * 100;
          const size = a.cluster_size ? Math.min(10, 6 + a.cluster_size / 100) : 6;
          const topOffset = 20 + jitter(a.id);
          return (
            <div
              key={a.id}
              className="absolute rounded-full opacity-75 hover:opacity-100 transition-opacity"
              style={{
                left: `${leftPct}%`,
                top: `${topOffset}px`,
                width: size,
                height: size,
                backgroundColor: TIER_COLORS[a.tier] || "#8a8a8a",
                transform: "translate(-50%, -50%)",
              }}
              title={`${a.name} (${a.position >= 0 ? "+" : ""}${a.position.toFixed(2)})`}
            />
          );
        })}
      </div>
      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-1.5 text-[10px] text-ki-on-surface-muted">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-ki-primary" />Elite</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: "#7c3aed" }} />Istituzionali</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-ki-warning" />Cittadini</span>
      </div>
    </div>
  );
}
