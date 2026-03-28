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
  1: "#3b82f6", // blue — elite
  2: "#8b5cf6", // purple — institutional
  3: "#f59e0b", // amber — citizen
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
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-3">
        Posizioni Agenti
      </p>
      {/* Axis labels above the strip */}
      <div className="flex justify-between text-[10px] text-gray-400 mb-1 px-1">
        <span className="truncate max-w-[45%]" title={negativeLabel}>{truncate(negativeLabel, 20)}</span>
        <span className="truncate max-w-[45%] text-right" title={positiveLabel}>{truncate(positiveLabel, 20)}</span>
      </div>
      {/* Strip */}
      <div className="relative h-10 bg-gradient-to-r from-red-50 via-gray-50 to-emerald-50 rounded-lg border border-gray-100 overflow-hidden">
        {/* Center line */}
        <div className="absolute left-1/2 top-1 bottom-1 w-px bg-gray-300" />
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
                backgroundColor: TIER_COLORS[a.tier] || "#6b7280",
                transform: "translate(-50%, -50%)",
              }}
              title={`${a.name} (${a.position >= 0 ? "+" : ""}${a.position.toFixed(2)})`}
            />
          );
        })}
      </div>
      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-2 text-[10px] text-gray-400">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" />Elite</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500" />Istituzionali</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500" />Cittadini</span>
      </div>
    </div>
  );
}
