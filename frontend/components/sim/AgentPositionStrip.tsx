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

export default function AgentPositionStrip({
  agents,
  negativeLabel = "-1",
  positiveLabel = "+1",
}: AgentPositionStripProps) {
  if (!agents || agents.length === 0) return null;

  const width = 280;
  const height = 40;
  const padX = 8;

  const toX = (pos: number) => padX + ((pos + 1) / 2) * (width - padX * 2);

  // Deterministic jitter based on agent id
  const jitter = (id: string) => {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
    return ((h % 100) / 100 - 0.5) * 12;
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
        Posizioni Agenti
      </p>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ maxHeight: 50 }}>
        {/* Axis */}
        <line x1={padX} y1={height / 2} x2={width - padX} y2={height / 2} stroke="#e5e7eb" strokeWidth={1} />
        <line x1={width / 2} y1={height / 2 - 6} x2={width / 2} y2={height / 2 + 6} stroke="#d1d5db" strokeWidth={1} />

        {/* Dots */}
        {agents.map((a) => {
          const cx = toX(a.position);
          const r = a.cluster_size ? Math.min(4, 2 + a.cluster_size / 50) : 2.5;
          return (
            <circle
              key={a.id}
              cx={cx}
              cy={height / 2 + jitter(a.id)}
              r={r}
              fill={TIER_COLORS[a.tier] || "#6b7280"}
              opacity={0.75}
            >
              <title>{a.name} ({a.position.toFixed(2)})</title>
            </circle>
          );
        })}

        {/* Labels */}
        <text x={padX} y={height - 2} fontSize={7} fill="#9ca3af" textAnchor="start">
          {negativeLabel}
        </text>
        <text x={width - padX} y={height - 2} fontSize={7} fill="#9ca3af" textAnchor="end">
          {positiveLabel}
        </text>
      </svg>
      <div className="flex items-center justify-center gap-3 mt-1 text-[9px] text-gray-400">
        <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-blue-500" />Elite</span>
        <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-purple-500" />Institutional</span>
        <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-amber-500" />Citizens</span>
      </div>
    </div>
  );
}
