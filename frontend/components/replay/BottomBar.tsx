"use client";

interface Agent {
  id: string;
  name: string;
  role: string;
  handle: string;
  avatarColor: string;
  position: number;
  influence: number;
}

interface Props {
  agents: Agent[];
  activeAgentIds: string[];
  currentRound: number;
  totalRounds: number;
}

function posColor(p: number): string {
  if (p > 0.3)  return "var(--pos)";
  if (p > 0.1)  return "color-mix(in oklch, var(--pos) 70%, white)";
  if (p > -0.1) return "var(--ink-3)";
  if (p > -0.3) return "color-mix(in oklch, var(--neg) 70%, white)";
  return "var(--neg)";
}

export default function BottomBar({ agents, activeAgentIds, currentRound, totalRounds }: Props) {
  const activeSet = new Set(activeAgentIds);
  const sorted = [...agents].sort((a, b) => b.influence - a.influence).slice(0, 30);

  return (
    <div className="h-12 flex items-center px-4 gap-3 overflow-x-auto scrollbar-thin">
      <span className="eyebrow flex-shrink-0 mr-1">Agents</span>
      {sorted.map((agent) => {
        const isActive = activeSet.has(agent.id);
        return (
          <div
            key={agent.id}
            className={`flex flex-col items-center gap-1 flex-shrink-0 transition-all duration-300 ${
              isActive ? "opacity-100" : "opacity-50"
            }`}
          >
            <div
              className={`w-7 h-7 rounded-full flex items-center justify-center text-[9px] font-medium text-white transition-all ${
                isActive ? "ring-2 ring-ki-on-surface ring-offset-1 ring-offset-ki-surface-raised" : ""
              }`}
              style={{ backgroundColor: posColor(agent.position) }}
            >
              {agent.name.split(" ").map((w) => w[0]).slice(0, 2).join("")}
            </div>
            <span className="font-data text-[9px] text-ki-on-surface-muted max-w-[52px] truncate text-center">
              {agent.name.split(" ").pop()}
            </span>
          </div>
        );
      })}
      <div className="flex-shrink-0 ml-auto font-data tabular text-[11px] text-ki-on-surface-muted">
        Round {currentRound}/{totalRounds}
      </div>
    </div>
  );
}
