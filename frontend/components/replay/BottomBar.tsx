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
  if (p > 0.3) return "#16a34a";
  if (p > 0.1) return "#4ade80";
  if (p > -0.1) return "#94a3b8";
  if (p > -0.3) return "#f87171";
  return "#dc2626";
}

export default function BottomBar({ agents, activeAgentIds, currentRound, totalRounds }: Props) {
  const activeSet = new Set(activeAgentIds);
  const sorted = [...agents].sort((a, b) => b.influence - a.influence).slice(0, 30);

  return (
    <div className="h-full flex items-center px-4 gap-3 overflow-x-auto scrollbar-thin">
      <span className="font-mono text-[9px] text-gray-400 uppercase flex-shrink-0 mr-1">
        Agents
      </span>
      {sorted.map((agent) => {
        const isActive = activeSet.has(agent.id);
        return (
          <div
            key={agent.id}
            className={`flex flex-col items-center gap-1 flex-shrink-0 transition-all duration-300 ${
              isActive ? "opacity-100 scale-105" : "opacity-40"
            }`}
          >
            <div
              className={`w-7 h-7 rounded-full flex items-center justify-center text-[8px] font-bold text-white transition-all ${
                isActive ? "ring-2 ring-blue-500 ring-offset-1 ring-offset-gray-50" : ""
              }`}
              style={{ backgroundColor: posColor(agent.position) }}
            >
              {agent.name
                .split(" ")
                .map((w) => w[0])
                .slice(0, 2)
                .join("")}
            </div>
            <span className="font-mono text-[7px] text-gray-400 max-w-[48px] truncate text-center">
              {agent.name.split(" ").pop()}
            </span>
          </div>
        );
      })}
      <div className="flex-shrink-0 ml-auto font-mono text-[9px] text-gray-400">
        Round {currentRound}/{totalRounds}
      </div>
    </div>
  );
}
