"use client";

import { AnimatePresence, motion } from "framer-motion";
import { agentToAvatarColor } from "@/lib/replay/constants";

interface ActiveAgent {
  id: string;
  name: string;
  timestamp: number;
}

interface Props {
  agents: ActiveAgent[];
  currentTime: number;
}

function timeAgo(now: number, ts: number): string {
  const diff = Math.max(0, now - ts);
  const sec = Math.floor(diff / 1000);
  if (sec < 5) return "now";
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  return `${min}m ago`;
}

export default function AgentActivityList({ agents, currentTime }: Props) {
  if (agents.length === 0) {
    return (
      <div className="text-[10px] font-mono text-gray-400 text-center py-2">
        No active agents
      </div>
    );
  }

  return (
    <div className="space-y-0.5">
      <AnimatePresence mode="popLayout">
        {agents.slice(0, 6).map((agent) => {
          const initials = agent.name
            .split(" ")
            .map((w) => w[0])
            .slice(0, 2)
            .join("");
          const color = agentToAvatarColor(agent.id);

          return (
            <motion.div
              key={agent.id}
              layout
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="flex items-center gap-2 py-1 px-2 rounded hover:bg-gray-100/60"
            >
              <div
                className="w-5 h-5 rounded-full flex items-center justify-center text-[7px] font-bold text-white flex-shrink-0"
                style={{ backgroundColor: color }}
              >
                {initials}
              </div>
              <span className="text-[11px] text-gray-700 truncate flex-1">
                {agent.name}
              </span>
              <span className="text-[9px] font-mono text-gray-400 flex-shrink-0">
                {timeAgo(currentTime, agent.timestamp)}
              </span>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
