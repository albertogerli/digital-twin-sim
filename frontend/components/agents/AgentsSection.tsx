"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import SectionHeader from "@/components/ui/SectionHeader";
import ScrollReveal from "@/components/ui/ScrollReveal";

interface Agent {
  id: string;
  name: string;
  role: string;
  archetype: string;
  tier: number;
  initial_position: number;
  final_position: number;
  position_delta: number;
  influence: number;
  emotional_state: string;
}

interface Props {
  agents: Agent[];
}

type TabKey = "elite" | "institutional" | "citizens";

const tierMap: Record<TabKey, number> = {
  elite: 1,
  institutional: 2,
  citizens: 3,
};

const tierLabels: Record<TabKey, string> = {
  elite: "Elite",
  institutional: "Institutional",
  citizens: "Citizen Clusters",
};

function PositionBar({ value }: { value: number }) {
  // value is -1 to +1
  const pct = ((value + 1) / 2) * 100;
  return (
    <div className="relative w-full h-2 rounded-full bg-gradient-to-r from-red-500/30 via-gray-300/30 to-green-500/30 overflow-hidden">
      <div
        className="absolute top-0 h-full w-1.5 rounded-full bg-gray-900 shadow-sm shadow-gray-900/50"
        style={{ left: `calc(${pct}% - 3px)` }}
      />
    </div>
  );
}

function AgentCard({ agent }: { agent: Agent }) {
  const emotionColor =
    agent.emotional_state === "angry" || agent.emotional_state === "frustrated"
      ? "text-red-600 bg-red-500/15 border-red-500/20"
      : agent.emotional_state === "hopeful" || agent.emotional_state === "optimistic"
        ? "text-green-600 bg-green-500/15 border-green-500/20"
        : "text-gray-500 bg-gray-500/15 border-gray-500/20";

  const deltaColor =
    agent.position_delta > 0.05
      ? "text-green-600"
      : agent.position_delta < -0.05
        ? "text-red-600"
        : "text-gray-400";

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4 hover:border-gray-300 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0 flex-1">
          <h4 className="font-display text-sm font-bold text-gray-900 truncate">
            {agent.name}
          </h4>
          <p className="font-body text-xs text-gray-400 truncate">{agent.role}</p>
        </div>
        <span className="ml-2 px-2 py-0.5 rounded-full bg-blue-500/15 border border-blue-500/20 text-cyan-600 font-mono text-[10px] uppercase flex-shrink-0">
          {agent.archetype}
        </span>
      </div>

      {/* Position bar */}
      <div className="mb-3">
        <div className="flex justify-between mb-1">
          <span className="font-mono text-[10px] text-gray-400">Position</span>
          <span className={`font-mono text-[10px] font-semibold ${deltaColor}`}>
            {agent.position_delta > 0 ? "+" : ""}
            {agent.position_delta.toFixed(2)}
          </span>
        </div>
        <PositionBar value={agent.final_position} />
        <div className="flex justify-between mt-0.5">
          <span className="font-mono text-[9px] text-red-600/60">-1</span>
          <span className="font-mono text-[9px] text-gray-500">0</span>
          <span className="font-mono text-[9px] text-green-600/60">+1</span>
        </div>
      </div>

      {/* Footer stats */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-mono text-[10px] text-gray-400">
          Influence: <span className="text-gray-700 font-semibold">{agent.influence.toFixed(2)}</span>
        </span>
        <span className={`px-1.5 py-0.5 rounded border text-[10px] font-mono ${emotionColor}`}>
          {agent.emotional_state}
        </span>
      </div>
    </div>
  );
}

export default function AgentsSection({ agents }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>("elite");

  const tabs = useMemo(() => {
    return (["elite", "institutional", "citizens"] as TabKey[]).map((key) => ({
      key,
      label: tierLabels[key],
      count: agents.filter((a) => a.tier === tierMap[key]).length,
    }));
  }, [agents]);

  const filteredAgents = useMemo(
    () => agents.filter((a) => a.tier === tierMap[activeTab]),
    [agents, activeTab]
  );

  return (
    <section id="agents" className="bg-white py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <SectionHeader
          title="Simulation Agents"
          subtitle="The autonomous agents that drive the simulation, organized by tier and influence level."
        />

        {/* Tabs */}
        <div className="flex gap-1 mb-10 bg-white rounded-lg p-1 w-fit border border-gray-200">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-5 py-2 text-sm font-sans rounded-md transition-all ${
                activeTab === tab.key
                  ? "bg-gray-100 text-gray-900 font-semibold"
                  : "text-gray-400 hover:text-gray-700"
              }`}
            >
              {tab.label}{" "}
              <span className="font-mono text-xs ml-1 opacity-50">({tab.count})</span>
            </button>
          ))}
        </div>

        {/* Agent Grid */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
        >
          {filteredAgents.map((agent, i) => (
            <ScrollReveal key={agent.id} delay={Math.min(i * 0.02, 0.3)}>
              <AgentCard agent={agent} />
            </ScrollReveal>
          ))}
        </motion.div>

        {filteredAgents.length === 0 && (
          <p className="text-center text-gray-400 py-12 font-body">
            No agents in this tier.
          </p>
        )}
      </div>
    </section>
  );
}
