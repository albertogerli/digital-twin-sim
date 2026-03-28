"use client";

import { useState } from "react";
import type { IndicatorState, RealWorldEffects } from "@/lib/replay/types";
import type { CoalitionData } from "@/lib/types";
import AnimatedCounter from "./AnimatedCounter";
import PolarizationGauge from "./PolarizationGauge";
import SentimentDonut from "./SentimentDonut";
import TrendingHashtags from "./TrendingHashtags";
import AgentActivityList from "./AgentActivityList";
import CoalitionBar from "./CoalitionBar";
import RealWorldEffectsPanel from "./RealWorldEffectsPanel";

interface Props {
  indicators: IndicatorState;
  coalitions: CoalitionData | null;
  realWorldEffects: RealWorldEffects | null;
}

export default function IndicatorPanel({ indicators, coalitions, realWorldEffects }: Props) {
  const [tab, setTab] = useState<"debate" | "effects">("debate");

  return (
    <div className="h-full flex flex-col">
      {/* Tab bar */}
      <div className="flex border-b border-gray-200 flex-shrink-0">
        <button
          onClick={() => setTab("debate")}
          className={`flex-1 py-1.5 text-[9px] font-mono uppercase tracking-wider transition-colors ${
            tab === "debate"
              ? "text-cyan-600 border-b-2 border-blue-500 bg-blue-900/20"
              : "text-gray-400 hover:text-gray-700"
          }`}
        >
          Debate
        </button>
        <button
          onClick={() => setTab("effects")}
          className={`flex-1 py-1.5 text-[9px] font-mono uppercase tracking-wider transition-colors ${
            tab === "effects"
              ? "text-cyan-600 border-b-2 border-blue-500 bg-blue-900/20"
              : "text-gray-400 hover:text-gray-700"
          }`}
        >
          Real Effects
        </button>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {tab === "debate" ? (
          <div className="divide-y divide-gray-200">
            {/* Counters */}
            <div className="px-3 py-3 flex gap-4">
              <AnimatedCounter value={indicators.postCount} label="Posts" size="md" />
              <AnimatedCounter value={indicators.reactionCount} label="Reactions" size="md" />
            </div>

            {/* Polarization Gauge */}
            <div className="px-3 py-3">
              <PolarizationGauge value={indicators.polarization} />
            </div>

            {/* Sentiment Donut */}
            <div className="px-3 py-3">
              <SentimentDonut
                positive={indicators.sentimentDistribution.positive}
                neutral={indicators.sentimentDistribution.neutral}
                negative={indicators.sentimentDistribution.negative}
              />
            </div>

            {/* Active Agents */}
            <div className="px-3 py-2">
              <div className="font-mono text-[9px] text-gray-500 uppercase tracking-wider mb-1.5">
                Active Agents
              </div>
              <AgentActivityList
                agents={indicators.activeAgents}
                currentTime={indicators.roundProgress * 30000}
              />
            </div>

            {/* Trending Hashtags */}
            <div className="px-3 py-2">
              <div className="font-mono text-[9px] text-gray-500 uppercase tracking-wider mb-1.5">
                Trending
              </div>
              <TrendingHashtags hashtags={indicators.trendingHashtags} />
            </div>

            {/* Coalition Composition */}
            <div className="px-3 py-2">
              <div className="font-mono text-[9px] text-gray-500 uppercase tracking-wider mb-1.5">
                Coalitions
              </div>
              <CoalitionBar
                coalitions={
                  coalitions
                    ? coalitions.coalitions.map((c) => ({
                        label: c.label,
                        size: c.members.length,
                        color: c.color,
                      }))
                    : indicators.coalitionSizes
                }
              />
            </div>
          </div>
        ) : (
          <div className="px-3">
            <RealWorldEffectsPanel effects={realWorldEffects} />
          </div>
        )}
      </div>
    </div>
  );
}
