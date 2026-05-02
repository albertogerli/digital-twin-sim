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
      <div className="flex border-b border-ki-border flex-shrink-0 bg-ki-surface-raised">
        <button
          onClick={() => setTab("debate")}
          className={`flex-1 h-8 eyebrow transition-colors ${
            tab === "debate"
              ? "text-ki-on-surface bg-ki-surface-raised border-b-2 border-ki-on-surface"
              : "text-ki-on-surface-muted hover:text-ki-on-surface-secondary hover:bg-ki-surface-hover"
          }`}
        >
          Debate
        </button>
        <button
          onClick={() => setTab("effects")}
          className={`flex-1 h-8 eyebrow transition-colors ${
            tab === "effects"
              ? "text-ki-on-surface bg-ki-surface-raised border-b-2 border-ki-on-surface"
              : "text-ki-on-surface-muted hover:text-ki-on-surface-secondary hover:bg-ki-surface-hover"
          }`}
        >
          Real effects
        </button>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {tab === "debate" ? (
          <div className="divide-y divide-ki-border">
            {/* Counters */}
            <div className="px-4 py-3 flex gap-5">
              <AnimatedCounter value={indicators.postCount} label="Posts" size="md" />
              <AnimatedCounter value={indicators.reactionCount} label="Reactions" size="md" />
            </div>

            <div className="px-4 py-3">
              <PolarizationGauge value={indicators.polarization} />
            </div>

            <div className="px-4 py-3">
              <SentimentDonut
                positive={indicators.sentimentDistribution.positive}
                neutral={indicators.sentimentDistribution.neutral}
                negative={indicators.sentimentDistribution.negative}
              />
            </div>

            <div className="px-4 py-3">
              <div className="eyebrow mb-2">Active agents</div>
              <AgentActivityList
                agents={indicators.activeAgents}
                currentTime={indicators.roundProgress * 30000}
              />
            </div>

            <div className="px-4 py-3">
              <div className="eyebrow mb-2">Trending</div>
              <TrendingHashtags hashtags={indicators.trendingHashtags} />
            </div>

            <div className="px-4 py-3">
              <div className="eyebrow mb-2">Coalitions</div>
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
          <div className="px-4 py-3">
            <RealWorldEffectsPanel effects={realWorldEffects} />
          </div>
        )}
      </div>
    </div>
  );
}
