"use client";

import { useEffect } from "react";
import type { ReplayMeta, RoundData } from "@/lib/types";
import type { ReplayRoundData } from "@/lib/replay/types";
import { useSimulationReplay } from "@/lib/replay/useSimulationReplay";
import TopBar from "./TopBar";
import BottomBar from "./BottomBar";
import LiveFeed from "./feed/LiveFeed";
import LiveNetworkGraph from "./network/LiveNetworkGraph";
import IndicatorPanel from "./indicators/IndicatorPanel";

interface Props {
  scenarioId: string;
  meta: ReplayMeta | null;
  rounds: RoundData[];
}

export default function CommandCenter({ scenarioId, meta, rounds }: Props) {
  // Cast RoundData[] to ReplayRoundData[] — they share the same shape
  const replayRounds = rounds as unknown as ReplayRoundData[];

  const {
    state,
    visiblePosts,
    graphSnapshot,
    currentEvent,
    keyInsight,
    indicators,
    coalitions,
    activeImpact,
    selectedPostId,
    realWorldEffects,
    controls,
  } = useSimulationReplay(replayRounds.length > 0 ? replayRounds : null);

  const selectedPostImpact = selectedPostId
    ? visiblePosts.find((p) => p.id === selectedPostId)?.impact ?? null
    : null;

  const totalRounds = meta?.totalRounds ?? rounds.length;

  // Keyboard shortcuts
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement) return;
      switch (e.code) {
        case "Space":
          e.preventDefault();
          controls.toggle();
          break;
        case "ArrowRight":
          if (state.currentRound < totalRounds) controls.seekToRound(state.currentRound + 1);
          break;
        case "ArrowLeft":
          if (state.currentRound > 1) controls.seekToRound(state.currentRound - 1);
          break;
        case "Digit1": controls.setSpeed(1); break;
        case "Digit2": controls.setSpeed(2); break;
        case "Digit3": controls.setSpeed(4); break;
        case "Digit4": controls.setSpeed(8); break;
        case "KeyR": controls.restart(); break;
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [controls, state.currentRound, totalRounds]);

  return (
    <div className="h-screen flex flex-col bg-gray-50 text-gray-900">
      {/* Top Bar */}
      <div className="flex-shrink-0 border-b border-gray-200 bg-white">
        <TopBar
          state={state}
          currentEvent={currentEvent}
          controls={controls}
          scenarioId={scenarioId}
          scenarioTitle={meta?.title ?? "Simulation"}
          totalRounds={totalRounds}
        />
      </div>

      {/* Main content — grid on desktop, stack on mobile */}
      <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-[1fr_400px_260px] overflow-auto md:overflow-hidden">
        {/* Left — Network Graph */}
        <div className="border-b md:border-b-0 md:border-r border-gray-200 overflow-hidden bg-gray-50 min-h-[300px] md:min-h-0">
          <LiveNetworkGraph
            snapshot={graphSnapshot}
            activeAgentIds={indicators.activeAgents.map((a) => a.id)}
            activeImpact={activeImpact}
            selectedPostId={selectedPostId}
            selectedPostImpact={selectedPostImpact}
          />
        </div>

        {/* Center — Post Feed */}
        <div className="overflow-hidden border-b md:border-b-0 md:border-r border-gray-200 bg-gray-50 min-h-[400px] md:min-h-0">
          <LiveFeed
            posts={visiblePosts}
            keyInsight={keyInsight}
            status={state.status}
            onPlay={controls.play}
            selectedPostId={selectedPostId}
            onSelectPost={controls.selectPost}
          />
        </div>

        {/* Right — Indicators */}
        <div className="overflow-y-auto scrollbar-thin bg-gray-50">
          <IndicatorPanel
            indicators={indicators}
            coalitions={coalitions}
            realWorldEffects={realWorldEffects}
          />
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white hidden md:block">
        <BottomBar
          agents={meta?.agents || []}
          activeAgentIds={indicators.activeAgents.map((a) => a.id)}
          currentRound={state.currentRound}
          totalRounds={totalRounds}
        />
      </div>
    </div>
  );
}
