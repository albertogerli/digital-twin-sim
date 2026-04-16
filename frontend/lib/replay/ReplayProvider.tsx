"use client";

import { useMemo } from "react";
import type { ReplayMeta, RoundData } from "@/lib/types";
import type { ReplayRoundData } from "./types";
import { useSimulationReplay } from "./useSimulationReplay";
import ReplayContext from "./ReplayContext";

interface Props {
  scenarioId: string;
  meta: ReplayMeta | null;
  rounds: RoundData[];
  children: React.ReactNode;
}

export default function ReplayProvider({
  scenarioId,
  meta,
  rounds,
  children,
}: Props) {
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
  const scenarioTitle = meta?.title ?? "Simulation";

  const value = useMemo(
    () => ({
      state,
      visiblePosts,
      graphSnapshot,
      currentEvent,
      keyInsight,
      indicators,
      coalitions,
      activeImpact,
      selectedPostId,
      selectedPostImpact,
      realWorldEffects,
      controls,
      totalRounds,
      scenarioId,
      scenarioTitle,
    }),
    [
      state,
      visiblePosts,
      graphSnapshot,
      currentEvent,
      keyInsight,
      indicators,
      coalitions,
      activeImpact,
      selectedPostId,
      selectedPostImpact,
      realWorldEffects,
      controls,
      totalRounds,
      scenarioId,
      scenarioTitle,
    ],
  );

  return (
    <ReplayContext.Provider value={value}>{children}</ReplayContext.Provider>
  );
}
