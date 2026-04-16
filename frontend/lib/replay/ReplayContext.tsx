"use client";

import { createContext, useContext } from "react";
import type {
  ReplayState,
  ReplayControls,
  VisiblePost,
  IndicatorState,
  ActiveImpact,
  RealWorldEffects,
} from "./types";
import type { GraphSnapshot, CoalitionData, PostImpact } from "@/lib/types";

export interface ReplayContextValue {
  state: ReplayState;
  visiblePosts: VisiblePost[];
  graphSnapshot: GraphSnapshot | null;
  currentEvent: {
    month: string;
    event: { event: string; shock_magnitude: number; shock_direction: number };
  } | null;
  keyInsight: string;
  indicators: IndicatorState;
  coalitions: CoalitionData | null;
  activeImpact: ActiveImpact | null;
  selectedPostId: string | null;
  selectedPostImpact: PostImpact | null;
  realWorldEffects: RealWorldEffects | null;
  controls: ReplayControls;
  totalRounds: number;
  scenarioId: string;
  scenarioTitle: string;
}

const ReplayContext = createContext<ReplayContextValue | null>(null);

export function useReplay(): ReplayContextValue {
  const ctx = useContext(ReplayContext);
  if (!ctx) {
    throw new Error("useReplay must be used within a ReplayProvider");
  }
  return ctx;
}

export default ReplayContext;
