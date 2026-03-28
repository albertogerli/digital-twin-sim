"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type {
  SimEvent,
  ReplayState,
  ReplayControls,
  VisiblePost,
  IndicatorState,
  ReplayRoundData,
  ActiveImpact,
  RealWorldEffects,
} from "./types";
import type { GraphSnapshot, CoalitionData, PostImpact } from "@/lib/types";
import { buildFullTimeline } from "./buildTimeline";
import { ROUND_DURATION, formatElapsed, DEFAULT_SPEED } from "./constants";

interface ReplayOutput {
  state: ReplayState;
  visiblePosts: VisiblePost[];
  graphSnapshot: GraphSnapshot | null;
  currentEvent: { month: string; event: { event: string; shock_magnitude: number; shock_direction: number } } | null;
  keyInsight: string;
  indicators: IndicatorState;
  coalitions: CoalitionData | null;
  activeImpact: ActiveImpact | null;
  selectedPostId: string | null;
  realWorldEffects: RealWorldEffects | null;
  controls: ReplayControls;
}

const INITIAL_INDICATORS: IndicatorState = {
  postCount: 0,
  reactionCount: 0,
  polarization: 5.5,
  sentimentDistribution: { positive: 0.33, neutral: 0.34, negative: 0.33 },
  activeAgents: [],
  trendingHashtags: [],
  coalitionSizes: [],
  roundProgress: 0,
};

export function useSimulationReplay(
  roundsData: ReplayRoundData[] | null,
): ReplayOutput {
  // Timeline
  const timelineRef = useRef<SimEvent[]>([]);
  const [state, setState] = useState<ReplayState>({
    status: "idle",
    speed: DEFAULT_SPEED,
    currentTime: 0,
    currentRound: 0,
    eventIndex: 0,
    totalEvents: 0,
    elapsedDisplay: "0:00",
  });

  // Derive total rounds dynamically from data
  const totalRounds = roundsData?.length ?? 0;

  // Visible data
  const [visiblePosts, setVisiblePosts] = useState<VisiblePost[]>([]);
  const [graphSnapshot, setGraphSnapshot] = useState<GraphSnapshot | null>(null);
  const [currentEvent, setCurrentEvent] = useState<ReplayOutput["currentEvent"]>(null);
  const [keyInsight, setKeyInsight] = useState("");
  const [indicators, setIndicators] = useState<IndicatorState>(INITIAL_INDICATORS);
  const [coalitions, setCoalitions] = useState<CoalitionData | null>(null);
  const [activeImpact, setActiveImpact] = useState<ActiveImpact | null>(null);
  const [selectedPostId, setSelectedPostId] = useState<string | null>(null);
  const [realWorldEffects, setRealWorldEffects] = useState<RealWorldEffects | null>(null);
  const activeImpactTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Animation frame
  const frameRef = useRef<number>(0);
  const lastTickRef = useRef<number>(0);
  const speedRef = useRef(DEFAULT_SPEED);
  const statusRef = useRef<ReplayState["status"]>("idle");
  const eventIndexRef = useRef(0);
  const currentTimeRef = useRef(0);
  const totalRoundsRef = useRef(totalRounds);
  totalRoundsRef.current = totalRounds;

  // Build timeline when data arrives
  useEffect(() => {
    if (!roundsData || roundsData.length === 0) return;
    const timeline = buildFullTimeline(roundsData);
    timelineRef.current = timeline;
    eventIndexRef.current = 0;
    currentTimeRef.current = 0;

    setState((s) => ({
      ...s,
      status: "paused",
      totalEvents: timeline.length,
      eventIndex: 0,
      currentTime: 0,
      currentRound: 0,
      elapsedDisplay: "0:00",
    }));
    statusRef.current = "paused";

    // Reset visible state
    setVisiblePosts([]);
    setGraphSnapshot(null);
    setCurrentEvent(null);
    setKeyInsight("");
    setIndicators(INITIAL_INDICATORS);
    setCoalitions(null);
    setRealWorldEffects(null);
  }, [roundsData]);

  // Process a single event
  const processEvent = useCallback((event: SimEvent) => {
    switch (event.type) {
      case "ROUND_START": {
        const p = event.payload as { month: string; event: { event: string; shock_magnitude: number; shock_direction: number }; key_insight: string };
        setCurrentEvent({ month: p.month, event: p.event });
        setKeyInsight(p.key_insight);
        break;
      }
      case "POST_APPEAR": {
        const post = event.payload as VisiblePost;
        setVisiblePosts((prev) => [post, ...prev.map(p => ({ ...p, isNew: false }))]);
        setIndicators((prev) => ({
          ...prev,
          postCount: prev.postCount + 1,
          reactionCount: prev.reactionCount + post.likes + post.reposts + post.replies,
          activeAgents: [
            { id: post.author_id, name: post.author_name, timestamp: event.timestamp },
            ...prev.activeAgents.filter((a) => a.id !== post.author_id).slice(0, 7),
          ],
        }));
        break;
      }
      case "POST_ENGAGE": {
        const { postId, progress } = event.payload as { postId: string; progress: number };
        setVisiblePosts((prev) =>
          prev.map((p) =>
            p.id === postId ? { ...p, engagementProgress: progress } : p,
          ),
        );
        break;
      }
      case "GRAPH_UPDATE": {
        setGraphSnapshot(event.payload as GraphSnapshot);
        break;
      }
      case "INDICATOR_UPDATE": {
        const ind = event.payload as {
          polarization: number;
          engagement: number;
          sentiment: { positive: number; neutral: number; negative: number };
          trendingHashtags: { tag: string; count: number }[];
          realWorldEffects?: RealWorldEffects;
        };
        setIndicators((prev) => ({
          ...prev,
          polarization: ind.polarization,
          sentimentDistribution: ind.sentiment,
        }));
        if (ind.realWorldEffects) {
          setRealWorldEffects(ind.realWorldEffects);
        }
        break;
      }
      case "TRENDING_UPDATE": {
        const tags = event.payload as { tag: string; count: number }[];
        const prevTags = new Set(
          indicators.trendingHashtags.map((h) => h.tag),
        );
        setIndicators((prev) => ({
          ...prev,
          trendingHashtags: [...tags]
            .sort((a, b) => b.count - a.count)
            .map((t) => ({
              tag: t.tag,
              count: t.count,
              trend: !prevTags.has(t.tag)
                ? ("new" as const)
                : ("up" as const),
            })),
        }));
        break;
      }
      case "POST_IMPACT": {
        const impact = event.payload as PostImpact;
        // Attach impact data to the corresponding post
        setVisiblePosts((prev) =>
          prev.map((p) =>
            p.id === impact.postId ? { ...p, impact } : p,
          ),
        );
        // Set active impact for graph animation
        const newActiveImpact: ActiveImpact = {
          postId: impact.postId,
          authorId: impact.authorId,
          influencedNodeIds: impact.influencedAgents.map((a) => a.agentId),
          affectedEdges: impact.edgeEffects.map((e) => ({ source: e.source, target: e.target })),
          nodeShifts: new Map(impact.influencedAgents.map((a) => [a.agentId, a.shift])),
          startTime: Date.now(),
        };
        setActiveImpact(newActiveImpact);
        // Auto-clear after 2500ms
        if (activeImpactTimerRef.current) clearTimeout(activeImpactTimerRef.current);
        activeImpactTimerRef.current = setTimeout(() => {
          setActiveImpact((prev) => prev?.postId === impact.postId ? null : prev);
        }, 2500);
        break;
      }
      case "COALITION_SHIFT": {
        setCoalitions(event.payload as CoalitionData);
        break;
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The tick loop
  const tick = useCallback((now: number) => {
    if (statusRef.current !== "playing") return;

    const delta = lastTickRef.current ? (now - lastTickRef.current) : 0;
    lastTickRef.current = now;

    const advanceMs = delta * speedRef.current;
    currentTimeRef.current += advanceMs;

    const timeline = timelineRef.current;
    let idx = eventIndexRef.current;

    // Process all events up to current time
    while (idx < timeline.length && timeline[idx].timestamp <= currentTimeRef.current) {
      processEvent(timeline[idx]);
      idx++;
    }
    eventIndexRef.current = idx;

    // Calculate current round — use dynamic total from data
    const maxRound = totalRoundsRef.current || 1;
    const currentRound = Math.min(
      Math.floor(currentTimeRef.current / ROUND_DURATION) + 1,
      maxRound,
    );
    const roundProgress =
      (currentTimeRef.current % ROUND_DURATION) / ROUND_DURATION;

    setState((s) => ({
      ...s,
      currentTime: currentTimeRef.current,
      eventIndex: idx,
      currentRound,
      elapsedDisplay: formatElapsed(currentTimeRef.current / speedRef.current),
    }));

    setIndicators((prev) => ({ ...prev, roundProgress }));

    // Check if finished
    if (idx >= timeline.length) {
      statusRef.current = "finished";
      setState((s) => ({ ...s, status: "finished" }));
      return;
    }

    frameRef.current = requestAnimationFrame(tick);
  }, [processEvent]);

  // Controls
  const play = useCallback(() => {
    if (statusRef.current === "finished") {
      // restart from beginning
      eventIndexRef.current = 0;
      currentTimeRef.current = 0;
      setVisiblePosts([]);
      setIndicators(INITIAL_INDICATORS);
    }
    statusRef.current = "playing";
    setState((s) => ({ ...s, status: "playing" }));
    lastTickRef.current = 0;
    frameRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const pause = useCallback(() => {
    statusRef.current = "paused";
    setState((s) => ({ ...s, status: "paused" }));
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
  }, []);

  const toggle = useCallback(() => {
    if (statusRef.current === "playing") pause();
    else play();
  }, [play, pause]);

  const setSpeed = useCallback((speed: number) => {
    speedRef.current = speed;
    setState((s) => ({ ...s, speed }));
  }, []);

  const seekToRound = useCallback((round: number) => {
    const targetTime = (round - 1) * ROUND_DURATION;
    currentTimeRef.current = targetTime;
    eventIndexRef.current = 0;

    // Reset and replay all events up to target time
    setVisiblePosts([]);
    setIndicators(INITIAL_INDICATORS);
    setGraphSnapshot(null);
    setCurrentEvent(null);
    setCoalitions(null);
    setActiveImpact(null);
    setSelectedPostId(null);
    setRealWorldEffects(null);

    const timeline = timelineRef.current;
    let idx = 0;
    while (idx < timeline.length && timeline[idx].timestamp <= targetTime) {
      processEvent(timeline[idx]);
      idx++;
    }
    eventIndexRef.current = idx;

    setState((s) => ({
      ...s,
      currentTime: targetTime,
      eventIndex: idx,
      currentRound: round,
      elapsedDisplay: formatElapsed(targetTime / speedRef.current),
    }));
  }, [processEvent]);

  const restart = useCallback(() => {
    pause();
    eventIndexRef.current = 0;
    currentTimeRef.current = 0;
    setVisiblePosts([]);
    setGraphSnapshot(null);
    setCurrentEvent(null);
    setKeyInsight("");
    setIndicators(INITIAL_INDICATORS);
    setCoalitions(null);
    setActiveImpact(null);
    setSelectedPostId(null);
    setRealWorldEffects(null);
    setState((s) => ({
      ...s,
      status: "paused",
      currentTime: 0,
      eventIndex: 0,
      currentRound: 0,
      elapsedDisplay: "0:00",
    }));
    statusRef.current = "paused";
  }, [pause]);

  const selectPost = useCallback((postId: string | null) => {
    setSelectedPostId(postId);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, []);

  return {
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
    controls: { play, pause, toggle, setSpeed, seekToRound, restart, selectPost },
  };
}
