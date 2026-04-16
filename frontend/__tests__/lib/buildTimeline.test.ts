import { describe, it, expect, beforeEach } from "vitest";
import { buildTimelineForRound, buildFullTimeline } from "@/lib/replay/buildTimeline";
import {
  ROUND_DURATION,
  GRAPH_UPDATE_DELAY,
  INDICATOR_UPDATE_DELAY,
  COALITION_UPDATE_DELAY,
  TRENDING_UPDATE_DELAY,
  POST_STAGGER,
  POST_IMPACT_DELAY,
  ENGAGE_TICKS,
} from "@/lib/replay/constants";
import type { ReplayRoundData } from "@/lib/replay/types";
import type { SimEventType } from "@/lib/replay/types";

function makeRoundData(overrides: Partial<ReplayRoundData> = {}): ReplayRoundData {
  return {
    round: 1,
    month: "Jan 2026",
    event: { event: "Test event", shock_magnitude: 0.5, shock_direction: -1 },
    posts: [],
    graphSnapshot: {
      round: 1,
      month: "Jan 2026",
      event_label: "Test event",
      nodes: [],
      edges: [],
      stats: { total_nodes: 0, total_edges: 0, avg_position: 0 },
    },
    indicators: {
      polarization: 0.3,
      engagement: 100,
      sentiment: { positive: 0.4, neutral: 0.3, negative: 0.3 },
      trendingHashtags: [{ tag: "#test", count: 5 }],
    },
    coalitions: { coalitions: [] },
    key_insight: "Test insight",
    ...overrides,
  };
}

function makePost(id: string, engagement: number) {
  return {
    id,
    author_id: `agent_${id}`,
    author_name: `Agent ${id}`,
    author_role: "citizen",
    tier: 1,
    platform: "twitter",
    text: `Post ${id} content #hashtag`,
    round: 1,
    likes: 10,
    reposts: 5,
    replies: 3,
    engagement_score: engagement,
    virality_tier: 1,
  };
}

describe("buildTimelineForRound", () => {
  it("produces correct base events for an empty-posts round", () => {
    const roundData = makeRoundData();
    const events = buildTimelineForRound(roundData);

    // Should have: ROUND_START, GRAPH_UPDATE, TRENDING_UPDATE, INDICATOR_UPDATE, COALITION_SHIFT
    expect(events.length).toBe(5);

    const types = events.map((e) => e.type);
    expect(types).toContain("ROUND_START");
    expect(types).toContain("GRAPH_UPDATE");
    expect(types).toContain("TRENDING_UPDATE");
    expect(types).toContain("INDICATOR_UPDATE");
    expect(types).toContain("COALITION_SHIFT");
  });

  it("events are sorted by timestamp", () => {
    const roundData = makeRoundData({
      posts: [makePost("p1", 100), makePost("p2", 50)],
    });
    const events = buildTimelineForRound(roundData);

    for (let i = 1; i < events.length; i++) {
      expect(events[i].timestamp).toBeGreaterThanOrEqual(events[i - 1].timestamp);
    }
  });

  it("ROUND_START is always the first event", () => {
    const roundData = makeRoundData();
    const events = buildTimelineForRound(roundData);
    expect(events[0].type).toBe("ROUND_START");
  });

  it("all events have the correct round number", () => {
    const roundData = makeRoundData({ round: 3 });
    const events = buildTimelineForRound(roundData);
    for (const event of events) {
      expect(event.round).toBe(3);
    }
  });

  it("event timestamps are within round bounds", () => {
    const round = 2;
    const roundData = makeRoundData({
      round,
      posts: [makePost("p1", 80)],
    });
    const events = buildTimelineForRound(roundData);
    const roundStart = (round - 1) * ROUND_DURATION;

    for (const event of events) {
      expect(event.timestamp).toBeGreaterThanOrEqual(roundStart);
      // Events should be within a reasonable range of the round
      // (ROUND_DURATION plus some buffer for post engage ticks)
      expect(event.timestamp).toBeLessThan(roundStart + ROUND_DURATION * 2);
    }
  });

  it("GRAPH_UPDATE is at roundStart + GRAPH_UPDATE_DELAY", () => {
    const roundData = makeRoundData({ round: 1 });
    const events = buildTimelineForRound(roundData);
    const graphEvent = events.find((e) => e.type === "GRAPH_UPDATE");
    expect(graphEvent).toBeDefined();
    expect(graphEvent!.timestamp).toBe(0 + GRAPH_UPDATE_DELAY);
  });

  it("INDICATOR_UPDATE is at roundStart + INDICATOR_UPDATE_DELAY", () => {
    const roundData = makeRoundData({ round: 1 });
    const events = buildTimelineForRound(roundData);
    const indicatorEvent = events.find((e) => e.type === "INDICATOR_UPDATE");
    expect(indicatorEvent!.timestamp).toBe(INDICATOR_UPDATE_DELAY);
  });

  it("COALITION_SHIFT is at roundStart + COALITION_UPDATE_DELAY", () => {
    const roundData = makeRoundData({ round: 1 });
    const events = buildTimelineForRound(roundData);
    const coalitionEvent = events.find((e) => e.type === "COALITION_SHIFT");
    expect(coalitionEvent!.timestamp).toBe(COALITION_UPDATE_DELAY);
  });

  it("TRENDING_UPDATE is at roundStart + TRENDING_UPDATE_DELAY", () => {
    const roundData = makeRoundData({ round: 1 });
    const events = buildTimelineForRound(roundData);
    const trendingEvent = events.find((e) => e.type === "TRENDING_UPDATE");
    expect(trendingEvent!.timestamp).toBe(TRENDING_UPDATE_DELAY);
  });

  it("every event has a unique id", () => {
    const roundData = makeRoundData({
      posts: [makePost("p1", 100), makePost("p2", 50), makePost("p3", 30)],
    });
    const events = buildTimelineForRound(roundData);
    const ids = events.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  describe("post events", () => {
    it("generates POST_APPEAR for each post", () => {
      const roundData = makeRoundData({
        posts: [makePost("p1", 100), makePost("p2", 50)],
      });
      const events = buildTimelineForRound(roundData);
      const postAppears = events.filter((e) => e.type === "POST_APPEAR");
      expect(postAppears.length).toBe(2);
    });

    it("generates POST_ENGAGE ticks for each post", () => {
      const roundData = makeRoundData({
        posts: [makePost("p1", 100)],
      });
      const events = buildTimelineForRound(roundData);
      const engageTicks = events.filter((e) => e.type === "POST_ENGAGE");
      expect(engageTicks.length).toBe(ENGAGE_TICKS);
    });

    it("POST_ENGAGE ticks have progressive progress values", () => {
      const roundData = makeRoundData({
        posts: [makePost("p1", 100)],
      });
      const events = buildTimelineForRound(roundData);
      const engageTicks = events.filter((e) => e.type === "POST_ENGAGE");
      engageTicks.forEach((tick, i) => {
        const payload = tick.payload as { postId: string; progress: number };
        expect(payload.progress).toBeCloseTo((i + 1) / ENGAGE_TICKS);
      });
    });

    it("posts are staggered by POST_STAGGER interval", () => {
      const roundData = makeRoundData({
        posts: [makePost("p1", 100), makePost("p2", 50), makePost("p3", 10)],
      });
      const events = buildTimelineForRound(roundData);
      const postAppears = events.filter((e) => e.type === "POST_APPEAR");
      // Posts sorted by engagement_score descending, then staggered
      for (let i = 1; i < postAppears.length; i++) {
        const diff = postAppears[i].timestamp - postAppears[i - 1].timestamp;
        expect(diff).toBe(POST_STAGGER);
      }
    });

    it("generates POST_IMPACT when postImpacts are present", () => {
      const roundData = makeRoundData({
        posts: [makePost("p1", 100)],
        postImpacts: [
          {
            postId: "p1",
            authorId: "agent_p1",
            influencedAgents: [],
            reach: 5,
            aggregateShift: 0.1,
            edgeEffects: [],
          },
        ],
      });
      const events = buildTimelineForRound(roundData);
      const impacts = events.filter((e) => e.type === "POST_IMPACT");
      expect(impacts.length).toBe(1);
    });

    it("POST_IMPACT occurs POST_IMPACT_DELAY ms after POST_APPEAR", () => {
      const roundData = makeRoundData({
        round: 1,
        posts: [makePost("p1", 100)],
        postImpacts: [
          {
            postId: "p1",
            authorId: "agent_p1",
            influencedAgents: [],
            reach: 5,
            aggregateShift: 0.1,
            edgeEffects: [],
          },
        ],
      });
      const events = buildTimelineForRound(roundData);
      const postAppear = events.find((e) => e.type === "POST_APPEAR")!;
      const postImpact = events.find((e) => e.type === "POST_IMPACT")!;
      expect(postImpact.timestamp - postAppear.timestamp).toBe(POST_IMPACT_DELAY);
    });

    it("does not generate POST_IMPACT when no postImpacts match", () => {
      const roundData = makeRoundData({
        posts: [makePost("p1", 100)],
        postImpacts: [
          {
            postId: "p_nonexistent",
            authorId: "agent_x",
            influencedAgents: [],
            reach: 0,
            aggregateShift: 0,
            edgeEffects: [],
          },
        ],
      });
      const events = buildTimelineForRound(roundData);
      const impacts = events.filter((e) => e.type === "POST_IMPACT");
      expect(impacts.length).toBe(0);
    });
  });

  it("ROUND_START payload includes month, event, and key_insight", () => {
    const roundData = makeRoundData({
      month: "Feb 2026",
      key_insight: "Markets react strongly",
    });
    const events = buildTimelineForRound(roundData);
    const roundStart = events.find((e) => e.type === "ROUND_START")!;
    const payload = roundStart.payload as Record<string, unknown>;
    expect(payload.month).toBe("Feb 2026");
    expect(payload.key_insight).toBe("Markets react strongly");
    expect(payload.event).toEqual(roundData.event);
  });
});

describe("buildFullTimeline", () => {
  it("returns empty array for empty rounds", () => {
    const events = buildFullTimeline([]);
    expect(events).toEqual([]);
  });

  it("produces events from all rounds", () => {
    const rounds = [makeRoundData({ round: 1 }), makeRoundData({ round: 2 })];
    const events = buildFullTimeline(rounds);

    const roundStarts = events.filter((e) => e.type === "ROUND_START");
    expect(roundStarts.length).toBe(2);
    expect(roundStarts[0].round).toBe(1);
    expect(roundStarts[1].round).toBe(2);
  });

  it("full timeline is sorted by timestamp across rounds", () => {
    const rounds = [
      makeRoundData({ round: 1 }),
      makeRoundData({ round: 2 }),
      makeRoundData({ round: 3 }),
    ];
    const events = buildFullTimeline(rounds);

    for (let i = 1; i < events.length; i++) {
      expect(events[i].timestamp).toBeGreaterThanOrEqual(events[i - 1].timestamp);
    }
  });

  it("resets event counter so IDs start from evt_1", () => {
    const rounds = [makeRoundData({ round: 1 })];
    const events = buildFullTimeline(rounds);
    expect(events[0].id).toBe("evt_1");
  });

  it("all expected event types are present for a single round", () => {
    const rounds = [
      makeRoundData({
        round: 1,
        posts: [makePost("p1", 100)],
        postImpacts: [
          {
            postId: "p1",
            authorId: "agent_p1",
            influencedAgents: [],
            reach: 3,
            aggregateShift: 0.05,
            edgeEffects: [],
          },
        ],
      }),
    ];
    const events = buildFullTimeline(rounds);
    const types = new Set(events.map((e) => e.type));

    const expectedTypes: SimEventType[] = [
      "ROUND_START",
      "POST_APPEAR",
      "POST_ENGAGE",
      "POST_IMPACT",
      "GRAPH_UPDATE",
      "INDICATOR_UPDATE",
      "COALITION_SHIFT",
      "TRENDING_UPDATE",
    ];
    for (const t of expectedTypes) {
      expect(types.has(t)).toBe(true);
    }
  });

  it("round 2 events start after round 1 events", () => {
    const rounds = [makeRoundData({ round: 1 }), makeRoundData({ round: 2 })];
    const events = buildFullTimeline(rounds);

    const round1Events = events.filter((e) => e.round === 1);
    const round2Events = events.filter((e) => e.round === 2);

    const maxRound1 = Math.max(...round1Events.map((e) => e.timestamp));
    const minRound2 = Math.min(...round2Events.map((e) => e.timestamp));

    expect(minRound2).toBeGreaterThanOrEqual(maxRound1);
  });
});
