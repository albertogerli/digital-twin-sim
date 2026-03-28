import type { SimEvent, ReplayRoundData, VisiblePost } from "./types";
import {
  ROUND_DURATION,
  POST_STAGGER,
  ENGAGE_TICK_INTERVAL,
  ENGAGE_TICKS,
  GRAPH_UPDATE_DELAY,
  INDICATOR_UPDATE_DELAY,
  COALITION_UPDATE_DELAY,
  TRENDING_UPDATE_DELAY,
  POST_IMPACT_DELAY,
  agentToHandle,
  agentToAvatarColor,
  extractHashtags,
} from "./constants";

let eventCounter = 0;
function eid(): string {
  return `evt_${++eventCounter}`;
}

export function buildTimelineForRound(
  roundData: ReplayRoundData,
): SimEvent[] {
  const events: SimEvent[] = [];
  const r = roundData.round;
  const roundStart = (r - 1) * ROUND_DURATION;

  // 1. ROUND_START
  events.push({
    id: eid(),
    type: "ROUND_START",
    timestamp: roundStart,
    round: r,
    payload: {
      month: roundData.month,
      event: roundData.event,
      key_insight: roundData.key_insight,
    },
  });

  // 2. GRAPH_UPDATE
  events.push({
    id: eid(),
    type: "GRAPH_UPDATE",
    timestamp: roundStart + GRAPH_UPDATE_DELAY,
    round: r,
    payload: roundData.graphSnapshot,
  });

  // 3. Posts, staggered
  const posts = roundData.posts.sort(
    (a, b) => b.engagement_score - a.engagement_score,
  );
  posts.forEach((post, i) => {
    const postTime = roundStart + 2000 + i * POST_STAGGER;

    // POST_APPEAR
    events.push({
      id: eid(),
      type: "POST_APPEAR",
      timestamp: postTime,
      round: r,
      payload: {
        ...post,
        handle: agentToHandle(post.author_name),
        avatarColor: agentToAvatarColor(post.author_id),
        hashtags: extractHashtags(post.text),
        engagementProgress: 0,
        isNew: true,
        virtualTimestamp: postTime,
      } as VisiblePost,
    });

    // POST_IMPACT — 500ms after POST_APPEAR
    const impactData = roundData.postImpacts?.find(imp => imp.postId === post.id);
    if (impactData) {
      events.push({
        id: eid(),
        type: "POST_IMPACT",
        timestamp: postTime + POST_IMPACT_DELAY,
        round: r,
        payload: impactData,
      });
    }

    // POST_ENGAGE ticks
    for (let tick = 1; tick <= ENGAGE_TICKS; tick++) {
      events.push({
        id: eid(),
        type: "POST_ENGAGE",
        timestamp: postTime + tick * ENGAGE_TICK_INTERVAL,
        round: r,
        payload: {
          postId: post.id,
          progress: tick / ENGAGE_TICKS,
        },
      });
    }
  });

  // 4. TRENDING_UPDATE
  events.push({
    id: eid(),
    type: "TRENDING_UPDATE",
    timestamp: roundStart + TRENDING_UPDATE_DELAY,
    round: r,
    payload: roundData.indicators.trendingHashtags,
  });

  // 5. INDICATOR_UPDATE (includes real-world effects)
  events.push({
    id: eid(),
    type: "INDICATOR_UPDATE",
    timestamp: roundStart + INDICATOR_UPDATE_DELAY,
    round: r,
    payload: { ...roundData.indicators, realWorldEffects: roundData.realWorldEffects },
  });

  // 6. COALITION_SHIFT
  events.push({
    id: eid(),
    type: "COALITION_SHIFT",
    timestamp: roundStart + COALITION_UPDATE_DELAY,
    round: r,
    payload: roundData.coalitions,
  });

  return events.sort((a, b) => a.timestamp - b.timestamp);
}

export function buildFullTimeline(rounds: ReplayRoundData[]): SimEvent[] {
  eventCounter = 0;
  const allEvents = rounds.flatMap((r) => buildTimelineForRound(r));
  return allEvents.sort((a, b) => a.timestamp - b.timestamp);
}
