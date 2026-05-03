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

/* ── Mock RAG citations ──────────────────────────────────────
   Deterministic, hash-based per post id. Until the backend ships
   real citations, this gives the UI something to render. */
const KB_DOCS = [
  { doc_id: "d1", title: "EU_energy_package_draft_v3.pdf",  pages: 84 },
  { doc_id: "d2", title: "Brandt_speech_2026-04-22.txt",    pages: 4  },
  { doc_id: "d3", title: "industry_position_paper.docx",     pages: 22 },
  { doc_id: "d4", title: "grid_investment_dataset_q1.csv",   pages: 0  },
  { doc_id: "d5", title: "council_voting_record_2025.json",  pages: 0  },
];

function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function mockCitations(postId: string, authorId: string) {
  const h = hashStr(postId + ":" + authorId);
  // ~30% of posts get citations
  if (h % 100 >= 30) return undefined;
  const n = 1 + (h % 3); // 1-3 citations
  const cs = [];
  for (let i = 0; i < n; i++) {
    const doc = KB_DOCS[(h + i * 7) % KB_DOCS.length];
    cs.push({
      doc_id: doc.doc_id,
      chunk_id: `${doc.doc_id}_${(h + i * 13) % 200}`,
      title: doc.title,
      score: 0.55 + ((h + i * 19) % 40) / 100,
      snippet: undefined,
    });
  }
  return cs;
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
    // Use real RAG citations from the backend payload when present;
    // fall back to deterministic mock citations for older scenarios where the
    // backend didn't yet thread citations through.
    const realCitations = (post as any).citations as VisiblePost["citations"] | undefined;
    const citations = realCitations && realCitations.length > 0
      ? realCitations
      : mockCitations(post.id, post.author_id);

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
        citations,
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
