import type {
  PostData,
  GraphSnapshot,
  CoalitionData,
  PostImpact,
  RealWorldEffects,
} from "@/lib/types";

// ─── Event Types ───
export type SimEventType =
  | "ROUND_START"
  | "POST_APPEAR"
  | "POST_ENGAGE"
  | "POST_IMPACT"
  | "GRAPH_UPDATE"
  | "INDICATOR_UPDATE"
  | "COALITION_SHIFT"
  | "TRENDING_UPDATE";

export interface SimEvent {
  id: string;
  type: SimEventType;
  timestamp: number; // virtual ms from simulation start
  round: number;
  payload: unknown;
}

// ─── Replay State ───
export type ReplayStatus = "idle" | "loading" | "playing" | "paused" | "finished";

export interface ReplayState {
  status: ReplayStatus;
  speed: number; // 1, 2, 4, 8
  currentTime: number;
  currentRound: number;
  eventIndex: number;
  totalEvents: number;
  elapsedDisplay: string;
}

// ─── Indicator State ───
export interface IndicatorState {
  postCount: number;
  reactionCount: number;
  polarization: number;
  sentimentDistribution: { positive: number; neutral: number; negative: number };
  activeAgents: { id: string; name: string; timestamp: number }[];
  trendingHashtags: { tag: string; count: number; trend: "up" | "down" | "new" }[];
  coalitionSizes: { label: string; size: number; color: string }[];
  roundProgress: number;
}

// ─── Real-World Effects (re-exported from lib/types) ───
export type { RealWorldEffects } from "@/lib/types";

// ─── KB citation (RAG retrieval — backend-provided when grounded) ───
export interface PostCitation {
  doc_id: string;
  chunk_id: string;
  title: string;       // human-readable doc name
  snippet?: string;    // optional preview
  score: number;       // 0-1 retrieval similarity
}

// ─── Visible Post (with engagement progress) ───
export interface VisiblePost extends PostData {
  engagementProgress: number; // 0-1, how much of final engagement to show
  isNew: boolean;
  virtualTimestamp: number;
  handle: string;
  avatarColor: string;
  hashtags: string[];
  impact?: PostImpact;
  citations?: PostCitation[];  // RAG chunks consulted for this post
}

// ─── Active Impact (transient graph animation state) ───
export interface ActiveImpact {
  postId: string;
  authorId: string;
  influencedNodeIds: string[];
  affectedEdges: { source: string; target: string }[];
  nodeShifts: Map<string, number>;
  startTime: number;
}

// ─── Replay Round Data (per-round JSON file format) ───
export interface ReplayRoundData {
  round: number;
  month: string;
  event: {
    event: string;
    shock_magnitude: number;
    shock_direction: number;
  };
  posts: PostData[];
  graphSnapshot: GraphSnapshot;
  indicators: {
    polarization: number;
    engagement: number;
    sentiment: { positive: number; neutral: number; negative: number };
    trendingHashtags: { tag: string; count: number }[];
  };
  coalitions: CoalitionData;
  key_insight: string;
  postImpacts?: PostImpact[];
  realWorldEffects?: RealWorldEffects;
}

// ─── Replay Metadata ───
export interface ReplayMetadata {
  scenario: string;
  title: string;
  totalRounds: number;
  totalAgents: number;
  agents: {
    id: string;
    name: string;
    role: string;
    handle: string;
    avatarColor: string;
    position: number;
    influence: number;
  }[];
}

// ─── Controls ───
export interface ReplayControls {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setSpeed: (speed: number) => void;
  seekToRound: (round: number) => void;
  restart: () => void;
  selectPost: (postId: string | null) => void;
}
