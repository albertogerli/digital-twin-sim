// ── scenarios.json ──────────────────────────────────────────────────
export interface ScenarioInfo {
  id: string;
  name: string;
  domain: string;
  description: string;
  num_rounds: number;
}

// ── metadata.json ───────────────────────────────────────────────────
export interface Metadata {
  scenario_id: string;
  scenario_name: string;
  num_rounds: number;
  domain: string;
  description: string;
}

// ── agents.json ─────────────────────────────────────────────────────
export interface AgentData {
  id: string;
  name: string;
  role: string;
  archetype: string;
  tier: number;
  initial_position: number;
  final_position: number;
  position_delta: number;
  influence: number;
  emotional_state: string;
}

// ── polarization.json ───────────────────────────────────────────────
export interface PolarizationPoint {
  round: number;
  polarization: number;
  avg_position: number;
  num_agents: number;
}

// ── coalitions.json ─────────────────────────────────────────────────
export interface Coalition {
  label: string;
  color: string;
  members: string[];
  avg_position: number;
  size: number;
}

export interface CoalitionRound {
  round: number;
  coalitions: Coalition[];
}

// ── top_posts.json ──────────────────────────────────────────────────
export interface TopPost {
  author_id: string;
  author_name: string;
  platform: string;
  text: string;
  round: number;
  likes: number;
  reposts: number;
  replies: number;
  total_engagement: number;
}

// ── replay_meta.json ────────────────────────────────────────────────
export interface ReplayAgent {
  id: string;
  name: string;
  role: string;
  handle: string;
  avatarColor: string;
  position: number;
  influence: number;
  archetype: string;
  tier: number;
}

export interface ReplayMeta {
  scenario: string;
  title: string;
  totalRounds: number;
  totalAgents: number;
  agents: ReplayAgent[];
}

// ── replay_round_N.json ─────────────────────────────────────────────

export interface RoundEvent {
  event: string;
  shock_magnitude: number;
  shock_direction: number;
}

export interface PostCitationRaw {
  doc_id: string;
  chunk_id: string;
  title: string;
  score: number;
  snippet?: string;
}

export interface PostData {
  id: string;
  author_id: string;
  author_name: string;
  author_role: string;
  tier: number;
  platform: string;
  text: string;
  round: number;
  likes: number;
  reposts: number;
  replies: number;
  engagement_score: number;
  virality_tier: number;
  citations?: PostCitationRaw[];
}

export interface GraphNode {
  id: string;
  name: string;
  type: string; // "persona" | "cluster"
  description: string;
  power_level: number;
  position: number;
  position_history: number[];
  delta: number;
  sentiment: string;
  category: string;
  clusterSize?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  type: string; // "influence" | "cluster_influence"
}

export interface GraphSnapshot {
  round: number;
  month: string;
  event_label: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    avg_position: number;
  };
}

export interface Indicators {
  polarization: number;
  engagement: number;
  sentiment: {
    positive: number;
    neutral: number;
    negative: number;
  };
  trendingHashtags: {
    tag: string;
    count: number;
  }[];
}

export interface CoalitionData {
  coalitions: Coalition[];
}

export interface PostImpact {
  postId: string;
  authorId: string;
  influencedAgents: {
    agentId: string;
    agentName: string;
    positionBefore: number;
    positionAfter: number;
    shift: number;
  }[];
  reach: number;
  aggregateShift: number;
  edgeEffects: {
    source: string;
    target: string;
    weightDelta: number;
  }[];
}

export interface RealWorldEffects {
  overview: {
    stability_index: number;
    tension_index: number;
    media_intensity: number;
    stakeholder_confidence: number;
  };
  opinion: {
    support_rate: number;
    opposition_rate: number;
    undecided_rate: number;
    avg_position: number;
  };
  engagement: {
    active_discussions: number;
    viral_content_pieces: number;
    coalition_count: number;
  };
}

export interface RoundData {
  round: number;
  month: string;
  event: RoundEvent;
  posts: PostData[];
  graphSnapshot: GraphSnapshot;
  indicators: Indicators;
  coalitions: CoalitionData;
  key_insight: string;
  postImpacts: PostImpact[];
  realWorldEffects: RealWorldEffects;
}
