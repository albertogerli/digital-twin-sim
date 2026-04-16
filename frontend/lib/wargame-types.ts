/** Types for the Wargame Terminal — shared between client and components. */

export type Sentiment = "positive" | "negative" | "neutral";
export type TickerDirection = "short" | "long";
export type FlashColor = "green" | "red" | "none";

export interface WgAgent {
  id: string;
  name: string;
  role: string;
  tier: number; // 1=elite 2=inst 3=citizen
  position: number; // -1..+1
  sentiment: Sentiment;
  cluster: string;
}

export interface WgPost {
  id: string;
  authorId: string;
  authorName: string;
  authorTier: number;
  text: string;
  platform: string;
  engagement: number;
  sentiment: Sentiment;
  round: number;
  ts: number;
}

export interface WgTicker {
  ticker: string;
  sector: string;
  t1: number;
  t3: number;
  t7: number;
  direction: TickerDirection;
  beta: number;
  confidence: number;
  flash: FlashColor;
}

export interface WgRoundState {
  round: number;
  totalRounds: number;
  polarization: number;
  contagionRisk: number;
  engagementScore: number;
  wave: number;
  warning: string;
  event: string;
  sentiment: { positive: number; neutral: number; negative: number };
  coalitions: { label: string; size: number; position: number }[];
}

export interface WgSitrep {
  round_completed: number;
  next_round: number;
  player_role: string;
  status: string;
  threats: string[];
  polarization: number;
  sentiment: { positive: number; neutral: number; negative: number };
  engagement_score: number;
  active_wave: number;
  contagion_risk: number;
  top_narratives: { author: string; text: string; engagement: number }[];
  coalitions: { label: string; size: number; avg_position: number }[];
  financial_impact: Record<string, unknown>;
  prompt: string;
  suggested_actions: string[];
}

/** Terminal phase state machine */
export type WargamePhase =
  | "idle"           // Not started
  | "configuring"    // POST sent, waiting for SSE
  | "briefing"       // brief_analyzed / pre-round phases
  | "running"        // Round in progress
  | "awaiting"       // Waiting for player intervention
  | "completed"      // All rounds done
  | "error";         // Something failed

export interface WargameConfig {
  brief: string;
  provider?: string;
  model?: string;
  domain?: string;
  rounds?: number;
  budget?: number;
  playerRole?: string;
  metricsToTrack?: string[];
}

export interface BriefingStep {
  phase: string;
  message: string;
  done: boolean;
}

export interface LogEntry {
  ts: number;
  type: string;
  message: string;
}
