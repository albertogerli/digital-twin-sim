// Timing constants (at 1x speed, in ms)
export const ROUND_DURATION = 30_000; // 30 seconds per round
export const POST_STAGGER = 2_500; // 2.5s between posts
export const ENGAGE_TICK_INTERVAL = 400; // 400ms between engagement ticks
export const ENGAGE_TICKS = 4; // 4 ticks to reach final engagement
export const GRAPH_UPDATE_DELAY = 1_000; // 1s after round start
export const INDICATOR_UPDATE_DELAY = 15_000; // midway through round
export const COALITION_UPDATE_DELAY = 25_000; // near end of round
export const TRENDING_UPDATE_DELAY = 10_000; // 10s after round start
export const POST_IMPACT_DELAY = 500; // ms after POST_APPEAR

export const SPEEDS = [1, 2, 4, 8] as const;
export const DEFAULT_SPEED = 1;

// Avatar colors for agents
export const AVATAR_COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#6366f1",
  "#84cc16", "#e11d48", "#0ea5e9", "#d946ef", "#10b981",
];

export function agentToHandle(name: string): string {
  return "@" + name.replace(/\s+/g, "").replace(/['.]/g, "");
}

export function agentToAvatarColor(id: string): string {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = ((hash << 5) - hash + id.charCodeAt(i)) | 0;
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length];
}

export function extractHashtags(text: string): string[] {
  const matches = text.match(/#\w+/g);
  return matches ? Array.from(new Set(matches)) : [];
}

export function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, "0")}`;
}
