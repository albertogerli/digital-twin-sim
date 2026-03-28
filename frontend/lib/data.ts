import type {
  ScenarioInfo,
  Metadata,
  AgentData,
  PolarizationPoint,
  CoalitionRound,
  TopPost,
  ReplayMeta,
  RoundData,
  GraphSnapshot,
} from "./types";

// ── Helpers ─────────────────────────────────────────────────────────

async function fetchJson<T>(path: string): Promise<T> {
  const res = await window.fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function fetchText(path: string): Promise<string> {
  const res = await window.fetch(path);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${path}: ${res.status} ${res.statusText}`);
  }
  return res.text();
}

function scenarioPath(id: string, file: string): string {
  return `/data/scenario_${id}/${file}`;
}

function apiPath(id: string, file: string): string {
  return `/api/scenarios/${id}/${file}`;
}

/**
 * Try API first, fall back to static path.
 */
async function fetchWithFallback<T>(
  id: string,
  file: string,
  parser: "json"
): Promise<T>;
async function fetchWithFallback(
  id: string,
  file: string,
  parser: "text"
): Promise<string>;
async function fetchWithFallback<T>(
  id: string,
  file: string,
  parser: "json" | "text"
): Promise<T | string> {
  try {
    if (parser === "text") {
      return await fetchText(apiPath(id, file));
    }
    return await fetchJson<T>(apiPath(id, file));
  } catch {
    // API not available, fall back to static
    if (parser === "text") {
      return await fetchText(scenarioPath(id, file));
    }
    return await fetchJson<T>(scenarioPath(id, file));
  }
}

// ── Public API ──────────────────────────────────────────────────────

/**
 * Load the list of all available scenarios.
 */
export async function loadScenarios(): Promise<ScenarioInfo[]> {
  return fetchJson<ScenarioInfo[]>("/data/scenarios.json");
}

/**
 * Load all editorial + replay-meta data for a single scenario.
 */
export async function loadScenarioData(id: string): Promise<{
  metadata: Metadata;
  agents: AgentData[];
  polarization: PolarizationPoint[];
  coalitions: CoalitionRound[];
  posts: TopPost[];
  replayMeta: ReplayMeta;
}> {
  const [metadata, agents, polarization, coalitions, posts, replayMeta] =
    await Promise.all([
      fetchWithFallback<Metadata>(id, "metadata.json", "json"),
      fetchWithFallback<AgentData[]>(id, "agents.json", "json"),
      fetchWithFallback<PolarizationPoint[]>(id, "polarization.json", "json"),
      fetchWithFallback<CoalitionRound[]>(id, "coalitions.json", "json"),
      fetchWithFallback<TopPost[]>(id, "top_posts.json", "json"),
      fetchWithFallback<ReplayMeta>(id, "replay_meta.json", "json"),
    ]);

  return { metadata, agents, polarization, coalitions, posts, replayMeta };
}

/**
 * Load a single round's replay data for a scenario.
 */
export async function loadRoundData(
  id: string,
  round: number
): Promise<RoundData> {
  return fetchWithFallback<RoundData>(id, `replay_round_${round}.json`, "json");
}

/**
 * Load the evolving knowledge graph for a scenario.
 */
export async function loadEvolvingGraph(
  id: string
): Promise<GraphSnapshot[]> {
  return fetchWithFallback<GraphSnapshot[]>(id, "evolving_graph.json", "json");
}

/**
 * Load the scenario report as markdown text.
 */
export async function loadReport(id: string): Promise<string> {
  return fetchWithFallback(id, "report.md", "text");
}

/**
 * Load all replay round data for a scenario in parallel.
 */
export async function loadAllReplayRounds(
  id: string,
  numRounds: number
): Promise<RoundData[]> {
  const promises = Array.from({ length: numRounds }, (_, i) =>
    loadRoundData(id, i + 1)
  );
  return Promise.all(promises);
}
