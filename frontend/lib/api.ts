/**
 * Typed API client with Zod validation.
 */

import { z } from "zod";
import { ScenarioInfoSchema, SimulationStatusSchema } from "./schemas";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function fetchJSON<T>(
  url: string,
  schema: z.ZodType<T>,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText}`);
  }

  const data = await res.json();
  const parsed = schema.safeParse(data);

  if (!parsed.success) {
    console.error("[API] Validation error:", parsed.error.issues);
    // Return data anyway but log the validation error
    return data as T;
  }

  return parsed.data;
}

export async function listScenarios() {
  return fetchJSON("/api/scenarios", z.array(ScenarioInfoSchema));
}

export async function listSimulations() {
  return fetchJSON("/api/simulations", z.array(SimulationStatusSchema));
}

export async function getSimulationStatus(simId: string) {
  return fetchJSON(`/api/simulations/${simId}`, SimulationStatusSchema);
}

export async function createSimulation(brief: string, options?: Record<string, unknown>) {
  return fetchJSON(
    "/api/simulations",
    z.object({ id: z.string(), status: z.string() }),
    {
      method: "POST",
      body: JSON.stringify({ brief, ...options }),
    },
  );
}
