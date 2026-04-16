/**
 * Zod schemas for API response validation.
 * Validates data at runtime to prevent crashes from malformed API responses.
 */

import { z } from "zod";

export const ScenarioInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  domain: z.string(),
  description: z.string().optional().default(""),
  num_rounds: z.number().int().optional().default(0),
});

export const AgentSchema = z.object({
  id: z.string(),
  name: z.string(),
  role: z.string().optional().default(""),
  position: z.number(),
  influence: z.number().optional().default(0.5),
});

export const SentimentSchema = z.object({
  positive: z.number(),
  neutral: z.number(),
  negative: z.number(),
});

export const RoundDataSchema = z.object({
  round: z.number().int(),
  event: z
    .object({
      event: z.string(),
      shock_magnitude: z.number().optional().default(0),
      shock_direction: z.number().optional().default(0),
    })
    .optional(),
  polarization: z.number().optional().default(0),
  sentiment: SentimentSchema.optional(),
  posts: z.array(z.any()).optional().default([]),
  network: z.any().optional(),
  coalitions: z.any().optional(),
});

export const SSEProgressEventSchema = z.object({
  type: z.string(),
  message: z.string().optional().default(""),
  round: z.number().nullable().optional(),
  phase: z.string().nullable().optional(),
  data: z.record(z.any()).optional().default({}),
});

export const SimulationStatusSchema = z.object({
  id: z.string(),
  status: z.enum([
    "queued",
    "analyzing",
    "configuring",
    "running",
    "awaiting_player",
    "exporting",
    "completed",
    "failed",
    "cancelled",
  ]),
  brief: z.string(),
  scenario_name: z.string().nullable().optional(),
  scenario_id: z.string().nullable().optional(),
  domain: z.string().nullable().optional(),
  current_round: z.number().optional().default(0),
  total_rounds: z.number().optional().default(0),
  cost: z.number().optional().default(0),
  created_at: z.string(),
  completed_at: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  agents_count: z.number().optional().default(0),
});

export type ScenarioInfo = z.infer<typeof ScenarioInfoSchema>;
export type SimulationStatus = z.infer<typeof SimulationStatusSchema>;
