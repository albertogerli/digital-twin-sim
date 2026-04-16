import { describe, it, expect } from "vitest";
import {
  ScenarioInfoSchema,
  AgentSchema,
  SentimentSchema,
  RoundDataSchema,
  SSEProgressEventSchema,
  SimulationStatusSchema,
} from "@/lib/schemas";

describe("ScenarioInfoSchema", () => {
  it("validates a complete valid object", () => {
    const data = {
      id: "scenario_1",
      name: "Test Scenario",
      domain: "financial",
      description: "A test scenario",
      num_rounds: 9,
    };
    const result = ScenarioInfoSchema.parse(data);
    expect(result.id).toBe("scenario_1");
    expect(result.name).toBe("Test Scenario");
    expect(result.num_rounds).toBe(9);
  });

  it("applies defaults for optional fields", () => {
    const data = { id: "s1", name: "Test", domain: "political" };
    const result = ScenarioInfoSchema.parse(data);
    expect(result.description).toBe("");
    expect(result.num_rounds).toBe(0);
  });

  it("rejects missing required field 'id'", () => {
    const data = { name: "Test", domain: "financial" };
    expect(() => ScenarioInfoSchema.parse(data)).toThrow();
  });

  it("rejects missing required field 'name'", () => {
    const data = { id: "s1", domain: "financial" };
    expect(() => ScenarioInfoSchema.parse(data)).toThrow();
  });

  it("rejects missing required field 'domain'", () => {
    const data = { id: "s1", name: "Test" };
    expect(() => ScenarioInfoSchema.parse(data)).toThrow();
  });

  it("rejects invalid type for num_rounds", () => {
    const data = { id: "s1", name: "Test", domain: "x", num_rounds: "nine" };
    expect(() => ScenarioInfoSchema.parse(data)).toThrow();
  });

  it("rejects non-integer num_rounds", () => {
    const data = { id: "s1", name: "Test", domain: "x", num_rounds: 3.5 };
    expect(() => ScenarioInfoSchema.parse(data)).toThrow();
  });
});

describe("AgentSchema", () => {
  it("validates a complete agent", () => {
    const data = {
      id: "agent_1",
      name: "Agent One",
      role: "elite",
      position: 0.7,
      influence: 0.9,
    };
    const result = AgentSchema.parse(data);
    expect(result.id).toBe("agent_1");
    expect(result.influence).toBe(0.9);
  });

  it("applies defaults for optional fields", () => {
    const data = { id: "a1", name: "Agent", position: 0.5 };
    const result = AgentSchema.parse(data);
    expect(result.role).toBe("");
    expect(result.influence).toBe(0.5);
  });

  it("rejects missing position (required)", () => {
    const data = { id: "a1", name: "Agent" };
    expect(() => AgentSchema.parse(data)).toThrow();
  });

  it("rejects invalid type for position", () => {
    const data = { id: "a1", name: "Agent", position: "high" };
    expect(() => AgentSchema.parse(data)).toThrow();
  });
});

describe("SentimentSchema", () => {
  it("validates correct sentiment distribution", () => {
    const data = { positive: 0.4, neutral: 0.3, negative: 0.3 };
    const result = SentimentSchema.parse(data);
    expect(result.positive).toBe(0.4);
  });

  it("rejects missing fields", () => {
    expect(() => SentimentSchema.parse({ positive: 0.5 })).toThrow();
    expect(() => SentimentSchema.parse({ positive: 0.5, neutral: 0.3 })).toThrow();
  });

  it("rejects non-numeric values", () => {
    expect(() =>
      SentimentSchema.parse({ positive: "high", neutral: 0.3, negative: 0.3 }),
    ).toThrow();
  });
});

describe("RoundDataSchema", () => {
  it("validates minimal round data", () => {
    const data = { round: 1 };
    const result = RoundDataSchema.parse(data);
    expect(result.round).toBe(1);
    expect(result.polarization).toBe(0);
    expect(result.posts).toEqual([]);
  });

  it("validates round data with event", () => {
    const data = {
      round: 3,
      event: { event: "Market crash", shock_magnitude: 0.8, shock_direction: -1 },
      polarization: 0.6,
    };
    const result = RoundDataSchema.parse(data);
    expect(result.event!.event).toBe("Market crash");
    expect(result.event!.shock_magnitude).toBe(0.8);
  });

  it("applies defaults within nested event object", () => {
    const data = {
      round: 2,
      event: { event: "Minor news" },
    };
    const result = RoundDataSchema.parse(data);
    expect(result.event!.shock_magnitude).toBe(0);
    expect(result.event!.shock_direction).toBe(0);
  });

  it("rejects non-integer round", () => {
    expect(() => RoundDataSchema.parse({ round: 1.5 })).toThrow();
  });

  it("rejects missing round field", () => {
    expect(() => RoundDataSchema.parse({})).toThrow();
  });
});

describe("SSEProgressEventSchema", () => {
  it("validates a complete SSE event", () => {
    const data = {
      type: "progress",
      message: "Running round 3",
      round: 3,
      phase: "simulation",
      data: { polarization: 0.5 },
    };
    const result = SSEProgressEventSchema.parse(data);
    expect(result.type).toBe("progress");
    expect(result.round).toBe(3);
  });

  it("applies defaults for optional fields", () => {
    const data = { type: "start" };
    const result = SSEProgressEventSchema.parse(data);
    expect(result.message).toBe("");
    expect(result.data).toEqual({});
  });

  it("allows null round and phase", () => {
    const data = { type: "info", round: null, phase: null };
    const result = SSEProgressEventSchema.parse(data);
    expect(result.round).toBeNull();
    expect(result.phase).toBeNull();
  });

  it("rejects missing type", () => {
    expect(() => SSEProgressEventSchema.parse({ message: "hello" })).toThrow();
  });
});

describe("SimulationStatusSchema", () => {
  const validStatus = {
    id: "sim_123",
    status: "running" as const,
    brief: "Test simulation",
    created_at: "2026-01-01T00:00:00Z",
  };

  it("validates a complete status", () => {
    const result = SimulationStatusSchema.parse(validStatus);
    expect(result.id).toBe("sim_123");
    expect(result.status).toBe("running");
    expect(result.current_round).toBe(0);
    expect(result.total_rounds).toBe(0);
    expect(result.cost).toBe(0);
    expect(result.agents_count).toBe(0);
  });

  it("validates all allowed status values", () => {
    const statuses = [
      "queued", "analyzing", "configuring", "running",
      "awaiting_player", "exporting", "completed", "failed", "cancelled",
    ];
    for (const s of statuses) {
      const result = SimulationStatusSchema.parse({ ...validStatus, status: s });
      expect(result.status).toBe(s);
    }
  });

  it("rejects invalid status value", () => {
    expect(() =>
      SimulationStatusSchema.parse({ ...validStatus, status: "unknown" }),
    ).toThrow();
  });

  it("rejects missing required fields", () => {
    expect(() => SimulationStatusSchema.parse({ id: "s1" })).toThrow();
    expect(() =>
      SimulationStatusSchema.parse({ id: "s1", status: "running" }),
    ).toThrow();
  });

  it("allows nullable optional fields", () => {
    const data = {
      ...validStatus,
      scenario_name: null,
      scenario_id: null,
      domain: null,
      completed_at: null,
      error: null,
    };
    const result = SimulationStatusSchema.parse(data);
    expect(result.scenario_name).toBeNull();
    expect(result.error).toBeNull();
  });

  it("applies numeric defaults", () => {
    const result = SimulationStatusSchema.parse(validStatus);
    expect(result.current_round).toBe(0);
    expect(result.total_rounds).toBe(0);
    expect(result.cost).toBe(0);
    expect(result.agents_count).toBe(0);
  });
});
