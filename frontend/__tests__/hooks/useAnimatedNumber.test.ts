import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";

// Mock requestAnimationFrame for test environment
vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
  return setTimeout(() => cb(performance.now()), 16) as unknown as number;
});
vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));

describe("useAnimatedNumber", () => {
  it("module exists and exports hook", async () => {
    const mod = await import("@/lib/replay/useAnimatedNumber");
    expect(mod.useAnimatedNumber).toBeDefined();
    expect(typeof mod.useAnimatedNumber).toBe("function");
  });
});
