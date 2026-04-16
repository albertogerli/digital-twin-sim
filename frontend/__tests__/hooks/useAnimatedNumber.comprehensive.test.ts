import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useAnimatedNumber } from "@/lib/replay/useAnimatedNumber";

// Mock requestAnimationFrame / cancelAnimationFrame with controllable timing
let rafCallbacks: Map<number, FrameRequestCallback>;
let rafId: number;
let mockNow: number;

beforeEach(() => {
  rafCallbacks = new Map();
  rafId = 0;
  mockNow = 0;

  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    const id = ++rafId;
    rafCallbacks.set(id, cb);
    return id;
  });

  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    rafCallbacks.delete(id);
  });

  vi.spyOn(performance, "now").mockImplementation(() => mockNow);
});

afterEach(() => {
  vi.restoreAllMocks();
});

/** Flush all pending rAF callbacks at the current mockNow time */
function flushRAF() {
  const pending = new Map(rafCallbacks);
  rafCallbacks.clear();
  for (const [, cb] of pending) {
    cb(mockNow);
  }
}

/** Advance time and flush rAF callbacks repeatedly to simulate animation */
function advanceTime(ms: number, steps = 10) {
  const stepSize = ms / steps;
  for (let i = 0; i < steps; i++) {
    mockNow += stepSize;
    flushRAF();
  }
}

describe("useAnimatedNumber", () => {
  it("returns the initial value immediately", () => {
    const { result } = renderHook(() => useAnimatedNumber(100));
    expect(result.current).toBe(100);
  });

  it("returns the target directly when delta is less than 0.5", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target),
      { initialProps: { target: 100 } },
    );

    // Change by less than 0.5 — should snap immediately
    act(() => {
      rerender({ target: 100.3 });
    });

    // Should snap to target without animation
    expect(result.current).toBe(100.3);
  });

  it("begins animating toward a new target value", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 100 });
    });

    // Before any animation tick, value should still be at start (0)
    // or the first tick might have fired — just check it's not at 100 yet
    // Actually the rAF hasn't been flushed yet
    expect(result.current).toBe(0);

    // Advance partway through animation
    act(() => {
      mockNow = 400; // half of 800ms duration
      flushRAF();
    });

    // Should be somewhere between 0 and 100 (ease-out cubic at 50%)
    expect(result.current).toBeGreaterThan(0);
    expect(result.current).toBeLessThan(100);
  });

  it("reaches the target value after full duration", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 200 });
    });

    // Advance past the full duration
    act(() => {
      advanceTime(900);
    });

    expect(result.current).toBe(200);
  });

  it("uses ease-out cubic easing (faster start, slower end)", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 1000),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 1000 });
    });

    // At 50% time, ease-out cubic = 1 - (1 - 0.5)^3 = 0.875
    act(() => {
      mockNow = 500;
      flushRAF();
    });

    const valueAtHalf = result.current;
    // Should be well past 50% of the target due to ease-out
    expect(valueAtHalf).toBeGreaterThan(500);
  });

  it("handles rapid value changes by animating from current position", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 0 } },
    );

    // Start animating to 100
    act(() => {
      rerender({ target: 100 });
    });

    // Advance partway
    act(() => {
      mockNow = 400;
      flushRAF();
    });

    const midValue = result.current;
    expect(midValue).toBeGreaterThan(0);
    expect(midValue).toBeLessThan(100);

    // Now rapidly change target to 200 — should animate from midValue, not from 0
    act(() => {
      rerender({ target: 200 });
    });

    // Advance a tiny bit
    act(() => {
      mockNow = 416; // one frame
      flushRAF();
    });

    // Value should be near midValue (just started animating toward 200)
    // It should not have jumped back to 0
    expect(result.current).toBeGreaterThanOrEqual(Math.floor(midValue) - 1);
  });

  it("cleans up animation frame on unmount", () => {
    const { result, rerender, unmount } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 500 });
    });

    // There should be a pending rAF callback
    expect(rafCallbacks.size).toBeGreaterThan(0);

    // Unmount should cancel pending frames
    act(() => {
      unmount();
    });

    expect(rafCallbacks.size).toBe(0);
  });

  it("cleans up previous animation when target changes", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 100 });
    });

    // One rAF should be pending
    const countBefore = rafCallbacks.size;

    // Change target — cleanup from previous effect should cancel old rAF
    act(() => {
      rerender({ target: 200 });
    });

    // After re-render, there should still be at most one pending rAF
    // (the old one canceled, new one scheduled)
    expect(rafCallbacks.size).toBeLessThanOrEqual(countBefore);
  });

  it("handles negative target values", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 50 } },
    );

    act(() => {
      rerender({ target: -50 });
    });

    act(() => {
      advanceTime(900);
    });

    expect(result.current).toBe(-50);
  });

  it("respects custom duration parameter", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 200),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 100 });
    });

    // At 200ms (full duration), should be at target
    act(() => {
      advanceTime(250);
    });

    expect(result.current).toBe(100);
  });

  it("rounds values to integers", () => {
    const { result, rerender } = renderHook(
      ({ target }) => useAnimatedNumber(target, 800),
      { initialProps: { target: 0 } },
    );

    act(() => {
      rerender({ target: 100 });
    });

    // Advance partway — value should be an integer (Math.round in the hook)
    act(() => {
      mockNow = 300;
      flushRAF();
    });

    expect(Number.isInteger(result.current)).toBe(true);
  });
});
