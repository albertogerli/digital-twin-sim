import { describe, it, expect } from "vitest";
import {
  positionColor,
  formatNumber,
  sentimentColor,
  domainColor,
  cn,
} from "@/lib/utils";

describe("positionColor", () => {
  it("returns green for strong positive", () => {
    expect(positionColor(0.5)).toBe("#22c55e");
  });

  it("returns light green for mild positive", () => {
    expect(positionColor(0.2)).toBe("#86efac");
  });

  it("returns gray for neutral", () => {
    expect(positionColor(0.0)).toBe("#94a3b8");
  });

  it("returns light red for mild negative", () => {
    expect(positionColor(-0.2)).toBe("#fca5a5");
  });

  it("returns red for strong negative", () => {
    expect(positionColor(-0.5)).toBe("#ef4444");
  });
});

describe("formatNumber", () => {
  it("formats millions", () => {
    expect(formatNumber(1500000)).toBe("1.5M");
  });

  it("formats thousands", () => {
    expect(formatNumber(2500)).toBe("2.5K");
  });

  it("returns raw number for small values", () => {
    expect(formatNumber(42)).toBe("42");
  });
});

describe("sentimentColor", () => {
  it("returns green for positive", () => {
    expect(sentimentColor("positive")).toBe("#22c55e");
  });

  it("returns red for negative", () => {
    expect(sentimentColor("negative")).toBe("#ef4444");
  });

  it("returns fallback for unknown", () => {
    expect(sentimentColor("unknown")).toBe("#64748b");
  });
});

describe("domainColor", () => {
  it("returns blue for financial", () => {
    expect(domainColor("financial")).toBe("#3b82f6");
  });

  it("returns fallback for unknown", () => {
    expect(domainColor("unknown_domain")).toBe("#64748b");
  });
});

describe("cn", () => {
  it("joins class names", () => {
    expect(cn("a", "b", "c")).toBe("a b c");
  });

  it("filters falsy values", () => {
    expect(cn("a", false, undefined, "b")).toBe("a b");
  });
});
