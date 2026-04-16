import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

// Mock next/navigation
const mockUsePathname = vi.fn(() => "/");
vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
  useRouter: vi.fn(() => ({ push: vi.fn() })),
}));

// Mock SideNav and TopBar to keep tests focused on AppShell logic
vi.mock("@/components/layout/SideNav", () => ({
  default: () => <div data-testid="side-nav">SideNav</div>,
}));

vi.mock("@/components/layout/TopBar", () => ({
  default: () => <div data-testid="top-bar">TopBar</div>,
}));

import AppShell from "@/components/layout/AppShell";

beforeEach(() => {
  mockUsePathname.mockReturnValue("/");
});

describe("AppShell", () => {
  it("renders children on a normal route", () => {
    mockUsePathname.mockReturnValue("/");
    render(
      <AppShell>
        <div>Page content</div>
      </AppShell>,
    );
    expect(screen.getByText("Page content")).toBeInTheDocument();
  });

  it("renders SideNav and TopBar on non-fullscreen routes", () => {
    mockUsePathname.mockReturnValue("/");
    render(
      <AppShell>
        <div>Home</div>
      </AppShell>,
    );
    expect(screen.getByTestId("side-nav")).toBeInTheDocument();
    expect(screen.getByTestId("top-bar")).toBeInTheDocument();
  });

  it("hides SideNav and TopBar on /replay route", () => {
    mockUsePathname.mockReturnValue("/replay");
    render(
      <AppShell>
        <div>Replay content</div>
      </AppShell>,
    );
    expect(screen.queryByTestId("side-nav")).not.toBeInTheDocument();
    expect(screen.queryByTestId("top-bar")).not.toBeInTheDocument();
    expect(screen.getByText("Replay content")).toBeInTheDocument();
  });

  it("hides SideNav and TopBar on /wargame route", () => {
    mockUsePathname.mockReturnValue("/wargame");
    render(
      <AppShell>
        <div>Wargame content</div>
      </AppShell>,
    );
    expect(screen.queryByTestId("side-nav")).not.toBeInTheDocument();
    expect(screen.queryByTestId("top-bar")).not.toBeInTheDocument();
  });

  it("hides SideNav and TopBar on /backtest route", () => {
    mockUsePathname.mockReturnValue("/backtest");
    render(
      <AppShell>
        <div>Backtest content</div>
      </AppShell>,
    );
    expect(screen.queryByTestId("side-nav")).not.toBeInTheDocument();
    expect(screen.queryByTestId("top-bar")).not.toBeInTheDocument();
  });

  it("hides SideNav and TopBar on /scenario/ routes", () => {
    mockUsePathname.mockReturnValue("/scenario/abc-123");
    render(
      <AppShell>
        <div>Scenario detail</div>
      </AppShell>,
    );
    expect(screen.queryByTestId("side-nav")).not.toBeInTheDocument();
    expect(screen.queryByTestId("top-bar")).not.toBeInTheDocument();
  });

  it("shows SideNav on /new route (not fullscreen)", () => {
    mockUsePathname.mockReturnValue("/new");
    render(
      <AppShell>
        <div>New sim</div>
      </AppShell>,
    );
    expect(screen.getByTestId("side-nav")).toBeInTheDocument();
  });

  it("shows SideNav on /paper route (not fullscreen)", () => {
    mockUsePathname.mockReturnValue("/paper");
    render(
      <AppShell>
        <div>Paper</div>
      </AppShell>,
    );
    expect(screen.getByTestId("side-nav")).toBeInTheDocument();
  });

  it("renders children even on fullscreen routes", () => {
    mockUsePathname.mockReturnValue("/replay");
    render(
      <AppShell>
        <div>Fullscreen child</div>
      </AppShell>,
    );
    expect(screen.getByText("Fullscreen child")).toBeInTheDocument();
  });
});
