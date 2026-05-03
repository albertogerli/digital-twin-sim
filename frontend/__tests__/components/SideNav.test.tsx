import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

// Mock next/navigation
const mockUsePathname = vi.fn(() => "/");
vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
  useRouter: vi.fn(() => ({ push: vi.fn() })),
}));

// Mock next/link to render a plain anchor that forwards aria-label/title
vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: { children: React.ReactNode; href: string; [key: string]: unknown }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

import SideNav from "@/components/layout/SideNav";

beforeEach(() => {
  mockUsePathname.mockReturnValue("/");
});

/* ───────────────────────────────────────────────────────────
   These tests target the icon-rail SideNav (Quiet Intelligence
   restyle): no visible labels, queried by aria-label / title.
   Active item gets `bg-ki-surface-raised` (no longer text-color).
   ─────────────────────────────────────────────────────────── */

describe("SideNav (icon rail)", () => {
  it("renders all main nav links by aria-label", () => {
    render(<SideNav />);
    expect(screen.getByLabelText("Dashboard")).toBeInTheDocument();
    expect(screen.getByLabelText("New simulation")).toBeInTheDocument();
    expect(screen.getByLabelText("Wargame")).toBeInTheDocument();
    expect(screen.getByLabelText("Backtest")).toBeInTheDocument();
    expect(screen.getByLabelText("Paper")).toBeInTheDocument();
  });

  it("renders the settings link", () => {
    render(<SideNav />);
    expect(screen.getByLabelText("Settings")).toBeInTheDocument();
  });

  it("renders the brand mark", () => {
    render(<SideNav />);
    expect(screen.getByText("DT")).toBeInTheDocument();
  });

  it("nav links have correct href attributes", () => {
    render(<SideNav />);
    expect(screen.getByLabelText("Dashboard")).toHaveAttribute("href", "/");
    expect(screen.getByLabelText("New simulation")).toHaveAttribute("href", "/new");
    expect(screen.getByLabelText("Wargame")).toHaveAttribute("href", "/wargame");
    expect(screen.getByLabelText("Backtest")).toHaveAttribute("href", "/backtest");
    expect(screen.getByLabelText("Paper")).toHaveAttribute("href", "/paper");
    expect(screen.getByLabelText("Settings")).toHaveAttribute("href", "/settings");
  });

  it("highlights the active Dashboard link when pathname is /", () => {
    mockUsePathname.mockReturnValue("/");
    render(<SideNav />);
    const dash = screen.getByLabelText("Dashboard");
    expect(dash.className).toContain("bg-ki-surface-raised");
  });

  it("does not highlight Dashboard when on another route", () => {
    mockUsePathname.mockReturnValue("/new");
    render(<SideNav />);
    const dash = screen.getByLabelText("Dashboard");
    expect(dash.className).not.toContain("bg-ki-surface-raised");
  });

  it("highlights New simulation when pathname starts with /new", () => {
    mockUsePathname.mockReturnValue("/new");
    render(<SideNav />);
    expect(screen.getByLabelText("New simulation").className).toContain("bg-ki-surface-raised");
  });

  it("highlights Backtest when pathname starts with /backtest", () => {
    mockUsePathname.mockReturnValue("/backtest/results");
    render(<SideNav />);
    expect(screen.getByLabelText("Backtest").className).toContain("bg-ki-surface-raised");
  });

  it("only one nav link is active at a time", () => {
    mockUsePathname.mockReturnValue("/wargame");
    render(<SideNav />);
    const allLinks = screen.getAllByRole("link");
    // The brand mark is also an <a>; filter to nav items via aria-label presence
    const navLinks = allLinks.filter((el) => el.getAttribute("aria-label"));
    const activeLinks = navLinks.filter((el) => el.className.includes("bg-ki-surface-raised"));
    expect(activeLinks.length).toBe(1);
    expect(activeLinks[0].getAttribute("aria-label")).toBe("Wargame");
  });

  it("renders material icons for each nav item", () => {
    render(<SideNav />);
    // New icon set after the Quiet Intelligence restyle
    expect(screen.getByText("grid_view")).toBeInTheDocument();
    expect(screen.getByText("add")).toBeInTheDocument();
    expect(screen.getByText("swords")).toBeInTheDocument();
    expect(screen.getByText("target")).toBeInTheDocument();
    expect(screen.getByText("description")).toBeInTheDocument();
    expect(screen.getByText("settings")).toBeInTheDocument();
  });
});
