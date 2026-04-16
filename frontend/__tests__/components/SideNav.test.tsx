import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

// Mock next/navigation
const mockUsePathname = vi.fn(() => "/");
vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
  useRouter: vi.fn(() => ({ push: vi.fn() })),
}));

// Mock next/link to render a plain anchor
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

describe("SideNav", () => {
  it("renders all main nav links", () => {
    render(<SideNav />);

    expect(screen.getByText("HOME")).toBeInTheDocument();
    expect(screen.getByText("NEW SIM")).toBeInTheDocument();
    expect(screen.getByText("WARGAME")).toBeInTheDocument();
    expect(screen.getByText("BACKTEST")).toBeInTheDocument();
    expect(screen.getByText("PAPER")).toBeInTheDocument();
  });

  it("renders the settings link", () => {
    render(<SideNav />);
    expect(screen.getByText("SETTINGS")).toBeInTheDocument();
  });

  it("renders the logo text", () => {
    render(<SideNav />);
    expect(screen.getByText("DTS")).toBeInTheDocument();
    expect(screen.getByText("SIM CONSOLE")).toBeInTheDocument();
  });

  it("nav links have correct href attributes", () => {
    render(<SideNav />);

    const homeLink = screen.getByText("HOME").closest("a");
    expect(homeLink).toHaveAttribute("href", "/");

    const newSimLink = screen.getByText("NEW SIM").closest("a");
    expect(newSimLink).toHaveAttribute("href", "/new");

    const wargameLink = screen.getByText("WARGAME").closest("a");
    expect(wargameLink).toHaveAttribute("href", "/wargame");

    const backtestLink = screen.getByText("BACKTEST").closest("a");
    expect(backtestLink).toHaveAttribute("href", "/backtest");

    const paperLink = screen.getByText("PAPER").closest("a");
    expect(paperLink).toHaveAttribute("href", "/paper");

    const settingsLink = screen.getByText("SETTINGS").closest("a");
    expect(settingsLink).toHaveAttribute("href", "/settings");
  });

  it("highlights the active HOME link when pathname is /", () => {
    mockUsePathname.mockReturnValue("/");
    render(<SideNav />);

    const homeLink = screen.getByText("HOME").closest("a");
    expect(homeLink?.className).toContain("text-ki-primary");
  });

  it("does not highlight HOME when on another route", () => {
    mockUsePathname.mockReturnValue("/new");
    render(<SideNav />);

    const homeLink = screen.getByText("HOME").closest("a");
    expect(homeLink?.className).not.toContain("text-ki-primary");
  });

  it("highlights NEW SIM when pathname starts with /new", () => {
    mockUsePathname.mockReturnValue("/new");
    render(<SideNav />);

    const newSimLink = screen.getByText("NEW SIM").closest("a");
    expect(newSimLink?.className).toContain("text-ki-primary");
  });

  it("highlights BACKTEST when pathname starts with /backtest", () => {
    mockUsePathname.mockReturnValue("/backtest/results");
    render(<SideNav />);

    const backtestLink = screen.getByText("BACKTEST").closest("a");
    expect(backtestLink?.className).toContain("text-ki-primary");
  });

  it("only one link is active at a time", () => {
    mockUsePathname.mockReturnValue("/wargame");
    render(<SideNav />);

    const allLinks = screen.getAllByRole("link");
    const activeLinks = allLinks.filter((link) =>
      link.className.includes("text-ki-primary"),
    );
    expect(activeLinks.length).toBe(1);
    expect(activeLinks[0].textContent).toContain("WARGAME");
  });

  it("renders material icons for each nav item", () => {
    render(<SideNav />);

    // Each nav item has a span with the icon name
    expect(screen.getByText("home")).toBeInTheDocument();
    expect(screen.getByText("add_chart")).toBeInTheDocument();
    expect(screen.getByText("swords")).toBeInTheDocument();
    expect(screen.getByText("history")).toBeInTheDocument();
    expect(screen.getByText("description")).toBeInTheDocument();
    expect(screen.getByText("settings")).toBeInTheDocument();
  });
});
