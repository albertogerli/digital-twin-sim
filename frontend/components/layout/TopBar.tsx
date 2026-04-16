"use client";

import { usePathname } from "next/navigation";

const PAGE_TITLES: Record<string, string> = {
  "/": "DASHBOARD",
  "/new": "NEW SIMULATION",
  "/wargame": "WARGAME",
  "/backtest": "BACKTEST",
  "/paper": "PAPER",
  "/settings": "SETTINGS",
};

export default function TopBar() {
  const pathname = usePathname();
  let title = PAGE_TITLES[pathname] || "DTS";
  if (pathname.startsWith("/scenario/")) title = "SCENARIO";
  if (pathname.startsWith("/sim/")) title = "MONITOR";

  return (
    <header className="w-full sticky top-0 z-30 bg-ki-surface-raised/90 backdrop-blur-md border-b border-ki-border flex justify-between items-center px-4 h-9">
      <span className="text-[10px] font-bold tracking-[0.1em] text-ki-on-surface-muted font-data">
        {title}
      </span>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
          <span className="font-data text-2xs text-ki-on-surface-muted">
            ONLINE
          </span>
        </div>
        <div className="w-px h-4 bg-ki-border" />
        <span className="material-symbols-outlined text-[16px] text-ki-on-surface-muted cursor-pointer hover:text-ki-on-surface">
          notifications
        </span>
      </div>
    </header>
  );
}
