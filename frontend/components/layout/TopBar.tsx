"use client";

import { useRouter, usePathname } from "next/navigation";
import { useState } from "react";

/* ───────────────────────────────────────────────────────────
   Quiet Intelligence Terminal — 44px header.
   Pattern: <eyebrow> / <page title> / <breadcrumb>  ·  <right slot>
   Right slot: live status dot + notifications + user chip.
   ─────────────────────────────────────────────────────────── */

const PAGE_TITLES: Record<string, string> = {
  "/":         "Dashboard",
  "/new":      "New simulation",
  "/wargame":  "Wargame",
  "/backtest": "Backtest",
  "/paper":    "Paper",
  "/settings": "Settings",
};

export default function TopBar() {
  const pathname = usePathname();
  const router = useRouter();
  const [signingOut, setSigningOut] = useState(false);

  async function handleLogout() {
    if (signingOut) return;
    setSigningOut(true);
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {
      /* even if request fails, clear-local-state-and-redirect */
    }
    router.replace("/login");
  }

  let title = PAGE_TITLES[pathname] || "DigitalTwinSim";
  let breadcrumb: string | null = null;

  if (pathname.startsWith("/scenario/")) {
    title = "Scenario";
    breadcrumb = pathname.replace("/scenario/", "");
  } else if (pathname.startsWith("/sim/")) {
    title = "Monitor";
    breadcrumb = pathname.replace("/sim/", "");
  }

  return (
    <header className="w-full sticky top-0 z-30 h-11 bg-ki-surface-raised/95 backdrop-blur border-b border-ki-border flex items-center px-4 gap-3 flex-shrink-0">
      {/* Left: eyebrow + title + breadcrumb */}
      <div className="flex items-baseline gap-2 min-w-0 flex-1 overflow-hidden">
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-faint shrink-0">
          DigitalTwinSim
        </span>
        <span className="text-ki-on-surface-faint shrink-0">/</span>
        <span className="text-[15px] font-medium text-ki-on-surface tracking-[-0.005em] shrink-0">
          {title}
        </span>
        {breadcrumb && (
          <>
            <span className="text-ki-on-surface-faint shrink-0">/</span>
            <span className="font-data text-[12px] text-ki-on-surface-secondary truncate">
              {breadcrumb}
            </span>
          </>
        )}
      </div>

      {/* Right: status + actions + user */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
          <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">
            Online
          </span>
        </div>
        <div className="w-px h-4 bg-ki-border" />
        <button
          aria-label="Notifications"
          className="w-7 h-7 grid place-items-center rounded-md text-ki-on-surface-muted hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors"
        >
          <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'wght' 400" }}>
            notifications
          </span>
        </button>
        <div className="flex items-center gap-2 pl-2 border-l border-ki-border">
          <div className="w-[22px] h-[22px] rounded-full bg-ki-on-surface text-ki-surface grid place-items-center font-data text-[10px] font-semibold">
            AG
          </div>
          <span className="text-[12px] text-ki-on-surface-secondary whitespace-nowrap">A. Gerli</span>
          <button
            onClick={handleLogout}
            disabled={signingOut}
            aria-label="Sign out"
            title="Sign out"
            className="ml-1 w-7 h-7 grid place-items-center rounded-md text-ki-on-surface-muted hover:bg-ki-surface-hover hover:text-ki-on-surface disabled:opacity-50 transition-colors"
          >
            <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'wght' 400" }}>
              {signingOut ? "hourglass_empty" : "logout"}
            </span>
          </button>
        </div>
      </div>
    </header>
  );
}
