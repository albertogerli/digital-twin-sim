"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/* ───────────────────────────────────────────────────────────
   Quiet Intelligence Terminal — 56px icon rail.
   Active item: subtle raised tile + 2px accent bar on the
   left edge. Tooltips via title attr (hover delay = native).
   ─────────────────────────────────────────────────────────── */

const NAV_ITEMS = [
  { href: "/",           label: "Dashboard",      icon: "grid_view" },
  { href: "/new",        label: "New simulation", icon: "add" },
  { href: "/wargame",    label: "Wargame",        icon: "swords" },
  { href: "/backtest",   label: "Backtest",       icon: "target" },
  { href: "/compliance", label: "Compliance",     icon: "verified_user" },
  { href: "/paper",      label: "Paper",          icon: "description" },
];

const BOTTOM_ITEMS = [
  { href: "/settings", label: "Settings", icon: "settings" },
];

export default function SideNav() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  const navLink = (item: typeof NAV_ITEMS[0]) => {
    const active = isActive(item.href);
    return (
      <Link
        key={item.href}
        href={item.href}
        title={item.label}
        aria-label={item.label}
        className={`relative w-9 h-9 grid place-items-center rounded-md transition-colors duration-100 ${
          active
            ? "bg-ki-surface-raised text-ki-on-surface shadow-[inset_0_0_0_1px_var(--line)]"
            : "text-ki-on-surface-muted hover:bg-ki-surface-hover hover:text-ki-on-surface"
        }`}
      >
        {/* Active rail — 2px indigo bar on the outer edge */}
        {active && (
          <span
            aria-hidden
            className="absolute -left-[10px] top-2 bottom-2 w-[2px] rounded-sm bg-ki-on-surface"
          />
        )}
        <span
          className="material-symbols-outlined text-[18px]"
          style={
            active
              ? { fontVariationSettings: "'FILL' 0, 'wght' 500" }
              : { fontVariationSettings: "'wght' 400" }
          }
        >
          {item.icon}
        </span>
      </Link>
    );
  };

  return (
    <aside className="h-screen w-14 fixed left-0 top-0 bg-ki-surface-sunken border-r border-ki-border flex flex-col items-center py-3 gap-1 z-40">
      {/* Brand mark */}
      <Link
        href="/"
        title="DigitalTwinSim"
        className="w-8 h-8 rounded-md bg-ki-on-surface text-ki-surface grid place-items-center font-data text-[12px] font-semibold tracking-tighter mb-3"
      >
        DT
      </Link>

      <nav className="flex-1 flex flex-col items-center gap-1">
        {NAV_ITEMS.map(navLink)}
      </nav>

      <div className="flex flex-col items-center gap-1">
        {BOTTOM_ITEMS.map(navLink)}
      </div>
    </aside>
  );
}
