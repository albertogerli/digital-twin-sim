"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "HOME", icon: "home" },
  { href: "/new", label: "NEW SIM", icon: "add_chart" },
  { href: "/wargame", label: "WARGAME", icon: "swords" },
  { href: "/backtest", label: "BACKTEST", icon: "history" },
  { href: "/paper", label: "PAPER", icon: "description" },
];

const BOTTOM_ITEMS = [
  { href: "/settings", label: "SETTINGS", icon: "settings" },
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
        className={`flex items-center gap-2.5 px-3 py-1.5 text-[10px] font-bold tracking-[0.08em] whitespace-nowrap transition-colors duration-100 ${
          active
            ? "text-ki-primary bg-ki-primary/[0.07]"
            : "text-ki-on-surface-muted hover:text-ki-on-surface hover:bg-ki-surface-hover"
        }`}
      >
        <span
          className="material-symbols-outlined text-[18px] shrink-0"
          style={active ? { fontVariationSettings: "'FILL' 1, 'wght' 500" } : { fontVariationSettings: "'wght' 300" }}
        >
          {item.icon}
        </span>
        {item.label}
      </Link>
    );
  };

  return (
    <aside className="h-screen w-48 fixed left-0 top-0 bg-ki-surface-raised border-r border-ki-border flex flex-col py-3 z-40">
      {/* Logo */}
      <div className="px-3 mb-4">
        <div className="text-sm font-extrabold tracking-tight text-ki-on-surface font-headline">
          DTS
        </div>
        <div className="text-2xs font-data text-ki-on-surface-muted tracking-wider">
          SIM CONSOLE
        </div>
      </div>

      <nav className="flex-1 flex flex-col gap-px">
        {NAV_ITEMS.map(navLink)}
      </nav>

      <div className="border-t border-ki-border pt-2 flex flex-col gap-px">
        {BOTTOM_ITEMS.map(navLink)}
      </div>
    </aside>
  );
}
