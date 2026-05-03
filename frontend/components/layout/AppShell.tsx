"use client";

import { usePathname } from "next/navigation";
import SideNav from "./SideNav";
import TopBar from "./TopBar";

const FULLSCREEN_ROUTES = ["/scenario/", "/replay", "/wargame", "/backtest", "/branches", "/login"];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isFullscreen = FULLSCREEN_ROUTES.some((r) => pathname.includes(r));

  if (isFullscreen) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen bg-ki-surface text-ki-on-surface">
      <div className="hidden md:block">
        <SideNav />
      </div>
      <main className="flex-1 md:ml-14 flex flex-col min-h-screen">
        <TopBar />
        <div className="flex-1">{children}</div>
      </main>
    </div>
  );
}
