"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface ScenarioInfo {
  id: string;
  name: string;
  domain: string;
  description: string;
  num_rounds: number;
}

interface SimStatus {
  id: string;
  status: string;
  brief: string;
  scenario_name?: string;
  scenario_id?: string;
  domain?: string;
  current_round: number;
  total_rounds: number;
  cost: number;
  created_at: string;
}

const ST: Record<string, { label: string; cls: string }> = {
  queued:      { label: "QUEUED",   cls: "text-ki-on-surface-muted" },
  analyzing:   { label: "ANLYZ",    cls: "text-ki-primary" },
  configuring: { label: "CONFIG",   cls: "text-ki-primary" },
  running:     { label: "LIVE",     cls: "text-ki-success font-black" },
  exporting:   { label: "EXPORT",   cls: "text-ki-warning" },
  completed:   { label: "DONE",     cls: "text-ki-success" },
  failed:      { label: "FAIL",     cls: "text-ki-error" },
  cancelled:   { label: "CANCEL",   cls: "text-ki-on-surface-muted" },
};

const DOMAIN_COLOR: Record<string, string> = {
  political: "text-domain-political",
  corporate: "text-domain-corporate",
  financial: "text-domain-financial",
  commercial: "text-domain-commercial",
  public_health: "text-domain-health",
  technology: "text-domain-technology",
};

export default function Home() {
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [simulations, setSimulations] = useState<SimStatus[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/api/scenarios").then((r) => r.ok ? r.json() : []).catch(() =>
        fetch("/data/scenarios.json").then((r) => r.json()).catch(() => [])
      ),
      fetch("/api/simulations").then((r) => r.ok ? r.json() : []).catch(() => []),
    ]).then(([sc, sims]) => {
      setScenarios(sc);
      setSimulations(sims);
      setLoading(false);
    });

    const interval = setInterval(() => {
      fetch("/api/simulations").then((r) => r.ok ? r.json() : []).then(setSimulations).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const runningSims = simulations.filter((s) =>
    ["queued", "analyzing", "configuring", "running", "exporting"].includes(s.status)
  );
  const recentSims = simulations
    .filter((s) => ["completed", "failed", "cancelled"].includes(s.status))
    .slice(0, 30);

  return (
    <div className="p-3 space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-baseline gap-3">
          <h2 className="text-base font-extrabold tracking-tight text-ki-on-surface font-headline">
            Simulations
          </h2>
          <span className="font-data text-2xs text-ki-on-surface-muted">
            {scenarios.length} scenarios &middot; {simulations.length} runs
          </span>
        </div>
        <Link
          href="/new"
          className="px-3 py-1 text-2xs font-bold bg-ki-primary text-ki-on-surface rounded-sm hover:bg-ki-primary-muted transition-colors"
        >
          + NEW SIM
        </Link>
      </div>

      {/* Quick nav strip */}
      <div className="flex gap-1">
        {[
          { href: "/wargame", label: "WARGAME", icon: "swords" },
          { href: "/backtest", label: "BACKTEST", icon: "history" },
          { href: "/paper", label: "PAPER", icon: "description" },
        ].map((n) => (
          <Link
            key={n.href}
            href={n.href}
            className="flex items-center gap-1.5 px-2.5 py-1 bg-ki-surface-raised border border-ki-border text-2xs font-bold text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors"
          >
            <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 300" }}>
              {n.icon}
            </span>
            {n.label}
          </Link>
        ))}
      </div>

      {/* Active simulations */}
      {runningSims.length > 0 && (
        <section>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-ki-success animate-pulse" />
            <span className="text-2xs font-bold text-ki-on-surface-muted tracking-[0.08em]">
              ACTIVE ({runningSims.length})
            </span>
          </div>
          <div className="border border-ki-border bg-ki-surface-raised">
            {runningSims.map((sim, i) => {
              const st = ST[sim.status] || ST.queued;
              const pct = sim.total_rounds > 0 ? (sim.current_round / sim.total_rounds) * 100 : 0;
              return (
                <Link
                  key={sim.id}
                  href={`/sim/${sim.id}`}
                  className={`flex items-center gap-2 px-2 py-1 hover:bg-ki-surface-hover transition-colors ${
                    i < runningSims.length - 1 ? "border-b border-ki-border" : ""
                  }`}
                >
                  <span className={`w-14 shrink-0 font-data text-2xs font-bold ${st.cls}`}>
                    {st.label}
                  </span>
                  <span className="font-body text-xs text-ki-on-surface truncate flex-1">
                    {sim.scenario_name || sim.brief.slice(0, 80)}
                  </span>
                  <span className="font-data text-2xs text-ki-on-surface-muted shrink-0">
                    {sim.total_rounds > 0 ? `${sim.current_round}/${sim.total_rounds}` : "..."}
                  </span>
                  <div className="w-20 h-1 bg-ki-sunken shrink-0">
                    <div className="h-full bg-ki-primary transition-all duration-500" style={{ width: `${pct}%` }} />
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {/* Recent simulations — dense table */}
      {recentSims.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-1">
            <span className="text-2xs font-bold text-ki-on-surface-muted tracking-[0.08em]">
              RECENT ({recentSims.length})
            </span>
          </div>
          {/* Table header */}
          <div className="border border-ki-border bg-ki-surface-raised">
            <div className="flex items-center gap-2 px-2 py-0.5 bg-ki-surface-sunken border-b border-ki-border text-2xs font-bold text-ki-on-surface-muted tracking-[0.06em]">
              <span className="w-12 shrink-0">STATUS</span>
              <span className="flex-1">SCENARIO</span>
              <span className="w-20 shrink-0 text-right hidden sm:block">DOMAIN</span>
              <span className="w-12 shrink-0 text-right">RNDS</span>
              <span className="w-16 shrink-0 text-right">COST</span>
              <span className="w-20 shrink-0 text-right">ACTIONS</span>
            </div>
            {recentSims.map((sim, i) => {
              const st = ST[sim.status] || ST.completed;
              return (
                <div
                  key={sim.id}
                  className={`flex items-center gap-2 px-2 py-[3px] hover:bg-ki-surface-hover transition-colors ${
                    i < recentSims.length - 1 ? "border-b border-ki-border/50" : ""
                  }`}
                >
                  <span className={`w-12 shrink-0 font-data text-2xs font-bold ${st.cls}`}>
                    {st.label}
                  </span>
                  <span className="font-body text-xs text-ki-on-surface truncate flex-1">
                    {sim.scenario_name || sim.brief.slice(0, 80)}
                  </span>
                  {sim.domain ? (
                    <span className={`w-20 shrink-0 text-right hidden sm:block font-data text-2xs font-bold ${DOMAIN_COLOR[sim.domain] || "text-ki-on-surface-muted"}`}>
                      {sim.domain.replace(/_/g, " ").toUpperCase().slice(0, 10)}
                    </span>
                  ) : (
                    <span className="w-20 shrink-0 hidden sm:block" />
                  )}
                  <span className="w-12 shrink-0 text-right font-data text-2xs text-ki-on-surface-muted">
                    {sim.total_rounds > 0 ? sim.total_rounds : "—"}
                  </span>
                  <span className="w-16 shrink-0 text-right font-data text-2xs text-ki-on-surface-muted">
                    ${sim.cost.toFixed(2)}
                  </span>
                  <div className="w-20 shrink-0 flex justify-end gap-1">
                    <Link
                      href={`/sim/${sim.id}`}
                      className="px-1.5 py-0.5 text-2xs font-bold text-ki-on-surface-muted hover:text-ki-on-surface border border-ki-border hover:bg-ki-surface-hover transition-colors"
                    >
                      LOG
                    </Link>
                    {sim.status === "completed" && sim.scenario_id && (
                      <Link
                        href={`/scenario/${sim.scenario_id}`}
                        className="px-1.5 py-0.5 text-2xs font-bold text-ki-primary hover:bg-ki-primary hover:text-ki-on-surface border border-ki-primary/30 transition-colors"
                      >
                        VIEW
                      </Link>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Scenarios grid — compact cards */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <span className="text-2xs font-bold text-ki-on-surface-muted tracking-[0.08em]">
            SCENARIOS ({scenarios.length})
          </span>
        </div>
        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-1">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="bg-ki-surface-raised border border-ki-border p-2 animate-pulse">
                <div className="h-3 bg-ki-surface-sunken w-16 mb-2" />
                <div className="h-3 bg-ki-surface-sunken w-full mb-1" />
                <div className="h-3 bg-ki-surface-sunken w-2/3" />
              </div>
            ))}
          </div>
        ) : scenarios.length === 0 ? (
          <div className="border border-dashed border-ki-border-strong bg-ki-surface-raised p-6 text-center">
            <span className="font-data text-xs text-ki-on-surface-muted">
              NO SCENARIOS — LAUNCH A SIMULATION TO BEGIN
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-1">
            {scenarios.map((s) => (
              <Link
                key={s.id}
                href={`/scenario/${s.id}`}
                className="group bg-ki-surface-raised border border-ki-border p-2 hover:bg-ki-surface-hover hover:border-ki-border-strong transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className={`font-data text-2xs font-bold ${DOMAIN_COLOR[s.domain] || "text-ki-on-surface-muted"}`}>
                    {s.domain.replace(/_/g, " ").toUpperCase()}
                  </span>
                  <span className="font-data text-2xs text-ki-on-surface-muted">
                    {s.id}
                  </span>
                </div>
                <div className="text-xs font-semibold text-ki-on-surface mb-0.5 truncate group-hover:text-ki-primary transition-colors">
                  {s.name}
                </div>
                <div className="text-2xs text-ki-on-surface-muted line-clamp-1">
                  {s.description}
                </div>
                <div className="mt-1 font-data text-2xs text-ki-on-surface-muted">
                  {s.num_rounds}R
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
