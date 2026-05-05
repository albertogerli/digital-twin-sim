"use client";

import { useEffect, useMemo, useState } from "react";
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

const STATUS: Record<string, { label: string; tone: "live" | "pos" | "warn" | "muted" }> = {
  queued:      { label: "Queued",      tone: "muted" },
  analyzing:   { label: "Analyzing",   tone: "live" },
  configuring: { label: "Configuring", tone: "live" },
  running:     { label: "Running",     tone: "live" },
  exporting:   { label: "Exporting",   tone: "live" },
  completed:   { label: "Completed",   tone: "pos"  },
  failed:      { label: "Failed",      tone: "warn" },
  cancelled:   { label: "Cancelled",   tone: "muted" },
};

const DOMAIN_CAP: Record<string, string> = {
  political:     "bg-domain-political",
  corporate:     "bg-domain-corporate",
  financial:     "bg-domain-financial",
  commercial:    "bg-domain-commercial",
  marketing:     "bg-domain-marketing",
  public_health: "bg-domain-health",
  technology:    "bg-domain-technology",
};

/* ── KPI cell ───────────────────────────────────────────── */
function KPI({
  label,
  value,
  delta,
  deltaPct,
  sub,
}: {
  label: string;
  value: string | number;
  delta?: number;
  deltaPct?: boolean;
  sub?: string;
}) {
  return (
    <div className="px-4 py-3 border-r border-ki-border flex-1 min-w-0">
      <div className="eyebrow">{label}</div>
      <div className="flex items-baseline gap-2 mt-1">
        <span className="font-data tabular text-[22px] font-medium tracking-tight2 text-ki-on-surface">
          {value}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span
            className={`font-data tabular text-[11px] ${
              delta > 0 ? "text-ki-success" : "text-ki-error"
            }`}
          >
            {delta > 0 ? "▲" : "▼"} {(deltaPct ? Math.abs(delta * 100).toFixed(2) + "%" : Math.abs(delta).toFixed(2))}
          </span>
        )}
      </div>
      {sub && <div className="text-[11px] text-ki-on-surface-muted mt-0.5">{sub}</div>}
    </div>
  );
}

/* ── Status dot ─────────────────────────────────────────── */
function StatusDot({ status }: { status: string }) {
  const s = STATUS[status] || STATUS.queued;
  const dotClass = {
    live:  "bg-ki-error animate-live-pulse",
    pos:   "bg-ki-success",
    warn:  "bg-ki-warning",
    muted: "bg-ki-on-surface-muted",
  }[s.tone];
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-ki-on-surface-secondary">
      <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
      {s.label}
    </span>
  );
}

export default function Home() {
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [simulations, setSimulations] = useState<SimStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "running" | "completed" | "queued" | "failed">("all");
  const [q, setQ] = useState("");

  useEffect(() => {
    Promise.all([
      fetch("/api/scenarios").then((r) => (r.ok ? r.json() : [])).catch(() =>
        fetch("/data/scenarios.json").then((r) => r.json()).catch(() => []),
      ),
      fetch("/api/simulations").then((r) => (r.ok ? r.json() : [])).catch(() => []),
    ]).then(([sc, sims]) => {
      setScenarios(sc);
      setSimulations(sims);
      setLoading(false);
    });

    const interval = setInterval(() => {
      fetch("/api/simulations")
        .then((r) => (r.ok ? r.json() : []))
        .then(setSimulations)
        .catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const counts = useMemo(() => {
    const c = { all: simulations.length, running: 0, completed: 0, queued: 0, failed: 0 };
    for (const s of simulations) {
      if (["queued", "analyzing", "configuring"].includes(s.status)) c.queued++;
      else if (["running", "exporting"].includes(s.status)) c.running++;
      else if (s.status === "completed") c.completed++;
      else if (s.status === "failed") c.failed++;
    }
    return c;
  }, [simulations]);

  const totalSpend = useMemo(
    () => simulations.reduce((acc, s) => acc + (s.cost || 0), 0),
    [simulations],
  );

  const filtered = useMemo(() => {
    return simulations.filter((s) => {
      if (filter === "running" && !["running", "exporting"].includes(s.status)) return false;
      if (filter === "completed" && s.status !== "completed") return false;
      if (filter === "queued" && !["queued", "analyzing", "configuring"].includes(s.status)) return false;
      if (filter === "failed" && s.status !== "failed") return false;
      const needle = q.toLowerCase();
      if (
        needle &&
        !(s.scenario_name || s.brief).toLowerCase().includes(needle) &&
        !s.id.toLowerCase().includes(needle)
      )
        return false;
      return true;
    });
  }, [simulations, filter, q]);

  const filterPill = (key: typeof filter, label: string, n: number) => (
    <button
      key={key}
      onClick={() => setFilter(key)}
      className={`inline-flex items-center gap-1.5 px-2.5 h-6 rounded-sm text-[11px] font-medium transition-colors ${
        filter === key
          ? "bg-ki-on-surface text-ki-surface"
          : "text-ki-on-surface-secondary hover:bg-ki-surface-hover"
      }`}
    >
      {label}
      <span className="font-data tabular opacity-70">{n}</span>
    </button>
  );

  return (
    <div className="flex flex-col h-[calc(100vh-44px)]">
      {/* KPI strip */}
      <div className="flex border-b border-ki-border bg-ki-surface-raised">
        <KPI label="Active simulations" value={counts.running} sub={`${counts.queued} queued`} />
        <KPI label="Completed" value={counts.completed} sub={`${counts.failed} failed`} />
        <KPI label="LLM spend (total)" value={`$${totalSpend.toFixed(2)}`} sub="all runs" />
        <KPI label="Scenarios" value={scenarios.length} sub="ready to launch" />
        <div className="px-4 py-3 w-[220px] flex flex-col justify-center">
          <Link
            href="/new"
            className="inline-flex items-center justify-center gap-1.5 h-8 rounded-sm bg-ki-on-surface text-ki-surface text-[12px] font-medium hover:bg-ki-on-surface-secondary transition-colors"
          >
            <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>
              add
            </span>
            New simulation
          </Link>
          <div className="text-[11px] text-ki-on-surface-muted mt-1.5 text-center">
            <span className="kbd">N</span> to launch
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-ki-border bg-ki-surface-raised">
        <div className="flex items-center gap-1">
          {filterPill("all", "All", counts.all)}
          {filterPill("running", "Running", counts.running)}
          {filterPill("completed", "Completed", counts.completed)}
          {filterPill("queued", "Queued", counts.queued)}
          {filterPill("failed", "Failed", counts.failed)}
        </div>
        <div className="flex-1" />
        <div className="relative w-[280px]">
          <span
            className="material-symbols-outlined absolute left-2 top-1/2 -translate-y-1/2 text-[14px] text-ki-on-surface-muted pointer-events-none"
            style={{ fontVariationSettings: "'wght' 400" }}
          >
            search
          </span>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by name or ID…"
            className="w-full h-7 pl-7 pr-2 text-[12px] bg-ki-surface-sunken border border-ki-border rounded-sm focus:outline-none focus:border-ki-primary placeholder:text-ki-on-surface-muted"
          />
        </div>
        <button className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm text-[11px] text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover">
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>
            tune
          </span>
          Filters
        </button>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto bg-ki-surface-raised">
        {loading ? (
          <div className="px-4 py-6 text-[12px] text-ki-on-surface-muted">Loading…</div>
        ) : filtered.length === 0 ? (
          <div className="px-4 py-12 text-center">
            <div className="text-[13px] text-ki-on-surface-secondary">No simulations match this filter.</div>
            <div className="text-[11px] text-ki-on-surface-muted mt-1">
              Try a different status or launch a new simulation.
            </div>
          </div>
        ) : (
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-ki-border bg-ki-surface-sunken">
                <th className="w-8" />
                <th className="text-left eyebrow font-medium px-3 py-1.5">Simulation</th>
                <th className="text-left eyebrow font-medium px-3 py-1.5 w-[110px]">Status</th>
                <th className="text-right eyebrow font-medium px-3 py-1.5 w-[140px]">Progress</th>
                <th className="text-left eyebrow font-medium px-3 py-1.5 w-[120px]">Domain</th>
                <th className="text-right eyebrow font-medium px-3 py-1.5 w-[80px]">Rounds</th>
                <th className="text-right eyebrow font-medium px-3 py-1.5 w-[80px]">Spend</th>
                <th className="w-[90px]" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((sim) => {
                const pct = sim.total_rounds > 0 ? sim.current_round / sim.total_rounds : 0;
                const cap = DOMAIN_CAP[sim.domain || ""] || "bg-ki-on-surface-faint";
                const isLive = ["running", "analyzing", "configuring", "exporting"].includes(sim.status);
                return (
                  <tr
                    key={sim.id}
                    className="border-b border-ki-border-faint hover:bg-ki-surface-hover transition-colors group"
                  >
                    <td className="p-0">
                      <div className="flex items-center h-8 pl-3">
                        <div className={`${cap} w-[2px] self-stretch rounded-sm`} />
                      </div>
                    </td>
                    <td className="px-3 py-1.5 max-w-[420px]">
                      <Link href={`/sim/${sim.id}`} className="flex items-center gap-2 min-w-0">
                        <span className="text-[13px] font-medium text-ki-on-surface truncate flex-1">
                          {sim.scenario_name || sim.brief.slice(0, 80)}
                        </span>
                        <span className="font-data text-[10px] text-ki-on-surface-muted shrink-0">
                          {sim.id.slice(0, 12)}
                        </span>
                      </Link>
                    </td>
                    <td className="px-3 py-1.5">
                      <StatusDot status={sim.status} />
                    </td>
                    <td className="px-3 py-1.5">
                      <div className="flex flex-col items-end gap-1">
                        <span className="font-data tabular text-[11px] text-ki-on-surface-secondary">
                          {sim.total_rounds > 0 ? `${sim.current_round}/${sim.total_rounds}` : "—"}
                        </span>
                        <div className="w-[80px] h-1 bg-ki-surface-sunken rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all duration-500 ${
                              sim.status === "failed"
                                ? "bg-ki-warning"
                                : isLive
                                ? "bg-ki-primary"
                                : "bg-ki-on-surface"
                            }`}
                            style={{ width: `${Math.max(2, pct * 100)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-1.5">
                      {sim.domain ? (
                        <span className="font-data text-[10px] uppercase tracking-[0.06em] text-ki-on-surface-secondary">
                          {sim.domain.replace(/_/g, " ")}
                        </span>
                      ) : (
                        <span className="text-ki-on-surface-faint">—</span>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-right font-data tabular text-[11px] text-ki-on-surface-secondary">
                      {sim.total_rounds > 0 ? sim.total_rounds : "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right font-data tabular text-[11px] text-ki-on-surface-secondary">
                      ${sim.cost.toFixed(2)}
                    </td>
                    <td className="px-2 py-1.5">
                      <div className="flex justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Link
                          href={`/sim/${sim.id}`}
                          className="inline-flex items-center justify-center h-6 px-2 rounded-sm text-[11px] text-ki-on-surface-secondary hover:bg-ki-surface-active"
                        >
                          Log
                        </Link>
                        {sim.status === "completed" && sim.scenario_id && (
                          <Link
                            href={`/scenario/${sim.scenario_id}`}
                            className="inline-flex items-center justify-center h-6 px-2 rounded-sm text-[11px] text-ki-primary hover:bg-ki-primary-soft"
                          >
                            Open
                          </Link>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Scenario shelf — all scenarios, scrollable when count > 8 rows */}
      {scenarios.length > 0 && (
        <div className="border-t border-ki-border bg-ki-surface-sunken">
          <div className="sticky top-0 z-10 flex items-center gap-2 px-4 py-1.5 bg-ki-surface-sunken border-b border-ki-border-faint">
            <span className="eyebrow">Scenarios</span>
            <span className="font-data text-[10px] text-ki-on-surface-muted">{scenarios.length}</span>
            <span className="ml-auto font-data text-[10px] text-ki-on-surface-muted">click any to open · scroll for more</span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-ki-border-faint max-h-[420px] overflow-y-auto">
            {scenarios.map((s) => {
              const cap = DOMAIN_CAP[s.domain] || "bg-ki-on-surface-faint";
              return (
                <Link
                  key={s.id}
                  href={`/scenario/${s.id}`}
                  className="flex bg-ki-surface-raised hover:bg-ki-surface-hover transition-colors"
                >
                  <div className={`${cap} w-[2px] self-stretch`} />
                  <div className="px-3 py-2 flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="font-data text-[10px] uppercase tracking-[0.06em] text-ki-on-surface-secondary">
                        {s.domain.replace(/_/g, " ")}
                      </span>
                      <span className="font-data text-[10px] text-ki-on-surface-muted">{s.num_rounds}R</span>
                    </div>
                    <div className="text-[12px] font-medium text-ki-on-surface truncate">{s.name}</div>
                    <div className="text-[11px] text-ki-on-surface-muted line-clamp-1 mt-0.5">{s.description}</div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      {/* Footer status bar */}
      <div className="flex items-center gap-3 h-7 px-4 border-t border-ki-border bg-ki-surface-sunken text-[11px] text-ki-on-surface-muted flex-shrink-0">
        <span className="font-data tabular">
          {filtered.length} of {simulations.length} simulations
        </span>
        <span className="text-ki-on-surface-faint">·</span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
          API connected
        </span>
        <div className="flex-1" />
        <span className="font-data tabular">DigitalTwinSim · v0.5</span>
      </div>
    </div>
  );
}
