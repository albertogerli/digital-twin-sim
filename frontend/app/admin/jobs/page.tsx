"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

type JobState = "running" | "completed" | "failed";

interface LastRun {
  state: JobState;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  output_tail: string[];
}

interface JobMeta {
  name: string;
  label: string;
  description: string;
  icon: string;
  last_run: LastRun | null;
}

const STATE_TONE: Record<JobState | "idle", { tone: string; dot: string; label: string }> = {
  running:   { tone: "bg-amber-50 text-amber-700 border-amber-200",    dot: "bg-amber-500 animate-pulse", label: "Running" },
  completed: { tone: "bg-emerald-50 text-emerald-700 border-emerald-200", dot: "bg-emerald-500",          label: "Completed" },
  failed:    { tone: "bg-red-50 text-red-700 border-red-200",          dot: "bg-red-500",                  label: "Failed" },
  idle:      { tone: "bg-gray-50 text-gray-600 border-gray-200",       dot: "bg-gray-400",                 label: "Never run" },
};

export default function AdminJobsPage() {
  const [jobs, setJobs] = useState<JobMeta[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [openLog, setOpenLog] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/admin/jobs`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setJobs(data.jobs ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    // Poll every 4s — cheap and lets us show "Running" → "Completed" without a refresh.
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [refresh]);

  async function runJob(name: string) {
    if (pending[name]) return;
    setPending((p) => ({ ...p, [name]: true }));
    try {
      const res = await fetch(`${API_BASE}/api/admin/jobs/${name}/run`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `HTTP ${res.status}`);
      }
      // Optimistic refresh
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending((p) => ({ ...p, [name]: false }));
    }
  }

  return (
    <div className="p-5 space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <div className="eyebrow">Admin</div>
        <h1 className="text-[20px] font-medium tracking-tight2 text-ki-on-surface mt-1">
          Background jobs
        </h1>
        <p className="text-[12px] text-ki-on-surface-secondary mt-1 leading-relaxed max-w-2xl">
          Manually trigger long-running maintenance jobs that normally run on a nightly cron:
          self-calibration forecasts, T+1/T+3/T+7 evaluation against realised yfinance returns,
          and the stakeholder-graph RSS crawl. Each job runs in a background thread on the API
          process; status auto-refreshes every 4 seconds.
        </p>
      </div>

      {error && (
        <div className="bg-ki-error-soft border border-ki-error/30 rounded p-2 text-[12px] text-ki-error">
          {error}
        </div>
      )}

      {jobs === null ? (
        <div className="text-[12px] text-ki-on-surface-muted">Loading jobs<span className="cursor-blink">_</span></div>
      ) : jobs.length === 0 ? (
        <div className="bg-ki-surface-raised border border-dashed border-ki-border rounded p-6 text-center">
          <span className="text-[12px] text-ki-on-surface-muted">
            No jobs registered on the backend.
          </span>
        </div>
      ) : (
        <div className="space-y-3">
          {jobs.map((j) => {
            const state = j.last_run?.state ?? "idle";
            const tone = STATE_TONE[state];
            const isRunning = state === "running";
            const isOpen = openLog === j.name;
            return (
              <div
                key={j.name}
                className="bg-ki-surface-raised border border-ki-border rounded overflow-hidden"
              >
                <div className="flex items-center gap-3 p-4">
                  <span className="w-9 h-9 grid place-items-center rounded-md bg-ki-primary-soft text-ki-primary flex-shrink-0">
                    <span className="material-symbols-outlined text-[18px]" style={{ fontVariationSettings: "'wght' 500" }}>
                      {j.icon}
                    </span>
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[13px] font-medium text-ki-on-surface">{j.label}</span>
                      <span className="font-data text-[10px] text-ki-on-surface-muted">{j.name}</span>
                      <span
                        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider border ${tone.tone}`}
                      >
                        <span className={`w-1.5 h-1.5 rounded-full ${tone.dot}`} />
                        {tone.label}
                        {j.last_run?.exit_code != null && state !== "running" && (
                          <span className="opacity-70">· exit {j.last_run.exit_code}</span>
                        )}
                      </span>
                    </div>
                    <div className="text-[11px] text-ki-on-surface-secondary mt-1 leading-snug">
                      {j.description}
                    </div>
                    {j.last_run && (
                      <div className="font-data tabular text-[10px] text-ki-on-surface-muted mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5">
                        <span>Started {fmtTs(j.last_run.started_at)}</span>
                        {j.last_run.finished_at && (
                          <>
                            <span className="text-ki-border-strong">·</span>
                            <span>Finished {fmtTs(j.last_run.finished_at)}</span>
                            {j.last_run.started_at && (
                              <>
                                <span className="text-ki-border-strong">·</span>
                                <span>{durationSec(j.last_run.started_at, j.last_run.finished_at)}s elapsed</span>
                              </>
                            )}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {j.last_run && j.last_run.output_tail.length > 0 && (
                      <button
                        onClick={() => setOpenLog(isOpen ? null : j.name)}
                        className="inline-flex items-center gap-1.5 h-8 px-2.5 rounded-sm border border-ki-border text-[11px] text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors"
                      >
                        <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>
                          {isOpen ? "expand_less" : "terminal"}
                        </span>
                        {isOpen ? "Hide log" : "Log"}
                      </button>
                    )}
                    <button
                      onClick={() => runJob(j.name)}
                      disabled={isRunning || pending[j.name]}
                      className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-ki-on-surface text-ki-surface text-[12px] font-medium hover:bg-ki-on-surface-secondary disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
                    >
                      {isRunning ? (
                        <>
                          <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                          Running…
                        </>
                      ) : (
                        <>
                          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 500" }}>
                            play_arrow
                          </span>
                          Run now
                        </>
                      )}
                    </button>
                  </div>
                </div>
                {isOpen && j.last_run && (
                  <pre className="border-t border-ki-border bg-ki-surface-sunken px-4 py-3 text-[11px] font-data text-ki-on-surface-secondary overflow-x-auto max-h-72 whitespace-pre-wrap leading-relaxed">
                    {j.last_run.output_tail.length === 0
                      ? "(no output captured)"
                      : j.last_run.output_tail.join("\n")}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Info box */}
      <div className="bg-ki-primary-soft rounded p-3 flex items-start gap-2">
        <span className="material-symbols-outlined text-[14px] text-ki-primary mt-0.5" style={{ fontVariationSettings: "'wght' 500" }}>
          info
        </span>
        <div className="text-[12px] text-ki-on-surface-secondary leading-relaxed">
          <p>
            <span className="text-ki-primary font-medium">Where they live.</span>{" "}
            Status is in-memory on the API process — restarting the backend clears the
            <em> last-run </em>history (the underlying SQLite registries persist).
            Hard cap: 600s wall-clock per command. Concurrent invocations of the same
            job are rejected with HTTP 409.
          </p>
          <p className="mt-1.5">
            <span className="text-ki-primary font-medium">CLI equivalents.</span>{" "}
            <code className="font-data text-[11px] bg-ki-surface-sunken px-1 rounded">python -m scripts.continuous_calibration forecast</code>
            ,{" "}
            <code className="font-data text-[11px] bg-ki-surface-sunken px-1 rounded">… evaluate --horizon {1}</code>
            ,{" "}
            <code className="font-data text-[11px] bg-ki-surface-sunken px-1 rounded">python -m stakeholder_graph.updater</code>
            .
          </p>
        </div>
      </div>

      <div className="flex gap-4">
        <Link
          href="/admin/invites"
          className="inline-flex items-center gap-1.5 text-[12px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors"
        >
          → Invite links
        </Link>
        <Link
          href="/settings"
          className="inline-flex items-center gap-1.5 text-[12px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors"
        >
          ← Settings
        </Link>
      </div>
    </div>
  );
}

function fmtTs(iso: string | null): string {
  if (!iso) return "—";
  try {
    return iso.replace("T", " ").replace("Z", "");
  } catch {
    return iso;
  }
}

function durationSec(start: string, end: string): string {
  try {
    const s = Date.parse(start);
    const e = Date.parse(end);
    if (!Number.isFinite(s) || !Number.isFinite(e)) return "—";
    return ((e - s) / 1000).toFixed(1);
  } catch {
    return "—";
  }
}
