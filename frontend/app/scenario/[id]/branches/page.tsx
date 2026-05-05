"use client";

import { useState, useMemo } from "react";
import Link from "next/link";

/* ───────────────────────────────────────────────────────────
   Scenario tree — what-if branches
   Mock data only (backend doesn't expose branches yet).
   Pattern: branch tree (left) + comparison view (right).
   ─────────────────────────────────────────────────────────── */

interface BranchOutcome {
  pol: number;
  sent: number;
  vote: number;
}

interface Branch {
  id: string;
  parent: string | null;
  name: string;
  divergence: string;
  tag: "baseline" | "current" | "what-if";
  posts: number;
  outcome: BranchOutcome;
  intervention?: string;
}

const MOCK_BRANCHES: Branch[] = [
  {
    id: "b0",
    parent: null,
    name: "Baseline",
    divergence: "—",
    tag: "baseline",
    posts: 4180,
    outcome: { pol: 0.71, sent: -0.18, vote: 0.37 },
  },
  {
    id: "b1",
    parent: "b0",
    name: "Industry concedes early",
    divergence: "R3",
    tag: "what-if",
    posts: 3920,
    outcome: { pol: 0.41, sent: 0.06, vote: 0.59 },
    intervention: "Vornex CEO accepts amendment package at R3",
  },
  {
    id: "b2",
    parent: "b0",
    name: "Crisis escalation",
    divergence: "R4",
    tag: "what-if",
    posts: 5240,
    outcome: { pol: 0.89, sent: -0.42, vote: 0.18 },
    intervention: "Industry lobby leaks confidential briefing at R4",
  },
  {
    id: "b3",
    parent: "b1",
    name: "Industry concedes + media amplification",
    divergence: "R5",
    tag: "current",
    posts: 4120,
    outcome: { pol: 0.32, sent: 0.18, vote: 0.74 },
    intervention: "Bloomberg cluster amplifies concession framing",
  },
  {
    id: "b4",
    parent: "b2",
    name: "Crisis + EU council emergency session",
    divergence: "R6",
    tag: "what-if",
    posts: 5680,
    outcome: { pol: 0.93, sent: -0.51, vote: 0.12 },
    intervention: "Council convenes emergency session, postpones vote",
  },
];

function BranchNode({
  branch,
  depth,
  active,
  onSelect,
  branches,
}: {
  branch: Branch;
  depth: number;
  active: string;
  onSelect: (id: string) => void;
  branches: Branch[];
}) {
  const children = branches.filter((b) => b.parent === branch.id);
  const isActive = active === branch.id;
  return (
    <div>
      <button
        onClick={() => onSelect(branch.id)}
        className={`flex items-center gap-2 w-full text-left px-2 py-2 rounded-sm transition-colors ${
          isActive ? "bg-ki-surface-hover" : "hover:bg-ki-surface-hover"
        }`}
        style={{ paddingLeft: 8 + depth * 18 }}
      >
        <svg
          className="flex-shrink-0"
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke={branch.tag === "current" ? "var(--accent)" : "var(--ink-3)"}
          strokeWidth="2"
        >
          <circle cx="6" cy="3" r="2" />
          <circle cx="6" cy="21" r="2" />
          <circle cx="18" cy="12" r="2" />
          <path d="M6 5v14" />
          <path d="M6 12h10" strokeDasharray={branch.tag === "what-if" || branch.tag === "current" ? "" : "0"} />
        </svg>
        <div className="flex flex-col flex-1 min-w-0">
          <span className={`text-[12px] font-medium truncate ${isActive ? "text-ki-on-surface" : "text-ki-on-surface-secondary"}`}>
            {branch.name}
          </span>
          <span className="font-data text-[10px] text-ki-on-surface-muted truncate">
            {branch.parent ? `from ${branch.divergence}` : "baseline"} · {branch.posts.toLocaleString()} posts
          </span>
        </div>
        <span className="font-data tabular text-[11px] text-ki-on-surface-secondary flex-shrink-0">
          {Math.round(branch.outcome.vote * 100)}%
        </span>
      </button>
      {children.map((c) => (
        <BranchNode key={c.id} branch={c} depth={depth + 1} active={active} onSelect={onSelect} branches={branches} />
      ))}
    </div>
  );
}

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
    <div className="px-4 py-3 border-r border-ki-border flex-1 min-w-0 last:border-r-0">
      <div className="eyebrow">{label}</div>
      <div className="flex items-baseline gap-2 mt-1">
        <span className="font-data tabular text-[20px] font-medium tracking-tight2 text-ki-on-surface">
          {value}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span className={`font-data tabular text-[11px] ${delta > 0 ? "text-ki-success" : "text-ki-error"}`}>
            {delta > 0 ? "▲" : "▼"} {deltaPct ? Math.abs(delta * 100).toFixed(0) + "%" : Math.abs(delta).toFixed(2)}
          </span>
        )}
      </div>
      {sub && <div className="text-[11px] text-ki-on-surface-muted mt-0.5">{sub}</div>}
    </div>
  );
}

export default function BranchesPage({ params }: { params: { id: string } }) {
  const [active, setActive] = useState("b3");
  const branches = MOCK_BRANCHES;
  const branch = branches.find((b) => b.id === active) || branches[0];
  const baseline = branches.find((b) => b.tag === "baseline")!;

  const voteRange = useMemo(() => {
    const votes = branches.map((b) => b.outcome.vote);
    return { min: Math.min(...votes), max: Math.max(...votes) };
  }, [branches]);

  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface flex flex-col">
      {/* Sub-toolbar */}
      <div className="sticky top-0 z-30 bg-ki-surface-raised/95 backdrop-blur border-b border-ki-border h-11 flex items-center px-4 gap-3">
        <Link href="/" className="inline-flex items-center gap-1 text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors group" title="Dashboard">
          <span className="material-symbols-outlined text-[14px] group-hover:-translate-x-0.5 transition-transform">arrow_back</span>
        </Link>
        <span className="text-ki-border-strong">/</span>
        <Link href={`/scenario/${params.id}`} className="inline-flex items-center gap-1.5 text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors">
          Scenario
        </Link>
        <span className="text-ki-border-strong">/</span>
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">Scenario tree</span>
        <span className="text-ki-border-strong">/</span>
        <span className="text-[13px] font-medium text-ki-on-surface truncate flex-1 min-w-0">
          {branches.length} branches · 1 baseline
        </span>
        <button className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors">
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>add</span>
          Fork what-if
        </button>
        <button className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border border-ki-border text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors">
          <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>download</span>
          Export comparison
        </button>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* Branch tree */}
        <aside className="w-[320px] border-r border-ki-border bg-ki-surface-sunken flex-shrink-0 p-4 overflow-y-auto">
          <div className="eyebrow mb-3">Branches</div>
          <div className="flex flex-col gap-0.5">
            {branches
              .filter((b) => b.parent === null)
              .map((b) => (
                <BranchNode key={b.id} branch={b} depth={0} active={active} onSelect={setActive} branches={branches} />
              ))}
          </div>
        </aside>

        {/* Comparison */}
        <div className="flex-1 overflow-y-auto p-6 max-w-[960px]">
          <div className="eyebrow">Selected branch</div>
          <h1 className="text-[24px] font-medium tracking-tight2 text-ki-on-surface mt-1">{branch.name}</h1>
          <p className="text-[12px] text-ki-on-surface-secondary mt-1">
            {branch.parent
              ? <>Forked from <span className="font-data text-ki-on-surface">{branches.find((b) => b.id === branch.parent)?.name}</span> at <span className="font-data tabular text-ki-on-surface">{branch.divergence}</span></>
              : "Baseline run · seed 1337"}
            {branch.intervention && (
              <span className="block mt-1 text-ki-on-surface-muted italic">{branch.intervention}</span>
            )}
          </p>

          {/* KPI strip */}
          <div className="mt-5 flex border border-ki-border rounded bg-ki-surface-raised">
            <KPI
              label="Polarization"
              value={branch.outcome.pol.toFixed(2)}
              delta={branch.outcome.pol - baseline.outcome.pol}
            />
            <KPI
              label="Net sentiment"
              value={(branch.outcome.sent > 0 ? "+" : "") + branch.outcome.sent.toFixed(2)}
              delta={branch.outcome.sent - baseline.outcome.sent}
            />
            <KPI
              label="Vote probability"
              value={`${Math.round(branch.outcome.vote * 100)}%`}
              delta={branch.outcome.vote - baseline.outcome.vote}
              deltaPct
            />
            <KPI label="Posts" value={branch.posts.toLocaleString()} sub="across 6 platforms" />
          </div>

          {/* Comparison table */}
          <div className="mt-6">
            <div className="eyebrow mb-2">Branch comparison</div>
            <div className="bg-ki-surface-raised border border-ki-border rounded overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="bg-ki-surface-sunken border-b border-ki-border">
                    <th className="text-left eyebrow font-medium px-3 py-2">Branch</th>
                    <th className="text-left eyebrow font-medium px-3 py-2 w-[100px]">Diverged</th>
                    <th className="text-right eyebrow font-medium px-3 py-2 w-[100px]">Polariz.</th>
                    <th className="text-right eyebrow font-medium px-3 py-2 w-[100px]">Sentiment</th>
                    <th className="text-right eyebrow font-medium px-3 py-2 w-[110px]">Vote prob.</th>
                    <th className="text-right eyebrow font-medium px-3 py-2 w-[110px]">Δ vs base</th>
                  </tr>
                </thead>
                <tbody>
                  {branches.map((b) => {
                    const isActive = b.id === active;
                    return (
                      <tr
                        key={b.id}
                        onClick={() => setActive(b.id)}
                        className={`border-b border-ki-border-faint last:border-b-0 cursor-pointer transition-colors ${
                          isActive ? "bg-ki-surface-hover" : "hover:bg-ki-surface-hover"
                        }`}
                      >
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <span className={`inline-flex items-center px-1.5 h-5 rounded-sm font-data text-[10px] ${
                              b.tag === "current" ? "bg-ki-primary-soft text-ki-primary" :
                              b.tag === "baseline" ? "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border" :
                              "bg-ki-surface-sunken text-ki-on-surface-muted border border-ki-border"
                            }`}>
                              {b.tag}
                            </span>
                            <span className="text-[12px] font-medium text-ki-on-surface">{b.name}</span>
                          </div>
                        </td>
                        <td className="px-3 py-2 font-data text-[11px] text-ki-on-surface-muted">
                          {b.parent ? b.divergence : "—"}
                        </td>
                        <td className="px-3 py-2 text-right font-data tabular text-[12px] text-ki-on-surface">
                          {b.outcome.pol.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right font-data tabular text-[12px] text-ki-on-surface">
                          {b.outcome.sent > 0 ? "+" : ""}{b.outcome.sent.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right font-data tabular text-[12px] text-ki-on-surface">
                          {Math.round(b.outcome.vote * 100)}%
                        </td>
                        <td className="px-3 py-2 text-right">
                          {b.id === baseline.id ? (
                            <span className="font-data text-[11px] text-ki-on-surface-muted">baseline</span>
                          ) : (
                            <span className={`font-data tabular text-[12px] ${
                              b.outcome.vote - baseline.outcome.vote > 0 ? "text-ki-success" : "text-ki-error"
                            }`}>
                              {b.outcome.vote - baseline.outcome.vote > 0 ? "▲" : "▼"} {Math.abs((b.outcome.vote - baseline.outcome.vote) * 100).toFixed(0)}%
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Narrative summary */}
          <div className="mt-6">
            <div className="eyebrow mb-2">Narrative summary</div>
            <div className="bg-ki-surface-raised border border-ki-border rounded p-4">
              <p className="text-[13px] text-ki-on-surface-secondary leading-[1.65] max-w-2xl">
                Across {branches.length - 1} counterfactuals, vote probability ranges from{" "}
                <span className="font-data tabular text-ki-on-surface">{Math.round(voteRange.min * 100)}%</span> to{" "}
                <span className="font-data tabular text-ki-on-surface">{Math.round(voteRange.max * 100)}%</span>.
                The dominant lever is industry positioning at R3–R4: when industry concedes early,
                polarization drops by <span className="font-data tabular text-ki-on-surface">0.30</span>{" "}
                and the vote becomes likely. Crisis escalation reverses this entirely.
                Baseline outcome remains the modal scenario at{" "}
                <span className="font-data tabular text-ki-on-surface">{Math.round(baseline.outcome.vote * 100)}%</span>{" "}
                vote probability.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
