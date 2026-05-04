"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import type React from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

type Mode = "OFF" | "LOG" | "STRICT" | "BLOCK";

interface ByodStatus {
  mode: Mode;
  n_audit_rows: number;
  by_site: Record<string, number>;
  by_category: Record<string, number>;
}

interface AuditRow {
  ts: string;
  site: string;
  mode: Mode;
  raw_chars: number;
  sanitized_chars: number;
  patterns: { category: string; count: number }[];
  tenant: string;
}

interface DoraPreview {
  sim_id: string;
  scenario_name: string;
  classification: {
    clients_affected: string;
    data_losses: string;
    reputational_impact: string;
    duration_downtime_hours: number;
    geographical_spread: string;
    economic_impact_eur_band: string;
    criticality_of_services_affected: string;
  };
  is_major: boolean;
  metrics_used: Record<string, number | string>;
}

interface ScenarioListItem {
  sim_id: string;
  name?: string;
  status: string;
  domain?: string;
  rounds?: number;
}

const MODE_TONE: Record<Mode, { label: string; tone: string; dot: string; help: string }> = {
  OFF: {
    label: "Off — passthrough",
    tone: "bg-gray-50 text-gray-700 border-gray-200",
    dot: "bg-gray-400",
    help: "No checking, no audit log. Use only in trusted single-tenant setups.",
  },
  LOG: {
    label: "Log — audit only",
    tone: "bg-blue-50 text-blue-700 border-blue-200",
    dot: "bg-blue-500",
    help: "Detect patterns, write audit log, do NOT modify the prompt.",
  },
  STRICT: {
    label: "Strict — redact + audit",
    tone: "bg-emerald-50 text-emerald-700 border-emerald-200",
    dot: "bg-emerald-500",
    help: "Production BYOD. Redacts and audits every detected pattern.",
  },
  BLOCK: {
    label: "Block — fail closed",
    tone: "bg-red-50 text-red-700 border-red-200",
    dot: "bg-red-500",
    help: "Paranoid mode. Raises BYODLeakError if any pattern is detected.",
  },
};

const LEVEL_TONE: Record<string, string> = {
  low: "bg-emerald-50 text-emerald-700 border-emerald-200",
  medium: "bg-amber-50 text-amber-700 border-amber-200",
  high: "bg-orange-50 text-orange-700 border-orange-200",
  critical: "bg-red-50 text-red-700 border-red-200",
  non_critical: "bg-gray-50 text-gray-700 border-gray-200",
  vital: "bg-red-50 text-red-700 border-red-200",
  local: "bg-gray-50 text-gray-700 border-gray-200",
  national: "bg-amber-50 text-amber-700 border-amber-200",
  cross_border: "bg-orange-50 text-orange-700 border-orange-200",
  eu_wide: "bg-red-50 text-red-700 border-red-200",
};

export default function CompliancePage() {
  const [tab, setTab] = useState<"byod" | "dora">("byod");

  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface">
      {/* Header */}
      <header className="border-b border-ki-border bg-ki-surface-raised">
        <div className="px-6 py-4 max-w-6xl mx-auto">
          <div className="flex items-baseline gap-3">
            <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">DigitalTwinSim</span>
            <span className="text-ki-border-strong">/</span>
            <h1 className="text-[20px] font-medium tracking-tight2">Compliance</h1>
            <span className="text-ki-border-strong">/</span>
            <span className="font-data text-[11px] text-ki-on-surface-secondary">CISO &amp; CRO console</span>
            <Link href="/" className="ml-auto text-[12px] text-ki-on-surface-muted hover:text-ki-on-surface underline">
              ← Dashboard
            </Link>
          </div>
          <p className="text-[12px] text-ki-on-surface-muted mt-1 max-w-3xl">
            Two regulatory tracks. <strong>BYOD</strong>: zero customer-sensitive data leaves the
            process to the LLM provider — sanitizer + audit log per Reg. (EU) 2022/2554. <strong>DORA</strong>:
            Major Incident Report XML auto-generated from any completed wargame scenario per
            EBA/EIOPA/ESMA JC 2024-43 (Art. 19-20).
          </p>
        </div>
        <div className="px-6 max-w-6xl mx-auto flex gap-1">
          <TabBtn active={tab === "byod"} onClick={() => setTab("byod")}>
            BYOD enclave
          </TabBtn>
          <TabBtn active={tab === "dora"} onClick={() => setTab("dora")}>
            DORA export
          </TabBtn>
        </div>
      </header>

      <main className="px-6 py-6 max-w-6xl mx-auto">
        {tab === "byod" ? <ByodPanel /> : <DoraPanel />}
      </main>
    </div>
  );
}

function TabBtn({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-[12px] font-medium border-b-2 transition-colors ${
        active
          ? "border-ki-primary text-ki-on-surface"
          : "border-transparent text-ki-on-surface-muted hover:text-ki-on-surface"
      }`}
    >
      {children}
    </button>
  );
}

/* ───────────────────────────────────────────────────────────
   BYOD panel
   ─────────────────────────────────────────────────────────── */

function ByodPanel() {
  const [status, setStatus] = useState<ByodStatus | null>(null);
  const [audit, setAudit] = useState<AuditRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/compliance/byod/status`).then((r) => r.json()),
      fetch(`${API_BASE}/api/compliance/byod/audit?limit=50`).then((r) => r.json()),
    ])
      .then(([s, a]) => {
        setStatus(s);
        setAudit(a.rows ?? []);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="text-[12px] text-ki-on-surface-muted">Loading BYOD status<span className="cursor-blink">_</span></div>;
  }
  if (error || !status) {
    return <div className="text-[12px] text-ki-on-surface-muted">BYOD status unavailable: {error}</div>;
  }

  const tone = MODE_TONE[status.mode];
  const totalDetections = Object.values(status.by_category).reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-6">
      {/* Mode badge + summary */}
      <section className="border border-ki-border rounded p-5 bg-ki-surface-raised">
        <div className="flex items-start justify-between gap-6">
          <div>
            <div className="eyebrow text-ki-primary mb-2">Current mode</div>
            <span
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded font-mono text-[12px] font-semibold uppercase tracking-wider border ${tone.tone}`}
            >
              <span className={`w-2 h-2 rounded-full ${tone.dot}`} />
              {tone.label}
            </span>
            <p className="text-[12px] text-ki-on-surface-muted mt-2 max-w-md">{tone.help}</p>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            <Stat label="Audit rows" value={status.n_audit_rows.toLocaleString()} />
            <Stat label="Total detections" value={totalDetections.toLocaleString()} />
            <Stat label="Active sites" value={Object.keys(status.by_site).length.toString()} />
          </div>
        </div>
      </section>

      {/* Sanitizer playground */}
      <SanitizerPlayground />

      {/* By category breakdown */}
      <section className="border border-ki-border rounded p-5">
        <div className="eyebrow text-ki-primary mb-3">Detections by category</div>
        {Object.keys(status.by_category).length === 0 ? (
          <div className="text-[12px] text-ki-on-surface-muted">No detections yet.</div>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left border-b border-ki-border">
                <th className="py-1.5 font-medium text-ki-on-surface-muted">Category</th>
                <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(status.by_category)
                .sort((a, b) => b[1] - a[1])
                .map(([cat, count]) => (
                  <tr key={cat} className="border-b border-ki-border last:border-0">
                    <td className="py-1.5 font-data">{cat}</td>
                    <td className="py-1.5 text-right font-data tabular-nums">{count.toLocaleString()}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Audit log tail */}
      <section className="border border-ki-border rounded p-5">
        <div className="flex items-baseline justify-between mb-3">
          <div className="eyebrow text-ki-primary">Audit log (last 50)</div>
          <a
            href={`${API_BASE}/api/compliance/byod/audit?limit=1000`}
            target="_blank" rel="noopener"
            className="text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface underline"
          >
            Raw JSON
          </a>
        </div>
        {audit.length === 0 ? (
          <div className="text-[12px] text-ki-on-surface-muted">
            No audit rows yet. Run a simulation with <code>BYOD_MODE=STRICT</code> to populate.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-left border-b border-ki-border">
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Timestamp</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Site</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Mode</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Tenant</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Patterns</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">Δ chars</th>
                </tr>
              </thead>
              <tbody>
                {audit.slice().reverse().map((row, i) => (
                  <tr key={i} className="border-b border-ki-border last:border-0">
                    <td className="py-1.5 font-data text-ki-on-surface-muted whitespace-nowrap">{row.ts.replace("T", " ").replace("Z", "")}</td>
                    <td className="py-1.5 font-data text-ki-on-surface">{row.site}</td>
                    <td className="py-1.5 font-data">{row.mode}</td>
                    <td className="py-1.5 font-data text-ki-on-surface-muted">{row.tenant}</td>
                    <td className="py-1.5">
                      {row.patterns.map((p, j) => (
                        <span key={j} className="inline-block bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 mr-1 rounded text-[10px] font-data">
                          {p.category} ×{p.count}
                        </span>
                      ))}
                    </td>
                    <td className="py-1.5 text-right font-data tabular-nums">
                      {row.raw_chars - row.sanitized_chars}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ComplianceLink href="https://github.com/albertogerli/digital-twin-sim/blob/main/docs/BYOD_ARCHITECTURE.md">
        Full BYOD architecture documentation →
      </ComplianceLink>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────
   Sanitizer playground (interactive)
   ─────────────────────────────────────────────────────────── */

const DEFAULT_PROMPT =
  "The bank's LCR is 95%, deposit balance reached €12,500,000, with client-12345 issuing a complaint via IBAN IT60 X05428 11101 000000123456. CET1 of 11.5% triggered the alert. Euribor 3M at 2.4%.";

function SanitizerPlayground() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [mode, setMode] = useState<Mode>("STRICT");
  const [result, setResult] = useState<{ sanitized_text?: string; detections?: { category: string; count: number; samples: string[] }[]; modified?: boolean; blocked?: boolean; reason?: string } | null>(null);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/compliance/byod/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, mode }),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ reason: String(e) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="border border-ki-border rounded p-5">
      <div className="eyebrow text-ki-primary mb-3">Sanitizer playground</div>
      <p className="text-[11px] text-ki-on-surface-muted mb-3">
        Paste any prompt and see what the BYOD sanitizer would do under each mode. Does NOT
        write to the production audit log.
      </p>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        className="w-full h-24 text-[12px] font-data border border-ki-border rounded p-2 bg-ki-surface-sunken"
      />
      <div className="flex items-center gap-2 mt-2">
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
          className="border border-ki-border rounded px-2 py-1 text-[12px]"
        >
          <option value="LOG">LOG (detect only)</option>
          <option value="STRICT">STRICT (redact + audit)</option>
          <option value="BLOCK">BLOCK (raise on leak)</option>
        </select>
        <button
          onClick={run}
          disabled={loading}
          className="bg-ki-primary text-white text-[12px] font-medium px-3 py-1 rounded hover:bg-ki-primary/90 disabled:opacity-50"
        >
          {loading ? "Sanitizing…" : "Sanitize"}
        </button>
      </div>

      {result && (
        <div className="mt-4 grid md:grid-cols-2 gap-3">
          <div>
            <div className="eyebrow mb-1">
              {result.blocked ? "BLOCKED" : "Output prompt"}
            </div>
            <pre className="text-[11px] font-data whitespace-pre-wrap border border-ki-border rounded p-2 bg-ki-surface-sunken min-h-[80px]">
              {result.blocked
                ? result.reason
                : result.sanitized_text || prompt}
            </pre>
          </div>
          <div>
            <div className="eyebrow mb-1">Detections</div>
            {(result.detections?.length ?? 0) === 0 ? (
              <div className="text-[12px] text-emerald-700">✓ No sensitive patterns detected.</div>
            ) : (
              <ul className="space-y-1">
                {result.detections?.map((d, i) => (
                  <li key={i} className="text-[11px] flex items-center justify-between border-b border-ki-border py-1">
                    <span className="font-data">{d.category}</span>
                    <span className="text-ki-on-surface-muted">×{d.count}</span>
                    <span className="font-data text-ki-on-surface-muted text-right truncate max-w-[180px]" title={d.samples.join(" | ")}>
                      {d.samples?.[0]}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

/* ───────────────────────────────────────────────────────────
   DORA panel
   ─────────────────────────────────────────────────────────── */

function DoraPanel() {
  const [scenarios, setScenarios] = useState<ScenarioListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [preview, setPreview] = useState<DoraPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/simulations`)
      .then((r) => r.json())
      .then((data) => {
        const list: ScenarioListItem[] = (data.simulations || data || [])
          .filter((s: any) => s.status === "completed");
        setScenarios(list);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setPreview(null);
    setPreviewLoading(true);
    fetch(`${API_BASE}/api/compliance/dora/preview/${selectedId}`)
      .then((r) => r.json())
      .then((data) => setPreview(data))
      .catch((e) => setError(String(e)))
      .finally(() => setPreviewLoading(false));
  }, [selectedId]);

  if (loading) {
    return <div className="text-[12px] text-ki-on-surface-muted">Loading scenarios<span className="cursor-blink">_</span></div>;
  }
  if (error) {
    return <div className="text-[12px] text-ki-on-surface-muted">{error}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-[280px_1fr] gap-6">
        {/* Scenario picker */}
        <aside className="border border-ki-border rounded">
          <div className="eyebrow text-ki-primary px-3 py-2 border-b border-ki-border">
            Completed simulations ({scenarios.length})
          </div>
          {scenarios.length === 0 ? (
            <div className="p-3 text-[12px] text-ki-on-surface-muted">
              No completed scenarios yet. Run a simulation first, then return here.
            </div>
          ) : (
            <ul className="max-h-[60vh] overflow-y-auto">
              {scenarios.map((s) => (
                <li key={s.sim_id}>
                  <button
                    onClick={() => setSelectedId(s.sim_id)}
                    className={`w-full text-left px-3 py-2 text-[12px] border-b border-ki-border last:border-0 hover:bg-ki-surface-sunken transition-colors ${
                      selectedId === s.sim_id ? "bg-ki-surface-sunken" : ""
                    }`}
                  >
                    <div className="font-medium truncate">{s.name || s.sim_id}</div>
                    <div className="text-[10px] font-data text-ki-on-surface-muted">
                      {s.domain || "—"} · {s.rounds || "—"}r
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        {/* Right pane: preview + export */}
        <div>
          {!selectedId ? (
            <div className="border border-ki-border rounded p-6 bg-ki-surface-sunken text-[12px] text-ki-on-surface-muted">
              Pick a completed simulation on the left to preview its DORA Major Incident Report
              classification (per RTS Annex I, 7-criterion grid) and download the XML wire format.
            </div>
          ) : previewLoading ? (
            <div className="text-[12px] text-ki-on-surface-muted">Computing classification<span className="cursor-blink">_</span></div>
          ) : preview ? (
            <DoraPreviewCard preview={preview} simId={selectedId} />
          ) : null}
        </div>
      </div>

      <ComplianceLink href="https://github.com/albertogerli/digital-twin-sim/blob/main/docs/DORA_EXPORT_SCOPE.md">
        DORA export scope &amp; honest limits documentation →
      </ComplianceLink>
    </div>
  );
}

function DoraPreviewCard({ preview, simId }: { preview: DoraPreview; simId: string }) {
  const c = preview.classification;
  const tones = LEVEL_TONE;
  return (
    <div className="border border-ki-border rounded">
      <div className="px-5 py-4 border-b border-ki-border flex items-baseline justify-between">
        <div>
          <div className="eyebrow text-ki-primary mb-1">Classification preview</div>
          <h3 className="text-[15px] font-medium">{preview.scenario_name}</h3>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono font-semibold uppercase border ${
            preview.is_major
              ? "bg-red-50 text-red-700 border-red-200"
              : "bg-emerald-50 text-emerald-700 border-emerald-200"
          }`}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${preview.is_major ? "bg-red-500" : "bg-emerald-500"}`} />
          {preview.is_major ? "Major incident — reportable" : "Not major"}
        </span>
      </div>

      <div className="p-5 grid sm:grid-cols-2 lg:grid-cols-4 gap-3 text-[11px]">
        <ClassCell label="Clients affected" value={c.clients_affected} tones={tones} />
        <ClassCell label="Data losses" value={c.data_losses} tones={tones} />
        <ClassCell label="Reputational impact" value={c.reputational_impact} tones={tones} />
        <ClassCell label="Downtime" value={`${c.duration_downtime_hours.toFixed(1)} h`} tones={tones} numeric />
        <ClassCell label="Geographical spread" value={c.geographical_spread} tones={tones} />
        <ClassCell label="Economic impact" value={c.economic_impact_eur_band} tones={tones} numeric />
        <ClassCell label="Service criticality" value={c.criticality_of_services_affected} tones={tones} />
      </div>

      <div className="px-5 pb-5">
        <div className="eyebrow mb-2">Source metrics from simulation</div>
        <pre className="text-[11px] font-data bg-ki-surface-sunken border border-ki-border rounded p-2 overflow-auto">
          {JSON.stringify(preview.metrics_used || {}, null, 2)}
        </pre>
      </div>

      <div className="px-5 pb-5 flex gap-2">
        <a
          href={`${API_BASE}/api/compliance/dora/export/${simId}`}
          download
          className="bg-ki-primary text-white text-[12px] font-medium px-3 py-1.5 rounded hover:bg-ki-primary/90"
        >
          Download XML
        </a>
        <a
          href={`${API_BASE}/api/compliance/dora/export/${simId}`}
          target="_blank" rel="noopener"
          className="border border-ki-border text-[12px] px-3 py-1.5 rounded hover:bg-ki-surface-sunken"
        >
          View in browser
        </a>
      </div>
    </div>
  );
}

function ClassCell({ label, value, tones, numeric }: { label: string; value: string; tones: Record<string, string>; numeric?: boolean }) {
  const tone = tones[value] || "bg-gray-50 text-gray-700 border-gray-200";
  return (
    <div className="border border-ki-border rounded p-3">
      <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted mb-1">{label}</div>
      {numeric ? (
        <div className="font-data tabular text-[13px] font-medium">{value}</div>
      ) : (
        <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider border ${tone}`}>
          {value}
        </span>
      )}
    </div>
  );
}

/* ───────────────────────────────────────────────────────────
   Shared bits
   ─────────────────────────────────────────────────────────── */

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-ki-border rounded p-2 min-w-[88px]">
      <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">{label}</div>
      <div className="font-data tabular text-[16px] font-medium">{value}</div>
    </div>
  );
}

function ComplianceLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <div className="text-[11px] text-ki-on-surface-muted">
      <a href={href} target="_blank" rel="noopener" className="underline hover:text-ki-on-surface">
        {children}
      </a>
    </div>
  );
}
