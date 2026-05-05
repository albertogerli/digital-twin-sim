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
  metrics_used: Record<string, any>;
}

interface EconomicImpactBreakdown {
  point_eur: number;
  low_eur: number;
  high_eur: number;
  selected_method: "anchor" | "ticker";
  detected_category: string | null;
  category_scores: Record<string, number>;
  anchor_estimate: {
    point_eur: number;
    low_eur: number;
    high_eur: number;
    inputs: {
      total_shock_units: number;
      alpha_eur_per_unit: number;
      sigma_residual_eur: number;
      r2_anchor_fit: number;
      n_reference_incidents: number;
      calibration_scope: string;
      requested_category?: string | null;
      fallback_overall_alpha_eur_per_unit?: number | null;
    };
    formula: string;
  };
  ticker_estimate: {
    point_eur: number;
    inputs: {
      tickers_priced: number;
      tickers_unknown: number;
      direct_loss_eur: number;
      contagion_multiplier: number;
      per_ticker?: { ticker: string; cum_pct: number; mcap_eur_m: number; loss_eur_m: number }[];
    };
    formula: string;
  };
  calibration_notes: string;
}

interface ScenarioListItem {
  // Backend /api/simulations returns id/scenario_name/total_rounds, NOT
  // sim_id/name/rounds. The earlier mismatch caused empty rows in the
  // DORA picker ("financial · -r" with no title and an undefined key).
  id: string;
  scenario_name?: string;
  status: string;
  domain?: string;
  total_rounds?: number;
  brief?: string;
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
  const [tab, setTab] = useState<"byod" | "dora" | "calibration" | "dora_validation">("byod");

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
            Three regulatory tracks. <strong>BYOD</strong>: zero customer-sensitive data leaves the
            process to the LLM provider. <strong>DORA</strong>: Major Incident Report XML
            auto-generated from any completed wargame scenario.
            <strong> Self-calibration</strong>: nightly shadow forecasts scored against realised
            yfinance returns — the data network effect that compounds with operating time.
          </p>
        </div>
        <div className="px-6 max-w-6xl mx-auto flex gap-1">
          <TabBtn active={tab === "byod"} onClick={() => setTab("byod")}>
            BYOD enclave
          </TabBtn>
          <TabBtn active={tab === "dora"} onClick={() => setTab("dora")}>
            DORA export
          </TabBtn>
          <TabBtn active={tab === "dora_validation"} onClick={() => setTab("dora_validation")}>
            DORA validation
          </TabBtn>
          <TabBtn active={tab === "calibration"} onClick={() => setTab("calibration")}>
            Self-calibration
          </TabBtn>
        </div>
      </header>

      <main className="px-6 py-6 max-w-6xl mx-auto">
        {tab === "byod" ? <ByodPanel /> :
         tab === "dora" ? <DoraPanel /> :
         tab === "dora_validation" ? <DoraValidationPanel /> :
         <CalibrationPanel />}
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
              {scenarios.map((s) => {
                const title = s.scenario_name || (s.brief ? s.brief.slice(0, 60) + "…" : s.id);
                return (
                  <li key={s.id}>
                    <button
                      onClick={() => setSelectedId(s.id)}
                      className={`w-full text-left px-3 py-2 text-[12px] border-b border-ki-border last:border-0 hover:bg-ki-surface-sunken transition-colors ${
                        selectedId === s.id ? "bg-ki-surface-sunken" : ""
                      }`}
                    >
                      <div className="font-medium truncate" title={title}>{title}</div>
                      <div className="text-[10px] font-data text-ki-on-surface-muted">
                        {s.domain || "—"} · {s.total_rounds ? `${s.total_rounds}r` : "—"} · <span className="text-ki-on-surface-muted">{s.id.slice(0, 6)}</span>
                      </div>
                    </button>
                  </li>
                );
              })}
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
  const breakdown: EconomicImpactBreakdown | null =
    (preview.metrics_used?.economic_impact_breakdown as EconomicImpactBreakdown) || null;
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

      {/* ── Hero KPI · Economic impact (the most-loaded number) ─────── */}
      {breakdown && (
        <EconomicImpactHero breakdown={breakdown} />
      )}

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
        <pre className="text-[11px] font-data bg-ki-surface-sunken border border-ki-border rounded p-2 overflow-x-auto max-h-64 whitespace-pre-wrap break-words">
          {JSON.stringify(
            // Drop the breakdown — already rendered in the hero;
            // including it here was bloating the pre block to ~80 lines.
            Object.fromEntries(
              Object.entries(preview.metrics_used || {})
                .filter(([k]) => k !== "economic_impact_breakdown")
            ),
            null,
            2,
          )}
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
          href={`${API_BASE}/api/compliance/dora/export/${simId}?inline=1`}
          target="_blank" rel="noopener"
          className="border border-ki-border text-[12px] px-3 py-1.5 rounded hover:bg-ki-surface-sunken"
        >
          View in browser
        </a>
      </div>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────
   DORA Validation panel — leave-one-out CV + OLS vs Huber
   ─────────────────────────────────────────────────────────── */

interface BacktestResult {
  id: string;
  label: string;
  category: string;
  shock_units: number;
  actual_eur_m: number;
  predicted_eur_m: number;
  error_eur_m: number;
  error_pct: number;
  within_50pct: boolean;
  within_100pct: boolean;
  within_200pct: boolean;
}

interface BacktestPayload {
  status: string;
  method: string;
  category_filter: string;
  n_total: number;
  mae_eur_m: number;
  rmse_eur_m: number;
  median_abs_pct_error: number;
  hit_rate_within_50pct: number;
  hit_rate_within_100pct: number;
  hit_rate_within_200pct: number;
  results: BacktestResult[];
}

interface DiagnosticsPayload {
  category: string;
  n_incidents: number;
  ols: { alpha_eur_m_per_unit: number; sigma_eur_m: number; r2: number };
  huber_robust: { alpha_eur_m_per_unit: number; sigma_eur_m: number; r2: number; epsilon: number };
  outliers_2sigma: { id: string; ols_residual_eur_m: number; outlier_z: number | null }[];
  alpha_drift_pct: number | null;
}

const DORA_CATEGORIES = ["overall", "banking_it", "banking_eu", "banking_us", "sovereign", "cyber", "telco", "energy"];

function DoraValidationPanel() {
  const [category, setCategory] = useState<string>("overall");
  const [diag, setDiag] = useState<DiagnosticsPayload | null>(null);
  const [bt, setBt] = useState<BacktestPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setDiag(null); setBt(null); setErr(null);
    const params = category === "overall" ? "" : `?category=${category}`;
    Promise.all([
      fetch(`${API_BASE}/api/compliance/dora/calibration/diagnostics${params}`).then(r => r.json()),
      fetch(`${API_BASE}/api/compliance/dora/backtest${params}`).then(r => r.json()),
    ])
      .then(([d, b]) => { setDiag(d); setBt(b); })
      .catch(e => setErr(String(e)));
  }, [category]);

  if (err) return <div className="text-[12px] text-ki-on-surface-muted">{err}</div>;
  if (!diag || !bt) {
    return <div className="text-[12px] text-ki-on-surface-muted">Loading validation<span className="cursor-blink">_</span></div>;
  }

  return (
    <div className="space-y-6">
      <section>
        <div className="eyebrow text-ki-primary mb-2">Validation slice</div>
        <p className="text-[12px] text-ki-on-surface-secondary mb-3 leading-relaxed max-w-3xl">
          Leave-one-out cross-validation on the {bt.n_total === 1 ? "incident" : `${bt.n_total} reference incidents`} per
          category — fit α on N-1, predict held-out, measure error. Plus OLS vs Huber robust regression
          comparison so a CRO can see the impact of outliers like Lehman 2008.
        </p>
        <div className="flex gap-1 flex-wrap">
          {DORA_CATEGORIES.map(c => (
            <button key={c}
              onClick={() => setCategory(c)}
              className={`px-2 py-1 text-[11px] rounded-sm border ${
                category === c
                  ? "bg-ki-on-surface text-ki-surface border-ki-on-surface"
                  : "border-ki-border text-ki-on-surface-secondary hover:bg-ki-surface-hover"
              }`}>
              {c.replace("_", " ")}
            </button>
          ))}
        </div>
      </section>

      {/* Diagnostics — OLS vs Huber */}
      <section className="border border-ki-border rounded p-5">
        <div className="eyebrow text-ki-primary mb-3">Calibration · OLS vs Huber robust</div>
        <div className="grid sm:grid-cols-2 gap-4">
          <div className="border border-ki-border rounded p-3">
            <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted mb-1">OLS no-intercept</div>
            <div className="font-data tabular text-[20px] font-medium text-ki-on-surface mt-0.5">
              €{(diag.ols.alpha_eur_m_per_unit / 1000).toFixed(2)}B<span className="text-[12px] text-ki-on-surface-muted">/unit</span>
            </div>
            <div className="text-[11px] text-ki-on-surface-muted mt-1">
              R² {diag.ols.r2.toFixed(3)} · σ €{(diag.ols.sigma_eur_m / 1000).toFixed(1)}B
            </div>
          </div>
          <div className="border border-ki-border rounded p-3">
            <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted mb-1">
              Huber robust (k=1.345)
              {diag.alpha_drift_pct !== null && (
                <span className={`ml-2 text-[10px] font-data ${diag.alpha_drift_pct < 0 ? "text-emerald-700" : "text-amber-700"}`}>
                  {diag.alpha_drift_pct >= 0 ? "+" : ""}{diag.alpha_drift_pct.toFixed(1)}%
                </span>
              )}
            </div>
            <div className="font-data tabular text-[20px] font-medium text-ki-on-surface mt-0.5">
              €{(diag.huber_robust.alpha_eur_m_per_unit / 1000).toFixed(2)}B<span className="text-[12px] text-ki-on-surface-muted">/unit</span>
            </div>
            <div className="text-[11px] text-ki-on-surface-muted mt-1">
              R² {diag.huber_robust.r2.toFixed(3)} · σ €{(diag.huber_robust.sigma_eur_m / 1000).toFixed(1)}B · used in production
            </div>
          </div>
        </div>
        {diag.outliers_2sigma.length > 0 && (
          <div className="mt-3">
            <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted mb-1.5">
              Outliers (|z| &gt; 2σ from OLS) · {diag.outliers_2sigma.length} {diag.outliers_2sigma.length === 1 ? "incident" : "incidents"}
            </div>
            <div className="grid sm:grid-cols-2 gap-1.5">
              {diag.outliers_2sigma.map(o => (
                <div key={o.id} className="text-[11px] font-data flex items-baseline justify-between bg-ki-surface-sunken border border-ki-border rounded-sm px-2 py-1">
                  <span className="text-ki-on-surface">{o.id}</span>
                  <span className="text-ki-on-surface-muted">z={o.outlier_z}</span>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-ki-on-surface-muted mt-2 leading-relaxed">
              Huber regression downweights these without dropping them — they still contribute, but not enough to drag
              α away from the bulk of the data.
            </p>
          </div>
        )}
      </section>

      {/* Backtest summary */}
      <section className="border border-ki-border rounded p-5">
        <div className="eyebrow text-ki-primary mb-3">Backtest · leave-one-out cross-validation</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <BTStat label="Hit ±50%" value={`${(bt.hit_rate_within_50pct * 100).toFixed(0)}%`} accent="primary" />
          <BTStat label="Hit ±100%" value={`${(bt.hit_rate_within_100pct * 100).toFixed(0)}%`} />
          <BTStat label="MAE" value={`€${(bt.mae_eur_m / 1000).toFixed(1)}B`} />
          <BTStat label="Median |%err|" value={`${bt.median_abs_pct_error.toFixed(0)}%`} />
        </div>
        <p className="text-[11px] text-ki-on-surface-muted mt-3 leading-relaxed">
          For each incident, α was re-fitted on the other {bt.n_total - 1} and used to predict the held-out cost.
          Method: <code className="font-data">{bt.method}</code> regression. Hit-rates are the share of out-of-sample
          predictions within the threshold.
        </p>
      </section>

      {/* Per-incident table */}
      <section className="border border-ki-border rounded">
        <div className="px-5 py-3 border-b border-ki-border flex items-baseline justify-between">
          <div className="eyebrow text-ki-primary">Per-incident predictions (worst → best)</div>
          <span className="font-data text-[11px] text-ki-on-surface-muted">{bt.results.length} rows</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead className="bg-ki-surface-sunken">
              <tr className="text-left border-b border-ki-border">
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Incident</th>
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Cat.</th>
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">Shock</th>
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">Actual €M</th>
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">Predicted €M</th>
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">% err</th>
                <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-center">Bands</th>
              </tr>
            </thead>
            <tbody>
              {bt.results.map((r, i) => {
                const tone = r.within_50pct ? "text-emerald-700" : r.within_100pct ? "text-amber-700" : "text-red-700";
                return (
                  <tr key={r.id} className={`border-b border-ki-border-faint last:border-0 ${i < 3 ? "bg-red-50/30" : ""}`}>
                    <td className="px-3 py-2 max-w-[320px]" title={r.label}>
                      <div className="text-ki-on-surface">{r.label}</div>
                      <div className="font-data text-[10px] text-ki-on-surface-muted">{r.id}</div>
                    </td>
                    <td className="px-3 py-2 font-data text-[10px] text-ki-on-surface-muted">{r.category}</td>
                    <td className="px-3 py-2 text-right font-data tabular">{r.shock_units.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right font-data tabular">{r.actual_eur_m.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right font-data tabular">{r.predicted_eur_m.toLocaleString()}</td>
                    <td className={`px-3 py-2 text-right font-data tabular font-medium ${tone}`}>
                      {r.error_pct >= 0 ? "+" : ""}{r.error_pct.toFixed(0)}%
                    </td>
                    <td className="px-3 py-2 text-center">
                      <span className={`inline-block w-3 h-3 rounded-full mr-1 ${r.within_50pct ? "bg-emerald-500" : "bg-ki-border-strong"}`} title="±50%" />
                      <span className={`inline-block w-3 h-3 rounded-full mr-1 ${r.within_100pct ? "bg-amber-500" : "bg-ki-border-strong"}`} title="±100%" />
                      <span className={`inline-block w-3 h-3 rounded-full ${r.within_200pct ? "bg-orange-500" : "bg-ki-border-strong"}`} title="±200%" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <ComplianceLink href="https://github.com/albertogerli/digital-twin-sim/blob/main/docs/DORA_ECONOMIC_IMPACT_PIPELINE.md">
        DORA economic-impact pipeline · full documentation →
      </ComplianceLink>
    </div>
  );
}

function BTStat({ label, value, accent }: { label: string; value: string; accent?: "primary" }) {
  return (
    <div className={`border rounded p-3 ${accent === "primary" ? "border-ki-primary/40 bg-ki-primary-soft" : "border-ki-border"}`}>
      <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">{label}</div>
      <div className="font-data tabular text-[20px] font-medium text-ki-on-surface mt-1">{value}</div>
    </div>
  );
}

function fmtEur(v: number): { num: string; suffix: string } {
  if (!Number.isFinite(v) || v <= 0) return { num: "0", suffix: "EUR" };
  if (v >= 1e9) return { num: (v / 1e9).toFixed(2), suffix: "B EUR" };
  if (v >= 1e6) return { num: (v / 1e6).toFixed(0), suffix: "M EUR" };
  if (v >= 1e3) return { num: (v / 1e3).toFixed(0), suffix: "K EUR" };
  return { num: v.toFixed(0), suffix: "EUR" };
}

function EconomicImpactHero({ breakdown }: { breakdown: EconomicImpactBreakdown }) {
  const [showMethods, setShowMethods] = useState(false);
  const point = fmtEur(breakdown.point_eur);
  const low = fmtEur(breakdown.low_eur);
  const high = fmtEur(breakdown.high_eur);
  // Tier colour based on point estimate magnitude
  const v = breakdown.point_eur;
  const tier = v >= 5e9 ? "critical" : v >= 1e9 ? "high" : v >= 100e6 ? "medium" : "low";
  const tierStyle: Record<string, { box: string; num: string; chip: string; label: string }> = {
    critical: { box: "bg-red-50 border-red-200", num: "text-red-700", chip: "bg-red-100 text-red-700 border-red-300", label: "≥5B · systemic / sovereign-class" },
    high:     { box: "bg-orange-50 border-orange-200", num: "text-orange-700", chip: "bg-orange-100 text-orange-700 border-orange-300", label: "1-5B · major bank or sector-wide" },
    medium:   { box: "bg-amber-50 border-amber-200", num: "text-amber-700", chip: "bg-amber-100 text-amber-700 border-amber-300", label: "100M-1B · single-firm / regional" },
    low:      { box: "bg-emerald-50 border-emerald-200", num: "text-emerald-700", chip: "bg-emerald-100 text-emerald-700 border-emerald-300", label: "<100M · contained / operational" },
  };
  const style = tierStyle[tier];
  const a = breakdown.anchor_estimate;
  const t = breakdown.ticker_estimate;
  return (
    <div className={`m-5 mb-0 p-4 border rounded-sm ${style.box}`}>
      <div className="flex items-baseline justify-between gap-3 mb-2">
        <div className="flex items-baseline gap-2 flex-wrap">
          <span className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted font-data">Economic impact estimate</span>
          <span className={`text-[9px] uppercase font-mono px-1.5 py-0.5 rounded-sm border ${style.chip}`}>{tier}</span>
          {breakdown.detected_category && (
            <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded-sm border bg-white/40 border-ki-border text-ki-on-surface-secondary" title="Category auto-detected from brief — Method A α conditioned on within-category incidents only">
              {breakdown.detected_category.replace("_", " ")}
            </span>
          )}
        </div>
        <button
          onClick={() => setShowMethods((v) => !v)}
          className="text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface underline decoration-dotted"
        >
          {showMethods ? "hide methodology" : "show methodology"}
        </button>
      </div>
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className={`font-data tabular text-[44px] leading-none font-medium ${style.num}`}>€{point.num}</span>
        <span className="font-data text-[14px] text-ki-on-surface-muted">{point.suffix}</span>
        <span className="ml-3 font-data text-[11px] text-ki-on-surface-muted">
          90% CI [€{low.num}{low.suffix.replace(" EUR","")} – €{high.num}{high.suffix.replace(" EUR","")}]
        </span>
      </div>
      <div className="mt-1 text-[11px] text-ki-on-surface-secondary">
        {style.label} · selected by <code className="font-data text-[10px] bg-white/40 px-1 rounded-sm">{breakdown.selected_method}</code> method
      </div>

      {showMethods && (
        <div className="mt-3 pt-3 border-t border-ki-border/40 space-y-3 text-[11px]">
          {/* Method A */}
          <div>
            <div className="font-data text-[10px] uppercase tracking-wider text-ki-on-surface-muted mb-1">
              Method A — calibrated shock anchor
              <span className="ml-2 text-[9px] normal-case text-ki-on-surface-muted">scope: {a.inputs.calibration_scope || "overall"}</span>
              {breakdown.selected_method === "anchor" && <span className="ml-2 text-[9px] text-ki-primary">(SELECTED)</span>}
            </div>
            <div className="text-ki-on-surface">
              <code className="font-data text-[10px] bg-white/40 px-1 rounded-sm">{a.formula}</code>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-1 mt-1 text-[10px] font-data text-ki-on-surface-secondary">
              <span>Σ shock: <span className="text-ki-on-surface">{a.inputs.total_shock_units.toFixed(3)}</span></span>
              <span>α: <span className="text-ki-on-surface">€{(a.inputs.alpha_eur_per_unit / 1e9).toFixed(2)}B/unit</span></span>
              <span>R²: <span className="text-ki-on-surface">{a.inputs.r2_anchor_fit}</span></span>
              <span>n incidents: <span className="text-ki-on-surface">{a.inputs.n_reference_incidents}</span></span>
            </div>
            {a.inputs.fallback_overall_alpha_eur_per_unit && (
              <div className="text-[10px] text-ki-on-surface-muted mt-1">
                Without category-conditioning, overall α would have been
                <span className="font-data ml-1">€{(a.inputs.fallback_overall_alpha_eur_per_unit / 1e9).toFixed(2)}B/unit</span>
                — using the tighter within-category fit.
              </div>
            )}
            <div className="font-data text-[11px] text-ki-on-surface mt-1">
              → <span className="font-medium">€{fmtEur(a.point_eur).num}{fmtEur(a.point_eur).suffix.replace(" EUR","")}</span>
            </div>
          </div>
          {/* Method B */}
          <div>
            <div className="font-data text-[10px] uppercase tracking-wider text-ki-on-surface-muted mb-1">
              Method B — ticker market-cap loss × contagion γ
              {breakdown.selected_method === "ticker" && <span className="ml-2 text-[9px] text-ki-primary">(SELECTED)</span>}
            </div>
            <div className="text-ki-on-surface">
              <code className="font-data text-[10px] bg-white/40 px-1 rounded-sm">{t.formula}</code>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-1 mt-1 text-[10px] font-data text-ki-on-surface-secondary">
              <span>Tickers priced: <span className="text-ki-on-surface">{t.inputs.tickers_priced}</span></span>
              <span>Unknown mcap: <span className="text-ki-on-surface">{t.inputs.tickers_unknown}</span></span>
              <span>γ contagion: <span className="text-ki-on-surface">{t.inputs.contagion_multiplier}×</span></span>
              <span>Direct loss: <span className="text-ki-on-surface">€{fmtEur(t.inputs.direct_loss_eur).num}{fmtEur(t.inputs.direct_loss_eur).suffix.replace(" EUR","")}</span></span>
            </div>
            {t.inputs.per_ticker && t.inputs.per_ticker.length > 0 && (
              <div className="mt-1.5 grid grid-cols-1 sm:grid-cols-2 gap-x-3 text-[10px] font-data">
                {t.inputs.per_ticker.slice(0, 6).map((row) => (
                  <div key={row.ticker} className="flex items-baseline gap-2 text-ki-on-surface-secondary">
                    <span className="w-14 text-ki-on-surface">{row.ticker}</span>
                    <span className={row.cum_pct < 0 ? "text-red-700" : "text-emerald-700"}>{row.cum_pct >= 0 ? "+" : ""}{row.cum_pct.toFixed(2)}%</span>
                    <span className="text-ki-on-surface-muted">× €{row.mcap_eur_m.toLocaleString()}M</span>
                    <span className="ml-auto text-ki-on-surface">€{row.loss_eur_m.toLocaleString()}M</span>
                  </div>
                ))}
              </div>
            )}
            <div className="font-data text-[11px] text-ki-on-surface mt-1">
              → <span className="font-medium">€{fmtEur(t.point_eur).num}{fmtEur(t.point_eur).suffix.replace(" EUR","")}</span>
            </div>
          </div>
          {/* Notes */}
          <div className="text-[10px] text-ki-on-surface-muted leading-relaxed pt-1 border-t border-ki-border/40">
            {breakdown.calibration_notes}
          </div>
        </div>
      )}
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

/* ───────────────────────────────────────────────────────────
   Self-calibration panel
   ─────────────────────────────────────────────────────────── */

interface CalibrationSummary {
  n_forecasts: number;
  n_evaluations: number;
  last_forecast_date: string | null;
  last_evaluation_date: string | null;
  mae_t1_running: number | null;
  mae_t3_running: number | null;
  mae_t7_running: number | null;
  direction_acc_t1: number | null;
  by_ticker: Record<string, { mae_t1?: number; mae_t3?: number; mae_t7?: number }>;
}

interface CalibrationRow {
  forecast_date: string;
  ticker: string;
  horizon_days: number;
  predicted_pct: number;
  realized_pct: number;
  abs_error_pp: number;
  evaluated_at: string;
}

function CalibrationPanel() {
  const [summary, setSummary] = useState<CalibrationSummary | null>(null);
  const [rows, setRows] = useState<CalibrationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/compliance/calibration/summary`).then((r) => r.json()),
      fetch(`${API_BASE}/api/compliance/calibration/recent?limit=30`).then((r) => r.json()),
    ])
      .then(([s, r]) => {
        setSummary(s);
        setRows(r.rows ?? []);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="text-[12px] text-ki-on-surface-muted">Loading calibration history<span className="cursor-blink">_</span></div>;
  }
  if (error || !summary) {
    return <div className="text-[12px] text-ki-on-surface-muted">Calibration history unavailable: {error}</div>;
  }

  const empty = summary.n_forecasts === 0;

  return (
    <div className="space-y-6">
      {/* Top stats strip */}
      <section className="border border-ki-border rounded p-5 bg-ki-surface-raised">
        <div className="eyebrow text-ki-primary mb-3">Loop status</div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Stat label="Forecasts recorded" value={summary.n_forecasts.toLocaleString()} />
          <Stat label="Evaluations scored" value={summary.n_evaluations.toLocaleString()} />
          <Stat label="Last forecast" value={summary.last_forecast_date ?? "—"} />
          <Stat label="Last evaluation" value={summary.last_evaluation_date?.slice(0, 10) ?? "—"} />
        </div>
        {empty && (
          <p className="text-[11px] text-ki-on-surface-muted mt-3">
            No forecasts yet. Run on the host machine:&nbsp;
            <code>python scripts/continuous_calibration.py forecast</code>.
            T+7 evaluations require waiting 7 trading days after the first forecast.
          </p>
        )}
      </section>

      {/* Running MAE cards */}
      <section className="border border-ki-border rounded p-5">
        <div className="eyebrow text-ki-primary mb-3">Running mean absolute error vs realised yfinance returns</div>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <MaeCard label="MAE T+1" value={summary.mae_t1_running} />
          <MaeCard label="MAE T+3" value={summary.mae_t3_running} />
          <MaeCard label="MAE T+7" value={summary.mae_t7_running} />
          <DirCard label="Direction acc T+1" value={summary.direction_acc_t1} />
        </div>
      </section>

      {/* By ticker */}
      <section className="border border-ki-border rounded p-5">
        <div className="eyebrow text-ki-primary mb-3">By ticker</div>
        {Object.keys(summary.by_ticker).length === 0 ? (
          <div className="text-[12px] text-ki-on-surface-muted">No per-ticker history yet.</div>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left border-b border-ki-border">
                <th className="py-1.5 font-medium text-ki-on-surface-muted">Ticker</th>
                <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">MAE T+1</th>
                <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">MAE T+3</th>
                <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">MAE T+7</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary.by_ticker)
                .sort()
                .map(([tk, m]) => (
                  <tr key={tk} className="border-b border-ki-border last:border-0">
                    <td className="py-1.5 font-data">{tk}</td>
                    <td className="py-1.5 text-right font-data tabular-nums">{m.mae_t1 != null ? `${m.mae_t1.toFixed(2)}pp` : "—"}</td>
                    <td className="py-1.5 text-right font-data tabular-nums">{m.mae_t3 != null ? `${m.mae_t3.toFixed(2)}pp` : "—"}</td>
                    <td className="py-1.5 text-right font-data tabular-nums">{m.mae_t7 != null ? `${m.mae_t7.toFixed(2)}pp` : "—"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Recent evaluations table */}
      <section className="border border-ki-border rounded p-5">
        <div className="eyebrow text-ki-primary mb-3">Recent evaluations</div>
        {rows.length === 0 ? (
          <div className="text-[12px] text-ki-on-surface-muted">No evaluations recorded yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-left border-b border-ki-border">
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Forecast date</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Ticker</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Horizon</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">Predicted</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">Realised</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted text-right">|err|</th>
                  <th className="py-1.5 font-medium text-ki-on-surface-muted">Scored</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="border-b border-ki-border last:border-0">
                    <td className="py-1.5 font-data">{r.forecast_date}</td>
                    <td className="py-1.5 font-data">{r.ticker}</td>
                    <td className="py-1.5 font-data">T+{r.horizon_days}</td>
                    <td className={`py-1.5 text-right font-data tabular-nums ${r.predicted_pct >= 0 ? "text-emerald-700" : "text-red-700"}`}>
                      {r.predicted_pct >= 0 ? "+" : ""}{r.predicted_pct.toFixed(2)}pp
                    </td>
                    <td className={`py-1.5 text-right font-data tabular-nums ${r.realized_pct >= 0 ? "text-emerald-700" : "text-red-700"}`}>
                      {r.realized_pct >= 0 ? "+" : ""}{r.realized_pct.toFixed(2)}pp
                    </td>
                    <td className="py-1.5 text-right font-data tabular-nums">{r.abs_error_pp.toFixed(2)}pp</td>
                    <td className="py-1.5 font-data text-ki-on-surface-muted">{r.evaluated_at?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ComplianceLink href="https://github.com/albertogerli/digital-twin-sim/blob/main/scripts/continuous_calibration.py">
        scripts/continuous_calibration.py — CLI scheduler →
      </ComplianceLink>
    </div>
  );
}

function MaeCard({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="border border-ki-border rounded p-3">
      <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">{label}</div>
      <div className="font-data tabular text-[20px] font-medium mt-1">
        {value != null ? `${value.toFixed(2)} pp` : "—"}
      </div>
    </div>
  );
}

function DirCard({ label, value }: { label: string; value: number | null }) {
  const pct = value != null ? Math.round(value * 100) : null;
  const tone = pct == null
    ? "bg-gray-50 text-gray-700 border-gray-200"
    : pct >= 60
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : pct >= 50
        ? "bg-amber-50 text-amber-700 border-amber-200"
        : "bg-red-50 text-red-700 border-red-200";
  return (
    <div className={`border rounded p-3 ${tone}`}>
      <div className="text-[10px] uppercase tracking-wider opacity-80">{label}</div>
      <div className="font-data tabular text-[20px] font-medium mt-1">
        {pct != null ? `${pct}%` : "—"}
      </div>
    </div>
  );
}
