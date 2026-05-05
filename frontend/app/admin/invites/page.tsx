"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

/* ───────────────────────────────────────────────────────────
   Admin · Invites — generate shareable login links + usage stats.
   Gated by middleware (any logged-in admin can access).
   Stateless invites: tokens are HMAC-signed; no server-side
   storage of issued invites. Usage stats come from two sources:
     1. Backend /api/admin/invites/stats aggregates the
        outputs/invite_redemptions.jsonl log + simulations table
        per tenant_id (= invite sub).
     2. The current-session "Generated" list below — in-memory,
        clears on refresh.
   ─────────────────────────────────────────────────────────── */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface CreatedInvite {
  url: string;
  label: string;
  expiresAt: string;
  sub: string;
  createdAt: string;
}

interface InviteUserStat {
  sub: string;
  label: string;
  first_redeemed: string | null;
  last_redeemed: string | null;
  redemption_count: number;
  sim_count: number;
  last_sim_at: string | null;
  total_cost: number;
  sim_status_breakdown: Record<string, number>;
}

interface InviteStats {
  total_redemptions: number;
  unique_invitees: number;
  redemptions_last_7d: number;
  redemptions_last_30d: number;
  users: InviteUserStat[];
  users_total: number;
}

const EXPIRY_OPTIONS = [
  { days: 1,  label: "1 day" },
  { days: 7,  label: "7 days" },
  { days: 30, label: "30 days" },
  { days: 90, label: "90 days" },
];

export default function AdminInvitesPage() {
  const [label, setLabel] = useState("");
  const [days, setDays] = useState(7);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [created, setCreated] = useState<CreatedInvite[]>([]);
  const [copiedSub, setCopiedSub] = useState<string | null>(null);
  const [stats, setStats] = useState<InviteStats | null>(null);
  const [statsErr, setStatsErr] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const res = await fetch("/api/auth/invite/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: label.trim(), expiresInDays: days }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data?.error || `HTTP ${res.status}`);
      } else {
        const data = await res.json();
        const invite: CreatedInvite = {
          url: data.url,
          label: data.label,
          expiresAt: data.expiresAt,
          sub: data.sub,
          createdAt: new Date().toISOString(),
        };
        setCreated((prev) => [invite, ...prev]);
        setLabel("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
    } finally {
      setSubmitting(false);
    }
  }

  async function copy(url: string, sub: string) {
    try {
      await navigator.clipboard.writeText(url);
      setCopiedSub(sub);
      setTimeout(() => setCopiedSub((s) => (s === sub ? null : s)), 1600);
    } catch {
      /* ignore — older browsers */
    }
  }

  const fmtExpiry = (iso: string) => {
    try {
      const d = new Date(iso);
      const now = Date.now();
      const ms = d.getTime() - now;
      const dys = Math.round(ms / (1000 * 60 * 60 * 24));
      if (dys <= 0) return "expired";
      if (dys === 1) return "in 1 day";
      return `in ${dys} days`;
    } catch {
      return iso;
    }
  };

  const fmtAbsolute = (iso: string | null) => {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      const ms = Date.now() - d.getTime();
      const h = Math.round(ms / 3600000);
      if (h < 1) return "<1h ago";
      if (h < 24) return `${h}h ago`;
      const days = Math.round(h / 24);
      if (days < 30) return `${days}d ago`;
      return d.toISOString().slice(0, 10);
    } catch { return iso; }
  };

  // Load usage stats on mount + poll every 20s so a redemption from an
  // invitee shows up without a hard refresh.
  useEffect(() => {
    let cancelled = false;
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/admin/invites/stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) {
          setStats(data);
          setStatsErr("");
        }
      } catch (e) {
        if (!cancelled) setStatsErr(e instanceof Error ? e.message : String(e));
      }
    };
    fetchStats();
    const id = setInterval(fetchStats, 20000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <div className="p-5 space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <div className="eyebrow">Admin</div>
        <h1 className="text-[20px] font-medium tracking-tight2 text-ki-on-surface mt-1">
          Invite links
        </h1>
        <p className="text-[12px] text-ki-on-surface-secondary mt-1 leading-relaxed max-w-xl">
          Generate a shareable URL that opens a 7-day session as soon as the
          recipient clicks it. No password required on their side.
        </p>
      </div>

      {/* Generator form */}
      <form onSubmit={handleSubmit} className="bg-ki-surface-raised border border-ki-border rounded p-4 flex flex-col gap-3">
        <div className="flex items-end gap-3">
          <div className="flex flex-col gap-1.5 flex-1 min-w-0">
            <label className="eyebrow">Label</label>
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="es. Marco Rossi · Banca Sella demo"
              maxLength={80}
              className="h-9 px-3 bg-ki-surface-sunken border border-ki-border rounded text-[13px] text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30"
            />
            <span className="text-[11px] text-ki-on-surface-muted">Optional · only for your own tracking; appears in welcome message.</span>
          </div>

          <div className="flex flex-col gap-1.5 w-[140px]">
            <label className="eyebrow">Expires</label>
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="h-9 px-2 bg-ki-surface-sunken border border-ki-border rounded text-[13px] text-ki-on-surface focus:outline-none focus:border-ki-primary"
            >
              {EXPIRY_OPTIONS.map((o) => (
                <option key={o.days} value={o.days}>{o.label}</option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="inline-flex items-center justify-center gap-1.5 h-9 px-4 rounded-sm bg-ki-on-surface text-ki-surface text-[13px] font-medium hover:bg-ki-on-surface-secondary disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? (
              <>
                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Generating…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>
                  add_link
                </span>
                Generate invite
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="bg-ki-error-soft border border-ki-error/30 rounded p-2 text-[12px] text-ki-error">
            {error}
          </div>
        )}
      </form>

      {/* ── Usage statistics across ALL invitees ─────────────────── */}
      <section className="bg-ki-surface-raised border border-ki-border rounded p-4">
        <div className="flex items-baseline justify-between mb-3">
          <div className="eyebrow text-ki-primary">Usage statistics</div>
          {stats && (
            <span className="font-data text-[11px] text-ki-on-surface-muted">
              {stats.users_total} invitees tracked
            </span>
          )}
        </div>
        {statsErr && (
          <div className="bg-ki-error-soft border border-ki-error/30 rounded p-2 text-[11px] text-ki-error mb-3">
            Stats unavailable: {statsErr}
          </div>
        )}
        {!stats ? (
          <div className="text-[12px] text-ki-on-surface-muted">Loading statistics<span className="cursor-blink">_</span></div>
        ) : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
              <div className="bg-ki-surface-sunken border border-ki-border rounded-sm p-2.5">
                <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">Redemptions</div>
                <div className="font-data tabular text-[18px] font-medium text-ki-on-surface mt-0.5">
                  {stats.total_redemptions.toLocaleString()}
                </div>
                <div className="text-[10px] text-ki-on-surface-muted mt-0.5">
                  {stats.unique_invitees} unique
                </div>
              </div>
              <div className="bg-ki-surface-sunken border border-ki-border rounded-sm p-2.5">
                <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">Last 7 days</div>
                <div className="font-data tabular text-[18px] font-medium text-ki-on-surface mt-0.5">
                  {stats.redemptions_last_7d.toLocaleString()}
                </div>
                <div className="text-[10px] text-ki-on-surface-muted mt-0.5">redemptions</div>
              </div>
              <div className="bg-ki-surface-sunken border border-ki-border rounded-sm p-2.5">
                <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">Last 30 days</div>
                <div className="font-data tabular text-[18px] font-medium text-ki-on-surface mt-0.5">
                  {stats.redemptions_last_30d.toLocaleString()}
                </div>
                <div className="text-[10px] text-ki-on-surface-muted mt-0.5">redemptions</div>
              </div>
              <div className="bg-ki-surface-sunken border border-ki-border rounded-sm p-2.5">
                <div className="text-[10px] uppercase tracking-wider text-ki-on-surface-muted">Total sims run</div>
                <div className="font-data tabular text-[18px] font-medium text-ki-on-surface mt-0.5">
                  {stats.users.reduce((a, u) => a + u.sim_count, 0).toLocaleString()}
                </div>
                <div className="text-[10px] text-ki-on-surface-muted mt-0.5">
                  ${stats.users.reduce((a, u) => a + u.total_cost, 0).toFixed(3)} spent
                </div>
              </div>
            </div>

            {stats.users.length === 0 ? (
              <div className="text-[12px] text-ki-on-surface-muted py-3 text-center">
                Nessun invito ancora utilizzato. Genera un link e condividilo: i numeri si popoleranno appena qualcuno lo apre.
              </div>
            ) : (
              <div className="overflow-x-auto -mx-4">
                <table className="w-full text-[11px]">
                  <thead className="bg-ki-surface-sunken">
                    <tr className="text-left border-y border-ki-border">
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Invitato</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted">ID</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">Click</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Primo</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Ultimo</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">Sim</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Stato sim</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted">Ultima attività</th>
                      <th className="px-3 py-2 font-medium text-ki-on-surface-muted text-right">Costo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.users.map((u) => {
                      const recentlyActive = u.last_sim_at && (Date.now() - new Date(u.last_sim_at).getTime() < 7 * 24 * 3600000);
                      return (
                        <tr key={u.sub} className="border-b border-ki-border-faint hover:bg-ki-surface-hover">
                          <td className="px-3 py-2 truncate max-w-[200px]" title={u.label}>
                            {u.label || <span className="text-ki-on-surface-muted italic">— (no label)</span>}
                          </td>
                          <td className="px-3 py-2 font-data text-[10px] text-ki-on-surface-muted">{u.sub}</td>
                          <td className="px-3 py-2 text-right font-data tabular">
                            {u.redemption_count > 0 ? u.redemption_count : "—"}
                          </td>
                          <td className="px-3 py-2 font-data text-[10px] text-ki-on-surface-muted whitespace-nowrap">
                            {fmtAbsolute(u.first_redeemed)}
                          </td>
                          <td className="px-3 py-2 font-data text-[10px] text-ki-on-surface-muted whitespace-nowrap">
                            {fmtAbsolute(u.last_redeemed)}
                          </td>
                          <td className="px-3 py-2 text-right font-data tabular">
                            {u.sim_count > 0 ? (
                              <span className="text-ki-on-surface font-medium">{u.sim_count}</span>
                            ) : <span className="text-ki-on-surface-muted">0</span>}
                          </td>
                          <td className="px-3 py-2">
                            {Object.keys(u.sim_status_breakdown).length === 0 ? (
                              <span className="text-ki-on-surface-muted">—</span>
                            ) : (
                              <span className="flex flex-wrap gap-1">
                                {Object.entries(u.sim_status_breakdown).map(([s, c]) => {
                                  const tone = s === "completed" ? "bg-ki-success-soft text-ki-success border-ki-success/30"
                                    : s === "failed" ? "bg-ki-error-soft text-ki-error border-ki-error/30"
                                    : "bg-ki-surface-sunken text-ki-on-surface-muted border-ki-border";
                                  return (
                                    <span key={s} className={`inline-block px-1.5 rounded-sm border text-[9px] font-data ${tone}`}>
                                      {s} {c}
                                    </span>
                                  );
                                })}
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2 font-data text-[10px] whitespace-nowrap">
                            {recentlyActive ? (
                              <span className="inline-flex items-center gap-1 text-ki-success">
                                <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
                                {fmtAbsolute(u.last_sim_at)}
                              </span>
                            ) : (
                              <span className="text-ki-on-surface-muted">{fmtAbsolute(u.last_sim_at)}</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right font-data tabular text-ki-on-surface-muted">
                            ${u.total_cost.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <p className="text-[10px] text-ki-on-surface-muted mt-3 leading-relaxed">
              Aggregato da <code className="font-data">outputs/invite_redemptions.jsonl</code> (click → backend log)
              + <code className="font-data">simulations</code> table (per <code>tenant_id</code>). Persiste tra
              redeploy se il volume Railway è montato su <code>/app/outputs</code>. Auto-refresh ogni 20s.
            </p>
          </>
        )}
      </section>

      {/* List of generated invites (this session) */}
      {created.length > 0 ? (
        <div>
          <div className="flex items-baseline justify-between mb-2">
            <div className="eyebrow">Generated this session</div>
            <span className="font-data tabular text-[11px] text-ki-on-surface-muted">{created.length}</span>
          </div>
          <div className="bg-ki-surface-raised border border-ki-border rounded overflow-hidden">
            {created.map((inv, i) => (
              <div
                key={inv.sub}
                className={`flex items-center gap-3 px-3 py-2.5 ${
                  i < created.length - 1 ? "border-b border-ki-border-faint" : ""
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[13px] font-medium text-ki-on-surface truncate">
                      {inv.label || <span className="text-ki-on-surface-muted">— (no label)</span>}
                    </span>
                    <span className="font-data tabular text-[10px] text-ki-on-surface-muted">{inv.sub}</span>
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="font-data tabular text-[11px] text-ki-on-surface-muted">
                      Expires {fmtExpiry(inv.expiresAt)}
                    </span>
                    <span className="text-ki-border-strong">·</span>
                    <span className="font-data tabular text-[11px] text-ki-on-surface-muted truncate">
                      {inv.url}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => copy(inv.url, inv.sub)}
                  className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-sm border text-[11px] transition-colors flex-shrink-0 ${
                    copiedSub === inv.sub
                      ? "bg-ki-success-soft border-ki-success text-ki-success"
                      : "border-ki-border text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface"
                  }`}
                >
                  <span className="material-symbols-outlined text-[13px]" style={{ fontVariationSettings: "'wght' 400" }}>
                    {copiedSub === inv.sub ? "check" : "content_copy"}
                  </span>
                  {copiedSub === inv.sub ? "Copied" : "Copy link"}
                </button>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-ki-on-surface-muted mt-2">
            Note · this list is in-memory and clears on refresh. The URLs themselves stay valid until they expire.
          </p>
        </div>
      ) : (
        <div className="bg-ki-surface-raised border border-dashed border-ki-border rounded p-6 text-center">
          <span className="text-[12px] text-ki-on-surface-muted">
            No invites generated yet — fill the form above and click <strong>Generate invite</strong>.
          </span>
        </div>
      )}

      {/* Info box */}
      <div className="bg-ki-primary-soft rounded p-3 flex items-start gap-2">
        <span className="material-symbols-outlined text-[14px] text-ki-primary mt-0.5" style={{ fontVariationSettings: "'wght' 500" }}>
          info
        </span>
        <div className="text-[12px] text-ki-on-surface-secondary leading-relaxed">
          <p>
            <span className="text-ki-primary font-medium">How it works.</span>{" "}
            The link contains an HMAC-signed token. When the recipient opens it,
            the server validates the signature and TTL, then mints a 7-day
            session cookie. They never see a password screen.
          </p>
          <p className="mt-1.5">
            <span className="text-ki-primary font-medium">Limits.</span>{" "}
            Tokens are stateless: <em>multi-use until expiry</em>, can&apos;t be
            revoked individually. To kill every active invite + session at
            once, rotate <code className="font-data text-[11px] bg-ki-surface-sunken px-1 rounded">DTS_AUTH_SECRET</code>.
          </p>
        </div>
      </div>

      <Link
        href="/settings"
        className="inline-flex items-center gap-1.5 text-[12px] text-ki-on-surface-muted hover:text-ki-on-surface transition-colors"
      >
        ← Settings
      </Link>
    </div>
  );
}
