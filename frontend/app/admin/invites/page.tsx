"use client";

import { useState } from "react";
import Link from "next/link";

/* ───────────────────────────────────────────────────────────
   Admin · Invites — generate shareable login links.
   Gated by middleware (any logged-in user can access).
   Stateless: tokens are HMAC-signed; no server-side storage,
   so we can't enforce single-use or revoke individual invites
   without rotating DTS_AUTH_SECRET (which kills ALL sessions).
   ─────────────────────────────────────────────────────────── */

interface CreatedInvite {
  url: string;
  label: string;
  expiresAt: string;
  sub: string;
  createdAt: string;
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

  return (
    <div className="p-5 space-y-6 max-w-3xl">
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
