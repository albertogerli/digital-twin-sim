"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";

/* ───────────────────────────────────────────────────────────
   Login — single-password gate (Quiet Intelligence styling).
   POSTs to /api/auth/login; on success follows ?next= or "/".
   ─────────────────────────────────────────────────────────── */

export default function LoginPage() {
  // useSearchParams must live under a Suspense boundary so the page can
  // statically prerender (Next 14 SSR-bailout requirement).
  return (
    <Suspense fallback={<LoginShell><div className="text-[12px] text-ki-on-surface-muted">Loading…</div></LoginShell>}>
      <LoginForm />
    </Suspense>
  );
}

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next") || "/";

  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  // Focus the password field on mount
  useEffect(() => {
    const el = document.getElementById("dts-password");
    el?.focus();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!password.trim() || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        // Replace so the back button doesn't return to /login
        router.replace(next);
      } else {
        const data = await res.json().catch(() => ({}));
        setError(data?.error || `HTTP ${res.status}`);
        setSubmitting(false);
        setPassword("");
        const el = document.getElementById("dts-password");
        (el as HTMLInputElement | null)?.focus();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Network error");
      setSubmitting(false);
    }
  }

  return (
    <LoginShell>
      <form onSubmit={handleSubmit} className="bg-ki-surface-raised border border-ki-border rounded p-5 flex flex-col gap-3 shadow-tint">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="dts-password" className="eyebrow">Password</label>
            <input
              id="dts-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting}
              required
              className="h-9 px-3 bg-ki-surface-sunken border border-ki-border rounded text-[14px] text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30 disabled:opacity-60"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <div className="bg-ki-error-soft border border-ki-error/30 rounded p-2 text-[12px] text-ki-error">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !password.trim()}
            className="inline-flex items-center justify-center gap-1.5 h-9 px-3 rounded-sm bg-ki-on-surface text-ki-surface text-[13px] font-medium hover:bg-ki-on-surface-secondary disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? (
              <>
                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Verifying…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>
                  login
                </span>
                Sign in
              </>
            )}
          </button>

          <p className="text-[11px] text-ki-on-surface-muted text-center mt-1">
            Sessione 7 giorni · cookie HttpOnly firmato HMAC-SHA256
          </p>
        </form>
    </LoginShell>
  );
}

function LoginShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface flex items-center justify-center p-6">
      <div className="w-full max-w-[360px]">
        {/* Brand mark */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-10 h-10 rounded-md bg-ki-on-surface text-ki-surface grid place-items-center font-data text-[14px] font-semibold tracking-tighter mb-3">
            DT
          </div>
          <div className="eyebrow">DigitalTwinSim</div>
          <div className="text-[18px] font-medium tracking-tight2 text-ki-on-surface mt-1">
            Sign in
          </div>
          <p className="text-[12px] text-ki-on-surface-muted mt-1 text-center">
            Inserisci la password per accedere alla piattaforma.
          </p>
        </div>

        {children}

        <p className="text-[11px] text-ki-on-surface-muted text-center mt-4">
          Tourbillon Tech Srl · DigitalTwinSim
        </p>
      </div>
    </div>
  );
}
