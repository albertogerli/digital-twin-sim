"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";

/* ───────────────────────────────────────────────────────────
   Public invite landing — recipient lands here from a link
   `?t=<invite-token>`. We POST to /api/auth/invite/redeem,
   the server validates and mints a session cookie, then we
   redirect to "/" (or `?next=`).
   ─────────────────────────────────────────────────────────── */

export default function InvitePage() {
  return (
    <Suspense fallback={<InviteShell><div className="text-[12px] text-ki-on-surface-muted">Loading…</div></InviteShell>}>
      <InviteRedeem />
    </Suspense>
  );
}

function InviteRedeem() {
  const router = useRouter();
  const params = useSearchParams();
  const token = params.get("t") || "";
  const next = params.get("next") || "/";

  const [status, setStatus] = useState<"validating" | "success" | "error">("validating");
  const [label, setLabel] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setError("Invite link is missing the token parameter.");
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/auth/invite/redeem", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }),
        });
        if (cancelled) return;
        if (res.ok) {
          const data = await res.json().catch(() => ({}));
          setLabel((data?.label as string) || "");
          setStatus("success");
          // Brief moment so the welcome flashes, then redirect
          setTimeout(() => router.replace(next), 900);
        } else {
          const data = await res.json().catch(() => ({}));
          setError(data?.error || `HTTP ${res.status}`);
          setStatus("error");
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Network error");
        setStatus("error");
      }
    })();
    return () => { cancelled = true; };
  }, [token, next, router]);

  return (
    <InviteShell>
      <div className="bg-ki-surface-raised border border-ki-border rounded p-5 flex flex-col gap-3 shadow-tint">
        {status === "validating" && (
          <div className="flex items-center gap-2 text-[13px] text-ki-on-surface-secondary">
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Validating invite…
          </div>
        )}

        {status === "success" && (
          <>
            <div className="flex items-center gap-2 text-[13px] text-ki-success">
              <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'wght' 500" }}>
                check_circle
              </span>
              Invite accepted{label ? ` — welcome, ${label}` : ""}.
            </div>
            <p className="text-[12px] text-ki-on-surface-muted">
              Redirecting…
            </p>
          </>
        )}

        {status === "error" && (
          <>
            <div className="bg-ki-error-soft border border-ki-error/30 rounded p-3">
              <div className="flex items-start gap-2 text-[12px] text-ki-error">
                <span className="material-symbols-outlined text-[16px] mt-0.5" style={{ fontVariationSettings: "'wght' 500" }}>
                  error
                </span>
                <div>
                  <div className="font-medium">Invite link rejected</div>
                  <div className="mt-1 text-ki-on-surface-secondary">{error}</div>
                </div>
              </div>
            </div>
            <a
              href="/login"
              className="inline-flex items-center justify-center gap-1.5 h-9 px-3 rounded-sm text-[12px] text-ki-on-surface border border-ki-border hover:bg-ki-surface-hover transition-colors"
            >
              Go to sign in
            </a>
          </>
        )}
      </div>
    </InviteShell>
  );
}

function InviteShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-ki-surface text-ki-on-surface flex items-center justify-center p-6">
      <div className="w-full max-w-[360px]">
        <div className="flex flex-col items-center mb-8">
          <div className="w-10 h-10 rounded-md bg-ki-on-surface text-ki-surface grid place-items-center font-data text-[14px] font-semibold tracking-tighter mb-3">
            DT
          </div>
          <div className="eyebrow">DigitalTwinSim</div>
          <div className="text-[18px] font-medium tracking-tight2 text-ki-on-surface mt-1">
            Invite
          </div>
          <p className="text-[12px] text-ki-on-surface-muted mt-1 text-center">
            Una sessione di 7 giorni viene aperta automaticamente.
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
