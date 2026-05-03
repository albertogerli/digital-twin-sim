"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface HealthStatus {
  status: string;
  postgres?: boolean;
  redis?: boolean;
  running?: number;
  max_concurrent?: number;
  version?: string;
}

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.ok ? r.json() : null)
      .catch(() => null)
      .then(setHealth);
  }, []);

  return (
    <div className="p-5 space-y-6 max-w-3xl">
      {/* Access management */}
      <section>
        <div className="eyebrow mb-2">Access</div>
        <Link
          href="/admin/invites"
          className="flex items-center gap-3 bg-ki-surface-raised border border-ki-border rounded p-3 hover:bg-ki-surface-hover transition-colors group"
        >
          <span className="w-8 h-8 grid place-items-center rounded-md bg-ki-primary-soft text-ki-primary flex-shrink-0">
            <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'wght' 500" }}>
              add_link
            </span>
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-[13px] font-medium text-ki-on-surface">Invite links</div>
            <div className="text-[11px] text-ki-on-surface-muted">
              Generate shareable URLs that auto-sign-in without a password.
            </div>
          </div>
          <span className="material-symbols-outlined text-[14px] text-ki-on-surface-muted group-hover:text-ki-on-surface" style={{ fontVariationSettings: "'wght' 400" }}>
            arrow_forward
          </span>
        </Link>
      </section>

      {/* Intelligence Hub */}
      <section>
        <div className="eyebrow mb-2">Intelligence hub</div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-ki-surface-raised border border-ki-border rounded p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[13px] font-medium text-ki-on-surface">Google Gemini</span>
              <span className="inline-flex items-center gap-1.5 text-[11px] text-ki-success">
                <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
                Active
              </span>
            </div>
            <div className="font-data text-[11px] text-ki-on-surface-secondary">
              gemini-3.1-flash-lite-preview
            </div>
            <div className="font-data tabular text-[11px] text-ki-on-surface-muted mt-0.5">
              $0.25 / 1M in · $1.50 / 1M out
            </div>
          </div>
          <div className="bg-ki-surface-raised border border-ki-border rounded p-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[13px] font-medium text-ki-on-surface">OpenAI</span>
              <span className="inline-flex items-center gap-1.5 text-[11px] text-ki-on-surface-muted">
                <span className="w-1.5 h-1.5 rounded-full bg-ki-on-surface-muted" />
                Standby
              </span>
            </div>
            <div className="font-data text-[11px] text-ki-on-surface-muted">
              Not configured
            </div>
          </div>
        </div>
      </section>

      {/* API Access */}
      <section>
        <div className="eyebrow mb-2">API access</div>
        <div className="bg-ki-surface-raised border border-ki-border rounded p-3">
          <div className="text-[12px] text-ki-on-surface-secondary mb-1.5 leading-relaxed">
            Configura via <code className="font-data text-[11px] bg-ki-surface-sunken px-1.5 py-0.5 rounded-sm text-ki-on-surface">DTS_API_KEYS</code> env
            o <code className="font-data text-[11px] bg-ki-surface-sunken px-1.5 py-0.5 rounded-sm text-ki-on-surface">DTS_KEY_MAP</code> per tenant mapping.
          </div>
          <div className="font-data text-[11px] text-ki-on-surface-muted">
            Header · X-API-Key
          </div>
        </div>
      </section>

      {/* System Health */}
      <section>
        <div className="eyebrow mb-2">System health</div>
        <div className="bg-ki-surface-raised border border-ki-border rounded overflow-hidden">
          {health ? (
            <table className="w-full">
              <tbody>
                {[
                  { label: "API server", value: health.status, ok: true },
                  { label: "PostgreSQL", value: health.postgres ? "Connected" : "Fallback (JSON)", ok: !!health.postgres },
                  { label: "Redis", value: health.redis ? "Connected" : "Local fallback", ok: !!health.redis },
                  ...(health.max_concurrent ? [{ label: "Slots", value: `${health.running || 0} / ${health.max_concurrent}`, ok: true }] : []),
                  ...(health.version ? [{ label: "Version", value: health.version, ok: true }] : []),
                ].map((row, i, arr) => (
                  <tr key={i} className={i < arr.length - 1 ? "border-b border-ki-border-faint" : ""}>
                    <td className="px-3 py-2 eyebrow w-36">{row.label}</td>
                    <td className="px-3 py-2">
                      <span className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${row.ok ? "bg-ki-success" : "bg-ki-on-surface-muted"}`} />
                        <span className="font-data tabular text-[12px] text-ki-on-surface">{row.value}</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-4 text-center">
              <span className="text-[12px] text-ki-on-surface-muted">
                Backend unreachable — esegui <code className="font-data bg-ki-surface-sunken px-1.5 py-0.5 rounded-sm text-ki-on-surface">python run_api.py</code>
              </span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
