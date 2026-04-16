"use client";

import { useEffect, useState } from "react";

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
    <div className="p-3 space-y-4 max-w-5xl">
      {/* Intelligence Hub */}
      <section>
        <div className="text-2xs font-bold text-ki-on-surface-muted tracking-[0.08em] mb-1">
          INTELLIGENCE HUB
        </div>
        <div className="grid grid-cols-2 gap-1">
          <div className="bg-ki-surface-raised border border-ki-border p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-bold text-ki-on-surface">Google Gemini</span>
              <span className="font-data text-2xs font-bold text-ki-success">ACTIVE</span>
            </div>
            <div className="font-data text-2xs text-ki-on-surface-muted">
              gemini-3.1-flash-lite-preview
            </div>
            <div className="font-data text-2xs text-ki-on-surface-muted mt-0.5">
              $0.25/1M in &middot; $1.50/1M out
            </div>
          </div>
          <div className="bg-ki-surface-raised border border-ki-border p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-bold text-ki-on-surface">OpenAI</span>
              <span className="font-data text-2xs text-ki-on-surface-muted">STANDBY</span>
            </div>
            <div className="font-data text-2xs text-ki-on-surface-muted">
              Not configured
            </div>
          </div>
        </div>
      </section>

      {/* API Access */}
      <section>
        <div className="text-2xs font-bold text-ki-on-surface-muted tracking-[0.08em] mb-1">
          API ACCESS
        </div>
        <div className="bg-ki-surface-raised border border-ki-border p-2">
          <div className="text-xs text-ki-on-surface-secondary mb-1">
            Set via <code className="font-data text-2xs bg-ki-surface-sunken px-1">DTS_API_KEYS</code> env
            or <code className="font-data text-2xs bg-ki-surface-sunken px-1">DTS_KEY_MAP</code> for tenant mapping.
          </div>
          <div className="font-data text-2xs text-ki-on-surface-muted">
            Header: X-API-Key
          </div>
        </div>
      </section>

      {/* System Health */}
      <section>
        <div className="text-2xs font-bold text-ki-on-surface-muted tracking-[0.08em] mb-1">
          SYSTEM HEALTH
        </div>
        <div className="bg-ki-surface-raised border border-ki-border">
          {health ? (
            <table className="w-full text-xs">
              <tbody>
                {[
                  { label: "API SERVER", value: health.status, ok: true },
                  { label: "POSTGRESQL", value: health.postgres ? "Connected" : "Fallback (JSON)", ok: !!health.postgres },
                  { label: "REDIS", value: health.redis ? "Connected" : "Local fallback", ok: !!health.redis },
                  ...(health.max_concurrent ? [{ label: "SLOTS", value: `${health.running || 0} / ${health.max_concurrent}`, ok: true }] : []),
                  ...(health.version ? [{ label: "VERSION", value: health.version, ok: true }] : []),
                ].map((row, i) => (
                  <tr key={i} className="border-b border-ki-border/50 last:border-0">
                    <td className="px-2 py-1 font-data text-2xs text-ki-on-surface-muted w-32">{row.label}</td>
                    <td className="px-2 py-1">
                      <span className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${row.ok ? "bg-ki-success" : "bg-ki-on-surface-muted"}`} />
                        <span className="font-data text-2xs text-ki-on-surface">{row.value}</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="p-3 text-center">
              <span className="font-data text-2xs text-ki-on-surface-muted">
                BACKEND UNREACHABLE — run <code className="bg-ki-surface-sunken px-1">python run_api.py</code>
              </span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
