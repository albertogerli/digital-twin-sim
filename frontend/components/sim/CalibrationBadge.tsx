"use client";

import { useState } from "react";

interface CalibrationBadgeProps {
  source: string; // e.g. "v2_domain", "v2_global", "v1_grid", "v1_defaults"
  showDetails?: boolean;
}

export default function CalibrationBadge({ source, showDetails = false }: CalibrationBadgeProps) {
  const [expanded, setExpanded] = useState(false);

  if (!source) return null;

  const parts = source.split("_");
  const version = parts[0] || "v1";
  const origin = parts.slice(1).join("_") || "default";

  const isV2 = version === "v2";

  const badgeColor = isV2
    ? "bg-ki-primary/10 text-ki-primary border-ki-primary/25"
    : "bg-ki-surface-sunken text-ki-on-surface-secondary border-ki-border";

  const versionLabel = isV2 ? "Bayesian v2" : "Grid Search v1";

  const originLabels: Record<string, string> = {
    domain: "domain-level posterior",
    global: "global posterior",
    grid: "grid search",
    defaults: "default parameters",
    default: "default parameters",
  };

  return (
    <div className="relative inline-block">
      <button
        onClick={() => showDetails && setExpanded(!expanded)}
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-sm text-[10px] font-medium border ${badgeColor} ${
          showDetails ? "cursor-pointer hover:opacity-80" : "cursor-default"
        }`}
      >
        <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
        {versionLabel}
        {isV2 && <span className="text-ki-primary-muted">({originLabels[origin] || origin})</span>}
        {showDetails && (
          <svg className={`w-2.5 h-2.5 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </button>

      {expanded && (
        <>
          {/* click-outside backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setExpanded(false)}
            aria-hidden
          />
          <div
            role="dialog"
            className="absolute top-full left-0 mt-1.5 z-50 w-64 p-2.5 bg-ki-surface border border-ki-border-strong rounded-sm shadow-lg text-[10px] text-ki-on-surface-secondary space-y-1"
          >
            <div className="flex items-baseline justify-between border-b border-ki-border-faint pb-1 mb-1">
              <span className="font-data text-[10px] uppercase tracking-wider text-ki-on-surface-muted">
                Calibration source
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); setExpanded(false); }}
                aria-label="Close"
                className="text-ki-on-surface-muted hover:text-ki-on-surface text-[12px] leading-none"
              >
                ×
              </button>
            </div>
            <div><span className="text-ki-on-surface-muted">Model:</span> <span className="text-ki-on-surface ml-1">{version} hierarchical Bayesian</span></div>
            <div><span className="text-ki-on-surface-muted">Source:</span> <span className="text-ki-on-surface ml-1">{originLabels[origin] || origin}</span></div>
            {isV2 && (
              <>
                <div><span className="text-ki-on-surface-muted">Training:</span> <span className="text-ki-on-surface ml-1">34 scenarios (8 test holdout)</span></div>
                <div><span className="text-ki-on-surface-muted">Test MAE:</span> <span className="text-ki-on-surface ml-1">12.6pp · 90% coverage 85.7%</span></div>
                <div><span className="text-ki-on-surface-muted">Discrepancy:</span> <span className="text-ki-on-surface ml-1">σ_within = 0.558 logit</span></div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
