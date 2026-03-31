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
    ? "bg-blue-50 text-blue-700 border-blue-200"
    : "bg-gray-50 text-gray-600 border-gray-200";

  const versionLabel = isV2 ? "Bayesian v2" : "Grid Search v1";

  const originLabels: Record<string, string> = {
    domain: "domain-level posterior",
    global: "global posterior",
    grid: "grid search",
    defaults: "default parameters",
    default: "default parameters",
  };

  return (
    <div className="inline-flex flex-col">
      <button
        onClick={() => showDetails && setExpanded(!expanded)}
        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium border ${badgeColor} ${
          showDetails ? "cursor-pointer hover:opacity-80" : "cursor-default"
        }`}
      >
        <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
        {versionLabel}
        {isV2 && <span className="text-blue-400">({originLabels[origin] || origin})</span>}
        {showDetails && (
          <svg className={`w-2.5 h-2.5 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        )}
      </button>

      {expanded && (
        <div className="mt-1 p-2 bg-white border border-gray-200 rounded text-[10px] text-gray-500 space-y-0.5">
          <div>Model: {version} hierarchical Bayesian</div>
          <div>Source: {originLabels[origin] || origin}</div>
          {isV2 && (
            <>
              <div>Training: 34 scenarios (8 test holdout)</div>
              <div>Test MAE: 12.6pp | 90% coverage: 85.7%</div>
              <div>Discrepancy: sigma_within = 0.558 logit</div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
