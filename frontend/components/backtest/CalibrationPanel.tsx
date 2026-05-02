"use client";

import { useMemo } from "react";
import type { BacktestScenario } from "@/lib/backtest-types";

/* ───────────────────────────────────────────────────────────
   Calibration panel — reliability diagram (binned predicted vs
   observed direction-hit rate). Pattern from screen-other.jsx.
   ─────────────────────────────────────────────────────────── */

interface Props {
  data: BacktestScenario[];
}

interface Bin {
  lo: number;
  hi: number;
  predicted: number;
  observed: number;
  n: number;
}

function buildBins(data: BacktestScenario[]): Bin[] {
  const bins: Bin[] = [
    { lo: 0.0, hi: 0.2, predicted: 0.1, observed: 0, n: 0 },
    { lo: 0.2, hi: 0.4, predicted: 0.3, observed: 0, n: 0 },
    { lo: 0.4, hi: 0.6, predicted: 0.5, observed: 0, n: 0 },
    { lo: 0.6, hi: 0.8, predicted: 0.7, observed: 0, n: 0 },
    { lo: 0.8, hi: 1.0, predicted: 0.9, observed: 0, n: 0 },
  ];
  for (const s of data) {
    const [c, t] = s.direction_accuracy.split("/").map(Number);
    if (!t) continue;
    const observed = c / t; // hit rate per scenario
    // approximate "stated confidence" from MAE: lower MAE = higher confidence
    const mae = s.mae_t1 ?? 5;
    const stated = Math.max(0.05, Math.min(0.95, 1 - mae / 10));
    const bin = bins.find((b) => stated >= b.lo && stated < b.hi) || bins[4];
    bin.observed = (bin.observed * bin.n + observed) / (bin.n + 1);
    bin.n += 1;
  }
  return bins;
}

export function CalibrationPanel({ data }: Props) {
  const bins = useMemo(() => buildBins(data), [data]);

  const W = 280, H = 200, P = 24;
  const x = (v: number) => P + v * (W - P * 2);
  const y = (v: number) => H - P - v * (H - P * 2);

  // overall stats
  const totalScenarios = bins.reduce((a, b) => a + b.n, 0);
  const weightedObs = bins.reduce((a, b) => a + b.observed * b.n, 0) / Math.max(1, totalScenarios);
  const weightedPred = bins.reduce((a, b) => a + b.predicted * b.n, 0) / Math.max(1, totalScenarios);
  const direction = weightedObs > weightedPred ? "under-confident" : "over-confident";

  return (
    <div className="border-b border-ki-border bg-ki-surface-raised">
      <div className="h-8 flex items-center px-3 border-b border-ki-border bg-ki-surface-sunken">
        <span className="eyebrow">Calibration</span>
        <span className="ml-auto font-data tabular text-[11px] text-ki-on-surface-muted">
          N={totalScenarios}
        </span>
      </div>

      <div className="p-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
          {/* Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={P} y2={H - P} stroke="var(--line-faint)" strokeDasharray="2,3" strokeWidth={0.5} />
              <line x1={P} x2={W - P} y1={y(v)} y2={y(v)} stroke="var(--line-faint)" strokeDasharray="2,3" strokeWidth={0.5} />
            </g>
          ))}
          {/* Diagonal (perfect calibration) */}
          <line
            x1={x(0)} y1={y(0)}
            x2={x(1)} y2={y(1)}
            stroke="var(--accent)"
            strokeDasharray="4,3"
            strokeWidth={1}
            opacity={0.6}
          />
          {/* Bin points */}
          {bins.filter((b) => b.n > 0).map((b, i, arr) => {
            const radius = 3 + Math.min(8, Math.sqrt(b.n) * 1.4);
            return (
              <g key={i}>
                {i > 0 && (() => {
                  const prev = arr[i - 1];
                  return (
                    <line
                      x1={x(prev.predicted)} y1={y(prev.observed)}
                      x2={x(b.predicted)}    y2={y(b.observed)}
                      stroke="var(--ink)" strokeWidth={1.2}
                    />
                  );
                })()}
                <circle
                  cx={x(b.predicted)}
                  cy={y(b.observed)}
                  r={radius}
                  fill="var(--accent-soft)"
                  stroke="var(--accent)"
                  strokeWidth={1.2}
                />
                <text
                  x={x(b.predicted) + radius + 4}
                  y={y(b.observed) + 3}
                  fontSize="9"
                  fontFamily="var(--font-mono)"
                  fill="var(--ink-3)"
                >
                  n={b.n}
                </text>
              </g>
            );
          })}
          {/* Axis labels */}
          <text x={W - P} y={y(0) + 14} textAnchor="end" fontSize="9" fontFamily="var(--font-mono)" fill="var(--ink-3)">
            stated confidence →
          </text>
          <text
            x={P - 4} y={P + 4}
            fontSize="9" fontFamily="var(--font-mono)" fill="var(--ink-3)"
            transform={`rotate(-90, ${P - 4}, ${P + 4})`}
          >
            observed accuracy
          </text>
          {/* Tick labels */}
          <text x={x(0)}    y={H - P + 12} textAnchor="middle" fontSize="8" fontFamily="var(--font-mono)" fill="var(--ink-3)">0.0</text>
          <text x={x(0.5)}  y={H - P + 12} textAnchor="middle" fontSize="8" fontFamily="var(--font-mono)" fill="var(--ink-3)">0.5</text>
          <text x={x(1)}    y={H - P + 12} textAnchor="middle" fontSize="8" fontFamily="var(--font-mono)" fill="var(--ink-3)">1.0</text>
        </svg>

        <p className="text-[11px] text-ki-on-surface-secondary leading-relaxed mt-2">
          Weighted observed accuracy{" "}
          <span className="font-data tabular text-ki-on-surface">{weightedObs.toFixed(2)}</span>{" "}
          vs stated confidence{" "}
          <span className="font-data tabular text-ki-on-surface">{weightedPred.toFixed(2)}</span>{" "}
          — model is{" "}
          <span className={direction === "under-confident" ? "text-ki-success font-medium" : "text-ki-warning font-medium"}>
            {direction}
          </span>
          .
        </p>
      </div>
    </div>
  );
}
