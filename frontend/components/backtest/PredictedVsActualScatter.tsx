"use client";

import { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ZAxis,
  Cell,
} from "recharts";
import { BacktestScenario } from "@/lib/backtest-types";

interface Pt {
  predicted: number;
  actual: number;
  ticker: string;
  scenario: string;
  error: number;
}

export function PredictedVsActualScatter({ data }: { data: BacktestScenario[] }) {
  const points = useMemo(() => {
    const pts: Pt[] = [];
    for (const s of data) {
      for (const [ticker, tr] of Object.entries(s.tickers)) {
        if (tr.actual_t1 !== 0 || tr.predicted_t1 !== 0) {
          pts.push({
            predicted: +tr.predicted_t1.toFixed(2),
            actual: +tr.actual_t1.toFixed(2),
            ticker,
            scenario: s.scenario,
            error: +Math.abs(tr.actual_t1 - tr.predicted_t1).toFixed(2),
          });
        }
      }
    }
    return pts;
  }, [data]);

  return (
    <div>
      <div className="h-6 flex items-center px-2 justify-between border-b border-ki-border-strong bg-ki-surface-sunken">
        <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
          Predicted vs Actual T+1 — {points.length} observations
        </span>
        <span className="font-data text-[9px] text-ki-on-surface-muted">
          DIAGONAL = PERFECT FIT
        </span>
      </div>
      <div className="h-[260px] p-1">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 4, right: 12, left: -12, bottom: 4 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#d4d4d4" />
            <XAxis
              dataKey="predicted"
              type="number"
              domain={[-12, 6]}
              tick={{ fill: "#8a8a8a", fontSize: 9, fontFamily: "var(--font-data), monospace" }}
              axisLine={{ stroke: "#d4d4d4" }}
              tickLine={false}
              label={{ value: "PREDICTED %", position: "bottom", offset: -4, style: { fill: "#8a8a8a", fontSize: 8, fontFamily: "var(--font-data)" } }}
            />
            <YAxis
              dataKey="actual"
              type="number"
              domain={[-15, 15]}
              tick={{ fill: "#8a8a8a", fontSize: 9, fontFamily: "var(--font-data), monospace" }}
              axisLine={false}
              tickLine={false}
              label={{ value: "ACTUAL %", angle: -90, position: "insideLeft", offset: 16, style: { fill: "#8a8a8a", fontSize: 8, fontFamily: "var(--font-data)" } }}
            />
            <ZAxis range={[12, 12]} />
            <ReferenceLine
              segment={[{ x: -15, y: -15 }, { x: 10, y: 10 }]}
              stroke="#b0b0b0"
              strokeDasharray="3 3"
            />
            <ReferenceLine x={0} stroke="#d4d4d4" />
            <ReferenceLine y={0} stroke="#d4d4d4" />
            <Tooltip
              content={({ payload }) => {
                if (!payload || !payload[0]) return null;
                const pt = payload[0].payload as Pt;
                return (
                  <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-2 font-data text-[10px]">
                    <div className="text-ki-on-surface-muted">{pt.ticker}</div>
                    <div className="text-ki-on-surface-muted text-[9px]">{pt.scenario}</div>
                    <div className="mt-1 flex gap-3">
                      <span>PRED <span className={pt.predicted < 0 ? "text-[#ff3b3b]" : "text-[#00d26a]"}>{pt.predicted > 0 ? "+" : ""}{pt.predicted}%</span></span>
                      <span>ACT <span className={pt.actual < 0 ? "text-[#ff3b3b]" : "text-[#00d26a]"}>{pt.actual > 0 ? "+" : ""}{pt.actual}%</span></span>
                    </div>
                    <div className="text-ki-on-surface-muted mt-0.5">ERR {pt.error}pp</div>
                  </div>
                );
              }}
            />
            <Scatter data={points}>
              {points.map((pt, i) => {
                // Green if direction correct, red if wrong
                const dirOk = (pt.predicted < 0 && pt.actual < -0.5) || (pt.predicted >= 0 && pt.actual > -0.5);
                return <Cell key={i} fill={dirOk ? "#00d26a" : "#ff3b3b"} fillOpacity={0.4} />;
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
