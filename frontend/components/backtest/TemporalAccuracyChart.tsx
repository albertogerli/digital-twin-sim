"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { BacktestScenario } from "@/lib/backtest-types";

export function TemporalAccuracyChart({ data }: { data: BacktestScenario[] }) {
  const catMap: Record<string, { t1: number[]; t3: number[]; t7: number[] }> = {};
  for (const s of data) {
    if (!catMap[s.category]) catMap[s.category] = { t1: [], t3: [], t7: [] };
    if (s.mae_t1 != null) catMap[s.category].t1.push(s.mae_t1);
    if (s.mae_t3 != null) catMap[s.category].t3.push(s.mae_t3);
    if (s.mae_t7 != null) catMap[s.category].t7.push(s.mae_t7);
  }

  const avg = (a: number[]) => a.length > 0 ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(2) : 0;

  const chartData = Object.entries(catMap)
    .map(([cat, v]) => ({
      cat: cat.length > 14 ? cat.slice(0, 12) + ".." : cat,
      "T+1": avg(v.t1),
      "T+3": avg(v.t3),
      "T+7": avg(v.t7),
    }))
    .sort((a, b) => a["T+1"] - b["T+1"]);

  return (
    <div className="border-b border-ki-border">
      <div className="h-6 flex items-center px-2 justify-between border-b border-ki-border-strong bg-ki-surface-sunken">
        <span className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider">
          MAE by Category &amp; Time Horizon
        </span>
        <div className="flex gap-3 font-data text-[9px]">
          <span className="text-ki-on-surface">T+1</span>
          <span className="text-ki-on-surface-muted">T+3</span>
          <span className="text-ki-on-surface-muted">T+7</span>
        </div>
      </div>
      <div className="h-[220px] p-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 44 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#d4d4d4" />
            <XAxis
              dataKey="cat"
              tick={{ fill: "#8a8a8a", fontSize: 9, fontFamily: "var(--font-data), monospace" }}
              angle={-40}
              textAnchor="end"
              height={50}
              axisLine={{ stroke: "#d4d4d4" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#8a8a8a", fontSize: 9, fontFamily: "var(--font-data), monospace" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#fafafa",
                border: "1px solid #d4d4d4",
                borderRadius: 2,
                padding: "4px 8px",
                fontFamily: "var(--font-data), monospace",
                fontSize: 10,
                color: "#1a1a1a",
              }}
              itemStyle={{ padding: 0, color: "#1a1a1a" }}
              labelStyle={{ color: "#8a8a8a", fontSize: 9, marginBottom: 2 }}
              cursor={{ fill: "rgba(0,0,0,0.03)" }}
            />
            <Bar dataKey="T+1" fill="#1a1a1a" radius={[1, 1, 0, 0]} maxBarSize={16} />
            <Bar dataKey="T+3" fill="#8a8a8a" radius={[1, 1, 0, 0]} maxBarSize={16} />
            <Bar dataKey="T+7" fill="#b0b0b0" radius={[1, 1, 0, 0]} maxBarSize={16} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
