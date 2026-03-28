"use client";

import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface SentimentEvolutionProps {
  rounds: { round: number; positive: number; neutral: number; negative: number }[];
}

export default function SentimentEvolution({ rounds }: SentimentEvolutionProps) {
  if (rounds.length === 0) return null;

  const data = rounds.map((r) => ({
    round: r.round,
    positive: Math.round(r.positive * 100),
    neutral: Math.round(r.neutral * 100),
    negative: Math.round(r.negative * 100),
  }));

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-3">
        Sentiment
      </p>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }} stackOffset="expand">
            <XAxis
              dataKey="round"
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, borderRadius: 8, border: "1px solid #e5e7eb" }}
              formatter={(v) => [`${v}%`]}
              labelFormatter={(l) => `Round ${l}`}
            />
            <Area type="monotone" dataKey="positive" stackId="1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} animationDuration={600} />
            <Area type="monotone" dataKey="neutral" stackId="1" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.4} animationDuration={600} />
            <Area type="monotone" dataKey="negative" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} animationDuration={600} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-center gap-4 mt-2 text-[10px]">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" />Positivo</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-400" />Neutrale</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" />Negativo</span>
      </div>
    </div>
  );
}
