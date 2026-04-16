"use client";

import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";

interface MonteCarloRound {
  round: number;
  polarization: { mean: number; std: number; ci_low: number; ci_high: number };
  avg_position: { mean: number; std: number };
  sentiment: { positive: number; neutral: number; negative: number };
}

interface MonteCarloData {
  n_runs: number;
  n_completed: number;
  final_polarization: { mean: number; std: number; ci_low: number; ci_high: number };
  final_position: { mean: number; std: number; ci_low: number; ci_high: number };
  outcome_probability: { pro_pct: number; against_pct: number };
  rounds: MonteCarloRound[];
  per_run: { run: number; final_polarization: number; final_avg_position: number }[];
}

interface MonteCarloProps {
  data: MonteCarloData;
  positiveLabel?: string;
  negativeLabel?: string;
}

export default function MonteCarloPanel({ data, positiveLabel = "Pro", negativeLabel = "Contro" }: MonteCarloProps) {
  if (!data || !data.rounds) return null;

  const chartData = data.rounds.map((r) => ({
    round: r.round,
    polMean: r.polarization.mean,
    polLow: r.polarization.ci_low,
    polHigh: r.polarization.ci_high,
    posMean: r.avg_position.mean,
  }));

  // Histogram of final positions
  const bins = 10;
  const binWidth = 2 / bins; // -1 to +1
  const histogram = Array.from({ length: bins }, (_, i) => {
    const lo = -1 + i * binWidth;
    const hi = lo + binWidth;
    const count = data.per_run.filter(
      (r) => r.final_avg_position >= lo && r.final_avg_position < hi
    ).length;
    return {
      label: lo.toFixed(1),
      count,
      color: lo + binWidth / 2 > 0 ? "#16a34a" : lo + binWidth / 2 < 0 ? "#dc2626" : "#8a8a8a",
    };
  });

  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-3 space-y-3">
      <div className="flex items-center gap-2">
        <h3 className="text-xs font-semibold text-ki-on-surface-secondary">Analisi Monte Carlo</h3>
        <span className="px-2 py-0.5 rounded-sm bg-domain-corporate/10 border border-domain-corporate/25 text-[10px] font-data text-domain-corporate">
          {data.n_completed}/{data.n_runs} runs
        </span>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <div className="bg-ki-surface-sunken rounded-sm p-2 text-center">
          <div className="text-[10px] font-data text-ki-on-surface-muted uppercase">Outcome</div>
          <div className="text-lg font-bold text-ki-success">{data.outcome_probability.pro_pct}%</div>
          <div className="text-[10px] text-ki-on-surface-muted">{positiveLabel}</div>
        </div>
        <div className="bg-ki-surface-sunken rounded-sm p-2 text-center">
          <div className="text-[10px] font-data text-ki-on-surface-muted uppercase">Outcome</div>
          <div className="text-lg font-bold text-ki-error">{data.outcome_probability.against_pct}%</div>
          <div className="text-[10px] text-ki-on-surface-muted">{negativeLabel}</div>
        </div>
        <div className="bg-ki-surface-sunken rounded-sm p-2 text-center">
          <div className="text-[10px] font-data text-ki-on-surface-muted uppercase">Polarizzazione</div>
          <div className="text-lg font-bold text-ki-on-surface">{data.final_polarization.mean}</div>
          <div className="text-[10px] text-ki-on-surface-muted">&plusmn;{data.final_polarization.std}</div>
        </div>
        <div className="bg-ki-surface-sunken rounded-sm p-2 text-center">
          <div className="text-[10px] font-data text-ki-on-surface-muted uppercase">Posizione</div>
          <div className="text-lg font-bold text-ki-on-surface">{data.final_position.mean > 0 ? "+" : ""}{data.final_position.mean}</div>
          <div className="text-[10px] text-ki-on-surface-muted">
            CI [{data.final_position.ci_low}, {data.final_position.ci_high}]
          </div>
        </div>
      </div>

      {/* Polarization with CI band */}
      <div>
        <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-1.5">
          Polarizzazione (95% CI)
        </p>
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <defs>
                <linearGradient id="mcCi" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#7c3aed" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <XAxis dataKey="round" tick={{ fontSize: 10, fill: "#8a8a8a" }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 10]} tick={{ fontSize: 10, fill: "#8a8a8a" }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 4, border: "1px solid #d4d4d4" }}
                labelFormatter={(l) => `Round ${l}`}
              />
              <Area
                type="monotone"
                dataKey="polHigh"
                stroke="none"
                fill="url(#mcCi)"
                animationDuration={600}
              />
              <Area
                type="monotone"
                dataKey="polLow"
                stroke="none"
                fill="#fafafa"
                animationDuration={600}
              />
              <Area
                type="monotone"
                dataKey="polMean"
                stroke="#7c3aed"
                strokeWidth={2}
                fill="none"
                animationDuration={600}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Outcome distribution */}
      <div>
        <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-1.5">
          Distribuzione Outcome
        </p>
        <div className="h-24">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogram} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <XAxis dataKey="label" tick={{ fontSize: 9, fill: "#8a8a8a" }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 9, fill: "#8a8a8a" }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 4, border: "1px solid #d4d4d4" }}
                formatter={(v) => [`${v} runs`]}
              />
              <Bar dataKey="count" animationDuration={600}>
                {histogram.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-between text-[9px] text-ki-on-surface-muted px-2 mt-1">
          <span>{negativeLabel}</span>
          <span>{positiveLabel}</span>
        </div>
      </div>
    </div>
  );
}
