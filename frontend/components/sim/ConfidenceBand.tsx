"use client";

interface ConfidenceInterval {
  pro_pct_mean: number;
  pro_pct_ci95_lo: number;
  pro_pct_ci95_hi: number;
  sigma_pp: number;
}

interface ConfidenceBandProps {
  rounds: { round: number; confidence_interval?: ConfidenceInterval | null }[];
  positionAxis?: { negative_label: string; positive_label: string } | null;
}

export default function ConfidenceBand({ rounds, positionAxis }: ConfidenceBandProps) {
  const dataWithCI = rounds.filter((r) => r.confidence_interval);
  if (dataWithCI.length < 2) return null;

  const positiveLabel = positionAxis?.positive_label || "Pro";
  const width = 320;
  const height = 160;
  const padX = 40;
  const padY = 20;
  const plotW = width - padX * 2;
  const plotH = height - padY * 2;

  const maxRound = Math.max(...dataWithCI.map((r) => r.round));
  const minRound = Math.min(...dataWithCI.map((r) => r.round));
  const xScale = (r: number) => padX + ((r - minRound) / Math.max(maxRound - minRound, 1)) * plotW;
  const yScale = (v: number) => padY + plotH - (v / 100) * plotH;

  // CI band path (upper → lower reversed)
  const upperPath = dataWithCI
    .map((r, i) => {
      const x = xScale(r.round);
      const y = yScale(r.confidence_interval!.pro_pct_ci95_hi);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const lowerPath = [...dataWithCI]
    .reverse()
    .map((r) => {
      const x = xScale(r.round);
      const y = yScale(r.confidence_interval!.pro_pct_ci95_lo);
      return `L ${x} ${y}`;
    })
    .join(" ");

  const bandPath = `${upperPath} ${lowerPath} Z`;

  // Mean line
  const meanPath = dataWithCI
    .map((r, i) => {
      const x = xScale(r.round);
      const y = yScale(r.confidence_interval!.pro_pct_mean);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const lastCI = dataWithCI[dataWithCI.length - 1].confidence_interval!;
  const ciWidth = lastCI.pro_pct_ci95_hi - lastCI.pro_pct_ci95_lo;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-3">
      <div className="flex items-center justify-between mb-1">
        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
          Confidence Band
        </h4>
        <span className="text-[10px] text-gray-500">
          95% CI: {ciWidth.toFixed(1)}pp
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height: 140 }}>
        {/* 50% reference line */}
        <line
          x1={padX} y1={yScale(50)} x2={width - padX} y2={yScale(50)}
          stroke="#e5e7eb" strokeDasharray="4 2" strokeWidth={0.5}
        />
        <text x={padX - 4} y={yScale(50) + 3} textAnchor="end" fontSize={8} fill="#9ca3af">50%</text>

        {/* CI band */}
        <path d={bandPath} fill="#3b82f6" fillOpacity={0.15} />

        {/* Mean line */}
        <path d={meanPath} fill="none" stroke="#3b82f6" strokeWidth={2} />

        {/* Data points */}
        {dataWithCI.map((r) => (
          <circle
            key={r.round}
            cx={xScale(r.round)}
            cy={yScale(r.confidence_interval!.pro_pct_mean)}
            r={2.5}
            fill="#3b82f6"
          />
        ))}

        {/* Y-axis labels */}
        <text x={padX - 4} y={yScale(100) + 3} textAnchor="end" fontSize={8} fill="#9ca3af">100</text>
        <text x={padX - 4} y={yScale(0) + 3} textAnchor="end" fontSize={8} fill="#9ca3af">0</text>

        {/* X-axis */}
        {dataWithCI.map((r) => (
          <text
            key={r.round}
            x={xScale(r.round)}
            y={height - 4}
            textAnchor="middle"
            fontSize={8}
            fill="#9ca3af"
          >
            R{r.round}
          </text>
        ))}

        {/* Last value annotation */}
        <text
          x={xScale(lastCI.pro_pct_mean > 50 ? maxRound : maxRound) + 6}
          y={yScale(lastCI.pro_pct_mean)}
          fontSize={9}
          fontWeight="bold"
          fill="#3b82f6"
        >
          {lastCI.pro_pct_mean.toFixed(1)}%
        </text>
      </svg>
      <div className="flex justify-between text-[10px] text-gray-400 mt-1 px-1">
        <span>{positiveLabel} estimate with model uncertainty (sigma={lastCI.sigma_pp.toFixed(1)}pp)</span>
      </div>
    </div>
  );
}
