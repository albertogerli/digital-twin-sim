"use client";

import { useAnimatedNumber } from "@/lib/replay/useAnimatedNumber";

interface Props {
  value: number; // 0-10
}

export default function PolarizationGauge({ value }: Props) {
  const animated = useAnimatedNumber(value, 1000);
  const clampedValue = Math.max(0, Math.min(10, animated));

  const cx = 60;
  const cy = 55;
  const radius = 40;

  // Needle angle: map 0-10 to 180-0 degrees
  const needleAngle = Math.PI - (clampedValue / 10) * Math.PI;
  const needleX = cx + Math.cos(needleAngle) * (radius - 8);
  const needleY = cy - Math.sin(needleAngle) * (radius - 8);

  const arcPath = (startA: number, endA: number, r: number) => {
    const x1 = cx + Math.cos(startA) * r;
    const y1 = cy - Math.sin(startA) * r;
    const x2 = cx + Math.cos(endA) * r;
    const y2 = cy - Math.sin(endA) * r;
    return `M ${x1} ${y1} A ${r} ${r} 0 0 0 ${x2} ${y2}`;
  };

  // Build colored arc segments
  const numSlices = 40;
  const slices = Array.from({ length: numSlices }, (_, i) => {
    const t = i / numSlices;
    const a1 = Math.PI - t * Math.PI;
    const a2 = Math.PI - ((i + 1) / numSlices) * Math.PI;
    let red: number, green: number, blue: number;
    if (t < 0.5) {
      const u = t * 2;
      red = Math.round(34 + u * 200);
      green = Math.round(197 - u * 30);
      blue = Math.round(94 - u * 70);
    } else {
      const u = (t - 0.5) * 2;
      red = Math.round(234 + u * 5);
      green = Math.round(167 - u * 99);
      blue = Math.round(24 - u * 20);
    }
    return { a1, a2, color: `rgb(${red},${green},${blue})` };
  });

  const activeColor = clampedValue > 6 ? "var(--neg)" : clampedValue > 3 ? "var(--warn)" : "var(--pos)";

  return (
    <div className="flex flex-col">
      <div className="flex items-baseline justify-between mb-1">
        <span className="eyebrow">Polarization</span>
        <span className="font-data text-[10px] text-ki-on-surface-muted">σ²(opinion)</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="font-data tabular text-[28px] font-medium tracking-tight2 text-ki-on-surface">
          {clampedValue.toFixed(2)}
        </span>
      </div>
      <svg viewBox="0 0 120 70" className="w-full max-w-[180px] mt-1">
        <path d={arcPath(Math.PI, 0, radius)} fill="none" stroke="var(--line)" strokeWidth="6" />
        {slices.map((s, i) => (
          <path key={i} d={arcPath(s.a1, s.a2, radius)} fill="none" stroke={s.color} strokeWidth="6" opacity="0.18" />
        ))}
        {clampedValue > 0.1 && (
          <path
            d={arcPath(Math.PI, Math.PI - (clampedValue / 10) * Math.PI, radius)}
            fill="none"
            stroke={activeColor}
            strokeWidth="6"
            opacity="0.85"
            strokeLinecap="round"
          />
        )}
        <line
          x1={cx} y1={cy} x2={needleX} y2={needleY}
          stroke="var(--ink)"
          strokeWidth="2"
          strokeLinecap="round"
          style={{ transition: "all 1s cubic-bezier(0.4, 0, 0.2, 1)" }}
        />
        <circle cx={cx} cy={cy} r="2.5" fill="var(--ink)" />
        <text x="14"  y="60" fill="var(--ink-3)" fontSize="7" fontFamily="var(--font-mono)">0</text>
        <text x="100" y="60" fill="var(--ink-3)" fontSize="7" fontFamily="var(--font-mono)">10</text>
      </svg>
      <div className="flex justify-between font-data tabular text-[10px] text-ki-on-surface-muted -mt-1">
        <span>cohesive</span><span>fragmented</span><span>polarized</span>
      </div>
    </div>
  );
}
