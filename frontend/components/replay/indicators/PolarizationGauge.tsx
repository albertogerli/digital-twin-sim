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

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 120 70" className="w-full max-w-[140px]">
        {/* Background arc */}
        <path
          d={arcPath(Math.PI, 0, radius)}
          fill="none"
          stroke="#1e293b"
          strokeWidth="10"
        />
        {/* Colored gradient segments */}
        {slices.map((s, i) => (
          <path
            key={i}
            d={arcPath(s.a1, s.a2, radius)}
            fill="none"
            stroke={s.color}
            strokeWidth="10"
            opacity="0.35"
          />
        ))}
        {/* Active arc */}
        {clampedValue > 0.1 && (
          <path
            d={arcPath(Math.PI, Math.PI - (clampedValue / 10) * Math.PI, radius)}
            fill="none"
            stroke={clampedValue > 6 ? "#ef4444" : clampedValue > 3 ? "#eab308" : "#22c55e"}
            strokeWidth="10"
            opacity="0.6"
            strokeLinecap="round"
          />
        )}
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needleX}
          y2={needleY}
          stroke="#e2e8f0"
          strokeWidth="2.5"
          strokeLinecap="round"
          style={{ transition: "all 1s cubic-bezier(0.4, 0, 0.2, 1)" }}
        />
        {/* Center dot */}
        <circle cx={cx} cy={cy} r="3.5" fill="#e2e8f0" />
        {/* Labels */}
        <text x="14" y="60" fill="#64748b" fontSize="7" fontFamily="monospace">0</text>
        <text x="100" y="60" fill="#64748b" fontSize="7" fontFamily="monospace">10</text>
      </svg>
      <div className="flex items-center gap-1.5 -mt-1">
        <span className="font-mono text-sm font-bold text-gray-800">
          {clampedValue.toFixed(1)}
        </span>
        <span className="font-mono text-[9px] text-gray-400 uppercase tracking-wider">
          polarization
        </span>
      </div>
    </div>
  );
}
