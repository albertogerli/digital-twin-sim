"use client";

interface ShockBadgeProps {
  magnitude: number;
  direction: number;
}

export default function ShockBadge({ magnitude, direction }: ShockBadgeProps) {
  if (!magnitude && magnitude !== 0) return null;

  const intensity =
    magnitude >= 0.7 ? "bg-red-100 text-red-700 border-red-200" :
    magnitude >= 0.4 ? "bg-amber-100 text-amber-700 border-amber-200" :
    "bg-gray-100 text-gray-600 border-gray-200";

  const arrow = direction > 0.1 ? "\u2197" : direction < -0.1 ? "\u2198" : "\u2192";

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold border ${intensity}`}>
      <span>{arrow}</span>
      <span>{magnitude.toFixed(1)}</span>
    </span>
  );
}
