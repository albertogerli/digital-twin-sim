"use client";

interface ShockBadgeProps {
  magnitude: number;
  direction: number;
}

export default function ShockBadge({ magnitude, direction }: ShockBadgeProps) {
  if (!magnitude && magnitude !== 0) return null;

  const intensity =
    magnitude >= 0.7 ? "bg-ki-error/15 text-ki-error border-ki-error/30" :
    magnitude >= 0.4 ? "bg-ki-warning/15 text-ki-warning border-ki-warning/30" :
    "bg-ki-surface-sunken text-ki-on-surface-secondary border-ki-border";

  const arrow = direction > 0.1 ? "\u2197" : direction < -0.1 ? "\u2198" : "\u2192";

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-sm text-[10px] font-data font-semibold border ${intensity}`}>
      <span>{arrow}</span>
      <span>{magnitude.toFixed(1)}</span>
    </span>
  );
}
