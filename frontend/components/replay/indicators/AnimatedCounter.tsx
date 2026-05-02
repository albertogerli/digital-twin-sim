"use client";

import { useAnimatedNumber } from "@/lib/replay/useAnimatedNumber";

interface Props {
  value: number;
  label: string;
  format?: (n: number) => string;
  size?: "sm" | "md" | "lg";
}

function defaultFormat(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toLocaleString();
}

export default function AnimatedCounter({ value, label, format = defaultFormat, size = "md" }: Props) {
  const animated = useAnimatedNumber(value, 800);

  const textSize = size === "lg" ? "text-[24px]" : size === "md" ? "text-[18px]" : "text-[13px]";

  return (
    <div className="flex flex-col">
      <span className={`font-data tabular ${textSize} font-medium text-ki-on-surface tracking-tight2`}>
        {format(animated)}
      </span>
      <span className="eyebrow mt-0.5">{label}</span>
    </div>
  );
}
