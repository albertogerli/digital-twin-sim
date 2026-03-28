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

  const textSize = size === "lg" ? "text-2xl" : size === "md" ? "text-lg" : "text-sm";

  return (
    <div className="flex flex-col">
      <span className={`font-mono ${textSize} font-bold text-gray-900 tabular-nums`}>
        {format(animated)}
      </span>
      <span className="font-mono text-[9px] text-gray-500 uppercase tracking-wider">
        {label}
      </span>
    </div>
  );
}
