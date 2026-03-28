export function positionColor(position: number): string {
  if (position > 0.3) return "#22c55e";
  if (position > 0.1) return "#86efac";
  if (position > -0.1) return "#94a3b8";
  if (position > -0.3) return "#fca5a5";
  return "#ef4444";
}

export function formatNumber(n: number): string {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return n.toString();
}

export function sentimentColor(sentiment: string): string {
  const map: Record<string, string> = {
    positive: "#22c55e",
    neutral: "#94a3b8",
    negative: "#ef4444",
    enthusiastic: "#22c55e",
    worried: "#f59e0b",
    angry: "#ef4444",
    frustrated: "#f97316",
    resigned: "#6b7280",
    combative: "#dc2626",
    satisfied: "#16a34a",
    disappointed: "#9333ea",
    indifferent: "#64748b",
  };
  return map[sentiment?.toLowerCase()] || "#64748b";
}

export function domainColor(domain: string): string {
  const map: Record<string, string> = {
    financial: "#3b82f6",
    commercial: "#22c55e",
    public_health: "#ef4444",
    corporate: "#8b5cf6",
    political: "#f59e0b",
    marketing: "#ec4899",
  };
  return map[domain?.toLowerCase()] || "#64748b";
}

export function cn(...classes: (string | undefined | false)[]): string {
  return classes.filter(Boolean).join(" ");
}
