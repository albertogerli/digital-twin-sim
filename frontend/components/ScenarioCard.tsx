import Link from "next/link";

interface ScenarioCardProps {
  id: string;
  name: string;
  domain: string;
  description: string;
  num_rounds: number;
}

const domainColors: Record<string, { bg: string; text: string; border: string }> = {
  financial:     { bg: "bg-blue-50",   text: "text-blue-700",   border: "hover:border-blue-400" },
  commercial:    { bg: "bg-green-50",  text: "text-green-700",  border: "hover:border-green-400" },
  public_health: { bg: "bg-red-50",    text: "text-red-700",    border: "hover:border-red-400" },
  corporate:     { bg: "bg-purple-50", text: "text-purple-700", border: "hover:border-purple-400" },
  political:     { bg: "bg-amber-50",  text: "text-amber-700",  border: "hover:border-amber-400" },
  marketing:     { bg: "bg-pink-50",   text: "text-pink-700",   border: "hover:border-pink-400" },
};

const defaultColor = { bg: "bg-gray-100", text: "text-gray-600", border: "hover:border-gray-400" };

export default function ScenarioCard({ id, name, domain, description, num_rounds }: ScenarioCardProps) {
  const color = domainColors[domain] ?? defaultColor;

  return (
    <Link href={`/scenario/${id}`}>
      <div
        className={`rounded-xl border border-gray-200 bg-white p-6 transition-colors ${color.border} cursor-pointer hover:shadow-md`}
      >
        {/* Domain badge */}
        <span
          className={`inline-block rounded-full px-3 py-0.5 text-xs font-medium ${color.bg} ${color.text}`}
        >
          {domain.replace(/_/g, " ")}
        </span>

        {/* Name */}
        <h2 className="mt-3 text-xl font-semibold text-gray-900">{name}</h2>

        {/* Description */}
        <p className="mt-2 text-sm text-gray-500">{description}</p>

        {/* Rounds */}
        <p className="mt-4 text-xs text-gray-400">
          {num_rounds} round{num_rounds !== 1 ? "s" : ""}
        </p>
      </div>
    </Link>
  );
}
