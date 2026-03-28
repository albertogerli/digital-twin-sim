"use client";

import { motion } from "framer-motion";
import Link from "next/link";

interface ScenarioMetadata {
  scenario_name: string;
  domain: string;
  description: string;
  num_rounds: number;
}

interface Props {
  metadata: ScenarioMetadata;
  scenarioId: string;
  agentsCount: number;
  finalPolarization: number;
}

const domainColors: Record<string, { bg: string; text: string; border: string }> = {
  politics: { bg: "bg-red-500/20", text: "text-red-600", border: "border-red-500/30" },
  economics: { bg: "bg-emerald-500/20", text: "text-emerald-600", border: "border-emerald-500/30" },
  technology: { bg: "bg-cyan-500/20", text: "text-cyan-600", border: "border-cyan-500/30" },
  health: { bg: "bg-pink-500/20", text: "text-pink-600", border: "border-pink-500/30" },
  environment: { bg: "bg-green-500/20", text: "text-green-600", border: "border-green-500/30" },
  social: { bg: "bg-violet-500/20", text: "text-violet-600", border: "border-violet-500/30" },
  education: { bg: "bg-amber-500/20", text: "text-amber-600", border: "border-amber-500/30" },
};

function getDomainStyle(domain: string) {
  const key = domain.toLowerCase();
  return domainColors[key] ?? { bg: "bg-blue-500/20", text: "text-cyan-600", border: "border-blue-500/30" };
}

export default function ScenarioHero({ metadata, scenarioId, agentsCount, finalPolarization }: Props) {
  const domainStyle = getDomainStyle(metadata.domain);

  const stats = [
    { label: "Rounds", value: metadata.num_rounds },
    { label: "Agents", value: agentsCount },
    { label: "Domain", value: metadata.domain },
    { label: "Final Polarization", value: finalPolarization.toFixed(2) },
  ];

  return (
    <section className="relative bg-white py-24 px-4 overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-cyan-50/30 via-white to-white" />

      <div className="relative max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.25, 0.1, 0.25, 1] }}
        >
          {/* Domain badge */}
          <span
            className={`inline-block px-3 py-1 rounded-full text-xs font-mono font-semibold uppercase tracking-wider border ${domainStyle.bg} ${domainStyle.text} ${domainStyle.border} mb-6`}
          >
            {metadata.domain}
          </span>

          {/* Title */}
          <h1 className="font-display text-4xl md:text-6xl lg:text-7xl font-bold text-gray-900 mb-6 leading-tight">
            {metadata.scenario_name}
          </h1>

          {/* Description */}
          <p className="font-body text-lg md:text-xl text-gray-500 max-w-3xl mb-10 leading-relaxed">
            {metadata.description}
          </p>

          {/* Stats ribbon */}
          <div className="flex flex-wrap gap-6 mb-10">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="bg-white border border-gray-200 rounded-lg px-5 py-3"
              >
                <p className="font-mono text-[10px] uppercase tracking-wider text-gray-400 mb-1">
                  {stat.label}
                </p>
                <p className="font-mono text-lg font-bold text-gray-800">
                  {stat.value}
                </p>
              </div>
            ))}
          </div>

          {/* CTA */}
          <Link
            href={`/scenario/${scenarioId}/replay`}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm transition-colors shadow-lg shadow-blue-600/20"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
            Avvia Replay
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
