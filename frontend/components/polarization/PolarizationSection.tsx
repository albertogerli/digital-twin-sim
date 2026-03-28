"use client";

import SectionHeader from "@/components/ui/SectionHeader";
import ScrollReveal from "@/components/ui/ScrollReveal";
import PolarizationChart from "./PolarizationChart";

interface PolarizationPoint {
  round: number;
  polarization: number;
  avg_position: number;
  num_agents?: number;
}

interface Props {
  polarization: PolarizationPoint[];
}

export default function PolarizationSection({ polarization }: Props) {
  const latest = polarization[polarization.length - 1];
  const first = polarization[0];

  const polDelta = latest ? latest.polarization - (first?.polarization ?? 0) : 0;
  const polDeltaColor =
    polDelta > 0.05 ? "text-red-600" : polDelta < -0.05 ? "text-green-600" : "text-gray-400";

  return (
    <section id="polarization" className="bg-white py-20 px-4">
      <div className="max-w-6xl mx-auto">
        <SectionHeader
          title="Polarization Analysis"
          subtitle="How group polarization and average position evolved across simulation rounds."
        />

        {/* Summary stats */}
        {latest && (
          <ScrollReveal>
            <div className="flex flex-wrap gap-4 mb-8">
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                <p className="font-mono text-[10px] uppercase tracking-wider text-gray-400 mb-1">
                  Final Polarization
                </p>
                <p className="font-mono text-lg font-bold text-gray-800">
                  {latest.polarization.toFixed(3)}
                </p>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                <p className="font-mono text-[10px] uppercase tracking-wider text-gray-400 mb-1">
                  Change
                </p>
                <p className={`font-mono text-lg font-bold ${polDeltaColor}`}>
                  {polDelta > 0 ? "+" : ""}
                  {polDelta.toFixed(3)}
                </p>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                <p className="font-mono text-[10px] uppercase tracking-wider text-gray-400 mb-1">
                  Final Avg Position
                </p>
                <p className="font-mono text-lg font-bold text-gray-800">
                  {latest.avg_position.toFixed(3)}
                </p>
              </div>
              {latest.num_agents && (
                <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                  <p className="font-mono text-[10px] uppercase tracking-wider text-gray-400 mb-1">
                    Agents
                  </p>
                  <p className="font-mono text-lg font-bold text-gray-800">
                    {latest.num_agents}
                  </p>
                </div>
              )}
            </div>
          </ScrollReveal>
        )}

        {/* Chart */}
        <ScrollReveal delay={0.1}>
          <div className="mb-4">
            <h3 className="font-sans text-xs uppercase tracking-wider text-gray-400 mb-3">
              Polarization &amp; Position Over Time
            </h3>
            <PolarizationChart data={polarization} />
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
