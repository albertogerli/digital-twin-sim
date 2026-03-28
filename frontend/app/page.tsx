"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ScenarioCard from "../components/ScenarioCard";
import DeliverablesSection from "../components/deliverables/DeliverablesSection";

interface ScenarioInfo {
  id: string;
  name: string;
  domain: string;
  description: string;
  num_rounds: number;
}

interface SimStatus {
  id: string;
  status: string;
  brief: string;
  scenario_name?: string;
  domain?: string;
  current_round: number;
  total_rounds: number;
  cost: number;
  created_at: string;
}

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  queued: { label: "In coda", color: "bg-gray-500" },
  analyzing: { label: "Analisi", color: "bg-blue-600 animate-pulse" },
  configuring: { label: "Configurazione", color: "bg-blue-600 animate-pulse" },
  running: { label: "In esecuzione", color: "bg-emerald-600 animate-pulse" },
  exporting: { label: "Export", color: "bg-amber-600 animate-pulse" },
  completed: { label: "Completata", color: "bg-emerald-600" },
  failed: { label: "Errore", color: "bg-red-600" },
  cancelled: { label: "Annullata", color: "bg-gray-500" },
};

export default function Home() {
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [simulations, setSimulations] = useState<SimStatus[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load scenarios from API (fallback to static)
    Promise.all([
      fetch("/api/scenarios").then((r) => r.ok ? r.json() : []).catch(() =>
        fetch("/data/scenarios.json").then((r) => r.json()).catch(() => [])
      ),
      fetch("/api/simulations").then((r) => r.ok ? r.json() : []).catch(() => []),
    ]).then(([sc, sims]) => {
      setScenarios(sc);
      setSimulations(sims);
      setLoading(false);
    });

    // Poll running simulations
    const interval = setInterval(() => {
      fetch("/api/simulations").then((r) => r.ok ? r.json() : []).then(setSimulations).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const runningSims = simulations.filter((s) =>
    ["queued", "analyzing", "configuring", "running", "exporting"].includes(s.status)
  );

  return (
    <main className="min-h-screen text-gray-900">
      {/* Header */}
      <header className="pt-12 pb-6 px-6 text-center">
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
          DigitalTwinSim
        </h1>
        <p className="mt-3 text-lg text-gray-500">
          Universal Digital Twin Simulation Platform
        </p>
      </header>

      {/* New Simulation CTA */}
      <section className="max-w-5xl mx-auto px-6 mb-8">
        <Link
          href="/new"
          className="block w-full p-6 rounded-xl border-2 border-dashed border-gray-300 hover:border-blue-500 hover:bg-gray-50 transition-all group text-center"
        >
          <div className="flex items-center justify-center gap-3">
            <svg className="w-8 h-8 text-gray-400 group-hover:text-blue-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span className="text-xl font-semibold text-gray-500 group-hover:text-blue-600 transition-colors">
              Nuova Simulazione
            </span>
          </div>
          <p className="mt-2 text-sm text-gray-400">
            Descrivi uno scenario e lancia una simulazione con agenti AI
          </p>
        </Link>
      </section>

      {/* Output & Deliverable */}
      <DeliverablesSection />

      {/* Running Simulations */}
      {runningSims.length > 0 && (
        <section className="max-w-5xl mx-auto px-6 mb-8">
          <h2 className="text-lg font-semibold mb-3 text-gray-700">Simulazioni in corso</h2>
          <div className="space-y-3">
            {runningSims.map((sim) => {
              const st = STATUS_LABELS[sim.status] || STATUS_LABELS.queued;
              return (
                <Link
                  key={sim.id}
                  href={`/sim/${sim.id}`}
                  className="block bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-500/50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium text-white ${st.color}`}>
                        {st.label}
                      </span>
                      <span className="font-medium text-sm">
                        {sim.scenario_name || sim.brief.slice(0, 60) + (sim.brief.length > 60 ? "..." : "")}
                      </span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {sim.total_rounds > 0
                        ? `Round ${sim.current_round}/${sim.total_rounds}`
                        : "..."}
                    </span>
                  </div>
                  {sim.total_rounds > 0 && (
                    <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full transition-all duration-500"
                        style={{ width: `${(sim.current_round / sim.total_rounds) * 100}%` }}
                      />
                    </div>
                  )}
                </Link>
              );
            })}
          </div>
        </section>
      )}

      {/* Scenario Grid */}
      <section className="max-w-5xl mx-auto px-6 pb-20">
        <h2 className="text-lg font-semibold mb-4 text-gray-700">
          Scenari completati
          {scenarios.length > 0 && (
            <span className="text-sm text-gray-400 font-normal ml-2">({scenarios.length})</span>
          )}
        </h2>
        {loading ? (
          <p className="text-center text-gray-400">Caricamento...</p>
        ) : scenarios.length === 0 ? (
          <p className="text-center text-gray-400 py-12">
            Nessuno scenario completato. Lancia la tua prima simulazione!
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {scenarios.map((s) => (
              <ScenarioCard
                key={s.id}
                id={s.id}
                name={s.name}
                domain={s.domain}
                description={s.description}
                num_rounds={s.num_rounds}
              />
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
