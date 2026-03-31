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
  scenario_id?: string;
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
  const recentSims = simulations
    .filter((s) => ["completed", "failed", "cancelled"].includes(s.status))
    .slice(0, 10);

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

      {/* Technical Paper CTA */}
      <section className="max-w-5xl mx-auto px-6 mb-8">
        <Link
          href="/paper"
          className="block w-full rounded-xl border border-blue-200 bg-blue-50 p-5 hover:bg-blue-100 transition-colors group"
        >
          <div className="flex items-center gap-4">
            <div className="shrink-0 w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-semibold text-gray-900 group-hover:text-blue-700 transition-colors">
                Technical Paper: Calibrating LLM-Driven Opinion Dynamics
              </h3>
              <p className="text-sm text-gray-500 mt-0.5">
                1,000 historical scenarios &middot; 14 domains &middot; 12.0% median error &middot; Grid search calibration methodology
              </p>
            </div>
            <svg className="w-5 h-5 text-gray-400 group-hover:text-blue-600 shrink-0 ml-auto transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </div>
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

      {/* Recent Simulations */}
      {recentSims.length > 0 && (
        <section className="max-w-5xl mx-auto px-6 mb-8">
          <h2 className="text-lg font-semibold mb-3 text-gray-700">Simulazioni recenti</h2>
          <div className="space-y-2">
            {recentSims.map((sim) => {
              const st = STATUS_LABELS[sim.status] || STATUS_LABELS.completed;
              return (
                <div
                  key={sim.id}
                  className="flex items-center gap-3 bg-white border border-gray-200 rounded-lg px-4 py-3"
                >
                  <span className={`shrink-0 px-2 py-0.5 rounded text-xs font-medium text-white ${st.color}`}>
                    {st.label}
                  </span>
                  <span className="font-medium text-sm text-gray-800 truncate flex-1">
                    {sim.scenario_name || sim.brief.slice(0, 60)}
                  </span>
                  {sim.domain && (
                    <span className="hidden sm:inline text-xs text-gray-400">
                      {sim.domain.replace(/_/g, " ")}
                    </span>
                  )}
                  <span className="text-xs text-gray-400 shrink-0">
                    {sim.total_rounds > 0 ? `${sim.total_rounds}R` : ""} ${sim.cost.toFixed(2)}
                  </span>
                  <div className="flex gap-1.5 shrink-0">
                    <Link
                      href={`/sim/${sim.id}`}
                      className="px-2.5 py-1 rounded text-xs font-medium bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors"
                    >
                      Log
                    </Link>
                    {sim.status === "completed" && sim.scenario_id && (
                      <Link
                        href={`/scenario/${sim.scenario_id}`}
                        className="px-2.5 py-1 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors"
                      >
                        Dashboard
                      </Link>
                    )}
                  </div>
                </div>
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
