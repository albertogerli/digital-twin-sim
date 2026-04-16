"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

interface Scenario {
  id: string;
  name: string;
  domain: string;
  num_rounds: number;
}

const DELIVERABLES = [
  {
    title: "Report Esecutivo Strategico",
    description:
      "Documento narrativo completo che sintetizza l'evoluzione dello scenario, identificando driver chiave, coalizioni e rischi reputazionali.",
    tags: ["TIMELINE", "NARRATIVE ANALYSIS", "RISK MAP"],
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    color: "blue",
    file: "report.md",
  },
  {
    title: "Dashboard Interattiva (Live)",
    description:
      "Esplora la simulazione con replay temporale, analisi granulare dei singoli agenti e visualizzazione dinamica del network.",
    tags: ["TIME REPLAY", "NETWORK GRAPH", "DRILL-DOWN"],
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    color: "emerald",
    path: "scenario",
  },
  {
    title: "Scenario Comparison & Data",
    description:
      "Confronto diretto tra varianti What-If (Base vs Best vs Worst Case) e accesso completo ai dati grezzi delle interazioni simulate.",
    tags: ["WHAT-IF BRANCHING", "JSON EXPORT", "SENSITIVITY ANALYSIS"],
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
    color: "amber",
    path: "scenario",
  },
];

const COLOR_MAP: Record<string, { bg: string; border: string; text: string; tagBg: string; tagText: string }> = {
  blue: { bg: "bg-blue-50", border: "border-blue-200", text: "text-blue-600", tagBg: "bg-blue-100", tagText: "text-blue-700" },
  emerald: { bg: "bg-emerald-50", border: "border-emerald-200", text: "text-emerald-600", tagBg: "bg-emerald-100", tagText: "text-emerald-700" },
  amber: { bg: "bg-amber-50", border: "border-amber-200", text: "text-amber-600", tagBg: "bg-amber-100", tagText: "text-amber-700" },
};

export default function DeliverablesSection() {
  const [latestScenario, setLatestScenario] = useState<Scenario | null>(null);

  useEffect(() => {
    fetch("/api/scenarios")
      .then((r) => (r.ok ? r.json() : []))
      .then((scenarios: Scenario[]) => {
        if (scenarios.length > 0) setLatestScenario(scenarios[0]);
      })
      .catch(() => {});
  }, []);

  return (
    <section className="max-w-5xl mx-auto px-6 pb-12">
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-700 mb-1">Output & Deliverable</h2>
        <p className="text-sm text-gray-400">
          Ogni simulazione produce automaticamente questi output
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {DELIVERABLES.map((d) => {
          const c = COLOR_MAP[d.color];
          const href = latestScenario
            ? `/scenario/${latestScenario.id}`
            : undefined;

          const content = (
            <div className={`rounded-xl border ${c.border} ${c.bg} p-6 flex flex-col h-full ${href ? "hover:shadow-md transition-shadow cursor-pointer" : ""}`}>
              <div className={`${c.text} mb-4`}>{d.icon}</div>
              <h3 className="text-base font-semibold text-gray-900 mb-2">{d.title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed mb-4 flex-1">{d.description}</p>
              <div className="flex flex-wrap gap-1.5">
                {d.tags.map((tag) => (
                  <span key={tag} className={`text-[10px] font-mono font-semibold tracking-wider px-2 py-0.5 rounded ${c.tagBg} ${c.tagText}`}>
                    {tag}
                  </span>
                ))}
              </div>
              {latestScenario && (
                <p className="text-[10px] text-gray-400 mt-3 font-mono truncate">
                  es. {latestScenario.name}
                </p>
              )}
            </div>
          );

          return href ? (
            <Link key={d.title} href={href}>{content}</Link>
          ) : (
            <div key={d.title}>{content}</div>
          );
        })}
      </div>
    </section>
  );
}
