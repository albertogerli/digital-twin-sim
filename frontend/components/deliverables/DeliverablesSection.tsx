"use client";

const DELIVERABLES = [
  {
    title: "Report Esecutivo Strategico",
    description:
      "Documento narrativo completo (PDF/PPT) che sintetizza l'evoluzione dello scenario, identificando i driver chiave, le coalizioni e i rischi reputazionali.",
    tags: ["TIMELINE", "NARRATIVE ANALYSIS", "RISK MAP"],
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
    color: "blue",
  },
  {
    title: "Dashboard Interattiva (Live)",
    description:
      "Accesso web sicuro per esplorare la simulazione. Permette il replay temporale, l'analisi granulare dei singoli agenti e la visualizzazione dinamica.",
    tags: ["TIME REPLAY", "NETWORK GRAPH", "DRILL-DOWN"],
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
        />
      </svg>
    ),
    color: "emerald",
  },
  {
    title: "Scenario Comparison & Data",
    description:
      "Confronto diretto tra varianti What-If (Base vs Best vs Worst Case) e accesso completo ai dati grezzi delle interazioni simulate.",
    tags: ["SQL EXPORT", "SENSITIVITY ANALYSIS", "WHAT-IF BRANCHING"],
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
        />
      </svg>
    ),
    color: "amber",
  },
];

const COLOR_MAP: Record<string, { bg: string; border: string; text: string; tagBg: string; tagText: string }> = {
  blue: {
    bg: "bg-blue-50",
    border: "border-blue-200",
    text: "text-blue-600",
    tagBg: "bg-blue-100",
    tagText: "text-blue-700",
  },
  emerald: {
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    text: "text-emerald-600",
    tagBg: "bg-emerald-100",
    tagText: "text-emerald-700",
  },
  amber: {
    bg: "bg-amber-50",
    border: "border-amber-200",
    text: "text-amber-600",
    tagBg: "bg-amber-100",
    tagText: "text-amber-700",
  },
};

export default function DeliverablesSection() {
  return (
    <section className="max-w-5xl mx-auto px-6 pb-12">
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-gray-700 mb-1">Output & Deliverable</h2>
        <p className="text-sm text-gray-400">
          Strumenti per decisioni informate e analisi granulare
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {DELIVERABLES.map((d) => {
          const c = COLOR_MAP[d.color];
          return (
            <div
              key={d.title}
              className={`rounded-xl border ${c.border} ${c.bg} p-6 flex flex-col`}
            >
              {/* Icon */}
              <div className={`${c.text} mb-4`}>{d.icon}</div>

              {/* Title */}
              <h3 className="text-base font-semibold text-gray-900 mb-2">
                {d.title}
              </h3>

              {/* Description */}
              <p className="text-sm text-gray-500 leading-relaxed mb-4 flex-1">
                {d.description}
              </p>

              {/* Tags */}
              <div className="flex flex-wrap gap-1.5">
                {d.tags.map((tag) => (
                  <span
                    key={tag}
                    className={`text-[10px] font-mono font-semibold tracking-wider px-2 py-0.5 rounded ${c.tagBg} ${c.tagText}`}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Dashboard preview mockup */}
      <div className="mt-8 rounded-xl border border-gray-200 bg-gray-50 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2.5 bg-white border-b border-gray-200">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-400" />
            <div className="w-3 h-3 rounded-full bg-amber-400" />
            <div className="w-3 h-3 rounded-full bg-emerald-400" />
          </div>
          <span className="text-xs text-gray-400 font-mono ml-2">
            app.digitaltwinsim.com/dashboard/analysis
          </span>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Left: simulation stats */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">
                  Simulation Status
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-700">
                  LIVE
                </span>
              </div>
              <div className="space-y-3">
                <div>
                  <p className="text-[10px] text-gray-400 font-mono uppercase">Round</p>
                  <p className="text-2xl font-bold text-gray-900 font-mono">6 / 9</p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-400 font-mono uppercase">
                    Polarization Index
                  </p>
                  <p className="text-2xl font-bold text-amber-600 font-mono">0.72</p>
                </div>
                <div className="w-full bg-gray-100 rounded-full h-2">
                  <div className="h-full bg-blue-500 rounded-full" style={{ width: "67%" }} />
                </div>
              </div>
            </div>

            {/* Center: What-If branching visual */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider block mb-3">
                What-If Branching Analysis
              </span>
              <div className="space-y-2">
                {/* Base scenario */}
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <div className="flex-1 h-1.5 bg-blue-100 rounded-full">
                    <div className="h-full bg-blue-500 rounded-full" style={{ width: "100%" }} />
                  </div>
                  <span className="text-[10px] font-mono text-gray-500 w-12 text-right">Base</span>
                </div>
                {/* Branch lines */}
                <div className="ml-1 border-l-2 border-dashed border-gray-300 h-4" />
                {/* Soft Landing */}
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <div className="flex-1 h-1.5 bg-emerald-100 rounded-full">
                    <div className="h-full bg-emerald-400 rounded-full" style={{ width: "78%" }} />
                  </div>
                  <span className="text-[10px] font-mono text-emerald-600 w-12 text-right">
                    Soft
                  </span>
                </div>
                {/* Crisis */}
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-500" />
                  <div className="flex-1 h-1.5 bg-red-100 rounded-full">
                    <div className="h-full bg-red-400 rounded-full" style={{ width: "45%" }} />
                  </div>
                  <span className="text-[10px] font-mono text-red-600 w-12 text-right">
                    Crisis
                  </span>
                </div>
              </div>
              <div className="mt-3 flex gap-2">
                <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-700 font-mono">
                  Scenario &quot;Soft Landing&quot;
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-red-100 text-red-700 font-mono">
                  Scenario &quot;Crisis&quot;
                </span>
              </div>
            </div>

            {/* Right: mini network + export */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <span className="text-xs font-mono text-gray-400 uppercase tracking-wider block mb-3">
                Network & Export
              </span>
              {/* Mini network visualization (CSS-only) */}
              <div className="relative h-28 mb-3">
                {[
                  { x: 20, y: 20, size: 12, color: "bg-blue-500" },
                  { x: 55, y: 15, size: 16, color: "bg-emerald-500" },
                  { x: 80, y: 30, size: 10, color: "bg-amber-500" },
                  { x: 35, y: 55, size: 14, color: "bg-red-400" },
                  { x: 65, y: 60, size: 11, color: "bg-violet-500" },
                  { x: 45, y: 35, size: 8, color: "bg-cyan-500" },
                  { x: 15, y: 70, size: 9, color: "bg-pink-400" },
                  { x: 75, y: 75, size: 10, color: "bg-blue-400" },
                ].map((node, i) => (
                  <div
                    key={i}
                    className={`absolute rounded-full ${node.color} opacity-70`}
                    style={{
                      left: `${node.x}%`,
                      top: `${node.y}%`,
                      width: node.size,
                      height: node.size,
                    }}
                  />
                ))}
                {/* Edges (decorative lines) */}
                <svg className="absolute inset-0 w-full h-full" xmlns="http://www.w3.org/2000/svg">
                  <line x1="22%" y1="23%" x2="57%" y2="18%" stroke="#d1d5db" strokeWidth="1" />
                  <line x1="57%" y1="18%" x2="82%" y2="33%" stroke="#d1d5db" strokeWidth="1" />
                  <line x1="37%" y1="58%" x2="67%" y2="63%" stroke="#d1d5db" strokeWidth="1" />
                  <line x1="22%" y1="23%" x2="37%" y2="58%" stroke="#d1d5db" strokeWidth="1" />
                  <line x1="47%" y1="38%" x2="57%" y2="18%" stroke="#d1d5db" strokeWidth="1" />
                  <line x1="47%" y1="38%" x2="67%" y2="63%" stroke="#d1d5db" strokeWidth="1" />
                </svg>
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Export Report (PDF)
                </div>
                <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                  </svg>
                  Export SQL / JSON
                </div>
                <div className="flex items-center gap-2 text-[10px] font-mono text-gray-500">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Sensitivity Analysis
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
