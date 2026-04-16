"use client";

import { useState } from "react";

interface SetupScreenProps {
  onStart: (brief: string, playerRole: string, provider: string, rounds?: number) => void;
}

const PRESETS = [
  {
    label: "Crisi Fiscale Italia",
    brief: "Il governo italiano annuncia una manovra finanziaria controversa con tagli a sanità e istruzione per finanziare un aumento della spesa militare. L'opposizione minaccia mozioni di sfiducia, i sindacati preparano scioperi generali, i mercati reagiscono con vendite su bancari italiani.",
    role: "Presidente del Consiglio",
  },
  {
    label: "Crisi CEO / Corporate",
    brief: "Il CEO di una grande multinazionale italiana è coinvolto in uno scandalo finanziario. Whistleblower rilasciano documenti interni che mostrano irregolarità contabili. Il titolo crolla in borsa, i sindacati chiedono dimissioni, CONSOB apre indagine.",
    role: "CEO",
  },
  {
    label: "Crisi Energetica EU",
    brief: "Una crisi energetica colpisce l'Europa: interruzione delle forniture di gas dalla Russia, prezzi dell'energia triplicano in un mese. L'Italia è particolarmente vulnerabile. Il governo deve decidere tra razionamento, sussidi emergenziali, o accelerazione sulle rinnovabili.",
    role: "Ministro della Transizione Ecologica",
  },
];

export function SetupScreen({ onStart }: SetupScreenProps) {
  const [brief, setBrief] = useState("");
  const [playerRole, setPlayerRole] = useState("");
  const [provider, setProvider] = useState("gemini");
  const [rounds, setRounds] = useState(9);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const canStart = brief.trim().length > 20;

  return (
    <div className="h-screen w-screen bg-ki-surface text-ki-on-surface flex flex-col overflow-hidden">
      {/* Header */}
      <div className="h-7 flex items-center px-2 border-b border-ki-border bg-ki-surface-sunken shrink-0">
        <span className="font-data text-[10px] text-ki-on-surface font-bold tracking-wider">WARGAME</span>
        <span className="font-data text-[9px] text-ki-on-surface-muted ml-2">SETUP</span>
        <span className="ml-auto font-data text-[9px] text-ki-on-surface-muted">DIGITAL TWIN SIMULATOR</span>
      </div>

      <div className="flex-1 flex items-center justify-center min-h-0 overflow-y-auto">
        <div className="max-w-2xl w-full px-3 py-4">
          {/* Title */}
          <div className="mb-3">
            <h1 className="font-data text-[14px] text-ki-on-surface font-bold tracking-wider mb-1">
              WARGAME TERMINAL
            </h1>
            <p className="font-data text-[10px] text-ki-on-surface-muted">
              DEPLOY A MULTI-AGENT SIMULATION. ALL AGENTS REACT TO YOUR MOVES IN REAL-TIME.
            </p>
          </div>

          {/* Presets */}
          <div className="mb-3">
            <div className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider mb-2">PRESETS</div>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => { setBrief(p.brief); setPlayerRole(p.role); }}
                  className="font-data text-[9px] px-3 py-1.5 border border-ki-border text-ki-on-surface-muted hover:text-ki-on-surface hover:border-ki-border-strong transition-colors"
                >
                  {p.label.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Brief */}
          <div className="mb-4">
            <label className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider block mb-1">
              SCENARIO BRIEF *
            </label>
            <textarea
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              placeholder="Descrivi lo scenario di crisi. Più dettagli dai, più ricca sarà la simulazione..."
              rows={5}
              className="w-full bg-ki-surface-sunken border border-ki-border focus:border-ki-border-strong font-data text-[11px] text-ki-on-surface placeholder:text-ki-on-surface-muted px-3 py-2 focus:outline-none resize-none"
            />
            <div className="font-data text-[8px] text-ki-on-surface-muted mt-0.5">
              {brief.length} chars — min 20
            </div>
          </div>

          {/* Player Role */}
          <div className="mb-4">
            <label className="font-data text-[9px] text-ki-on-surface-muted uppercase tracking-wider block mb-1">
              YOUR ROLE
            </label>
            <input
              type="text"
              value={playerRole}
              onChange={(e) => setPlayerRole(e.target.value)}
              placeholder='es. "Presidente del Consiglio", "CEO di Stellantis", "Segretario CGIL"'
              className="w-full bg-ki-surface-sunken border border-ki-border focus:border-ki-border-strong font-data text-[11px] text-ki-on-surface placeholder:text-ki-on-surface-muted px-3 py-1.5 focus:outline-none"
            />
          </div>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="font-data text-[8px] text-ki-on-surface-muted hover:text-ki-on-surface-muted mb-3 uppercase tracking-wider"
          >
            {showAdvanced ? "▼" : "▶"} ADVANCED OPTIONS
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-3 mb-4 p-3 border border-ki-border-strong bg-ki-surface-sunken">
              <div>
                <label className="font-data text-[8px] text-ki-on-surface-muted uppercase block mb-1">PROVIDER</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-ki-surface-sunken border border-ki-border font-data text-[10px] text-ki-on-surface px-2 py-1 focus:outline-none"
                >
                  <option value="gemini">Gemini</option>
                  <option value="openai">OpenAI</option>
                </select>
              </div>
              <div>
                <label className="font-data text-[8px] text-ki-on-surface-muted uppercase block mb-1">ROUNDS</label>
                <input
                  type="number"
                  min={3}
                  max={15}
                  value={rounds}
                  onChange={(e) => setRounds(parseInt(e.target.value) || 9)}
                  className="w-full bg-ki-surface-sunken border border-ki-border font-data text-[10px] text-ki-on-surface px-2 py-1 focus:outline-none"
                />
              </div>
            </div>
          )}

          {/* Launch */}
          <button
            onClick={() => canStart && onStart(brief, playerRole, provider, rounds)}
            disabled={!canStart}
            className="w-full h-10 font-data text-[11px] font-bold tracking-wider border transition-colors disabled:opacity-20 disabled:cursor-not-allowed border-ki-success/25 text-ki-success hover:bg-ki-success/5 hover:border-ki-success/50"
          >
            ▶ DEPLOY SIMULATION
          </button>

          <div className="font-data text-[8px] text-ki-on-surface-muted mt-2 text-center">
            BACKEND MUST BE RUNNING ON localhost:8000 — ALL AGENTS WILL REACT TO YOUR MOVES
          </div>
        </div>
      </div>
    </div>
  );
}
