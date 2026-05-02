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
      {/* Sub-toolbar */}
      <div className="h-11 flex items-center px-4 gap-2 border-b border-ki-border bg-ki-surface-raised shrink-0">
        <span className="font-data text-[10px] uppercase tracking-[0.08em] text-ki-on-surface-muted">
          DigitalTwinSim
        </span>
        <span className="text-ki-border-strong">/</span>
        <span className="text-[14px] font-medium text-ki-on-surface">Wargame</span>
        <span className="text-ki-border-strong">/</span>
        <span className="font-data text-[11px] text-ki-on-surface-secondary">Setup</span>
        <span className="ml-auto font-data text-[11px] text-ki-on-surface-muted">
          Multi-agent crisis simulator
        </span>
      </div>

      <div className="flex-1 flex items-center justify-center min-h-0 overflow-y-auto">
        <div className="max-w-2xl w-full px-6 py-8">
          {/* Title */}
          <div className="mb-6">
            <div className="eyebrow mb-1">Briefing</div>
            <h1 className="text-[20px] font-medium tracking-tight2 text-ki-on-surface mb-1">
              Deploy a wargame simulation
            </h1>
            <p className="text-[12px] text-ki-on-surface-secondary leading-relaxed">
              Definisci scenario di crisi e ruolo. Tutti gli agenti reagiscono alle tue mosse in tempo reale.
            </p>
          </div>

          {/* Presets */}
          <div className="mb-5">
            <div className="eyebrow mb-2">Presets</div>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => { setBrief(p.brief); setPlayerRole(p.role); }}
                  className="inline-flex items-center h-7 px-2.5 rounded-sm border border-ki-border bg-ki-surface-raised text-[12px] text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface hover:border-ki-border-strong transition-colors"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Brief */}
          <div className="mb-5">
            <label className="block eyebrow mb-1.5">
              Scenario brief <span className="text-ki-error">*</span>
            </label>
            <textarea
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              placeholder="Descrivi lo scenario di crisi. Più dettagli dai, più ricca sarà la simulazione…"
              rows={5}
              className="w-full bg-ki-surface-raised border border-ki-border rounded text-[13px] text-ki-on-surface placeholder:text-ki-on-surface-muted px-3 py-2 focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30 resize-none leading-relaxed"
            />
            <div className="font-data tabular text-[11px] text-ki-on-surface-muted mt-1">
              {brief.length} chars · min 20
            </div>
          </div>

          {/* Player Role */}
          <div className="mb-5">
            <label className="block eyebrow mb-1.5">Your role</label>
            <input
              type="text"
              value={playerRole}
              onChange={(e) => setPlayerRole(e.target.value)}
              placeholder='es. "Presidente del Consiglio", "CEO di Stellantis", "Segretario CGIL"'
              className="w-full bg-ki-surface-raised border border-ki-border rounded text-[13px] text-ki-on-surface placeholder:text-ki-on-surface-muted px-3 h-8 focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30"
            />
          </div>

          {/* Advanced toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="inline-flex items-center gap-1 text-[11px] text-ki-on-surface-muted hover:text-ki-on-surface-secondary mb-3 transition-colors"
          >
            <span className={`material-symbols-outlined text-[14px] transition-transform ${showAdvanced ? "rotate-90" : ""}`} style={{ fontVariationSettings: "'wght' 400" }}>
              chevron_right
            </span>
            Advanced options
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-3 mb-5 p-3 rounded border border-ki-border bg-ki-surface-sunken">
              <div>
                <label className="block eyebrow mb-1">Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-ki-surface-raised border border-ki-border rounded-sm text-[12px] text-ki-on-surface px-2 h-7 focus:outline-none focus:border-ki-primary"
                >
                  <option value="gemini">Gemini</option>
                  <option value="openai">OpenAI</option>
                </select>
              </div>
              <div>
                <label className="block eyebrow mb-1">Rounds</label>
                <input
                  type="number"
                  min={3}
                  max={15}
                  value={rounds}
                  onChange={(e) => setRounds(parseInt(e.target.value) || 9)}
                  className="w-full bg-ki-surface-raised border border-ki-border rounded-sm font-data tabular text-[12px] text-ki-on-surface px-2 h-7 focus:outline-none focus:border-ki-primary"
                />
              </div>
            </div>
          )}

          {/* Launch */}
          <button
            onClick={() => canStart && onStart(brief, playerRole, provider, rounds)}
            disabled={!canStart}
            className="w-full h-9 inline-flex items-center justify-center gap-2 rounded-sm bg-ki-on-surface text-ki-surface text-[13px] font-medium hover:bg-ki-on-surface-secondary disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
          >
            <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'wght' 500" }}>
              play_arrow
            </span>
            Deploy simulation
          </button>

          <div className="text-[11px] text-ki-on-surface-muted mt-3 text-center">
            Tutti gli agenti reagiscono alle tue mosse in tempo reale
          </div>
        </div>
      </div>
    </div>
  );
}
