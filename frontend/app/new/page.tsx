"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const DOMAINS = [
  { id: "", label: "Auto-detect (dall'LLM)" },
  { id: "political", label: "Politico" },
  { id: "commercial", label: "Commerciale" },
  { id: "marketing", label: "Marketing" },
  { id: "corporate", label: "Corporate" },
  { id: "public_health", label: "Salute Pubblica" },
  { id: "financial", label: "Finanziario" },
];

const PROVIDERS = [
  { id: "gemini", label: "Google Gemini Flash Lite" },
  { id: "openai", label: "OpenAI (GPT-5.4-mini)" },
];

const KPI_SUGGESTIONS: Record<string, string[]> = {
  "": [
    "Consenso pubblico",
    "Fiducia istituzionale",
    "Polarizzazione sociale",
    "Viralità mediatica",
    "Rischio reputazionale",
  ],
  political: [
    "Consenso elettorale",
    "Fiducia istituzionale",
    "Mobilitazione di piazza",
    "Copertura mediatica favorevole",
    "Coesione della coalizione",
    "Rischio di crisi governativa",
  ],
  commercial: [
    "Brand sentiment",
    "Intenzione d'acquisto",
    "Passaparola positivo",
    "Rischio boicottaggio",
    "Quota di voce mediatica",
    "Fidelizzazione clienti",
  ],
  marketing: [
    "Brand awareness",
    "Engagement rate",
    "Sentiment positivo",
    "Viralità campagna",
    "Conversione stimata",
    "Share of voice",
  ],
  corporate: [
    "Fiducia degli investitori",
    "Morale dipendenti",
    "Rischio reputazionale",
    "Fiducia dei clienti",
    "Stabilità operativa",
    "Attrattività per talenti",
  ],
  public_health: [
    "Compliance pubblica",
    "Fiducia nella scienza",
    "Diffusione disinformazione",
    "Adesione vaccinale",
    "Pressione sul sistema sanitario",
    "Coesione sociale",
  ],
  financial: [
    "Fiducia del mercato",
    "Appetito al rischio",
    "Volatilità percepita",
    "Fiducia nei regolatori",
    "Rischio contagio",
    "Sentiment retail",
  ],
};

const EXAMPLE_BRIEFS = [
  "L'Italia introduce una tassa del 5% su tutte le transazioni crypto. Come reagiscono mercati, cittadini e istituzioni?",
  "Nike pubblica una campagna generata interamente con AI. Scoppia la polemica sui social.",
  "Un'azienda Fortune 500 annuncia che sostituirà il 40% del middle management con agenti AI.",
  "L'OMS raccomanda un nuovo vaccino obbligatorio per viaggiare. Reazioni globali.",
  "Apple raddoppia il prezzo dell'iPhone in Europa a causa dei dazi.",
];

export default function NewSimulation() {
  const router = useRouter();
  const [brief, setBrief] = useState("");
  const [provider, setProvider] = useState("gemini");
  const [domain, setDomain] = useState("");
  const [rounds, setRounds] = useState(5);
  const [budget, setBudget] = useState(2.0);
  const [eliteOnly, setEliteOnly] = useState(false);
  const [monteCarlo, setMonteCarlo] = useState(false);
  const [monteCarloRuns, setMonteCarloRuns] = useState(20);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [selectedKpis, setSelectedKpis] = useState<string[]>([]);
  const [suggestedKpisFromLlm, setSuggestedKpisFromLlm] = useState<string[]>([]);
  const [loadingKpis, setLoadingKpis] = useState(false);
  const [customKpi, setCustomKpi] = useState("");

  const [dragOver, setDragOver] = useState(false);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) {
      setFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  }

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }

  function toggleKpi(kpi: string) {
    setSelectedKpis((prev) =>
      prev.includes(kpi) ? prev.filter((k) => k !== kpi) : [...prev, kpi]
    );
  }

  function addCustomKpi() {
    const trimmed = customKpi.trim();
    if (trimmed && !selectedKpis.includes(trimmed)) {
      setSelectedKpis((prev) => [...prev, trimmed]);
      setCustomKpi("");
    }
  }

  async function suggestKpis() {
    if (!brief.trim()) return;
    setLoadingKpis(true);
    try {
      const res = await fetch("/api/suggest-kpis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ brief: brief.trim(), domain: domain || undefined }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.kpis?.length) {
          setSuggestedKpisFromLlm(data.kpis);
        }
      }
    } catch {
      // fallback to static suggestions
    } finally {
      setLoadingKpis(false);
    }
  }

  const suggestedKpis = suggestedKpisFromLlm.length > 0
    ? suggestedKpisFromLlm
    : KPI_SUGGESTIONS[domain] || KPI_SUGGESTIONS[""];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!brief.trim()) return;

    setSubmitting(true);
    setError("");

    try {
      let res: Response;

      if (files.length > 0) {
        // Multipart upload with documents
        const formData = new FormData();
        formData.append("brief", brief.trim());
        formData.append("provider", provider);
        if (domain) formData.append("domain", domain);
        formData.append("rounds", String(rounds));
        formData.append("budget", String(budget));
        formData.append("elite_only", String(eliteOnly));
        formData.append("monte_carlo", String(monteCarlo));
        if (monteCarlo) formData.append("monte_carlo_runs", String(monteCarloRuns));
        if (selectedKpis.length > 0) formData.append("metrics_to_track", JSON.stringify(selectedKpis));
        for (const file of files) {
          formData.append("documents", file);
        }
        res = await fetch("/api/simulations/with-documents", {
          method: "POST",
          body: formData,
        });
      } else {
        // Simple JSON request
        res = await fetch("/api/simulations", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            brief: brief.trim(),
            provider,
            domain: domain || undefined,
            rounds,
            budget,
            elite_only: eliteOnly,
            monte_carlo: monteCarlo,
            monte_carlo_runs: monteCarlo ? monteCarloRuns : undefined,
            metrics_to_track: selectedKpis.length > 0 ? selectedKpis : undefined,
          }),
        });
      }

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      router.push(`/sim/${data.id}`);
    } catch (err: any) {
      setError(err.message || "Errore di connessione. Il server API è attivo?");
      setSubmitting(false);
    }
  }

  return (
    <main className="text-ki-on-surface">
      <div className="p-3">
        <h1 className="text-base font-extrabold font-headline uppercase tracking-wide mb-1">Nuova Simulazione</h1>
        <p className="text-xs text-ki-on-surface-muted mb-3 font-data">
          Descrivi lo scenario che vuoi simulare. L&apos;AI analizzer&agrave; il brief, generer&agrave; gli agenti e lancer&agrave; la simulazione.
        </p>

        <form onSubmit={handleSubmit} className="space-y-3">
          {/* Brief textarea */}
          <div>
            <label className="block text-xs font-semibold text-ki-on-surface-secondary mb-1 uppercase tracking-wide font-headline">
              Scenario Brief
            </label>
            <textarea
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              rows={4}
              className="w-full bg-ki-surface-raised border border-ki-border rounded-sm px-3 py-2 text-sm text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary resize-none font-data"
              placeholder="Descrivi lo scenario che vuoi simulare..."
              required
            />
            <div className="mt-1.5 flex flex-wrap gap-1">
              {EXAMPLE_BRIEFS.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setBrief(ex)}
                  className="text-[10px] px-2 py-1 rounded-sm bg-ki-surface-sunken text-ki-on-surface-muted hover:bg-ki-border hover:text-ki-on-surface-secondary transition-colors truncate max-w-[280px] font-data"
                >
                  {ex.slice(0, 60)}...
                </button>
              ))}
            </div>
          </div>

          {/* Document upload (RAG) */}
          <div>
            <label className="block text-xs font-semibold text-ki-on-surface-secondary mb-1 uppercase tracking-wide font-headline">
              Documentazione <span className="text-ki-on-surface-muted font-normal normal-case">(opzionale)</span>
            </label>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`relative border border-dashed rounded-sm p-3 text-center transition-colors ${
                dragOver
                  ? "border-ki-primary bg-ki-primary/[0.07]"
                  : "border-ki-border hover:border-ki-on-surface-muted"
              }`}
            >
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.md,.csv,.json"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <svg className="w-6 h-6 mx-auto text-ki-on-surface-muted mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-xs text-ki-on-surface-muted font-data">
                Trascina file o <span className="text-ki-primary font-semibold">sfoglia</span>
              </p>
              <p className="text-[10px] text-ki-on-surface-muted mt-0.5 font-data">
                PDF, DOCX, TXT, MD, CSV, JSON — contenuto usato come contesto RAG
              </p>
            </div>

            {files.length > 0 && (
              <ul className="mt-2 space-y-1">
                {files.map((file, i) => (
                  <li
                    key={`${file.name}-${i}`}
                    className="flex items-center justify-between bg-ki-surface-sunken border border-ki-border rounded-sm px-2 py-1.5"
                  >
                    <div className="flex items-center gap-1.5 min-w-0">
                      <svg className="w-3.5 h-3.5 text-ki-on-surface-muted flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="text-xs text-ki-on-surface-secondary truncate font-data">{file.name}</span>
                      <span className="text-[10px] text-ki-on-surface-muted flex-shrink-0 font-data">
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(i)}
                      className="text-ki-on-surface-muted hover:text-ki-error transition-colors ml-2"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* KPI Selection */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs font-semibold text-ki-on-surface-secondary uppercase tracking-wide font-headline">
                KPI da monitorare <span className="text-ki-on-surface-muted font-normal normal-case">(opzionale)</span>
              </label>
              <button
                type="button"
                onClick={suggestKpis}
                disabled={!brief.trim() || loadingKpis}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-sm text-[10px] font-semibold bg-ki-primary/[0.07] text-ki-primary hover:bg-ki-primary/[0.15] disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-data"
              >
                {loadingKpis ? (
                  <>
                    <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Analisi...
                  </>
                ) : (
                  <>
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Suggerisci KPI dal brief
                  </>
                )}
              </button>
            </div>
            <p className="text-[10px] text-ki-on-surface-muted mb-2 font-data">
              {suggestedKpisFromLlm.length > 0
                ? "KPI suggeriti dall'AI in base al tuo brief. Clicca per selezionare."
                : "Scrivi il brief e clicca \"Suggerisci KPI\" per ottenere metriche personalizzate, oppure scegli tra quelle generiche."}
            </p>
            <div className="flex flex-wrap gap-1 mb-2">
              {suggestedKpis.map((kpi) => (
                <button
                  key={kpi}
                  type="button"
                  onClick={() => toggleKpi(kpi)}
                  className={`px-2 py-1 rounded-sm text-[10px] font-semibold transition-colors font-data ${
                    selectedKpis.includes(kpi)
                      ? "bg-ki-primary text-ki-on-surface"
                      : "bg-ki-surface-sunken text-ki-on-surface-muted hover:bg-ki-border"
                  }`}
                >
                  {selectedKpis.includes(kpi) && (
                    <span className="mr-0.5">&#10003;</span>
                  )}
                  {kpi}
                </button>
              ))}
            </div>
            {/* Custom KPI pills (user-added, not in suggestions) */}
            {selectedKpis.filter((k) => !suggestedKpis.includes(k)).length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {selectedKpis
                  .filter((k) => !suggestedKpis.includes(k))
                  .map((kpi) => (
                    <button
                      key={kpi}
                      type="button"
                      onClick={() => toggleKpi(kpi)}
                      className="px-2 py-1 rounded-sm text-[10px] font-semibold bg-ki-primary text-ki-on-surface transition-colors font-data"
                    >
                      &#10003; {kpi}
                    </button>
                  ))}
              </div>
            )}
            <div className="flex gap-1.5">
              <input
                type="text"
                value={customKpi}
                onChange={(e) => setCustomKpi(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addCustomKpi();
                  }
                }}
                placeholder="Aggiungi KPI personalizzato..."
                className="flex-1 bg-ki-surface-raised border border-ki-border rounded-sm px-2 py-1.5 text-xs text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary font-data"
              />
              <button
                type="button"
                onClick={addCustomKpi}
                disabled={!customKpi.trim()}
                className="px-3 py-1.5 rounded-sm text-xs font-semibold bg-ki-surface-sunken text-ki-on-surface-secondary hover:bg-ki-border disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-data"
              >
                Aggiungi
              </button>
            </div>
            {selectedKpis.length > 0 && (
              <p className="text-[10px] text-ki-primary mt-1 font-data font-semibold">
                {selectedKpis.length} KPI selezionati
              </p>
            )}
          </div>

          {/* Advanced settings toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-ki-on-surface-muted hover:text-ki-on-surface-secondary transition-colors font-headline uppercase tracking-wide"
          >
            <svg
              className={`w-3 h-3 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Impostazioni avanzate
          </button>

          {showAdvanced && (
            <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-3 space-y-3">
              {/* Provider */}
              <div>
                <label className="block text-xs font-semibold text-ki-on-surface-secondary mb-1 uppercase tracking-wide font-headline">Provider LLM</label>
                <div className="flex gap-1.5">
                  {PROVIDERS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setProvider(p.id)}
                      className={`px-2.5 py-1.5 rounded-sm text-xs font-semibold transition-colors font-data ${
                        provider === p.id
                          ? "bg-ki-primary text-ki-on-surface"
                          : "bg-ki-surface-sunken text-ki-on-surface-muted hover:bg-ki-border"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Domain */}
              <div>
                <label className="block text-xs font-semibold text-ki-on-surface-secondary mb-1 uppercase tracking-wide font-headline">Dominio</label>
                <select
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  className="w-full bg-ki-surface-sunken border border-ki-border rounded-sm px-2.5 py-1.5 text-xs text-ki-on-surface focus:outline-none focus:border-ki-primary font-data"
                >
                  {DOMAINS.map((d) => (
                    <option key={d.id} value={d.id}>{d.label}</option>
                  ))}
                </select>
              </div>

              {/* Rounds */}
              <div>
                <label className="block text-xs font-semibold text-ki-on-surface-secondary mb-1 font-headline uppercase tracking-wide">
                  Rounds: <span className="text-ki-primary font-extrabold font-data">{rounds}</span>
                </label>
                <input
                  type="range"
                  min={2}
                  max={12}
                  value={rounds}
                  onChange={(e) => setRounds(Number(e.target.value))}
                  className="w-full accent-ki-primary"
                />
                <div className="flex justify-between text-[10px] text-ki-on-surface-muted mt-0.5 font-data">
                  <span>2 (veloce)</span>
                  <span>12 (dettagliato)</span>
                </div>
              </div>

              {/* Budget */}
              <div>
                <label className="block text-xs font-semibold text-ki-on-surface-secondary mb-1 font-headline uppercase tracking-wide">
                  Budget massimo: <span className="text-ki-primary font-extrabold font-data">${budget.toFixed(1)}</span>
                </label>
                <input
                  type="range"
                  min={0.5}
                  max={10}
                  step={0.5}
                  value={budget}
                  onChange={(e) => setBudget(Number(e.target.value))}
                  className="w-full accent-ki-primary"
                />
              </div>

              {/* Elite only */}
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={eliteOnly}
                  onChange={(e) => setEliteOnly(e.target.checked)}
                  className="w-3.5 h-3.5 rounded-sm border-ki-border bg-ki-surface-raised accent-ki-primary"
                />
                <div>
                  <span className="text-xs text-ki-on-surface-secondary font-data">Solo agenti Elite</span>
                  <p className="text-[10px] text-ki-on-surface-muted font-data">Salta agenti istituzionali e cittadini (pi&ugrave; veloce ed economico)</p>
                </div>
              </label>

              {/* Monte Carlo */}
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={monteCarlo}
                  onChange={(e) => setMonteCarlo(e.target.checked)}
                  className="w-3.5 h-3.5 rounded-sm border-ki-border bg-ki-surface-raised accent-ki-primary"
                />
                <div>
                  <span className="text-xs text-ki-on-surface-secondary font-data">Analisi Monte Carlo</span>
                  <p className="text-[10px] text-ki-on-surface-muted font-data">Esegue N simulazioni sintetiche con parametri perturbati per intervalli di confidenza (costo zero)</p>
                </div>
              </label>
              {monteCarlo && (
                <div className="ml-5">
                  <label className="text-[10px] text-ki-on-surface-muted font-data">
                    Numero di run: <span className="text-ki-primary font-extrabold">{monteCarloRuns}</span>
                  </label>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    step={5}
                    value={monteCarloRuns}
                    onChange={(e) => setMonteCarloRuns(Number(e.target.value))}
                    className="w-full accent-ki-primary"
                  />
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-ki-error/[0.07] border border-ki-error/30 rounded-sm p-2 text-ki-error text-xs font-data">
              {error}
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={submitting || !brief.trim()}
            className="w-full py-2.5 rounded-sm bg-ki-primary hover:bg-ki-primary-muted disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted text-ki-on-surface font-extrabold text-sm transition-colors flex items-center justify-center gap-2 font-headline uppercase tracking-wide"
          >
            {submitting ? (
              <>
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Avvio in corso...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Lancia Simulazione
              </>
            )}
          </button>
        </form>
      </div>
    </main>
  );
}
