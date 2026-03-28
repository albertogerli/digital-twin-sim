"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

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
    <main className="min-h-screen text-gray-900">
      <div className="max-w-3xl mx-auto px-6 py-12">
        {/* Back */}
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-gray-500 hover:text-blue-600 transition-colors mb-8 text-sm"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Torna alla home
        </Link>

        <h1 className="text-3xl font-bold mb-2">Nuova Simulazione</h1>
        <p className="text-gray-500 mb-8">
          Descrivi lo scenario che vuoi simulare. L&apos;AI analizzerà il brief, genererà gli agenti e lancerà la simulazione.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Brief textarea */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Scenario Brief
            </label>
            <textarea
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              rows={5}
              className="w-full bg-white border border-gray-300 rounded-lg px-4 py-3 text-gray-900 placeholder-gray-400 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 resize-none"
              placeholder="Descrivi lo scenario che vuoi simulare..."
              required
            />
            <div className="mt-2 flex flex-wrap gap-2">
              {EXAMPLE_BRIEFS.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setBrief(ex)}
                  className="text-xs px-3 py-1.5 rounded-full bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-800 transition-colors truncate max-w-[300px]"
                >
                  {ex.slice(0, 60)}...
                </button>
              ))}
            </div>
          </div>

          {/* Document upload (RAG) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Documentazione <span className="text-gray-400 font-normal">(opzionale)</span>
            </label>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                dragOver
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-300 hover:border-gray-400"
              }`}
            >
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.md,.csv,.json"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <svg className="w-8 h-8 mx-auto text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-sm text-gray-500">
                Trascina file o <span className="text-blue-600 font-medium">sfoglia</span>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                PDF, DOCX, TXT, MD, CSV, JSON — il contenuto viene usato come contesto RAG
              </p>
            </div>

            {files.length > 0 && (
              <ul className="mt-3 space-y-2">
                {files.map((file, i) => (
                  <li
                    key={`${file.name}-${i}`}
                    className="flex items-center justify-between bg-gray-50 border border-gray-200 rounded-lg px-4 py-2"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="text-sm text-gray-700 truncate">{file.name}</span>
                      <span className="text-xs text-gray-400 flex-shrink-0">
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(i)}
                      className="text-gray-400 hover:text-red-500 transition-colors ml-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Advanced settings toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-800 transition-colors"
          >
            <svg
              className={`w-4 h-4 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Impostazioni avanzate
          </button>

          {showAdvanced && (
            <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-5">
              {/* Provider */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Provider LLM</label>
                <div className="flex gap-3">
                  {PROVIDERS.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => setProvider(p.id)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                        provider === p.id
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Domain */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Dominio</label>
                <select
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  className="w-full bg-gray-50 border border-gray-300 rounded-lg px-4 py-2.5 text-gray-900 focus:outline-none focus:border-blue-500"
                >
                  {DOMAINS.map((d) => (
                    <option key={d.id} value={d.id}>{d.label}</option>
                  ))}
                </select>
              </div>

              {/* Rounds */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Rounds: <span className="text-blue-600 font-bold">{rounds}</span>
                </label>
                <input
                  type="range"
                  min={2}
                  max={12}
                  value={rounds}
                  onChange={(e) => setRounds(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>2 (veloce)</span>
                  <span>12 (dettagliato)</span>
                </div>
              </div>

              {/* Budget */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Budget massimo: <span className="text-blue-600 font-bold">${budget.toFixed(1)}</span>
                </label>
                <input
                  type="range"
                  min={0.5}
                  max={10}
                  step={0.5}
                  value={budget}
                  onChange={(e) => setBudget(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* Elite only */}
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={eliteOnly}
                  onChange={(e) => setEliteOnly(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-300 bg-white accent-blue-500"
                />
                <div>
                  <span className="text-sm text-gray-700">Solo agenti Elite</span>
                  <p className="text-xs text-gray-400">Salta agenti istituzionali e cittadini (più veloce ed economico)</p>
                </div>
              </label>

              {/* Monte Carlo */}
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={monteCarlo}
                  onChange={(e) => setMonteCarlo(e.target.checked)}
                  className="w-4 h-4 rounded border-gray-300 bg-white accent-blue-500"
                />
                <div>
                  <span className="text-sm text-gray-700">Analisi Monte Carlo</span>
                  <p className="text-xs text-gray-400">Esegue N simulazioni sintetiche con parametri perturbati per ottenere intervalli di confidenza (costo zero aggiuntivo)</p>
                </div>
              </label>
              {monteCarlo && (
                <div className="ml-7">
                  <label className="text-xs text-gray-500">
                    Numero di run: <span className="text-blue-600 font-bold">{monteCarloRuns}</span>
                  </label>
                  <input
                    type="range"
                    min={5}
                    max={100}
                    step={5}
                    value={monteCarloRuns}
                    onChange={(e) => setMonteCarloRuns(Number(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-600 text-sm">
              {error}
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={submitting || !brief.trim()}
            className="w-full py-4 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-200 disabled:text-gray-400 text-white font-semibold text-lg transition-colors flex items-center justify-center gap-3"
          >
            {submitting ? (
              <>
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Avvio in corso...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
