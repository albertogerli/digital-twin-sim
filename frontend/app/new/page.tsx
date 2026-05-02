"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const DOMAINS = [
  { id: "",              label: "Auto-detect",      desc: "L'AI sceglie il dominio dal brief" },
  { id: "political",     label: "Political",        desc: "Elezioni, regolazione, geopolitica" },
  { id: "corporate",     label: "Corporate",        desc: "Comms, layoff, leadership" },
  { id: "financial",     label: "Financial",        desc: "M&A, IPO, eventi di mercato" },
  { id: "commercial",    label: "Commercial",       desc: "Lanci prodotto, pricing" },
  { id: "marketing",     label: "Marketing",        desc: "Campagne, brand crisis" },
  { id: "public_health", label: "Public health",    desc: "Vaccini, outbreak, policy" },
];

const PROVIDERS = [
  { id: "gemini", label: "Google Gemini Flash Lite" },
  { id: "openai", label: "OpenAI (GPT-5.4-mini)" },
];

const KPI_SUGGESTIONS: Record<string, string[]> = {
  "":            ["Consenso pubblico", "Fiducia istituzionale", "Polarizzazione sociale", "Viralità mediatica", "Rischio reputazionale"],
  political:     ["Consenso elettorale", "Fiducia istituzionale", "Mobilitazione di piazza", "Copertura mediatica favorevole", "Coesione coalizione", "Rischio crisi governativa"],
  commercial:    ["Brand sentiment", "Intenzione d'acquisto", "Passaparola positivo", "Rischio boicottaggio", "Quota di voce", "Fidelizzazione clienti"],
  marketing:     ["Brand awareness", "Engagement rate", "Sentiment positivo", "Viralità campagna", "Conversione stimata", "Share of voice"],
  corporate:     ["Fiducia investitori", "Morale dipendenti", "Rischio reputazionale", "Fiducia clienti", "Stabilità operativa", "Attrattività talenti"],
  public_health: ["Compliance pubblica", "Fiducia nella scienza", "Diffusione disinformazione", "Adesione vaccinale", "Pressione sistema sanitario", "Coesione sociale"],
  financial:     ["Fiducia mercato", "Appetito al rischio", "Volatilità percepita", "Fiducia regolatori", "Rischio contagio", "Sentiment retail"],
};

const EXAMPLE_BRIEFS = [
  "L'Italia introduce una tassa del 5% su tutte le transazioni crypto. Come reagiscono mercati, cittadini e istituzioni?",
  "Nike pubblica una campagna generata interamente con AI. Scoppia la polemica sui social.",
  "Un'azienda Fortune 500 annuncia che sostituirà il 40% del middle management con agenti AI.",
  "L'OMS raccomanda un nuovo vaccino obbligatorio per viaggiare. Reazioni globali.",
  "Apple raddoppia il prezzo dell'iPhone in Europa a causa dei dazi.",
];

type StepNum = 1 | 2 | 3 | 4;

const STEPS: { n: StepNum; label: string }[] = [
  { n: 1, label: "Domain & brief" },
  { n: 2, label: "Knowledge base" },
  { n: 3, label: "Engine" },
  { n: 4, label: "Review & launch" },
];

function fmtSize(b: number) {
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  return (b / 1024 / 1024).toFixed(1) + " MB";
}

export default function NewSimulation() {
  const router = useRouter();

  // Wizard state
  const [step, setStep] = useState<StepNum>(1);
  const [name, setName] = useState("");

  // Existing fields
  const [brief, setBrief] = useState("");
  const [provider, setProvider] = useState("gemini");
  const [domain, setDomain] = useState("");
  const [rounds, setRounds] = useState(5);
  const [budget, setBudget] = useState(2.0);
  const [eliteOnly, setEliteOnly] = useState(false);
  const [monteCarlo, setMonteCarlo] = useState(false);
  const [monteCarloRuns, setMonteCarloRuns] = useState(20);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [selectedKpis, setSelectedKpis] = useState<string[]>([]);
  const [suggestedKpisFromLlm, setSuggestedKpisFromLlm] = useState<string[]>([]);
  const [loadingKpis, setLoadingKpis] = useState(false);
  const [customKpi, setCustomKpi] = useState("");
  const [dragOver, setDragOver] = useState(false);

  // RAG settings (mock — backend doesn't use these yet, but the UI matches the design)
  const [rag, setRag] = useState({
    mode: "hybrid" as "dense" | "sparse" | "hybrid",
    embedder: "text-embedding-3-large",
    chunkSize: 800,
    overlap: 120,
    topK: 6,
  });

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
      prev.includes(kpi) ? prev.filter((k) => k !== kpi) : [...prev, kpi],
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
        if (data.kpis?.length) setSuggestedKpisFromLlm(data.kpis);
      }
    } catch {
      /* fallback to static */
    } finally {
      setLoadingKpis(false);
    }
  }

  const suggestedKpis = suggestedKpisFromLlm.length > 0 ? suggestedKpisFromLlm : KPI_SUGGESTIONS[domain] || KPI_SUGGESTIONS[""];
  const customKpis = selectedKpis.filter((k) => !suggestedKpis.includes(k));

  const totalSize = files.reduce((a, f) => a + f.size, 0);
  const estTokens = Math.ceil(brief.split(/\s+/).filter(Boolean).length * 1.33);
  const estimatedRuntime = Math.max(2, Math.round((rounds * (eliteOnly ? 30 : 90)) / 60));
  const estimatedCost = Math.min(budget, rounds * (eliteOnly ? 0.05 : 0.18));

  async function handleLaunch() {
    if (!brief.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      let res: Response;
      if (files.length > 0) {
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
        for (const file of files) formData.append("documents", file);
        res = await fetch("/api/simulations/with-documents", { method: "POST", body: formData });
      } else {
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

  const canContinue =
    step === 1 ? brief.trim().length > 0 :
    step === 2 ? true :
    step === 3 ? true :
    true;

  const next = () => setStep((s) => (s < 4 ? ((s + 1) as StepNum) : s));
  const prev = () => setStep((s) => (s > 1 ? ((s - 1) as StepNum) : s));

  return (
    <div className="flex h-[calc(100vh-44px)] text-ki-on-surface overflow-hidden">
      {/* LEFT RAIL — steps + estimated cost */}
      <aside className="w-[260px] border-r border-ki-border bg-ki-surface-sunken flex flex-col py-5 px-4 flex-shrink-0">
        <div className="eyebrow">Workflow</div>
        <div className="text-[15px] font-medium text-ki-on-surface tracking-[-0.005em] mt-1">Briefing</div>
        <p className="text-[12px] text-ki-on-surface-muted mt-1 leading-relaxed">
          Trasforma un brief reale in uno scenario simulabile.
        </p>

        <div className="mt-6 flex flex-col gap-1">
          {STEPS.map((s) => {
            const active = s.n === step;
            const done = s.n < step;
            return (
              <button
                key={s.n}
                onClick={() => setStep(s.n)}
                className={`flex items-center gap-3 px-2 py-2 rounded-sm text-left transition-colors ${
                  active
                    ? "bg-ki-surface-raised shadow-[inset_0_0_0_1px_var(--line)]"
                    : "hover:bg-ki-surface-hover"
                }`}
              >
                <span
                  className={`w-[22px] h-[22px] rounded-full grid place-items-center font-data text-[11px] flex-shrink-0 ${
                    done
                      ? "bg-ki-on-surface text-ki-surface border border-ki-on-surface"
                      : "border border-ki-border-strong text-ki-on-surface-secondary"
                  }`}
                >
                  {done ? "✓" : s.n}
                </span>
                <span className={`text-[13px] ${active ? "text-ki-on-surface font-medium" : "text-ki-on-surface-secondary"}`}>
                  {s.label}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex-1" />

        {/* Estimated card */}
        <div className="bg-ki-surface-raised border border-ki-border rounded p-3">
          <div className="eyebrow">Estimated</div>
          <div className="flex justify-between items-baseline mt-2">
            <span className="text-[12px] text-ki-on-surface-secondary">Runtime</span>
            <span className="font-data tabular text-[12px] text-ki-on-surface">~{estimatedRuntime} min</span>
          </div>
          <div className="flex justify-between items-baseline mt-1">
            <span className="text-[12px] text-ki-on-surface-secondary">LLM cost</span>
            <span className="font-data tabular text-[12px] text-ki-on-surface">${estimatedCost.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-baseline mt-1">
            <span className="text-[12px] text-ki-on-surface-secondary">Tokens</span>
            <span className="font-data tabular text-[12px] text-ki-on-surface">~{(rounds * (eliteOnly ? 4 : 12)).toLocaleString()}k</span>
          </div>
        </div>
      </aside>

      {/* MAIN — step content + footer */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-y-auto px-8 py-6">
          <div className="max-w-[880px]">
            {/* ── STEP 1 ───────────────────────────────────── */}
            {step === 1 && (
              <div className="flex flex-col gap-6">
                <div>
                  <div className="eyebrow">Step 1 of 4</div>
                  <h1 className="text-[28px] font-medium tracking-tight2 text-ki-on-surface mt-1 leading-[1.1]">
                    Cosa stiamo simulando?
                  </h1>
                  <p className="text-[13px] text-ki-on-surface-secondary leading-relaxed max-w-xl mt-2">
                    Scegli un dominio, descrivi la situazione in linguaggio naturale.
                    Il briefing engine estrarrà stakeholder, fazioni e topologia.
                  </p>
                </div>

                {/* Name */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-[11px] text-ki-on-surface-secondary">Simulation name <span className="text-ki-on-surface-muted">(opzionale)</span></label>
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="es. Aurora — EU energy package reaction"
                    className="h-9 px-3 bg-ki-surface-raised border border-ki-border rounded text-[14px] text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30"
                  />
                </div>

                {/* Domain selector */}
                <div className="flex flex-col gap-2">
                  <label className="text-[11px] text-ki-on-surface-secondary">Domain</label>
                  <div className="grid grid-cols-3 gap-2">
                    {DOMAINS.map((d) => {
                      const sel = domain === d.id;
                      const cap =
                        d.id === "political" ? "bg-domain-political" :
                        d.id === "corporate" ? "bg-domain-corporate" :
                        d.id === "financial" ? "bg-domain-financial" :
                        d.id === "commercial" ? "bg-domain-commercial" :
                        d.id === "marketing" ? "bg-domain-marketing" :
                        d.id === "public_health" ? "bg-domain-health" :
                        "bg-ki-on-surface-faint";
                      return (
                        <button
                          key={d.id || "auto"}
                          onClick={() => setDomain(d.id)}
                          className={`bg-ki-surface-raised border rounded p-3 text-left transition-colors ${
                            sel ? "border-ki-on-surface shadow-[inset_0_0_0_1px_var(--ink)]" : "border-ki-border hover:border-ki-border-strong"
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <div className={`${cap} w-[2px] h-3 rounded-sm`} />
                            <span className="text-[13px] font-medium text-ki-on-surface">{d.label}</span>
                          </div>
                          <div className="text-[11px] text-ki-on-surface-muted mt-1 pl-3">{d.desc}</div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Brief */}
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-baseline">
                    <label className="text-[11px] text-ki-on-surface-secondary">Brief</label>
                  </div>
                  <textarea
                    value={brief}
                    onChange={(e) => setBrief(e.target.value)}
                    rows={6}
                    placeholder="Descrivi lo scenario in linguaggio naturale…"
                    className="bg-ki-surface-raised border border-ki-border rounded text-[13px] text-ki-on-surface placeholder-ki-on-surface-muted px-3 py-2 focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30 resize-none leading-relaxed"
                  />
                  <div className="flex items-center gap-2 font-data tabular text-[11px] text-ki-on-surface-muted">
                    <span>{brief.length} chars</span>
                    <span>·</span>
                    <span>~{estTokens} tokens</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <span className="text-[11px] text-ki-on-surface-muted self-center mr-1">Esempi:</span>
                    {EXAMPLE_BRIEFS.map((ex, i) => (
                      <button
                        key={i}
                        onClick={() => setBrief(ex)}
                        className="text-[11px] px-2 h-6 rounded-sm bg-ki-surface-sunken text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors truncate max-w-[300px]"
                      >
                        {ex.slice(0, 56)}…
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── STEP 2 — Knowledge base + KPIs ───────────── */}
            {step === 2 && (
              <div className="flex flex-col gap-6">
                <div>
                  <div className="eyebrow">Step 2 of 4</div>
                  <h1 className="text-[28px] font-medium tracking-tight2 text-ki-on-surface mt-1 leading-[1.1]">
                    Knowledge base & metriche
                  </h1>
                  <p className="text-[13px] text-ki-on-surface-secondary leading-relaxed max-w-xl mt-2">
                    Documenti grounded nel reasoning degli agenti via retrieval. Brief, speech, dataset, analisi pregresse.
                  </p>
                </div>

                {/* KB upload */}
                <section className="flex flex-col gap-2">
                  <div className="flex items-end justify-between">
                    <div>
                      <div className="eyebrow">Knowledge base</div>
                      <p className="text-[11px] text-ki-on-surface-muted mt-0.5">
                        PDF · DOCX · TXT · MD · CSV · JSON · HTML — fino a 50 MB ciascuno
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5 font-data tabular text-[11px] text-ki-on-surface-muted">
                      <span>{files.length} docs</span>
                      <span>·</span>
                      <span>{fmtSize(totalSize)}</span>
                    </div>
                  </div>

                  <div
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    className={`relative flex items-center gap-3 border border-dashed rounded p-4 transition-colors ${
                      dragOver
                        ? "border-ki-on-surface bg-ki-surface-raised"
                        : "border-ki-border-strong bg-ki-surface-sunken"
                    }`}
                  >
                    <input
                      type="file"
                      multiple
                      accept=".pdf,.docx,.doc,.txt,.md,.csv,.json,.html"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <div className="w-8 h-8 border border-ki-border-strong grid place-items-center bg-ki-surface-raised flex-shrink-0">
                      <span className="material-symbols-outlined text-[16px] text-ki-on-surface-secondary" style={{ fontVariationSettings: "'wght' 400" }}>
                        upload_file
                      </span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[13px] font-medium text-ki-on-surface">Drop files o paste URLs</div>
                      <div className="text-[11px] text-ki-on-surface-muted">
                        Embedded con <span className="font-data text-ki-on-surface-secondary">{rag.embedder}</span>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        const inp = (e.currentTarget.parentElement?.querySelector('input[type="file"]') as HTMLInputElement | null);
                        inp?.click();
                      }}
                      className="inline-flex items-center gap-1 h-7 px-2.5 rounded-sm border border-ki-border bg-ki-surface-raised text-[11px] text-ki-on-surface hover:bg-ki-surface-hover transition-colors relative z-10"
                    >
                      <span className="material-symbols-outlined text-[12px]" style={{ fontVariationSettings: "'wght' 400" }}>add</span>
                      Add files
                    </button>
                  </div>

                  {files.length > 0 && (
                    <div className="bg-ki-surface-raised border border-ki-border rounded overflow-hidden">
                      <div className="flex items-center px-3 h-7 border-b border-ki-border-faint bg-ki-surface-sunken">
                        <span className="eyebrow flex-1">Source</span>
                        <span className="eyebrow w-16 text-right">Size</span>
                        <span className="eyebrow w-24 text-right">Status</span>
                        <span className="w-7" />
                      </div>
                      {files.map((f, i) => {
                        const ext = f.name.split(".").pop()?.toUpperCase() || "FILE";
                        return (
                          <div key={`${f.name}-${i}`} className={`flex items-center px-3 py-2 ${i < files.length - 1 ? "border-b border-ki-border-faint" : ""}`}>
                            <div className="flex items-center gap-3 flex-1 min-w-0">
                              <span className="font-data text-[10px] text-ki-on-surface-secondary border border-ki-border w-9 text-center py-[1px] flex-shrink-0">
                                {ext.slice(0, 4)}
                              </span>
                              <span className="text-[12px] text-ki-on-surface truncate">{f.name}</span>
                            </div>
                            <span className="font-data tabular text-[11px] text-ki-on-surface-muted w-16 text-right">
                              {fmtSize(f.size)}
                            </span>
                            <span className="flex items-center gap-1.5 w-24 justify-end">
                              <span className="w-1.5 h-1.5 rounded-full bg-ki-success" />
                              <span className="font-data text-[11px] text-ki-success">ready</span>
                            </span>
                            <button
                              onClick={() => removeFile(i)}
                              className="w-7 h-7 grid place-items-center text-ki-on-surface-muted hover:text-ki-error transition-colors"
                              aria-label="Rimuovi"
                            >
                              <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 400" }}>close</span>
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Retrieval settings collapsible */}
                  <details className="mt-1">
                    <summary className="flex items-center gap-2 cursor-pointer py-2 list-none">
                      <span className="material-symbols-outlined text-[13px] text-ki-on-surface-muted" style={{ fontVariationSettings: "'wght' 400" }}>tune</span>
                      <span className="text-[12px] text-ki-on-surface-secondary">Retrieval settings</span>
                      <span className="font-data text-[11px] text-ki-on-surface-muted">
                        {rag.mode} · chunk {rag.chunkSize} · overlap {rag.overlap} · top-{rag.topK}
                      </span>
                    </summary>
                    <div className="bg-ki-surface-raised border border-ki-border rounded p-4 mt-2 grid grid-cols-2 gap-4">
                      <div className="flex flex-col gap-1.5">
                        <div className="eyebrow">Retrieval mode</div>
                        <div className="flex gap-1">
                          {(["dense", "sparse", "hybrid"] as const).map((m) => (
                            <button
                              key={m}
                              onClick={() => setRag((r) => ({ ...r, mode: m }))}
                              className={`inline-flex items-center px-2.5 h-6 rounded-sm text-[11px] transition-colors ${
                                rag.mode === m ? "bg-ki-primary-soft text-ki-primary" : "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover"
                              }`}
                            >
                              {m}
                            </button>
                          ))}
                        </div>
                        <p className="text-[11px] text-ki-on-surface-muted">Hybrid combina BM25 + dense embeddings.</p>
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="eyebrow">Embedder</div>
                        <select
                          value={rag.embedder}
                          onChange={(e) => setRag((r) => ({ ...r, embedder: e.target.value }))}
                          className="h-7 px-2 bg-ki-surface-sunken border border-ki-border rounded-sm text-[12px] text-ki-on-surface focus:outline-none focus:border-ki-primary"
                        >
                          <option>text-embedding-3-large</option>
                          <option>text-embedding-3-small</option>
                          <option>voyage-3-large</option>
                          <option>nomic-embed-text-v1.5 (local)</option>
                        </select>
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex justify-between"><div className="eyebrow">Chunk size</div><span className="font-data tabular text-[11px] text-ki-on-surface-secondary">{rag.chunkSize} tok</span></div>
                        <input type="range" min={200} max={2000} step={50} value={rag.chunkSize} onChange={(e) => setRag((r) => ({ ...r, chunkSize: +e.target.value }))} className="accent-ki-primary" />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex justify-between"><div className="eyebrow">Overlap</div><span className="font-data tabular text-[11px] text-ki-on-surface-secondary">{rag.overlap} tok</span></div>
                        <input type="range" min={0} max={400} step={10} value={rag.overlap} onChange={(e) => setRag((r) => ({ ...r, overlap: +e.target.value }))} className="accent-ki-primary" />
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex justify-between"><div className="eyebrow">Top-K per query</div><span className="font-data tabular text-[11px] text-ki-on-surface-secondary">{rag.topK}</span></div>
                        <input type="range" min={1} max={20} step={1} value={rag.topK} onChange={(e) => setRag((r) => ({ ...r, topK: +e.target.value }))} className="accent-ki-primary" />
                      </div>
                    </div>
                  </details>
                </section>

                {/* KPIs */}
                <section className="flex flex-col gap-2">
                  <div className="flex items-end justify-between">
                    <div>
                      <div className="eyebrow">KPI da monitorare</div>
                      <p className="text-[11px] text-ki-on-surface-muted mt-0.5">
                        Metriche custom emerse dal brief, valutate per round
                      </p>
                    </div>
                    <button
                      onClick={suggestKpis}
                      disabled={!brief.trim() || loadingKpis}
                      className="inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] text-ki-primary border border-ki-primary/25 hover:bg-ki-primary-soft disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      {loadingKpis ? "Analisi…" : (
                        <>
                          <span className="material-symbols-outlined text-[12px]" style={{ fontVariationSettings: "'wght' 500" }}>auto_awesome</span>
                          Suggerisci dal brief
                        </>
                      )}
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {suggestedKpis.map((kpi) => {
                      const sel = selectedKpis.includes(kpi);
                      return (
                        <button
                          key={kpi}
                          onClick={() => toggleKpi(kpi)}
                          className={`inline-flex items-center gap-1 px-2 h-6 rounded-sm text-[11px] transition-colors ${
                            sel
                              ? "bg-ki-primary-soft text-ki-primary border border-transparent"
                              : "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover"
                          }`}
                        >
                          {sel && <span className="text-[10px]">✓</span>}
                          {kpi}
                        </button>
                      );
                    })}
                  </div>
                  {customKpis.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {customKpis.map((kpi) => (
                        <button
                          key={kpi}
                          onClick={() => toggleKpi(kpi)}
                          className="inline-flex items-center gap-1 px-2 h-6 rounded-sm text-[11px] bg-ki-primary-soft text-ki-primary"
                        >
                          <span className="text-[10px]">✓</span>
                          {kpi}
                        </button>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-1.5">
                    <input
                      value={customKpi}
                      onChange={(e) => setCustomKpi(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addCustomKpi(); } }}
                      placeholder="Aggiungi KPI personalizzato…"
                      className="flex-1 bg-ki-surface-sunken border border-ki-border rounded-sm px-2 h-7 text-[12px] text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary"
                    />
                    <button
                      onClick={addCustomKpi}
                      disabled={!customKpi.trim()}
                      className="inline-flex items-center h-7 px-2.5 rounded-sm text-[11px] text-ki-on-surface border border-ki-border hover:bg-ki-surface-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      Aggiungi
                    </button>
                  </div>
                </section>
              </div>
            )}

            {/* ── STEP 3 — Engine ──────────────────────────── */}
            {step === 3 && (
              <div className="flex flex-col gap-6">
                <div>
                  <div className="eyebrow">Step 3 of 4</div>
                  <h1 className="text-[28px] font-medium tracking-tight2 text-ki-on-surface mt-1 leading-[1.1]">
                    Engine configuration
                  </h1>
                  <p className="text-[13px] text-ki-on-surface-secondary leading-relaxed max-w-xl mt-2">
                    Time horizon, modello LLM, moduli computazionali.
                  </p>
                </div>

                {/* Rounds */}
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between"><label className="text-[11px] text-ki-on-surface-secondary">Rounds</label><span className="font-data tabular text-[12px] text-ki-on-surface">{rounds}</span></div>
                  <input type="range" min={2} max={12} value={rounds} onChange={(e) => setRounds(+e.target.value)} className="accent-ki-primary" />
                  <div className="flex justify-between font-data tabular text-[11px] text-ki-on-surface-muted">
                    <span>2 — fast</span><span>6 — balanced</span><span>12 — high-fidelity</span>
                  </div>
                </div>

                {/* Budget */}
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between"><label className="text-[11px] text-ki-on-surface-secondary">Budget massimo</label><span className="font-data tabular text-[12px] text-ki-on-surface">${budget.toFixed(1)}</span></div>
                  <input type="range" min={0.5} max={10} step={0.5} value={budget} onChange={(e) => setBudget(+e.target.value)} className="accent-ki-primary" />
                </div>

                {/* Provider */}
                <div className="flex flex-col gap-2">
                  <label className="text-[11px] text-ki-on-surface-secondary">LLM provider</label>
                  <div className="flex gap-1.5">
                    {PROVIDERS.map((p) => (
                      <button
                        key={p.id}
                        onClick={() => setProvider(p.id)}
                        className={`inline-flex items-center px-3 h-7 rounded-sm text-[12px] transition-colors ${
                          provider === p.id ? "bg-ki-primary-soft text-ki-primary" : "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover"
                        }`}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Modules */}
                <div className="flex flex-col gap-2">
                  <label className="text-[11px] text-ki-on-surface-secondary">Modules</label>
                  <div className="bg-ki-surface-raised border border-ki-border rounded">
                    <label className="flex items-start gap-3 px-4 py-3 border-b border-ki-border-faint cursor-pointer">
                      <input type="checkbox" checked={eliteOnly} onChange={(e) => setEliteOnly(e.target.checked)} className="mt-0.5 w-3.5 h-3.5 accent-ki-primary" />
                      <div className="flex-1">
                        <div className="text-[13px] font-medium text-ki-on-surface">Solo agenti Elite</div>
                        <div className="text-[11px] text-ki-on-surface-muted">Salta istituzionali e cittadini · più veloce ed economico</div>
                      </div>
                    </label>
                    <label className="flex items-start gap-3 px-4 py-3 cursor-pointer">
                      <input type="checkbox" checked={monteCarlo} onChange={(e) => setMonteCarlo(e.target.checked)} className="mt-0.5 w-3.5 h-3.5 accent-ki-primary" />
                      <div className="flex-1">
                        <div className="text-[13px] font-medium text-ki-on-surface">Monte Carlo bands</div>
                        <div className="text-[11px] text-ki-on-surface-muted">Run paralleli per intervalli di confidenza · costo zero</div>
                        {monteCarlo && (
                          <div className="mt-2">
                            <div className="flex justify-between"><span className="text-[11px] text-ki-on-surface-secondary">N° run</span><span className="font-data tabular text-[11px] text-ki-on-surface">{monteCarloRuns}</span></div>
                            <input type="range" min={5} max={100} step={5} value={monteCarloRuns} onChange={(e) => setMonteCarloRuns(+e.target.value)} className="w-full accent-ki-primary" />
                          </div>
                        )}
                      </div>
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* ── STEP 4 — Review ──────────────────────────── */}
            {step === 4 && (
              <div className="flex flex-col gap-6">
                <div>
                  <div className="eyebrow">Step 4 of 4</div>
                  <h1 className="text-[28px] font-medium tracking-tight2 text-ki-on-surface mt-1 leading-[1.1]">
                    Review & launch
                  </h1>
                </div>

                <div className="bg-ki-surface-raised border border-ki-border rounded">
                  {([
                    ["Name",         name || <span className="text-ki-on-surface-muted">— (auto-generated)</span>],
                    ["Domain",       <span className="inline-flex items-center gap-2"><span className="w-[2px] h-3 rounded-sm" style={{ background: domain ? `var(--d-${domain === "public_health" ? "health" : domain})` : "var(--ink-3)" }} /> {DOMAINS.find((d) => d.id === domain)?.label || "Auto-detect"}</span>],
                    ["Brief",        brief.length > 80 ? brief.slice(0, 80) + "…" : brief || <span className="text-ki-on-surface-muted">—</span>],
                    ["Knowledge base", `${files.length} doc${files.length === 1 ? "" : "s"} · ${fmtSize(totalSize)} · ${rag.mode} retrieval, top-${rag.topK}`],
                    ["KPI tracked",  selectedKpis.length > 0 ? `${selectedKpis.length} metriche custom` : <span className="text-ki-on-surface-muted">— (auto)</span>],
                    ["Engine",       `${PROVIDERS.find((p) => p.id === provider)?.label} · ${rounds} rounds${eliteOnly ? " · elite-only" : ""}${monteCarlo ? ` · Monte Carlo (n=${monteCarloRuns})` : ""}`],
                    ["Budget",       `$${budget.toFixed(2)}`],
                    ["Estimated runtime", `~${estimatedRuntime} min`],
                  ] as Array<[string, React.ReactNode]>).map(([k, v], i, arr) => (
                    <div key={k} className={`flex px-4 py-3 ${i < arr.length - 1 ? "border-b border-ki-border-faint" : ""}`}>
                      <span className="w-[180px] text-[11px] text-ki-on-surface-secondary uppercase tracking-[0.06em] font-medium">{k}</span>
                      <span className="text-[13px] text-ki-on-surface flex-1 min-w-0">{v}</span>
                    </div>
                  ))}
                </div>

                <div className="bg-ki-primary-soft border border-transparent rounded p-3 flex items-start gap-2">
                  <span className="material-symbols-outlined text-[14px] text-ki-primary flex-shrink-0 mt-0.5" style={{ fontVariationSettings: "'wght' 500" }}>info</span>
                  <span className="text-[12px] text-ki-primary">
                    <strong className="font-medium">Briefing engine ready.</strong> Stakeholder e fazioni saranno inferiti dal brief e confermati prima del round 1.
                  </span>
                </div>

                {error && (
                  <div className="bg-ki-error-soft border border-ki-error/30 rounded p-3 text-[12px] text-ki-error">
                    {error}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-8 py-3 border-t border-ki-border bg-ki-surface-raised">
          <button
            onClick={() => router.push("/")}
            className="inline-flex items-center h-8 px-3 rounded-sm text-[12px] text-ki-on-surface-secondary hover:bg-ki-surface-hover transition-colors"
          >
            Cancel
          </button>
          <div className="flex items-center gap-2">
            {step > 1 && (
              <button
                onClick={prev}
                className="inline-flex items-center h-8 px-3 rounded-sm text-[12px] text-ki-on-surface border border-ki-border hover:bg-ki-surface-hover transition-colors"
              >
                Back
              </button>
            )}
            {step < 4 && (
              <button
                onClick={next}
                disabled={!canContinue}
                className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-ki-on-surface text-ki-surface text-[12px] font-medium hover:bg-ki-on-surface-secondary disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
              >
                Continue
                <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>arrow_forward</span>
              </button>
            )}
            {step === 4 && (
              <button
                onClick={handleLaunch}
                disabled={submitting || !brief.trim()}
                className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-ki-primary text-white text-[12px] font-medium hover:bg-ki-primary-muted disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
              >
                {submitting ? (
                  <>
                    <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Launching…
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>play_arrow</span>
                    Launch simulation
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
