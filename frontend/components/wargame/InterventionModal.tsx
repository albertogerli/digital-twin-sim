"use client";

import { useState, useMemo, useRef } from "react";

/* ───────────────────────────────────────────────────────────
   Wargame Intervene modal — actor + action + message/KB-doc
   + Monte Carlo predicted consequences.
   Pattern from screen-other.jsx + design v2 chat (KB inject).
   ─────────────────────────────────────────────────────────── */

interface Props {
  open: boolean;
  round: number;
  onClose: () => void;
  onSubmit: (payload: {
    actor: string;
    action: string;
    message: string;
    /** When action === "inject_kb": the document to ingest into the live RAG store. */
    kbDoc?: {
      name: string;
      size: number;
      sourceType: "file" | "url";
      sourceRef: string;
      text: string;        // Raw text body sent to backend rag_store.add_document()
    };
  }) => void;
  submitting?: boolean;
}

const ACTORS = [
  { id: "commission", label: "EU Commission" },
  { id: "industry",   label: "Vornex Industries" },
  { id: "advocacy",   label: "Climate Coalition" },
  { id: "media",      label: "Reuters wire" },
  { id: "custom",     label: "+ Custom actor" },
];

const ACTIONS = [
  { id: "press_release", label: "Press release",   desc: "Public statement, structured framing" },
  { id: "leak",          label: "Strategic leak",  desc: "Anonymous to wire service" },
  { id: "concede",       label: "Concession",      desc: "Accept amendment package" },
  { id: "escalate",      label: "Escalation",      desc: "Threaten retaliatory measure" },
  { id: "inject_kb",     label: "Inject KB doc",   desc: "Add document to live agent knowledge base" },
];

// Toy "predictor": deterministic shifts based on actor+action combo
function predictConsequences(actor: string, action: string) {
  const base = {
    polDelta:  -0.05,
    sentDelta: +0.05,
    voteDelta: +0.05,
    confidence: 0.72,
  };
  if (action === "concede")    return { polDelta: -0.18, sentDelta: +0.14, voteDelta: +0.22, confidence: 0.78 };
  if (action === "escalate")   return { polDelta: +0.21, sentDelta: -0.16, voteDelta: -0.18, confidence: 0.71 };
  if (action === "leak")       return { polDelta: +0.08, sentDelta: -0.04, voteDelta: -0.06, confidence: 0.64 };
  if (action === "inject_kb")  return { polDelta: -0.06, sentDelta: +0.07, voteDelta: +0.10, confidence: 0.69 };
  if (actor === "media")       return { ...base, voteDelta: +0.12, confidence: 0.81 };
  return base;
}

function fmtSize(b: number) {
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  return (b / 1024 / 1024).toFixed(1) + " MB";
}

export function InterventionModal({ open, round, onClose, onSubmit, submitting }: Props) {
  const [actor, setActor] = useState("commission");
  const [action, setAction] = useState("press_release");
  const [message, setMessage] = useState(
    "The Commission will table a revised draft of Article 14 by end-of-week, addressing industrial timeline concerns while preserving 2030 grid investment targets."
  );
  const [kbFile, setKbFile] = useState<File | null>(null);
  const [kbUrl, setKbUrl] = useState("");
  const [kbMode, setKbMode] = useState<"file" | "url">("file");
  const fileRef = useRef<HTMLInputElement>(null);

  const pred = useMemo(() => predictConsequences(actor, action), [actor, action]);
  const isKbInject = action === "inject_kb";
  const canSubmit = isKbInject
    ? (kbMode === "file" ? !!kbFile : kbUrl.trim().length > 4)
    : message.trim().length > 0;

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center backdrop-blur-sm"
      style={{ background: "oklch(0 0 0 / 0.4)" }}
      onClick={onClose}
    >
      <div
        className="bg-ki-surface-raised border border-ki-border rounded-lg w-[720px] max-h-[85vh] flex flex-col shadow-[0_20px_60px_oklch(0_0_0_/_0.2)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between px-5 py-4 border-b border-ki-border">
          <div className="flex flex-col">
            <span className="eyebrow">Wargame · R{round}</span>
            <span className="text-[20px] font-medium tracking-tight2 text-ki-on-surface mt-0.5">
              Player intervention
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-7 h-7 grid place-items-center rounded-sm text-ki-on-surface-muted hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors"
          >
            <span className="material-symbols-outlined text-[16px]" style={{ fontVariationSettings: "'wght' 400" }}>close</span>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5">
          <p className="text-[13px] text-ki-on-surface-secondary leading-relaxed mb-5">
            Simulation paused. Inject a real-world action; the engine will resume and propagate consequences for the next 24 ticks before pausing again.
          </p>

          {/* Acting party */}
          <div className="flex flex-col gap-2 mb-5">
            <label className="text-[11px] text-ki-on-surface-secondary">Acting party</label>
            <div className="flex flex-wrap gap-1.5">
              {ACTORS.map((a) => (
                <button
                  key={a.id}
                  onClick={() => setActor(a.id)}
                  className={`inline-flex items-center px-3 h-7 rounded-sm text-[12px] transition-colors ${
                    actor === a.id
                      ? "bg-ki-primary-soft text-ki-primary"
                      : "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover"
                  }`}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          {/* Action type */}
          <div className="flex flex-col gap-2 mb-5">
            <label className="text-[11px] text-ki-on-surface-secondary">Action type</label>
            <div className="grid grid-cols-2 gap-2">
              {ACTIONS.map((a) => (
                <button
                  key={a.id}
                  onClick={() => setAction(a.id)}
                  className={`bg-ki-surface-raised border rounded p-3 text-left transition-colors ${
                    action === a.id
                      ? "border-ki-on-surface shadow-[inset_0_0_0_1px_var(--ink)]"
                      : "border-ki-border hover:border-ki-border-strong"
                  }`}
                >
                  <div className="text-[13px] font-medium text-ki-on-surface">{a.label}</div>
                  <div className="text-[11px] text-ki-on-surface-muted mt-1">{a.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Message OR KB document */}
          {!isKbInject ? (
            <div className="flex flex-col gap-2 mb-5">
              <label className="text-[11px] text-ki-on-surface-secondary">Message</label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={4}
                className="bg-ki-surface-sunken border border-ki-border rounded text-[13px] text-ki-on-surface px-3 py-2 focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30 resize-none leading-relaxed"
              />
            </div>
          ) : (
            <div className="flex flex-col gap-2 mb-5">
              <div className="flex items-baseline justify-between">
                <label className="text-[11px] text-ki-on-surface-secondary">Document source</label>
                <div className="flex gap-1">
                  <button
                    onClick={() => setKbMode("file")}
                    className={`inline-flex items-center px-2.5 h-6 rounded-sm text-[11px] transition-colors ${
                      kbMode === "file"
                        ? "bg-ki-primary-soft text-ki-primary"
                        : "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover"
                    }`}
                  >
                    File
                  </button>
                  <button
                    onClick={() => setKbMode("url")}
                    className={`inline-flex items-center px-2.5 h-6 rounded-sm text-[11px] transition-colors ${
                      kbMode === "url"
                        ? "bg-ki-primary-soft text-ki-primary"
                        : "bg-ki-surface-sunken text-ki-on-surface-secondary border border-ki-border hover:bg-ki-surface-hover"
                    }`}
                  >
                    URL
                  </button>
                </div>
              </div>

              {kbMode === "file" ? (
                <div className="relative border border-dashed border-ki-border-strong bg-ki-surface-sunken rounded p-4 flex items-center gap-3">
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".pdf,.docx,.doc,.txt,.md,.csv,.json,.html"
                    onChange={(e) => setKbFile(e.target.files?.[0] || null)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <div className="w-8 h-8 border border-ki-border-strong grid place-items-center bg-ki-surface-raised flex-shrink-0">
                    <span className="material-symbols-outlined text-[16px] text-ki-on-surface-secondary" style={{ fontVariationSettings: "'wght' 400" }}>
                      upload_file
                    </span>
                  </div>
                  <div className="flex-1 min-w-0">
                    {kbFile ? (
                      <>
                        <div className="text-[12px] text-ki-on-surface truncate">{kbFile.name}</div>
                        <div className="font-data tabular text-[11px] text-ki-on-surface-muted">{fmtSize(kbFile.size)}</div>
                      </>
                    ) : (
                      <>
                        <div className="text-[12px] font-medium text-ki-on-surface">Drop file or browse</div>
                        <div className="text-[11px] text-ki-on-surface-muted">PDF · DOCX · TXT · MD · CSV · JSON · HTML</div>
                      </>
                    )}
                  </div>
                  {kbFile && (
                    <button
                      onClick={(e) => { e.preventDefault(); setKbFile(null); }}
                      className="text-ki-on-surface-muted hover:text-ki-error transition-colors relative z-10"
                      aria-label="Remove file"
                    >
                      <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 400" }}>close</span>
                    </button>
                  )}
                </div>
              ) : (
                <input
                  value={kbUrl}
                  onChange={(e) => setKbUrl(e.target.value)}
                  placeholder="https://example.com/breaking-news/article"
                  className="h-9 px-3 bg-ki-surface-sunken border border-ki-border rounded text-[13px] text-ki-on-surface placeholder-ki-on-surface-muted focus:outline-none focus:border-ki-primary focus:ring-1 focus:ring-ki-primary/30"
                />
              )}

              <div className="bg-ki-warning-soft border border-ki-warning/30 rounded p-2 flex items-start gap-2">
                <span className="material-symbols-outlined text-[14px] text-ki-warning mt-0.5" style={{ fontVariationSettings: "'wght' 500" }}>info</span>
                <span className="text-[11px] text-ki-on-surface-secondary leading-relaxed">
                  Documento ingerito → embedded → indicizzato e disponibile nel KB degli agenti dal prossimo round. Es. breaking news, leak, statement.
                </span>
              </div>
            </div>
          )}

          {/* Predicted consequences (Monte Carlo) */}
          <div className="bg-ki-primary-soft rounded p-4">
            <div className="eyebrow text-ki-primary mb-2">Predicted consequences · Monte Carlo n=32</div>
            <div className="grid grid-cols-4 gap-3 mt-2">
              <div className="flex flex-col">
                <span className="text-[11px] text-ki-on-surface-secondary">Polarization Δ</span>
                <span className={`font-data tabular text-[16px] font-medium tracking-tight2 ${pred.polDelta < 0 ? "text-ki-success" : "text-ki-error"}`}>
                  {pred.polDelta > 0 ? "+" : ""}{pred.polDelta.toFixed(2)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[11px] text-ki-on-surface-secondary">Sentiment Δ</span>
                <span className={`font-data tabular text-[16px] font-medium tracking-tight2 ${pred.sentDelta > 0 ? "text-ki-success" : "text-ki-error"}`}>
                  {pred.sentDelta > 0 ? "+" : ""}{pred.sentDelta.toFixed(2)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[11px] text-ki-on-surface-secondary">Vote prob. Δ</span>
                <span className={`font-data tabular text-[16px] font-medium tracking-tight2 ${pred.voteDelta > 0 ? "text-ki-success" : "text-ki-error"}`}>
                  {pred.voteDelta > 0 ? "+" : ""}{Math.round(pred.voteDelta * 100)}%
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[11px] text-ki-on-surface-secondary">Confidence</span>
                <span className="font-data tabular text-[16px] font-medium tracking-tight2 text-ki-on-surface">
                  {pred.confidence.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-ki-border bg-ki-surface-sunken">
          <span className="text-[11px] text-ki-on-surface-muted">
            Action will appear in the replay timeline as a player intervention.
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="inline-flex items-center h-8 px-3 rounded-sm text-[12px] text-ki-on-surface border border-ki-border hover:bg-ki-surface-hover transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={async () => {
                if (isKbInject) {
                  if (kbMode === "file" && kbFile) {
                    // Read the file as text (text/markdown/csv/json/html). For PDF/DOCX,
                    // backend extraction is best — but here we only have the browser, so
                    // we send raw text and let the backend store it as-is.
                    const text = await kbFile.text();
                    onSubmit({
                      actor,
                      action,
                      message: `[KB inject] ${kbFile.name}`,
                      kbDoc: {
                        name: kbFile.name,
                        size: kbFile.size,
                        sourceType: "file",
                        sourceRef: kbFile.name,
                        text: text.slice(0, 200_000),   // hard cap matches backend
                      },
                    });
                  } else if (kbMode === "url" && kbUrl.trim()) {
                    // Fetch the URL via the backend later; for now ship URL+placeholder so
                    // ingestion still happens (backend can swap in a fetch step).
                    const slug = kbUrl.replace(/^https?:\/\//, "").slice(0, 60);
                    onSubmit({
                      actor,
                      action,
                      message: `[KB inject] ${slug}`,
                      kbDoc: {
                        name: slug,
                        size: 0,
                        sourceType: "url",
                        sourceRef: kbUrl,
                        text: `Source URL: ${kbUrl}\n\n(Document body to be fetched server-side.)`,
                      },
                    });
                  }
                } else {
                  onSubmit({ actor, action, message });
                }
              }}
              disabled={submitting || !canSubmit}
              className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-ki-primary text-white text-[12px] font-medium hover:bg-ki-primary-muted disabled:bg-ki-surface-sunken disabled:text-ki-on-surface-muted disabled:cursor-not-allowed transition-colors"
            >
              <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>
                {isKbInject ? "library_add" : "bolt"}
              </span>
              {submitting ? "Submitting…" : isKbInject ? "Inject doc & resume" : "Inject & resume"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
