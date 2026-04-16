"use client";

import { useState, useRef, useEffect } from "react";

interface CommandBarProps {
  onSubmit: (text: string) => void;
  processing: boolean;
  lastIntervention: string;
  round: number;
  disabled?: boolean;
  awaiting?: boolean;
}

export function CommandBar({ onSubmit, processing, lastIntervention, round, disabled, awaiting }: CommandBarProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (awaiting) inputRef.current?.focus();
  }, [awaiting, processing]);

  const handleSubmit = () => {
    if (!input.trim() || processing || disabled) return;
    onSubmit(input.trim());
    setInput("");
  };

  const isLocked = processing || disabled;

  return (
    <div className="border-t border-ki-border bg-ki-surface-sunken shrink-0">
      {/* Last intervention echo */}
      {lastIntervention && (
        <div className="h-5 flex items-center px-2 border-b border-ki-border-strong gap-2">
          <span className="font-data text-[9px] text-ki-on-surface-muted">LAST INTERVENTION R{round}:</span>
          <span className="font-data text-[9px] text-[#00d26a] truncate">{lastIntervention}</span>
        </div>
      )}

      {/* Input row */}
      <div className="flex items-center h-9 px-2 gap-2">
        <span className="font-data text-[10px] text-ki-on-surface-muted shrink-0">
          {processing ? (
            <span className="text-[#ffaa00] animate-pulse">PROCESSING...</span>
          ) : awaiting ? (
            <span className="text-[#ff3b3b] animate-pulse">AWAITING YOUR MOVE R{round + 1} $</span>
          ) : disabled ? (
            <span className="text-ki-on-surface-muted">SIMULATION ENDED</span>
          ) : (
            <>INTERVENTION R{round + 1} $</>
          )}
        </span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          disabled={isLocked}
          placeholder={awaiting
            ? 'SCRIVI LA TUA CONTROMOSSA — comunicato stampa, azione politica, dichiarazione...'
            : 'es. "Emetto un comunicato stampa di scuse profonde e ritiro la manovra controversa"'}
          className="flex-1 bg-transparent font-data text-[11px] text-ki-on-surface placeholder:text-ki-on-surface-muted focus:outline-none disabled:opacity-40"
        />
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || isLocked}
          className={`font-data text-[9px] px-3 py-1 border transition-colors disabled:opacity-20 ${
            awaiting
              ? "border-[#ff3b3b20] text-[#ff3b3b] hover:bg-[#ff3b3b08] hover:border-[#ff3b3b50]"
              : "border-ki-border text-ki-on-surface-muted hover:text-ki-on-surface hover:border-ki-border-strong"
          }`}
        >
          EXECUTE
        </button>
      </div>

      {/* Help hint */}
      <div className="h-4 flex items-center px-2 border-t border-ki-border-strong">
        <span className="font-data text-[8px] text-ki-on-surface-muted">
          {awaiting
            ? "⚠ SIMULATION PAUSED — TYPE YOUR COUNTER-MOVE AND PRESS ENTER. ALL AGENTS WILL REACT."
            : "TYPE YOUR COUNTER-MOVE. ALL AGENTS WILL REACT. ENTER TO SUBMIT."}
        </span>
      </div>
    </div>
  );
}
