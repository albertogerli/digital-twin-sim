"use client";

import { useEffect, useRef, useState } from "react";

/* ───────────────────────────────────────────────────────────
   Reusable info tooltip — small (i) glyph that opens a
   floating panel with technical-but-readable explanation.

   Usage:
     <InfoTip
       title="Polarization"
       body={
         <>
           Standard deviation of agent positions × 10, capped at 10.
           <br />
           <span className="text-ki-on-surface-muted">0 = consensus, 10 = full split</span>
         </>
       }
     />

   Click toggles. Click-outside closes. Position auto-flips left
   if the popover would overflow viewport on the right side.
   ─────────────────────────────────────────────────────────── */

interface InfoTipProps {
  /** Short bold title shown at the top of the popover. */
  title?: string;
  /** Body content — string or JSX. Keep < ~60 words. */
  body: React.ReactNode;
  /** Pixel size of the (i) glyph, default 11. */
  size?: number;
  /** Tailwind class for the glyph color — default ki-on-surface-muted. */
  className?: string;
  /** Force the popover side. Auto-flips by default. */
  side?: "auto" | "left" | "right";
}

export default function InfoTip({
  title,
  body,
  size = 11,
  className = "text-ki-on-surface-muted hover:text-ki-on-surface",
  side = "auto",
}: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const [flipLeft, setFlipLeft] = useState(false);
  const wrapRef = useRef<HTMLSpanElement | null>(null);

  // Decide left/right placement based on viewport on open
  useEffect(() => {
    if (!open) return;
    if (side === "left") { setFlipLeft(true); return; }
    if (side === "right") { setFlipLeft(false); return; }
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const POPOVER_W = 240;
    setFlipLeft(rect.left + POPOVER_W > window.innerWidth - 16);
  }, [open, side]);

  return (
    <span ref={wrapRef} className="relative inline-flex items-center align-middle">
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        aria-label={title ? `Spiegazione ${title}` : "Spiegazione"}
        aria-expanded={open}
        className={`inline-flex items-center justify-center rounded-full border border-ki-border-faint hover:border-ki-border ${className}`}
        style={{ width: size + 4, height: size + 4, lineHeight: 1 }}
      >
        <span style={{ fontSize: size, fontWeight: 600, fontFamily: "Georgia, serif" }}>i</span>
      </button>
      {open && (
        <>
          <span
            aria-hidden
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          />
          <span
            role="dialog"
            className={`absolute top-full mt-1.5 z-50 w-60 p-2.5 bg-ki-surface border border-ki-border-strong rounded-sm shadow-lg text-[11px] leading-relaxed text-ki-on-surface-secondary ${
              flipLeft ? "right-0" : "left-0"
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            {title && (
              <div className="font-data text-[10px] uppercase tracking-wider text-ki-on-surface-muted border-b border-ki-border-faint pb-1 mb-1.5">
                {title}
              </div>
            )}
            <div>{body}</div>
          </span>
        </>
      )}
    </span>
  );
}
