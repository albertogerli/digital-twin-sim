"use client";

interface BriefingStep {
  phase: string;
  message: string;
  done: boolean;
}

interface BriefingProgressProps {
  steps: BriefingStep[];
  visible: boolean;
}

const PHASE_ICONS: Record<string, string> = {
  web_research: "\uD83C\uDF10",
  documents: "\uD83D\uDCC4",
  entity_research: "\uD83D\uDD0D",
  agent_generation: "\uD83E\uDDD1\u200D\uD83D\uDCBB",
  brief_analysis: "\uD83D\uDCCA",
};

const PHASE_LABELS: Record<string, string> = {
  web_research: "Ricerca Web",
  documents: "Documenti",
  entity_research: "Ricerca Entit\u00E0",
  agent_generation: "Generazione Agenti",
  brief_analysis: "Analisi Brief",
};

export default function BriefingProgress({ steps, visible }: BriefingProgressProps) {
  if (!visible || steps.length === 0) return null;

  return (
    <div className="bg-ki-surface-raised border border-ki-border rounded-sm p-3 mb-4">
      <p className="font-data text-[10px] text-ki-on-surface-muted uppercase tracking-wider mb-3">
        Preparazione Scenario
      </p>
      <div className="space-y-2">
        {steps.map((step, i) => {
          const icon = PHASE_ICONS[step.phase] || "\u2699\uFE0F";
          const label = PHASE_LABELS[step.phase] || step.phase;
          const isLast = i === steps.length - 1;
          const isActive = isLast && !step.done;

          return (
            <div key={`${step.phase}-${i}`} className="flex items-start gap-2">
              {/* Stepper dot */}
              <div className="flex flex-col items-center">
                <div className={`w-5 h-5 rounded-sm flex items-center justify-center text-xs flex-shrink-0 ${
                  step.done
                    ? "bg-ki-success/15 text-ki-success"
                    : isActive
                    ? "bg-ki-primary/15 text-ki-primary animate-pulse"
                    : "bg-ki-surface-sunken text-ki-on-surface-muted"
                }`}>
                  {step.done ? "\u2713" : icon}
                </div>
                {i < steps.length - 1 && (
                  <div className={`w-px h-3 ${step.done ? "bg-ki-success/30" : "bg-ki-border"}`} />
                )}
              </div>
              {/* Text */}
              <div className="pt-0.5">
                <p className={`text-xs font-semibold ${step.done ? "text-ki-on-surface-secondary" : isActive ? "text-ki-primary" : "text-ki-on-surface-muted"}`}>
                  {label}
                </p>
                <p className="text-[11px] text-ki-on-surface-muted mt-0.5">{step.message}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
