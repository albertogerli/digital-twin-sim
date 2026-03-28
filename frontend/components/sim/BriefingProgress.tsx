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
    <div className="bg-white border border-gray-200 rounded-xl p-5 mb-6">
      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-4">
        Preparazione Scenario
      </p>
      <div className="space-y-3">
        {steps.map((step, i) => {
          const icon = PHASE_ICONS[step.phase] || "\u2699\uFE0F";
          const label = PHASE_LABELS[step.phase] || step.phase;
          const isLast = i === steps.length - 1;
          const isActive = isLast && !step.done;

          return (
            <div key={`${step.phase}-${i}`} className="flex items-start gap-3">
              {/* Stepper dot */}
              <div className="flex flex-col items-center">
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs flex-shrink-0 ${
                  step.done
                    ? "bg-emerald-100 text-emerald-600"
                    : isActive
                    ? "bg-blue-100 text-blue-600 animate-pulse"
                    : "bg-gray-100 text-gray-400"
                }`}>
                  {step.done ? "\u2713" : icon}
                </div>
                {i < steps.length - 1 && (
                  <div className={`w-px h-4 ${step.done ? "bg-emerald-200" : "bg-gray-200"}`} />
                )}
              </div>
              {/* Text */}
              <div className="pt-0.5">
                <p className={`text-xs font-semibold ${step.done ? "text-gray-700" : isActive ? "text-blue-700" : "text-gray-400"}`}>
                  {label}
                </p>
                <p className="text-[11px] text-gray-400 mt-0.5">{step.message}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
