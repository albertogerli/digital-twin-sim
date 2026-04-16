"use client";

interface RoundPhaseIndicatorProps {
  phaseIndex: number;
  totalPhases: number;
  message: string;
}

const PHASE_NAMES = [
  "Evento",
  "Elite",
  "Istituzionali",
  "Cittadini",
  "Platform",
  "Opinioni",
  "Checkpoint",
];

export default function RoundPhaseIndicator({ phaseIndex, totalPhases, message }: RoundPhaseIndicatorProps) {
  if (phaseIndex <= 0) return null;

  const phases = totalPhases || 7;

  return (
    <div className="flex items-center gap-1.5 mt-1.5">
      {Array.from({ length: phases }, (_, i) => {
        const step = i + 1;
        const done = step < phaseIndex;
        const active = step === phaseIndex;
        return (
          <div key={i} className="flex items-center gap-1.5">
            <div
              className={`h-1 rounded-full transition-all duration-300 ${
                done
                  ? "bg-ki-success w-6"
                  : active
                  ? "bg-ki-primary w-8 animate-pulse"
                  : "bg-ki-surface-sunken w-6"
              }`}
              title={PHASE_NAMES[i] || `Phase ${step}`}
            />
          </div>
        );
      })}
      <span className="text-[10px] text-ki-on-surface-muted font-data ml-2 truncate">{message}</span>
    </div>
  );
}
