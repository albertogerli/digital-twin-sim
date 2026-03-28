"use client";

import type { RealWorldEffects } from "@/lib/replay/types";

interface Props {
  effects: RealWorldEffects | null;
}

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
      <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
    </div>
  );
}

function Metric({ label, value, unit, delta, color }: { label: string; value: string | number; unit?: string; delta?: number; color?: string }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="text-[9px] text-gray-400">{label}</span>
      <div className="flex items-center gap-1">
        <span className={`text-[10px] font-mono font-bold tabular-nums ${color ?? "text-gray-800"}`}>
          {value}{unit && <span className="text-gray-400 font-normal ml-0.5">{unit}</span>}
        </span>
        {delta !== undefined && delta !== 0 && (
          <span className={`text-[8px] font-mono ${delta > 0 ? "text-green-600" : "text-red-600"}`}>
            {delta > 0 ? "+" : ""}{typeof delta === "number" ? delta.toFixed(1) : delta}
          </span>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="py-3">
      <div className="flex items-center gap-1.5 mb-2">
        <span className="font-mono text-[9px] text-gray-500 uppercase tracking-wider font-bold">{title}</span>
      </div>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

export default function RealWorldEffectsPanel({ effects }: Props) {
  if (!effects) {
    return (
      <div className="h-full flex items-center justify-center py-8">
        <span className="text-[10px] font-mono text-gray-400">Waiting for data...</span>
      </div>
    );
  }

  const { overview, opinion, engagement } = effects;

  return (
    <div className="h-full flex flex-col divide-y divide-gray-200 overflow-y-auto">
      {/* Overview */}
      <Section title="Overview">
        <Metric
          label="Stability Index"
          value={overview.stability_index}
          unit="/100"
          color={overview.stability_index > 60 ? "text-green-600" : overview.stability_index > 40 ? "text-amber-600" : "text-red-600"}
        />
        <MiniBar
          value={overview.stability_index}
          max={100}
          color={overview.stability_index > 60 ? "#22c55e" : overview.stability_index > 40 ? "#f59e0b" : "#ef4444"}
        />

        <Metric
          label="Tension Index"
          value={overview.tension_index}
          unit="/100"
          color={overview.tension_index > 60 ? "text-red-600" : overview.tension_index > 40 ? "text-amber-600" : "text-green-600"}
        />
        <MiniBar
          value={overview.tension_index}
          max={100}
          color={overview.tension_index > 60 ? "#ef4444" : overview.tension_index > 40 ? "#f59e0b" : "#22c55e"}
        />

        <Metric
          label="Media Intensity"
          value={overview.media_intensity}
          unit="/100"
        />

        <Metric
          label="Stakeholder Confidence"
          value={overview.stakeholder_confidence}
          unit="/100"
          color={overview.stakeholder_confidence > 60 ? "text-green-600" : overview.stakeholder_confidence > 40 ? "text-amber-600" : "text-red-600"}
        />
      </Section>

      {/* Opinion */}
      <Section title="Public Opinion">
        <div className="grid grid-cols-3 gap-1 mb-2">
          {[
            { label: "Support", value: opinion.support_rate, color: "text-green-600" },
            { label: "Opposition", value: opinion.opposition_rate, color: "text-red-600" },
            { label: "Undecided", value: opinion.undecided_rate, color: "text-gray-500" },
          ].map((item) => (
            <div key={item.label} className="text-center">
              <div className={`text-[11px] font-mono font-bold ${item.color}`}>
                {Math.round(item.value)}%
              </div>
              <div className="text-[7px] text-gray-400 leading-tight">{item.label}</div>
            </div>
          ))}
        </div>

        {/* Opinion bar */}
        <div className="h-3 rounded-full overflow-hidden flex bg-gray-100">
          <div
            className="h-full bg-green-500 transition-all duration-700"
            style={{ width: `${opinion.support_rate}%` }}
          />
          <div
            className="h-full bg-gray-400 transition-all duration-700"
            style={{ width: `${opinion.undecided_rate}%` }}
          />
          <div
            className="h-full bg-red-500 transition-all duration-700"
            style={{ width: `${opinion.opposition_rate}%` }}
          />
        </div>

        <Metric
          label="Avg Position"
          value={opinion.avg_position > 0 ? `+${opinion.avg_position.toFixed(2)}` : opinion.avg_position.toFixed(2)}
          color={opinion.avg_position > 0 ? "text-green-600" : opinion.avg_position < 0 ? "text-red-600" : "text-gray-500"}
        />
      </Section>

      {/* Engagement */}
      <Section title="Engagement Metrics">
        <Metric label="Active Discussions" value={engagement.active_discussions} />
        <Metric label="Viral Content" value={engagement.viral_content_pieces} />
        <Metric label="Coalition Count" value={engagement.coalition_count} />
      </Section>
    </div>
  );
}
