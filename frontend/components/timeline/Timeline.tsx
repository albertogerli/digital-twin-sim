"use client";

import SectionHeader from "@/components/ui/SectionHeader";
import RoundChapter from "./RoundChapter";

interface RoundEvent {
  event: string;
  shock_magnitude?: number;
  shock_direction?: number;
}

interface RoundPost {
  id: string;
  author_name: string;
  platform: string;
  text: string;
  total_engagement: number;
}

export interface RoundData {
  round: number;
  label?: string;
  event: RoundEvent;
  top_posts: RoundPost[];
  polarization: number;
  avg_position?: number;
  coalition_snapshot?: Record<string, number>;
  key_insight?: string;
}

interface GraphSnapshot {
  nodes: any[];
  edges: any[];
}

interface Props {
  rounds: RoundData[];
  graphSnapshots?: GraphSnapshot[];
}

export default function Timeline({ rounds }: Props) {
  return (
    <section id="timeline" className="bg-white py-20 px-4">
      <div className="max-w-5xl mx-auto">
        <SectionHeader
          title="Simulation Timeline"
          subtitle="Round-by-round breakdown of events, viral content, and shifting dynamics."
        />

        <div className="space-y-4">
          {rounds.map((round, i) => (
            <RoundChapter
              key={round.round}
              round={round}
              defaultOpen={i === 0}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
