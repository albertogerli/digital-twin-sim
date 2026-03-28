"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import type { ReplayMeta, RoundData } from "@/lib/types";

const CommandCenter = dynamic(
  () => import("@/components/replay/CommandCenter"),
  {
    ssr: false,
    loading: () => <ReplayLoadingSkeleton />,
  }
);

function ReplayLoadingSkeleton() {
  return (
    <div className="h-screen bg-gray-50 flex flex-col items-center justify-center gap-4">
      <div className="w-16 h-16 rounded-full border-2 border-blue-500/30 border-t-blue-500 animate-spin" />
      <p className="text-gray-500 text-sm font-mono">Loading simulation replay...</p>
    </div>
  );
}

async function fetchWithFallback<T>(
  id: string,
  file: string
): Promise<T> {
  const apiUrl = `/api/scenarios/${id}/${file}`;
  const staticUrl = `/data/scenario_${id}/${file}`;
  try {
    const r = await fetch(apiUrl);
    if (r.ok) return r.json();
  } catch {
    // API not available
  }
  const r2 = await fetch(staticUrl);
  if (!r2.ok) throw new Error(`Failed to fetch ${file} (status ${r2.status})`);
  return r2.json();
}

export default function ReplayPage({
  params,
}: {
  params: { id: string };
}) {
  const { id } = params;

  const [meta, setMeta] = useState<ReplayMeta | null>(null);
  const [rounds, setRounds] = useState<RoundData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        // Load metadata first to know round count
        const replayMeta = await fetchWithFallback<ReplayMeta>(id, "replay_meta.json");
        if (cancelled) return;
        setMeta(replayMeta);

        // Load all rounds in parallel
        const numRounds = replayMeta.totalRounds;
        const roundPromises = Array.from({ length: numRounds }, (_, i) =>
          fetchWithFallback<RoundData>(id, `replay_round_${i + 1}.json`).catch(() => null)
        );
        const rawRounds = await Promise.all(roundPromises);
        if (cancelled) return;

        const validRounds = rawRounds
          .filter((r): r is RoundData => r != null)
          .sort((a, b) => a.round - b.round);

        setRounds(validRounds);
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        console.error("Failed to load replay data:", err);
        setError(err instanceof Error ? err.message : "Failed to load replay data.");
        setLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [id]);

  if (loading) {
    return <ReplayLoadingSkeleton />;
  }

  if (error || !meta) {
    return (
      <div className="h-screen bg-gray-50 flex flex-col items-center justify-center gap-6 px-4">
        <div className="bg-red-950/30 border border-red-900/50 rounded-xl p-8 max-w-md text-center">
          <p className="text-red-300 text-lg font-medium mb-2">Failed to Load Replay</p>
          <p className="text-red-600/70 text-sm mb-6">{error || "Replay data not found."}</p>
          <Link
            href={`/scenario/${id}`}
            className="inline-flex items-center gap-2 text-cyan-600 hover:text-blue-600 transition-colors text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to scenario
          </Link>
        </div>
      </div>
    );
  }

  return (
    <CommandCenter
      scenarioId={id}
      meta={meta}
      rounds={rounds}
    />
  );
}
