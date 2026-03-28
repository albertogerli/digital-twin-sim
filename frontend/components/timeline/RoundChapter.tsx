"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

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

interface RoundData {
  round: number;
  label?: string;
  event: RoundEvent;
  top_posts: RoundPost[];
  polarization: number;
  avg_position?: number;
  coalition_snapshot?: Record<string, number>;
  key_insight?: string;
}

interface Props {
  round: RoundData;
  defaultOpen?: boolean;
}

function formatEngagement(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

export default function RoundChapter({ round, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  const shockMag = round.event.shock_magnitude ?? 0;
  const shockDir = round.event.shock_direction ?? 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.5, delay: 0.05 }}
      className="relative"
    >
      {/* Timeline line */}
      <div className="absolute left-6 top-0 bottom-0 w-px bg-gray-200 hidden md:block" />

      {/* Clickable header */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full text-left flex items-center gap-4 group cursor-pointer"
      >
        <div
          className={`w-12 h-12 rounded-full border flex items-center justify-center flex-shrink-0 relative z-10 transition-colors ${
            open
              ? "bg-blue-500/15 border-blue-500/40"
              : "bg-white border-gray-300 group-hover:border-blue-500/30"
          }`}
        >
          <span
            className={`font-mono text-sm font-bold transition-colors ${
              open ? "text-cyan-600" : "text-gray-400 group-hover:text-cyan-600"
            }`}
          >
            {round.round}
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <h3 className="font-display text-lg md:text-xl font-bold text-gray-900">
              {round.label ?? `Round ${round.round}`}
            </h3>
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-full bg-gray-100 font-mono text-[10px] text-gray-400">
                {round.top_posts.length} posts
              </span>
              <span className="px-2 py-0.5 rounded-full bg-gray-100 font-mono text-[10px] text-gray-400">
                pol. {round.polarization.toFixed(2)}
              </span>
              {shockMag > 0 && (
                <span
                  className={`px-2 py-0.5 rounded-full font-mono text-[10px] ${
                    shockMag > 0.6
                      ? "bg-red-500/15 text-red-600"
                      : "bg-amber-500/15 text-amber-600"
                  }`}
                >
                  shock {shockMag.toFixed(1)}
                </span>
              )}
            </div>
          </div>
          <p className="font-body text-sm text-gray-400 mt-0.5">
            {round.event.event}
          </p>
        </div>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform flex-shrink-0 ${
            open ? "rotate-180" : ""
          }`}
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </button>

      {/* Collapsible content */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="md:ml-16 mt-6 space-y-6">
              {/* Event detail */}
              <div className="bg-white border border-gray-200 rounded-xl p-4">
                <p className="font-body text-sm text-gray-700 leading-relaxed">
                  {round.event.event}
                </p>
                {(shockMag > 0 || shockDir !== 0) && (
                  <div className="flex gap-4 mt-3">
                    <span className="font-mono text-[10px] text-gray-400">
                      Shock: {shockMag.toFixed(1)}
                    </span>
                    <span
                      className="font-mono text-[10px]"
                      style={{ color: shockDir > 0 ? "#4ade80" : "#f87171" }}
                    >
                      Dir: {shockDir > 0 ? "+" : ""}
                      {shockDir.toFixed(1)}
                    </span>
                  </div>
                )}
              </div>

              {/* Content grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Top posts */}
                <div className="lg:col-span-2 space-y-3">
                  <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider">
                    Top Posts This Round
                  </p>
                  {round.top_posts.slice(0, 3).map((post) => (
                    <div
                      key={post.id}
                      className="bg-gray-50 border border-gray-200 rounded-lg p-3"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 font-mono text-[10px] font-bold">
                          {post.author_name.charAt(0).toUpperCase()}
                        </div>
                        <span className="font-display text-xs font-semibold text-gray-700">
                          {post.author_name}
                        </span>
                        <span className="px-1.5 py-0.5 rounded bg-gray-100 font-mono text-[9px] text-gray-400 uppercase">
                          {post.platform}
                        </span>
                        <span className="ml-auto font-mono text-[10px] text-cyan-600">
                          {formatEngagement(post.total_engagement)}
                        </span>
                      </div>
                      <p className="font-body text-xs text-gray-500 leading-relaxed line-clamp-3">
                        {post.text}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Sidebar */}
                <div className="space-y-6">
                  {/* Coalition snapshot */}
                  {round.coalition_snapshot && (
                    <div>
                      <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                        Coalitions
                      </p>
                      <div className="space-y-1.5">
                        {Object.entries(round.coalition_snapshot).map(([name, size]) => (
                          <div key={name} className="flex items-center gap-2">
                            <span className="font-body text-xs text-gray-500 truncate flex-1">
                              {name}
                            </span>
                            <span className="font-mono text-[10px] text-gray-700 font-semibold">
                              {size as number}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Polarization bar */}
                  <div>
                    <p className="font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                      Polarization
                    </p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-amber-500"
                          style={{
                            width: `${Math.min(round.polarization * 100, 100)}%`,
                          }}
                        />
                      </div>
                      <span className="font-mono text-sm text-cyan-600">
                        {round.polarization.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Key insight */}
              {round.key_insight && (
                <div className="border-l-2 border-amber-500 pl-4 py-2">
                  <p className="font-body text-sm text-gray-500 italic">
                    {round.key_insight}
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
