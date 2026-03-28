"use client";

import { useMemo } from "react";
import SectionHeader from "@/components/ui/SectionHeader";
import ScrollReveal from "@/components/ui/ScrollReveal";

interface ViralPost {
  id: string;
  author_name: string;
  author_avatar?: string;
  platform: string;
  round: number;
  text: string;
  total_engagement: number;
  likes?: number;
  reposts?: number;
  replies?: number;
}

interface Props {
  posts: ViralPost[];
}

const platformStyles: Record<string, { bg: string; text: string }> = {
  social: { bg: "bg-blue-500/15", text: "text-cyan-600" },
  social_media: { bg: "bg-blue-500/15", text: "text-cyan-600" },
  forum: { bg: "bg-violet-500/15", text: "text-violet-600" },
  press: { bg: "bg-amber-500/15", text: "text-amber-600" },
  tv: { bg: "bg-red-500/15", text: "text-red-600" },
  official: { bg: "bg-emerald-500/15", text: "text-emerald-600" },
  street: { bg: "bg-orange-500/15", text: "text-orange-600" },
  trade_press: { bg: "bg-yellow-500/15", text: "text-yellow-600" },
  blog: { bg: "bg-cyan-500/15", text: "text-cyan-600" },
};

function getPlatformStyle(platform: string) {
  const key = platform.toLowerCase();
  return platformStyles[key] ?? { bg: "bg-gray-500/15", text: "text-gray-500" };
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

export default function ViralShowcase({ posts }: Props) {
  const topPosts = useMemo(
    () => [...posts].sort((a, b) => b.total_engagement - a.total_engagement).slice(0, 10),
    [posts]
  );

  return (
    <section id="viral-posts" className="bg-white py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <SectionHeader
          title="Viral Posts"
          subtitle="The most engaging content produced during the simulation, ranked by total engagement."
        />

        <div className="space-y-4">
          {topPosts.map((post, i) => {
            const pStyle = getPlatformStyle(post.platform);

            return (
              <ScrollReveal key={post.id} delay={Math.min(i * 0.04, 0.3)}>
                <div className="relative flex gap-4">
                  {/* Rank number */}
                  <div className="flex-shrink-0 w-8 pt-4">
                    <span className="font-mono text-xs text-gray-500 font-bold">
                      #{i + 1}
                    </span>
                  </div>

                  {/* Card */}
                  <div className="flex-1 bg-white border border-gray-200 rounded-xl p-4 hover:border-gray-300 transition-colors">
                    {/* Author row */}
                    <div className="flex items-center gap-3 mb-3">
                      {/* Avatar */}
                      <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-gray-500 font-mono text-xs font-bold flex-shrink-0">
                        {post.author_avatar ? (
                          <span>{post.author_avatar}</span>
                        ) : (
                          <span>{post.author_name.charAt(0).toUpperCase()}</span>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <span className="font-display text-sm font-semibold text-gray-800 truncate block">
                          {post.author_name}
                        </span>
                      </div>
                      {/* Platform badge */}
                      <span
                        className={`px-2 py-0.5 rounded-full font-mono text-[10px] uppercase ${pStyle.bg} ${pStyle.text}`}
                      >
                        {post.platform}
                      </span>
                      {/* Round */}
                      <span className="px-2 py-0.5 rounded-full bg-gray-100 font-mono text-[10px] text-gray-400">
                        R{post.round}
                      </span>
                    </div>

                    {/* Post text */}
                    <p className="font-body text-sm text-gray-700 leading-relaxed mb-3">
                      {post.text}
                    </p>

                    {/* Engagement metrics */}
                    <div className="flex items-center gap-4">
                      {post.likes != null && (
                        <span className="flex items-center gap-1 font-mono text-[11px] text-gray-400">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-red-600/70">
                            <path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z" />
                          </svg>
                          {formatNumber(post.likes)}
                        </span>
                      )}
                      {post.reposts != null && (
                        <span className="flex items-center gap-1 font-mono text-[11px] text-gray-400">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-green-600/70">
                            <path d="M17 1l4 4-4 4" />
                            <path d="M3 11V9a4 4 0 014-4h14" />
                            <path d="M7 23l-4-4 4-4" />
                            <path d="M21 13v2a4 4 0 01-4 4H3" />
                          </svg>
                          {formatNumber(post.reposts)}
                        </span>
                      )}
                      {post.replies != null && (
                        <span className="flex items-center gap-1 font-mono text-[11px] text-gray-400">
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-cyan-600/70">
                            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                          </svg>
                          {formatNumber(post.replies)}
                        </span>
                      )}
                      <span className="ml-auto font-mono text-[11px] text-cyan-600 font-semibold">
                        {formatNumber(post.total_engagement)} engagement
                      </span>
                    </div>
                  </div>
                </div>
              </ScrollReveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
