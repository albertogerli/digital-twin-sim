"use client";

import { useEffect, useRef } from "react";
import { type WgPost } from "@/lib/wargame-types";

function PostCard({ post }: { post: WgPost }) {
  const sentColor = post.sentiment === "negative" ? "#ff3b3b" : post.sentiment === "positive" ? "#00d26a" : "#6b7280";
  const tierLabel = post.authorTier === 1 ? "ELITE" : post.authorTier === 2 ? "INST" : "";
  const tierColor = post.authorTier === 1 ? "#ffaa00" : post.authorTier === 2 ? "#6b7280" : "#2a2d35";
  const platformColor = post.platform === "bloomberg" ? "#ff7700" : post.platform === "official" ? "#ffaa00" : "#6b7280";

  return (
    <div className={`px-2 py-1.5 border-b border-ki-border-strong hover:bg-ki-surface-sunken transition-colors ${post.sentiment === "negative" ? "flash-red" : post.sentiment === "positive" ? "flash-green" : ""}`}>
      {/* Header row */}
      <div className="flex items-center gap-1.5 mb-0.5">
        {/* Sentiment dot */}
        <span className="w-1 h-1 rounded-full shrink-0" style={{ background: sentColor }} />
        {/* Author */}
        <span className="font-data text-[10px] text-ki-on-surface font-medium truncate">
          {post.authorName}
        </span>
        {/* Tier badge */}
        {tierLabel && (
          <span className="font-data text-[8px] px-1 py-0 border rounded-sm shrink-0"
            style={{ color: tierColor, borderColor: tierColor + "40" }}>
            {tierLabel}
          </span>
        )}
        {/* Platform */}
        <span className="font-data text-[8px] ml-auto shrink-0" style={{ color: platformColor }}>
          {post.platform.toUpperCase()}
        </span>
      </div>

      {/* Text */}
      <p className="text-[11px] text-ki-on-surface-muted leading-tight pl-2.5">
        {post.text}
      </p>

      {/* Engagement */}
      <div className="flex items-center gap-2 mt-1 pl-2.5">
        <span className="font-data text-[9px] text-ki-on-surface-muted">
          {post.engagement > 1000 ? `${(post.engagement / 1000).toFixed(1)}K` : post.engagement} eng
        </span>
        <span className="font-data text-[9px]" style={{ color: sentColor }}>
          {post.sentiment.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

export function AgentFeed({ posts }: { posts: WgPost[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [posts.length]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0">
      {posts.length === 0 && (
        <div className="flex items-center justify-center h-full">
          <span className="font-data text-[10px] text-ki-on-surface-muted">WAITING FOR ROUND DATA<span className="cursor-blink">_</span></span>
        </div>
      )}
      {posts.filter(Boolean).map((post, i) => (
        <PostCard key={post.id || i} post={post} />
      ))}
    </div>
  );
}
