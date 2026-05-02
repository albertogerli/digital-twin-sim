"use client";

import { useRef, useEffect } from "react";
import type { VisiblePost, ReplayStatus } from "@/lib/replay/types";
import LivePostCard from "./LivePostCard";

interface Props {
  posts: VisiblePost[];
  keyInsight: string;
  status: ReplayStatus;
  onPlay: () => void;
  selectedPostId: string | null;
  onSelectPost: (postId: string | null) => void;
}

export default function LiveFeed({ posts, keyInsight, status, onPlay, selectedPostId, onSelectPost }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  // Auto-scroll to top on new posts
  useEffect(() => {
    if (posts.length > prevCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
    prevCountRef.current = posts.length;
  }, [posts.length]);

  if (status === "idle" || (status === "paused" && posts.length === 0)) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4 px-8 bg-ki-surface">
        <div className="w-12 h-12 rounded-full bg-ki-surface-sunken border border-ki-border flex items-center justify-center">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-ki-on-surface-secondary">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-[14px] text-ki-on-surface font-medium">Ready to simulate</p>
          <p className="text-[11px] text-ki-on-surface-muted mt-1">
            Premi <span className="kbd">Space</span> o clicca sotto
          </p>
        </div>
        <button
          onClick={onPlay}
          className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-ki-on-surface text-ki-surface text-[12px] font-medium hover:bg-ki-on-surface-secondary transition-colors"
        >
          <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: "'wght' 500" }}>play_arrow</span>
          Start simulation
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Key insight banner */}
      {keyInsight && (
        <div className="px-3 py-2 bg-ki-warning-soft border-b border-ki-warning/30">
          <p className="text-[12px] text-ki-on-surface">
            <span className="eyebrow text-ki-warning mr-1.5">Insight</span>
            {keyInsight}
          </p>
        </div>
      )}

      {/* Header */}
      <div className="px-3 h-8 border-b border-ki-border flex items-center justify-between bg-ki-surface-raised">
        <div className="flex items-center gap-2">
          <span className="eyebrow">Live feed</span>
          <span className="font-data tabular text-[11px] text-ki-on-surface-muted">{posts.length} posts</span>
        </div>
        {status === "finished" && (
          <span className="font-data text-[11px] text-ki-warning">Completed</span>
        )}
      </div>

      {/* Post list */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto bg-ki-surface">
        {posts.length > 0 ? (
          posts.map((post) => (
            <LivePostCard
              key={post.id}
              post={post}
              isSelected={post.id === selectedPostId}
              onSelect={onSelectPost}
            />
          ))
        ) : status === "playing" ? (
          <div className="flex items-center justify-center h-32">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-ki-primary animate-pulse" />
              <span className="text-[12px] text-ki-on-surface-muted font-data">Waiting for posts…</span>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
