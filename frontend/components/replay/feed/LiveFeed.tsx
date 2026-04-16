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
      <div className="h-full flex flex-col items-center justify-center gap-4 px-8">
        <div className="w-14 h-14 rounded-full bg-ki-surface-sunken border border-ki-border flex items-center justify-center">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-ki-on-surface-muted">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm text-ki-on-surface-secondary font-medium">Ready to simulate</p>
          <p className="text-xs text-ki-on-surface-muted mt-1 font-mono">
            Press Space or click below
          </p>
        </div>
        <button
          onClick={onPlay}
          className="px-4 py-2 rounded-lg bg-ki-primary/[0.07] border border-ki-primary/30 text-ki-primary text-xs font-mono hover:bg-ki-primary/20 transition-all"
        >
          Start simulation
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Key insight banner */}
      {keyInsight && (
        <div className="px-3 py-1.5 bg-amber-900/30 border-b border-amber-700/40">
          <p className="text-[10px] font-mono text-amber-300">
            <span className="font-bold">INSIGHT</span> — {keyInsight}
          </p>
        </div>
      )}

      {/* Post counter */}
      <div className="px-3 py-1.5 border-b border-gray-200 flex items-center justify-between">
        <span className="font-mono text-[10px] text-gray-400">
          LIVE FEED — {posts.length} posts
        </span>
        {status === "finished" && (
          <span className="font-mono text-[10px] text-amber-600">Completed</span>
        )}
      </div>

      {/* Post list */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
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
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              <span className="text-xs text-gray-400 font-mono">Waiting for posts...</span>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
