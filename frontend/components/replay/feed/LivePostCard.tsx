"use client";

import { motion } from "framer-motion";
import type { VisiblePost } from "@/lib/replay/types";
import PlatformBadge from "./PlatformBadge";
import PostEngagement from "./PostEngagement";

interface Props {
  post: VisiblePost;
  isSelected: boolean;
  onSelect: (postId: string | null) => void;
}

function highlightText(text: string): React.ReactNode[] {
  const parts = text.split(/(#\w+|@\w+)/g);
  return parts.map((part, i) => {
    if (part.startsWith("#")) {
      return <span key={i} className="text-ki-primary font-medium">{part}</span>;
    }
    if (part.startsWith("@")) {
      return <span key={i} className="text-ki-primary">{part}</span>;
    }
    return <span key={i}>{part}</span>;
  });
}

export default function LivePostCard({ post, isSelected, onSelect }: Props) {
  const initials = post.author_name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("");

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 400, damping: 30, mass: 0.8 }}
      onClick={() => onSelect(isSelected ? null : post.id)}
      className={`px-3 py-3 border-b border-ki-border-faint cursor-pointer transition-colors ${
        isSelected
          ? "bg-ki-primary-soft shadow-[inset_2px_0_0_var(--accent)]"
          : "hover:bg-ki-surface-hover"
      }`}
    >
      <div className="flex gap-2.5">
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center text-[9px] font-medium text-white flex-shrink-0"
          style={{ backgroundColor: post.avatarColor }}
        >
          {initials}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-[12px] font-medium text-ki-on-surface truncate">{post.author_name}</span>
            <span className="font-data text-[11px] text-ki-on-surface-muted truncate">{post.handle}</span>
            <span className="text-ki-on-surface-faint text-[10px]">·</span>
            <span className="font-data tabular text-[10px] text-ki-on-surface-muted">R{post.round}</span>
            <div className="ml-auto flex-shrink-0">
              <PlatformBadge platform={post.platform} />
            </div>
          </div>

          <p className="mt-1.5 text-[12.5px] text-ki-on-surface-secondary leading-[1.55]">
            {highlightText(post.text)}
          </p>

          {post.author_role && (
            <div className="mt-1.5">
              <span className="inline-block px-1.5 py-0.5 rounded-sm font-data text-[10px] bg-ki-surface-sunken text-ki-on-surface-muted border border-ki-border">
                {post.author_role}
              </span>
            </div>
          )}

          <PostEngagement
            likes={post.likes}
            reposts={post.reposts}
            replies={post.replies}
            progress={post.engagementProgress}
          />

          {post.impact && (
            <div className="mt-2 flex items-center gap-1.5">
              <div className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm bg-ki-primary-soft text-ki-primary font-data tabular text-[10px]">
                <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
                {post.impact.reach} agents
              </div>
              <div className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-sm font-data tabular text-[10px] ${
                post.impact.aggregateShift > 0
                  ? "bg-ki-success-soft text-ki-success"
                  : post.impact.aggregateShift < 0
                  ? "bg-ki-error-soft text-ki-error"
                  : "bg-ki-surface-sunken text-ki-on-surface-muted"
              }`}>
                {post.impact.aggregateShift > 0 ? "+" : ""}
                {post.impact.aggregateShift.toFixed(2)} shift
              </div>
            </div>
          )}

          {/* RAG citations — chunks consulted by the agent */}
          {post.citations && post.citations.length > 0 && (
            <div className="mt-2 flex items-center gap-1 flex-wrap">
              <span className="font-data text-[9px] uppercase tracking-[0.06em] text-ki-on-surface-muted mr-0.5">
                cites
              </span>
              {post.citations.map((c) => (
                <span
                  key={c.chunk_id}
                  title={`${c.title} · ${c.chunk_id} · score ${c.score.toFixed(2)}`}
                  className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm bg-ki-surface-sunken border border-ki-border text-ki-on-surface-secondary hover:bg-ki-surface-hover hover:text-ki-on-surface transition-colors cursor-help font-data text-[10px]"
                >
                  <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                  </svg>
                  <span className="truncate max-w-[120px]">{c.title.replace(/\.[a-z]+$/i, "")}</span>
                  <span className="text-ki-on-surface-muted">·</span>
                  <span className="tabular">{c.score.toFixed(2)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
