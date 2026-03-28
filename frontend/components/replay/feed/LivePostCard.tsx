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
      return (
        <span key={i} className="text-cyan-600 font-medium">
          {part}
        </span>
      );
    }
    if (part.startsWith("@")) {
      return (
        <span key={i} className="text-blue-600">
          {part}
        </span>
      );
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
      initial={{ opacity: 0, y: -16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 400, damping: 30, mass: 0.8 }}
      onClick={() => onSelect(isSelected ? null : post.id)}
      className={`px-3 py-3 border-b border-gray-200 cursor-pointer transition-colors ${
        isSelected
          ? "bg-blue-50 ring-1 ring-blue-600"
          : "hover:bg-white"
      }`}
    >
      <div className="flex gap-2.5">
        {/* Avatar */}
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-[9px] font-bold text-white flex-shrink-0"
          style={{ backgroundColor: post.avatarColor }}
        >
          {initials}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Header row */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-semibold text-xs text-gray-900 truncate">
              {post.author_name}
            </span>
            <span className="text-gray-400 text-[10px] font-mono truncate">
              {post.handle}
            </span>
            <span className="text-gray-500 text-[10px]">&middot;</span>
            <span className="text-gray-400 text-[10px] font-mono">R{post.round}</span>
            <div className="ml-auto flex-shrink-0">
              <PlatformBadge platform={post.platform} />
            </div>
          </div>

          {/* Post text */}
          <p className="mt-1 text-[12px] text-gray-700 leading-relaxed">
            {highlightText(post.text)}
          </p>

          {/* Role badge */}
          {post.author_role && (
            <div className="mt-1">
              <span className="inline-block px-1.5 py-0.5 rounded text-[8px] font-mono bg-gray-100 text-gray-500 border border-gray-300">
                {post.author_role}
              </span>
            </div>
          )}

          {/* Engagement */}
          <PostEngagement
            likes={post.likes}
            reposts={post.reposts}
            replies={post.replies}
            progress={post.engagementProgress}
          />

          {/* Impact badge */}
          {post.impact && (
            <div className="mt-1.5 flex items-center gap-2">
              <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-50 border border-blue-700/50">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
                <span className="text-[9px] font-mono text-blue-600">
                  {post.impact.reach} agents
                </span>
              </div>
              <div className={`flex items-center gap-0.5 px-1.5 py-0.5 rounded border ${
                post.impact.aggregateShift > 0
                  ? "bg-green-50 border-green-700/50 text-green-300"
                  : post.impact.aggregateShift < 0
                  ? "bg-red-50 border-red-700/50 text-red-300"
                  : "bg-gray-100 border-gray-300 text-gray-500"
              }`}>
                <span className="text-[9px] font-mono font-bold">
                  {post.impact.aggregateShift > 0 ? "+" : ""}
                  {post.impact.aggregateShift.toFixed(2)}
                </span>
                <span className="text-[8px] font-mono opacity-70">shift</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
