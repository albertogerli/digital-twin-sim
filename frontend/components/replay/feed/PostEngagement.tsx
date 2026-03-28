"use client";

import { useAnimatedNumber } from "@/lib/replay/useAnimatedNumber";

interface Props {
  likes: number;
  reposts: number;
  replies: number;
  progress: number;
}

function formatCount(n: number): string {
  if (n >= 10000) return (n / 1000).toFixed(1) + "K";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return n.toString();
}

export default function PostEngagement({ likes, reposts, replies, progress }: Props) {
  const animLikes = useAnimatedNumber(Math.round(likes * progress), 600);
  const animReposts = useAnimatedNumber(Math.round(reposts * progress), 600);
  const animReplies = useAnimatedNumber(Math.round(replies * progress), 600);

  return (
    <div className="flex items-center gap-4 mt-1.5 text-[10px] text-gray-400">
      <button className="flex items-center gap-1 hover:text-pink-400 transition-colors group">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="group-hover:fill-pink-400/20">
          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
        </svg>
        <span className={animLikes > 0 ? "text-gray-700" : ""}>{formatCount(animLikes)}</span>
      </button>

      <button className="flex items-center gap-1 hover:text-green-600 transition-colors group">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M17 1l4 4-4 4" />
          <path d="M3 11V9a4 4 0 0 1 4-4h14" />
          <path d="M7 23l-4-4 4-4" />
          <path d="M21 13v2a4 4 0 0 1-4 4H3" />
        </svg>
        <span className={animReposts > 0 ? "text-gray-700" : ""}>{formatCount(animReposts)}</span>
      </button>

      <button className="flex items-center gap-1 hover:text-cyan-600 transition-colors group">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span className={animReplies > 0 ? "text-gray-700" : ""}>{formatCount(animReplies)}</span>
      </button>
    </div>
  );
}
