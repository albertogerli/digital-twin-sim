"use client";

import { AnimatePresence, motion } from "framer-motion";

interface Hashtag {
  tag: string;
  count: number;
  trend: "up" | "down" | "new";
}

interface Props {
  hashtags: Hashtag[];
}

export default function TrendingHashtags({ hashtags }: Props) {
  if (hashtags.length === 0) {
    return (
      <div className="text-[10px] font-mono text-gray-400 text-center py-2">
        Waiting for hashtags...
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <AnimatePresence mode="popLayout">
        {hashtags.slice(0, 6).map((h, i) => (
          <motion.div
            key={h.tag}
            layout
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
            transition={{ duration: 0.3, delay: i * 0.05 }}
            className="flex items-center justify-between py-1 px-2 rounded hover:bg-gray-100/60"
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-gray-400 font-mono text-[10px] w-3 flex-shrink-0">
                {i + 1}
              </span>
              <span className="text-cyan-600 text-xs font-medium truncate">
                {h.tag}
              </span>
              {h.trend === "new" && (
                <span className="px-1 py-0.5 rounded text-[7px] font-mono font-bold bg-blue-50 text-cyan-600 border border-blue-700/50 flex-shrink-0">
                  NEW
                </span>
              )}
            </div>
            <div className="flex items-center gap-1 flex-shrink-0 ml-2">
              <span className="font-mono text-[10px] text-gray-500 tabular-nums">
                {h.count}
              </span>
              {h.trend === "up" && (
                <svg width="8" height="8" viewBox="0 0 8 8" className="text-green-500">
                  <path d="M4 1L7 5H1L4 1Z" fill="currentColor" />
                </svg>
              )}
              {h.trend === "down" && (
                <svg width="8" height="8" viewBox="0 0 8 8" className="text-red-500">
                  <path d="M4 7L1 3H7L4 7Z" fill="currentColor" />
                </svg>
              )}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
