"use client";

interface Props {
  platform: string;
}

const BADGE_CONFIG: Record<string, { label: string; icon: React.ReactNode; bgClass: string; textClass: string }> = {
  social: {
    label: "Social",
    bgClass: "bg-gray-100 border-gray-300",
    textClass: "text-gray-500",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
    ),
  },
  xsim: {
    label: "X-Sim",
    bgClass: "bg-gray-100 border-gray-300",
    textClass: "text-gray-500",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
    ),
  },
  forum: {
    label: "Forum",
    bgClass: "bg-gray-100 border-gray-300",
    textClass: "text-gray-500",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
      </svg>
    ),
  },
  press: {
    label: "Press",
    bgClass: "bg-amber-900/40 border-amber-700/50",
    textClass: "text-amber-600",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z" />
      </svg>
    ),
  },
  stampa: {
    label: "Press",
    bgClass: "bg-amber-900/40 border-amber-700/50",
    textClass: "text-amber-600",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z" />
      </svg>
    ),
  },
  tv: {
    label: "TV",
    bgClass: "bg-purple-900/40 border-purple-700/50",
    textClass: "text-purple-400",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M21 3H3c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h5v2h8v-2h5c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 14H3V5h18v12z" />
      </svg>
    ),
  },
  institutional: {
    label: "Official",
    bgClass: "bg-blue-50 border-blue-700/50",
    textClass: "text-cyan-600",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2L3 7v2h18V7L12 2zm0 2.18L17.36 7H6.64L12 4.18zM5 10v7h2v-7H5zm4 0v7h2v-7H9zm4 0v7h2v-7h-2zm4 0v7h2v-7h-2zM3 18v2h18v-2H3z" />
      </svg>
    ),
  },
  istituzionale: {
    label: "Official",
    bgClass: "bg-blue-50 border-blue-700/50",
    textClass: "text-cyan-600",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2L3 7v2h18V7L12 2zm0 2.18L17.36 7H6.64L12 4.18zM5 10v7h2v-7H5zm4 0v7h2v-7H9zm4 0v7h2v-7h-2zm4 0v7h2v-7h-2zM3 18v2h18v-2H3z" />
      </svg>
    ),
  },
  public: {
    label: "Public",
    bgClass: "bg-rose-900/40 border-rose-700/50",
    textClass: "text-rose-400",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z" />
      </svg>
    ),
  },
  piazza: {
    label: "Public",
    bgClass: "bg-rose-900/40 border-rose-700/50",
    textClass: "text-rose-400",
    icon: (
      <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor">
        <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z" />
      </svg>
    ),
  },
};

export default function PlatformBadge({ platform }: Props) {
  const config = BADGE_CONFIG[platform] || BADGE_CONFIG.social;
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[8px] font-mono ${config.bgClass} ${config.textClass}`}>
      {config.icon}
      {config.label}
    </span>
  );
}
