import type { Config } from "tailwindcss";

/* ───────────────────────────────────────────────────────────
   Quiet Intelligence Terminal — design tokens
   Token names preserved (ki-*, domain-*) so existing
   components auto-pick the new palette without rewrites.
   Source of truth: /tmp/dts_design/digitaltwinsim/project/styles.css
   ─────────────────────────────────────────────────────────── */

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: { dark: "#0a0a0a", warm: "#f5f0eb" },
        accent: {
          pro: "oklch(0.58 0.13 150)",
          con: "oklch(0.55 0.18 25)",
          gold: "oklch(0.7 0.13 75)",
          blue: "oklch(0.52 0.17 264)",
          cyan: "oklch(0.55 0.14 220)",
          amber: "oklch(0.7 0.13 75)",
        },
        // Quiet Intelligence palette — neutral cool, OKLCH source
        ki: {
          primary: "oklch(0.52 0.17 264)",
          "primary-muted": "oklch(0.46 0.18 264)",
          "primary-soft": "oklch(0.94 0.04 264)",
          surface: "oklch(0.985 0.003 90)",
          "surface-raised": "oklch(1 0 0)",
          "surface-sunken": "oklch(0.965 0.004 90)",
          "surface-hover": "oklch(0.955 0.004 90)",
          "surface-active": "oklch(0.93 0.005 90)",
          border: "oklch(0.9 0.005 90)",
          "border-strong": "oklch(0.82 0.006 90)",
          "border-faint": "oklch(0.94 0.004 90)",
          "on-surface": "oklch(0.2 0.012 260)",
          "on-surface-secondary": "oklch(0.42 0.012 260)",
          "on-surface-muted": "oklch(0.6 0.01 260)",
          "on-surface-faint": "oklch(0.72 0.008 260)",
          error: "oklch(0.55 0.18 25)",
          "error-soft": "oklch(0.94 0.04 25)",
          success: "oklch(0.58 0.13 150)",
          "success-soft": "oklch(0.94 0.04 150)",
          warning: "oklch(0.7 0.13 75)",
          "warning-soft": "oklch(0.95 0.05 75)",
        },
        domain: {
          political: "oklch(0.55 0.14 50)",
          corporate: "oklch(0.48 0.16 290)",
          financial: "oklch(0.52 0.17 264)",
          commercial: "oklch(0.55 0.13 165)",
          marketing: "oklch(0.58 0.18 340)",
          health: "oklch(0.55 0.18 25)",
          technology: "oklch(0.55 0.14 220)",
        },
      },
      fontFamily: {
        // Quiet Intelligence: Geist as primary, with Inter fallback
        headline: ["Geist", "Inter", "system-ui", "sans-serif"],
        body: ["Geist", "Inter", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "JetBrains Mono", "ui-monospace", "monospace"],
        data: ["Geist Mono", "JetBrains Mono", "ui-monospace", "monospace"],
        // Legacy
        display: ["Playfair Display", "Georgia", "serif"],
        serif: ["Source Serif 4", "Georgia", "serif"],
        sans: ["Geist", "Inter", "system-ui", "sans-serif"],
      },
      fontSize: {
        // Density-tuned scale
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }], // 10px / 14px — eyebrow
        mini: ["0.6875rem", { lineHeight: "0.95rem" }],   // 11px / 15px — caption
      },
      letterSpacing: {
        eyebrow: "0.08em",
        tight2: "-0.02em",
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "5px",
        lg: "8px",
      },
      boxShadow: {
        // Hairline-first: shadows are nearly imperceptible
        tint: "0 1px 2px rgb(0 0 0 / 0.04)",
        hairline: "0 0 0 1px oklch(0.9 0.005 90)",
      },
      keyframes: {
        streamFade: {
          from: { opacity: "0", transform: "translateY(-4px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        livePulse: {
          "0%":   { boxShadow: "0 0 0 0 oklch(0.55 0.18 25 / 0.45)" },
          "70%":  { boxShadow: "0 0 0 6px oklch(0.55 0.18 25 / 0)" },
          "100%": { boxShadow: "0 0 0 0 oklch(0.55 0.18 25 / 0)" },
        },
      },
      animation: {
        "stream-in": "streamFade 0.25s ease-out",
        "live-pulse": "livePulse 1.6s infinite",
      },
    },
  },
  plugins: [],
};
export default config;
