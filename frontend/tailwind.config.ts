import type { Config } from "tailwindcss";

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
        // Legacy (keep for existing components)
        bg: { dark: "#0a0a0a", warm: "#f5f0eb" },
        accent: {
          pro: "#22c55e",
          con: "#ef4444",
          gold: "#d4a574",
          blue: "#3b82f6",
          cyan: "#06b6d4",
          amber: "#f59e0b",
        },
        // Industrial palette — sterile, clinical
        ki: {
          primary: "#1a6dff",
          "primary-muted": "#3d7eff",
          surface: "#f5f5f5",
          "surface-raised": "#fafafa",
          "surface-sunken": "#ebebeb",
          "surface-hover": "#e8e8e8",
          border: "#d4d4d4",
          "border-strong": "#b0b0b0",
          "on-surface": "#1a1a1a",
          "on-surface-secondary": "#525252",
          "on-surface-muted": "#8a8a8a",
          error: "#dc2626",
          success: "#16a34a",
          warning: "#d97706",
        },
        domain: {
          political: "#d97706",
          corporate: "#7c3aed",
          financial: "#1a6dff",
          commercial: "#059669",
          marketing: "#db2777",
          health: "#dc2626",
          technology: "#0891b2",
        },
      },
      fontFamily: {
        headline: ["Inter Tight", "Inter", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
        data: ["JetBrains Mono", "Roboto Mono", "monospace"],
        // Legacy
        display: ["Playfair Display", "Georgia", "serif"],
        serif: ["Source Serif 4", "Georgia", "serif"],
        sans: ["DM Sans", "system-ui", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      boxShadow: {
        tint: "0 1px 3px rgba(0,0,0,0.06)",
      },
    },
  },
  plugins: [],
};
export default config;
