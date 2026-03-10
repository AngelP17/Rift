import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "var(--color-text)",
        muted: "var(--color-muted)",
        panel: "var(--color-panel)",
        line: "var(--color-line)",
        accent: "var(--color-accent)",
        good: "var(--color-good)",
        warn: "var(--color-warn)",
        bad: "var(--color-bad)"
      },
      borderRadius: {
        "4xl": "2rem"
      },
      boxShadow: {
        glass: "0 30px 80px rgba(0, 0, 0, 0.32)",
        card: "0 18px 48px rgba(0, 0, 0, 0.24)"
      },
      transitionTimingFunction: {
        fluid: "cubic-bezier(0.25, 0.46, 0.45, 0.94)"
      },
      fontFamily: {
        display: ["var(--font-display)"],
        body: ["var(--font-body)"],
        mono: ["var(--font-mono)"]
      }
    }
  },
  plugins: []
};

export default config;
