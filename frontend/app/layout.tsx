import type { Metadata } from "next";
import { JetBrains_Mono, Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
  variable: "--font-display"
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
  variable: "--font-mono"
});

export const metadata: Metadata = {
  title: "Rift Fraud Intelligence",
  description:
    "Local-first graph fraud intelligence with replayable audit evidence and an operations console."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html className={`${outfit.variable} ${jetbrainsMono.variable}`} lang="en">
      <body>{children}</body>
    </html>
  );
}
