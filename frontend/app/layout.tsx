import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rift Fraud Intelligence",
  description: "Local-first graph fraud intelligence with replayable audit evidence and a modern operations dashboard."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
