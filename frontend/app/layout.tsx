import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rift Dashboard",
  description: "Modern React dashboard for Rift fraud detection operations."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
