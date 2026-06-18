import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "DocChat RAG",
  description: "Chat with your documents — grounded answers with citations.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
