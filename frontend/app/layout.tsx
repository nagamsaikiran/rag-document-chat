import "./globals.css";
import type { ReactNode } from "react";
import Script from "next/script";

export const metadata = {
  title: "DocChat RAG",
  description: "Chat with your documents — grounded answers with citations.",
};

// Google Analytics 4. Set NEXT_PUBLIC_GA_ID (e.g. G-XXXXXXXX) to enable; if it's
// unset (local dev), nothing loads — so analytics only run where you want them.
const GA_ID = process.env.NEXT_PUBLIC_GA_ID;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        {GA_ID && (
          <>
            <Script
              src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`}
              strategy="afterInteractive"
            />
            <Script id="ga4" strategy="afterInteractive">
              {`
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', '${GA_ID}');
              `}
            </Script>
          </>
        )}
      </head>
      <body>{children}</body>
    </html>
  );
}
