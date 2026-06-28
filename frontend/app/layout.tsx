import type { ReactNode } from "react";
import Script from "next/script";

export const metadata = {
  title: "DocChat RAG",
  description: "Chat with your documents — grounded answers with citations.",
};

// Google Analytics 4 — loads only in production builds (keeps local dev traffic
// out of your stats). A GA Measurement ID isn't secret (it's exposed in the
// client-side gtag on any analytics-enabled site), so hardcoding it is fine.
const GA_ID =
  process.env.NODE_ENV === "production"
    ? process.env.NEXT_PUBLIC_GA_ID || "G-1RN8CNKNNZ"
    : undefined;

// Global styles are INLINED here (instead of `import "./globals.css"`) on
// purpose: with Next.js `output: "export"`, React's stylesheet-precedence
// management drops the external <link> during hydration (causing a flash of
// styling that then disappears). A plain inline <style> isn't float-managed,
// so it survives hydration and renders identically everywhere.
const GLOBAL_CSS = `
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #0f1115; color: #e7e9ee; }
.wrap { max-width: 820px; margin: 0 auto; padding: 32px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #9aa0ab; margin: 0 0 24px; font-size: 14px; }
.card { background: #171a21; border: 1px solid #262b36; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
input[type="text"] { flex: 1; min-width: 200px; padding: 11px 13px; border-radius: 8px; border: 1px solid #2c3340; background: #0f1115; color: #e7e9ee; font-size: 15px; }
button { padding: 11px 16px; border-radius: 8px; border: 0; cursor: pointer; background: #4c7dff; color: white; font-size: 14px; font-weight: 600; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
button.ghost { background: transparent; border: 1px solid #3a4150; color: #aab2c0; font-weight: 500; }
button.ghost:hover { border-color: #ff8088; color: #ff8088; }
.msg { padding: 12px 14px; border-radius: 10px; margin: 10px 0; white-space: pre-wrap; line-height: 1.5; }
.user { background: #1d2530; }
.assistant { background: #14181f; border: 1px solid #232a36; }
.label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #8089ff; margin-bottom: 6px; }
.cites { margin-top: 10px; font-size: 13px; color: #aab2c0; }
.cite { padding: 6px 8px; border-left: 3px solid #4c7dff; background: #11151c; border-radius: 4px; margin: 6px 0; }
.muted { color: #767d8a; font-size: 13px; }
.err-text { color: #ff8088; font-size: 13px; }
.toggle { display: flex; align-items: center; gap: 7px; margin-top: 12px; font-size: 13px; color: #aab2c0; cursor: pointer; user-select: none; }
.toggle input { width: 15px; height: 15px; cursor: pointer; }
.pill { font-size: 12px; background: #1d2530; padding: 4px 9px; border-radius: 999px; color: #9aa0ab; }
.banner { padding: 11px 14px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; }
.banner.err { background: #2a1518; border: 1px solid #5a2128; color: #ff9ba1; }
`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <style dangerouslySetInnerHTML={{ __html: GLOBAL_CSS }} />
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
