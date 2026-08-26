import type { ReactNode } from "react";
import Script from "next/script";
import { Hanken_Grotesk } from "next/font/google";

export const metadata = {
  title: "DocChat RAG",
  description: "Chat with your documents — grounded answers with citations.",
};

// Self-hosted via next/font (served from our own origin at build time), so it
// loads under the app's strict same-origin CSP with no external request.
const sans = Hanken_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  variable: "--font-hanken",
  display: "swap",
});

// Google Analytics 4 — loads only in production builds. A GA Measurement ID
// isn't secret, so hardcoding is fine.
const GA_ID =
  process.env.NODE_ENV === "production"
    ? process.env.NEXT_PUBLIC_GA_ID || "G-1RN8CNKNNZ"
    : undefined;

// Styles are INLINED here (not `import "./globals.css"`) on purpose: with
// Next.js `output: "export"`, React's stylesheet-precedence management drops an
// external <link> during hydration. A plain inline <style> survives hydration.
//
// Design: "Clean & professional" — a two-column app shell. A left sidebar holds
// the brand, a short "how it works", and GitHub/LinkedIn links; the main column
// has a top toolbar (upload + options), the conversation, and a docked
// composer. One calm steel-blue accent on a neutral cool-grey system, hairline
// borders, restrained motion. On narrow screens the shell collapses to a single
// column: the sidebar becomes a compact top bar (icon links), the toolbar wraps,
// and everything fits the width with vertical scroll only.
const GLOBAL_CSS = `
:root{
  --bg:#f7f8fa; --surface:#ffffff; --surface-2:#f9fafb;
  --ink:#1a1d23; --ink-soft:#3b4048; --muted:#6b7280;
  --line:#e6e8ee; --line-strong:#d7dae2;
  --accent:#2f5aa8; --accent-ink:#234785; --accent-weak:#eaf0f9; --accent-weak-2:#f2f6fc;
  --danger:#b4433a;
  --sh-sm:0 1px 2px rgba(20,24,40,.05);
  --sh-md:0 1px 2px rgba(20,24,40,.05), 0 8px 20px -10px rgba(20,24,40,.16);
  --sh-lg:0 2px 6px rgba(20,24,40,.06), 0 18px 36px -14px rgba(20,24,40,.20);
  --ring:0 0 0 3px rgba(47,90,168,.18);
  --font-sans: var(--font-hanken), system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  --r:12px; --r-sm:9px;
}
*{ box-sizing:border-box; }
html{ -webkit-text-size-adjust:100%; }
html,body{ height:100%; overflow:hidden; }
body{
  margin:0; background:var(--bg); color:var(--ink); font-family:var(--font-sans);
  line-height:1.6; -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}

/* ---- app shell: a fixed frame; only the conversation (.content) scrolls ---- */
.shell{ display:grid; grid-template-columns:250px 1fr; height:100vh; height:100dvh; overflow:hidden; }

/* ---- sidebar (fixed) ---- */
.sidebar{
  background:var(--surface); border-right:1px solid var(--line);
  display:flex; flex-direction:column; gap:22px; padding:20px 18px;
  height:100vh; height:100dvh; overflow-y:auto;
}
.brand{ display:flex; align-items:center; gap:11px; }
.brand-mark{
  width:38px; height:38px; border-radius:10px; flex:none; display:grid; place-items:center;
  background:var(--accent); box-shadow:var(--sh-sm);
}
.brand-mark svg{ width:22px; height:22px; display:block; }
.brand h1{ font-size:16.5px; font-weight:800; letter-spacing:-.01em; line-height:1.1; margin:0; color:var(--ink); }
.brand .tag{ font-size:11px; color:var(--muted); font-weight:600; }

.s-title{ font-size:11px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); font-weight:700; margin-bottom:11px; }
.hiw{ display:flex; flex-direction:column; gap:11px; }
.hiw-item{ display:flex; gap:9px; font-size:12.5px; color:var(--ink-soft); line-height:1.4; }
.hiw-item .d{ width:6px; height:6px; border-radius:50%; background:var(--accent); margin-top:6px; flex:none; }

.social{ margin-top:auto; border-top:1px solid var(--line); padding-top:16px; display:flex; flex-direction:column; gap:8px; }
.social a{
  display:inline-flex; align-items:center; gap:9px; font-size:13px; font-weight:600; color:var(--ink-soft);
  border:1px solid var(--line-strong); border-radius:9px; padding:9px 12px; text-decoration:none;
  transition:color .15s, background .15s, border-color .15s;
}
.social a:hover{ color:var(--accent); background:var(--accent-weak-2); border-color:var(--accent); }
.social a svg{ width:16px; height:16px; color:var(--muted); flex:none; }
.social a:hover svg{ color:var(--accent); }

/* compact icon links, shown only in the mobile top bar */
.m-social{ display:none; gap:6px; }
.m-social a{
  width:36px; height:36px; border-radius:8px; display:grid; place-items:center;
  color:var(--muted); border:1px solid var(--line-strong); text-decoration:none;
  transition:color .15s, background .15s, border-color .15s;
}
.m-social a:hover{ color:var(--accent); background:var(--accent-weak-2); border-color:var(--accent); }
.m-social a svg{ width:17px; height:17px; display:block; }

/* ---- main column: toolbar + composer are fixed; the middle scrolls ---- */
.main{ display:flex; flex-direction:column; min-width:0; min-height:0; height:100vh; height:100dvh; overflow:hidden; }

.toolbar{
  flex:none;
  background:var(--surface); border-bottom:1px solid var(--line);
  padding:14px 22px; display:flex; flex-direction:column; gap:11px;
}
.tb-row{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.tb-opts{ display:flex; gap:20px; align-items:center; flex-wrap:wrap; font-size:13px; color:var(--muted); }
.tb-opts .toggle{ margin:0; }
.tb-sources{ display:flex; align-items:center; gap:7px; flex-wrap:wrap; }

/* the only scrollable region */
.content{ flex:1 1 auto; min-height:0; overflow-y:auto; padding:24px 22px; }
/* when there are no messages yet, center the placeholder in the open space */
.content:has(.empty){ display:flex; align-items:center; justify-content:center; }
.thread{ width:100%; max-width:720px; margin:0 auto; }
.empty{ color:var(--muted); font-size:15px; text-align:center; max-width:38ch; margin:0 auto; padding:24px 16px; line-height:1.55; }

.foot{ flex:none; padding:16px 22px 20px; display:flex; justify-content:center; background:var(--bg); border-top:1px solid var(--line); }
.composer{
  width:100%; max-width:720px;
  background:var(--surface); border:1px solid var(--line-strong); border-radius:14px;
  padding:8px 8px 8px 6px; box-shadow:var(--sh-lg); display:flex; gap:9px; align-items:center;
}
.composer input[type="text"]{ border:0; box-shadow:none; background:transparent; padding:11px 12px; }
.composer input[type="text"]:focus{ box-shadow:none; }
.composer button{ flex:none; }

/* ---- form controls ---- */
input[type="text"]{
  flex:1; min-width:180px; padding:12px 14px; border-radius:var(--r-sm);
  border:1px solid var(--line-strong); background:var(--surface); color:var(--ink);
  font-size:15px; font-family:var(--font-sans); transition:border-color .15s, box-shadow .15s;
}
input[type="text"]::placeholder{ color:var(--muted); }
input[type="text"]:focus{ outline:none; border-color:var(--accent); box-shadow:var(--ring); }

input[type="file"]{ font-family:var(--font-sans); font-size:13px; color:var(--muted); max-width:100%; }
input[type="file"]::file-selector-button{
  font-family:var(--font-sans); font-weight:600; font-size:13px; cursor:pointer;
  padding:8px 13px; margin-right:11px; border-radius:8px;
  border:1px solid var(--line-strong); background:var(--surface-2); color:var(--ink);
  transition:border-color .15s, color .15s;
}
input[type="file"]::file-selector-button:hover{ border-color:var(--accent); color:var(--accent); }

button{
  font-family:var(--font-sans); font-size:14px; font-weight:600; cursor:pointer;
  padding:10px 16px; border-radius:var(--r-sm); border:1px solid var(--accent); color:#fff;
  background:var(--accent); box-shadow:var(--sh-sm);
  transition:background .15s, border-color .15s, transform .12s ease, box-shadow .15s;
}
button:hover{ background:var(--accent-ink); border-color:var(--accent-ink); transform:translateY(-1px); box-shadow:var(--sh-md); }
button:active{ transform:translateY(0); box-shadow:var(--sh-sm); }
button:focus-visible{ outline:none; box-shadow:var(--ring); }
button:disabled{ opacity:.5; cursor:not-allowed; transform:none; box-shadow:var(--sh-sm); }
button.ghost{ background:var(--surface); color:var(--ink-soft); border:1px solid var(--line-strong); font-weight:500; }
button.ghost:hover{ background:var(--surface); color:var(--accent); border-color:var(--accent); }

.pill{
  font-size:12px; font-weight:600; background:var(--accent-weak); color:var(--accent-ink);
  padding:6px 12px; border-radius:999px; display:inline-flex; align-items:center; gap:7px;
}
.pill::before{ content:""; width:6px; height:6px; border-radius:50%; background:var(--accent); }

.toggle{ display:flex; align-items:center; gap:7px; font-size:13px; color:var(--muted); cursor:pointer; user-select:none; }
.toggle input{ width:15px; height:15px; cursor:pointer; accent-color:var(--accent); flex:none; }

.src{
  display:inline-flex; align-items:center; gap:2px; font-size:13px; font-weight:500;
  background:var(--surface-2); color:var(--ink-soft); border:1px solid var(--line-strong);
  border-radius:999px; padding:4px 5px 4px 12px;
}
.src-x{
  background:transparent; border:0; box-shadow:none; color:var(--muted); font-size:15px; font-weight:700;
  cursor:pointer; padding:0 7px; line-height:1; border-radius:999px; transition:color .15s;
}
.src-x:hover{ background:transparent; color:var(--danger); transform:none; box-shadow:none; }

/* ---- conversation ---- */
.msg{ display:flex; gap:12px; padding:16px 0; border-top:1px solid var(--line); animation:fade .28s ease both; }
.msg:first-child{ border-top:0; }
@keyframes fade{ from{ opacity:0; transform:translateY(6px); } to{ opacity:1; transform:none; } }

.avatar{
  width:32px; height:32px; border-radius:8px; flex:none; display:grid; place-items:center;
  font-size:11px; font-weight:700; letter-spacing:.02em; color:#fff; box-shadow:var(--sh-sm);
}
.msg.user .avatar{ background:#4b5563; }
.msg.assistant .avatar{ background:var(--accent); }
.bubble{ flex:1; min-width:0; }
.label{ font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:700; margin-bottom:5px; }
.msg.assistant .label{ color:var(--accent); }
.answer{ margin:0; color:var(--ink-soft); font-size:15px; line-height:1.65; white-space:pre-wrap; overflow-wrap:break-word; }
.msg.user .answer{ color:var(--ink); font-weight:600; font-size:16px; }

.answer.streaming::after{
  content:""; display:inline-block; width:2px; height:1.05em; margin-left:2px;
  background:var(--accent); border-radius:2px; vertical-align:-2px; animation:blink 1s steps(2) infinite;
}
@keyframes blink{ 50%{ opacity:0; } }
.typing{ display:inline-flex; gap:5px; padding:4px 0; }
.typing span{ width:6px; height:6px; border-radius:50%; background:var(--muted); opacity:.5; animation:bounce 1.1s ease-in-out infinite; }
.typing span:nth-child(2){ animation-delay:.15s; }
.typing span:nth-child(3){ animation-delay:.3s; }
@keyframes bounce{ 0%,80%,100%{ transform:translateY(0); opacity:.4; } 40%{ transform:translateY(-4px); opacity:.9; } }

/* citations */
.cites{ margin-top:12px; font-size:13px; }
.cites > summary{
  cursor:pointer; user-select:none; list-style:none; display:inline-flex; align-items:center; gap:7px;
  font-size:12px; font-weight:600; letter-spacing:.02em; color:var(--accent);
  background:var(--accent-weak-2); border:1px solid var(--line-strong); padding:6px 12px; border-radius:8px;
  transition:background .15s, border-color .15s;
}
.cites > summary:hover{ background:var(--accent-weak); border-color:var(--accent); }
.cites > summary::-webkit-details-marker{ display:none; }
.cites > summary::before{ content:"▸"; font-size:10px; transition:transform .15s; }
.cites[open] > summary::before{ transform:rotate(90deg); }
.cites[open] > summary{ margin-bottom:10px; }
.cite{
  padding:11px 14px; border:1px solid var(--line); border-left:3px solid var(--accent);
  background:var(--surface-2); border-radius:0 8px 8px 0; margin:8px 0;
}
.cite-head{ font-size:12.5px; color:var(--accent-ink); font-weight:700; }
.muted{ color:var(--muted); font-size:13px; }
.cite .muted{ margin-top:4px; line-height:1.5; }
.err-text{ color:var(--danger); font-size:13px; }
.status-msg{ font-size:13px; margin:2px 0 0; }
.status-msg.ok{ color:var(--muted); }
.status-msg.bad{ color:var(--danger); }

/* follow-up chips */
.followups{ margin-top:16px; }
.fu-label{ font-size:11px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); margin-bottom:9px; font-weight:700; }
.fu-chips{ display:flex; flex-wrap:wrap; gap:9px; }
.chip{
  font-size:13px; font-weight:500; cursor:pointer; color:var(--ink-soft);
  padding:9px 13px; border-radius:8px; line-height:1.3; text-align:left; max-width:100%;
  background:var(--surface); border:1px solid var(--line-strong); box-shadow:var(--sh-sm);
  transition:border-color .15s, color .15s, background .15s, transform .12s ease;
}
.chip:hover{ border-color:var(--accent); color:var(--accent-ink); background:var(--accent-weak-2); transform:translateY(-1px); }
.chip:active{ transform:translateY(0); }
.chip.general{ border-style:dashed; color:var(--muted); }
.chip.general:hover{ border-color:var(--accent); color:var(--accent-ink); background:var(--accent-weak-2); }
.chip:disabled{ opacity:.55; cursor:not-allowed; transform:none; }

/* backend-down banner */
.banner{ margin:14px 22px 0; padding:12px 15px; border-radius:10px; font-size:14px; box-shadow:var(--sh-sm); }
.banner.err{ background:#fbeeec; border:1px solid #eecac5; color:#8c352c; }

/* ---- responsive: collapse to a single column ---- */
@media (max-width:820px){
  .shell{ display:flex; flex-direction:column; }
  .sidebar{
    flex:none; height:auto; overflow:visible;
    flex-direction:row; align-items:center; justify-content:space-between;
    border-right:0; border-bottom:1px solid var(--line);
    padding:11px 15px; gap:12px;
  }
  .brand .tag{ display:none; }
  .hiwrap{ display:none; }
  .social{ display:none; }
  .m-social{ display:flex; }
  .main{ flex:1 1 auto; height:auto; min-height:0; }
  .toolbar{ padding:12px 15px; }
  .content{ padding:16px 15px; }
  .foot{ padding:12px 15px 16px; }
}
@media (min-width:821px){ .m-social{ display:none; } }

@media (prefers-reduced-motion: reduce){
  *{ animation:none !important; transition:none !important; }
}
`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={sans.variable}>
      <head>
        <link rel="icon" type="image/svg+xml" href="/icon.svg" />
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
