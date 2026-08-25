"use client";

import { useEffect, useRef, useState } from "react";

// Resolve the backend base URL at runtime (in the browser), so it's correct
// whether running locally or deployed — no reliance on build-time env vars:
//   - explicit NEXT_PUBLIC_API_URL wins if set to a real value
//   - on localhost (dev) -> the separate backend on :8000
//   - anywhere else (deployed) -> same origin as the served frontend ("")
function resolveApiBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_URL;
  if (fromEnv) return fromEnv;
  if (typeof window !== "undefined") {
    const h = window.location.hostname;
    if (h !== "localhost" && h !== "127.0.0.1") return ""; // same-origin in prod
  }
  return "http://localhost:8000";
}
const API = resolveApiBase();

// Fire a Google Analytics event if GA is loaded (no-op otherwise).
function track(event: string, params: Record<string, unknown> = {}) {
  (window as any).gtag?.("event", event, params);
}

// A per-browser session id, persisted in localStorage. Sent with every request
// so the backend keeps each visitor's documents isolated (no login needed).
function sessionId(): string {
  if (typeof window === "undefined") return "public";
  const existing = localStorage.getItem("docchat_sid");
  if (existing) return existing;
  const fresh: string =
    (crypto as any).randomUUID?.() ||
    Math.random().toString(36).slice(2) + Date.now().toString(36);
  localStorage.setItem("docchat_sid", fresh);
  return fresh;
}
function sessionHeader(): Record<string, string> {
  return { "X-Session-Id": sessionId() };
}

type Citation = { marker: number; source: string; page: number; snippet: string };
type Suggestion = { question: string; scope: "in_document" | "general" };
type Message = {
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
  suggestions?: Suggestion[];
};

// Where "general" follow-ups (beyond the uploaded document) are sent. Google AI
// Mode gives an AI-style answer for questions the document can't cover.
function webSearchUrl(q: string): string {
  return "https://www.google.com/search?udm=50&q=" + encodeURIComponent(q);
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [indexed, setIndexed] = useState(0);
  const [sources, setSources] = useState<string[]>([]);
  const [vision, setVision] = useState(false);
  // Ask the backend to propose follow-up questions after each answer. On by
  // default; turning it off skips one LLM call per question (saves free-tier
  // quota). Preference is remembered per browser.
  const [suggest, setSuggest] = useState(true);
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      const h = await fetch(`${API}/health`, { headers: sessionHeader() }).then((r) => r.json());
      setIndexed(h.indexed_chunks ?? 0);
      setBackendUp(true);
      const s = await fetch(`${API}/sources`, { headers: sessionHeader() }).then((r) => r.json());
      setSources(s.sources ?? []);
    } catch {
      setBackendUp(false);
    }
  }
  useEffect(() => {
    refresh();
    // Load the saved follow-up preference (in an effect, so server and first
    // client render match and there's no hydration mismatch).
    try {
      const v = localStorage.getItem("docchat_suggest");
      if (v !== null) setSuggest(v === "1");
    } catch {}
  }, []);

  async function upload() {
    const files = fileRef.current?.files;
    if (!files || files.length === 0) {
      setStatus({ msg: "Pick a file first (PDF, DOCX, TXT, MD, HTML).", ok: false });
      return;
    }
    setBusy(true);
    setStatus({
      msg: vision ? "Uploading and reading with vision (slower)…" : "Uploading and indexing…",
      ok: true,
    });
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    fd.append("multimodal", String(vision));
    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: fd, headers: sessionHeader() });
      const data = await res.json();
      const results = data.results ?? [];
      const failed = results.filter((r: any) => r.error);
      const indexedNow = results.reduce(
        (n: number, r: any) => n + (r.chunks_indexed ?? 0),
        0
      );
      if (failed.length > 0) {
        // Surface the backend's actual reason (bad key, rate limit, scanned PDF…)
        setStatus({ msg: `Upload failed: ${failed[0].error}`, ok: false });
        track("pdf_upload_failed", { reason: failed[0].error });
      } else {
        setStatus({ msg: `Indexed ${indexedNow} chunks. Ask a question below.`, ok: true });
        track("pdf_uploaded", { chunks: indexedNow, files: results.length });
      }
      await refresh();
    } catch (e: any) {
      setStatus({
        msg: `Could not reach the backend at ${API}. Is 2-start-backend.bat running?`,
        ok: false,
      });
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function deleteSource(name: string) {
    if (busy) return;
    if (!confirm(`Remove "${name}" from the index?`)) return;
    setBusy(true);
    try {
      await fetch(`${API}/sources/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...sessionHeader() },
        body: JSON.stringify({ source: name }),
      });
      setStatus({ msg: `Removed ${name}.`, ok: true });
      await refresh();
    } catch {
      setStatus({ msg: "Could not reach the backend to delete.", ok: false });
    } finally {
      setBusy(false);
    }
  }

  async function clearAll() {
    if (busy) return;
    if (!confirm("Remove all indexed documents?")) return;
    setBusy(true);
    setStatus({ msg: "Clearing…", ok: true });
    try {
      await fetch(`${API}/clear`, { method: "POST", headers: sessionHeader() });
      setMessages([]);
      setStatus({ msg: "Cleared. Upload a PDF to start fresh.", ok: true });
      await refresh();
    } catch {
      setStatus({ msg: "Could not reach the backend to clear.", ok: false });
    } finally {
      setBusy(false);
    }
  }

  // Clicking a follow-up chip: in-document questions go to our RAG app;
  // "general" ones open Google AI Mode in a new tab (the app only knows the
  // uploaded document, so it can't answer those).
  function followUp(s: Suggestion) {
    if (busy) return;
    if (s.scope === "general") {
      track("followup_web", { question: s.question });
      window.open(webSearchUrl(s.question), "_blank", "noopener,noreferrer");
      return;
    }
    track("followup_document", { question: s.question });
    ask(s.question);
  }

  async function ask(preset?: string) {
    const q = (preset ?? input).trim();
    if (!q || busy) return;
    if (preset === undefined) setInput("");
    // Send recent turns so the backend can resolve follow-ups
    // ("what about section 3?") into standalone retrieval queries.
    const history = messages.slice(-12).map((m) => ({
      role: m.role,
      content: m.text.slice(0, 4000),
    }));
    setMessages((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "" }]);
    setBusy(true);
    track("question_asked", { length: q.length });

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...sessionHeader() },
        body: JSON.stringify({ question: q, history, suggest }),
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let evt: any;
          try {
            evt = JSON.parse(line.slice(6));
          } catch {
            continue; // one malformed event shouldn't kill the whole stream
          }
          // Immutable updates only: never mutate an existing message object.
          // React Strict Mode (next dev) invokes updaters twice to surface impure
          // code — a mutating `text += ...` would double every streamed chunk.
          if (evt.type === "token") {
            const delta = evt.data as string;
            setMessages((m) =>
              m.map((msg, idx) =>
                idx === m.length - 1 ? { ...msg, text: msg.text + delta } : msg
              )
            );
          } else if (evt.type === "citations") {
            const cites = evt.data as Citation[];
            setMessages((m) =>
              m.map((msg, idx) =>
                idx === m.length - 1 ? { ...msg, citations: cites } : msg
              )
            );
          } else if (evt.type === "suggestions") {
            const sugg = evt.data as Suggestion[];
            setMessages((m) =>
              m.map((msg, idx) =>
                idx === m.length - 1 ? { ...msg, suggestions: sugg } : msg
              )
            );
          }
        }
      }
    } catch {
      const errText = "Error reaching the backend. Is it running on " + API + "?";
      setMessages((m) =>
        m.map((msg, idx) => (idx === m.length - 1 ? { ...msg, text: errText } : msg))
      );
    } finally {
      setBusy(false);
    }
  }

  const githubUrl = "https://github.com/nagamsaikiran/rag-document-chat";
  const linkedinUrl = "https://www.linkedin.com/in/saikirannagam";
  const GithubIcon = (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.5 11.5 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222 0 1.606-.014 2.898-.014 3.293 0 .322.216.694.825.576C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
  const LinkedinIcon = (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
              <path d="M13.5 3.75H7.25A1.75 1.75 0 0 0 5.5 5.5v13A1.75 1.75 0 0 0 7.25 20.25h9.5A1.75 1.75 0 0 0 18.5 18.5V8.75z" />
              <path d="M13.25 3.75V9h5" />
              <path d="M8.75 13h6.5M8.75 16h4" />
            </svg>
          </span>
          <div>
            <h1>DocChat RAG</h1>
            <div className="tag">Grounded doc Q&amp;A</div>
          </div>
        </div>

        <div className="hiwrap">
          <div className="s-title">How it works</div>
          <div className="hiw">
            <div className="hiw-item"><span className="d" /><span>Answers grounded in your files, with citations</span></div>
            <div className="hiw-item"><span className="d" /><span>Follow-up suggestions after each answer</span></div>
            <div className="hiw-item"><span className="d" /><span>Private to your browser session</span></div>
          </div>
        </div>

        <nav className="social">
          <a href={githubUrl} target="_blank" rel="noopener noreferrer" title="View source on GitHub">
            {GithubIcon}GitHub
          </a>
          <a href={linkedinUrl} target="_blank" rel="noopener noreferrer" title="LinkedIn profile">
            {LinkedinIcon}LinkedIn
          </a>
        </nav>

        <nav className="m-social" aria-label="Links">
          <a href={githubUrl} target="_blank" rel="noopener noreferrer" title="View source on GitHub" aria-label="View source on GitHub">
            {GithubIcon}
          </a>
          <a href={linkedinUrl} target="_blank" rel="noopener noreferrer" title="LinkedIn profile" aria-label="LinkedIn profile">
            {LinkedinIcon}
          </a>
        </nav>
      </aside>

      <div className="main">
        <div className="toolbar">
          <div className="tb-row">
            <input ref={fileRef} type="file" accept=".pdf,.docx,.txt,.md,.html,.htm" multiple />
            <button onClick={upload} disabled={busy}>Upload &amp; index</button>
            <span className="pill">{indexed} chunks indexed</span>
            {indexed > 0 && (
              <button onClick={clearAll} disabled={busy} className="ghost" style={{ marginLeft: "auto" }}>
                Clear all
              </button>
            )}
          </div>
          <div className="tb-opts">
            <label className="toggle" title="Reads tables, charts, and figures with a vision model. Slower and uses more quota.">
              <input
                type="checkbox"
                checked={vision}
                onChange={(e) => setVision(e.target.checked)}
                disabled={busy}
              />
              Read tables &amp; images
            </label>
            <label className="toggle" title="Turns off the extra AI call that proposes follow-up questions after each answer.">
              <input
                type="checkbox"
                checked={suggest}
                onChange={(e) => {
                  setSuggest(e.target.checked);
                  try {
                    localStorage.setItem("docchat_suggest", e.target.checked ? "1" : "0");
                  } catch {}
                }}
                disabled={busy}
              />
              Suggest follow-up questions
            </label>
            {sources.length > 0 && (
              <div className="tb-sources">
                <span>Sources:</span>
                {sources.map((s) => (
                  <span key={s} className="src">
                    {s}
                    <button
                      className="src-x"
                      title={`Remove ${s}`}
                      onClick={() => deleteSource(s)}
                      disabled={busy}
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
          {status && (
            <p className={`status-msg ${status.ok ? "ok" : "bad"}`}>{status.msg}</p>
          )}
        </div>

        {backendUp === false && (
          <div className="banner err">
            Backend not reachable at {API}. Start it with 2-start-backend.bat and refresh.
          </div>
        )}

        <div className="content">
          <div className="thread">
            {messages.length === 0 && (
              <p className="empty">Ask something about your uploaded documents…</p>
            )}
            {messages.map((m, i) => {
              const isLast = i === messages.length - 1;
              const streaming = busy && isLast && m.role === "assistant";
              return (
                <div key={i} className={`msg ${m.role}`}>
                  <div className="avatar" aria-hidden="true">{m.role === "user" ? "You" : "AI"}</div>
                  <div className="bubble">
                    <div className="label">{m.role === "user" ? "You" : "Assistant"}</div>
                    {m.text ? (
                      <div className={streaming ? "answer streaming" : "answer"}>{m.text}</div>
                    ) : streaming ? (
                      <div className="typing" aria-label="Thinking"><span></span><span></span><span></span></div>
                    ) : (
                      <div className="answer"></div>
                    )}
                    {m.citations && m.citations.length > 0 && (
                      <details className="cites">
                        <summary>
                          {m.citations.length} source{m.citations.length > 1 ? "s" : ""} · click to view
                        </summary>
                        {m.citations.map((c) => (
                          <div key={c.marker} className="cite">
                            <span className="cite-head">[{c.marker}] {c.source} — p.{c.page}</span>
                            <div className="muted">{c.snippet}</div>
                          </div>
                        ))}
                      </details>
                    )}
                    {m.suggestions && m.suggestions.length > 0 && (
                      <div className="followups">
                        <div className="fu-label">Follow-up</div>
                        <div className="fu-chips">
                          {m.suggestions.map((s, j) => (
                            <button
                              key={j}
                              className={`chip ${s.scope}`}
                              onClick={() => followUp(s)}
                              disabled={busy}
                              title={
                                s.scope === "general"
                                  ? "Beyond this document — opens Google AI Mode in a new tab"
                                  : "Answered from your document"
                              }
                            >
                              {s.question}
                              {s.scope === "general" ? " ↗" : ""}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="foot">
          <div className="composer">
            <input
              type="text"
              value={input}
              placeholder="Ask a question…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
            />
            <button onClick={() => ask()} disabled={busy}>Ask</button>
          </div>
        </div>
      </div>
    </div>
  );
}
