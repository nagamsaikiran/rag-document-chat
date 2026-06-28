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
  let id = localStorage.getItem("docchat_sid");
  if (!id) {
    id =
      (crypto as any).randomUUID?.() ??
      Math.random().toString(36).slice(2) + Date.now().toString(36);
    localStorage.setItem("docchat_sid", id);
  }
  return id;
}
function sessionHeader(): Record<string, string> {
  return { "X-Session-Id": sessionId() };
}

type Citation = { marker: number; source: string; page: number; snippet: string };
type Message = { role: "user" | "assistant"; text: string; citations?: Citation[] };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [indexed, setIndexed] = useState(0);
  const [sources, setSources] = useState<string[]>([]);
  const [vision, setVision] = useState(false);
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
  }, []);

  async function upload() {
    const files = fileRef.current?.files;
    if (!files || files.length === 0) {
      setStatus({ msg: "Pick a PDF first.", ok: false });
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

  async function ask() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "" }]);
    setBusy(true);
    track("question_asked", { length: q.length });

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...sessionHeader() },
        body: JSON.stringify({ question: q }),
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
          const evt = JSON.parse(line.slice(6));
          if (evt.type === "token") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1].text += evt.data;
              return copy;
            });
          } else if (evt.type === "citations") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1].citations = evt.data;
              return copy;
            });
          }
        }
      }
    } catch {
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1].text = "Error reaching the backend. Is it running on " + API + "?";
        return copy;
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap">
      <h1>DocChat RAG</h1>
      <p className="sub">Upload PDFs and ask questions. Answers are grounded in your documents with citations.</p>

      {backendUp === false && (
        <div className="banner err">
          Backend not reachable at {API}. Start it with 2-start-backend.bat and refresh.
        </div>
      )}

      <div className="card">
        <div className="row">
          <input ref={fileRef} type="file" accept="application/pdf" multiple />
          <button onClick={upload} disabled={busy}>Upload &amp; index</button>
          <span className="pill">{indexed} chunks indexed</span>
          {indexed > 0 && (
            <button onClick={clearAll} disabled={busy} className="ghost">Clear all</button>
          )}
        </div>
        <label className="toggle" title="Reads tables, charts, and figures with a vision model. Slower and uses more quota.">
          <input
            type="checkbox"
            checked={vision}
            onChange={(e) => setVision(e.target.checked)}
            disabled={busy}
          />
          Read tables &amp; images
        </label>
        {status && (
          <p className={status.ok ? "muted" : "err-text"} style={{ marginTop: 10 }}>
            {status.msg}
          </p>
        )}
        {sources.length > 0 && (
          <p className="muted" style={{ marginTop: 10 }}>Sources: {sources.join(", ")}</p>
        )}
      </div>

      <div className="card">
        {messages.length === 0 && <p className="muted">Ask something about your uploaded documents…</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="label">{m.role}</div>
            <div>{m.text || (busy && i === messages.length - 1 ? "…" : "")}</div>
            {m.citations && m.citations.length > 0 && (
              <div className="cites">
                {m.citations.map((c) => (
                  <div key={c.marker} className="cite">
                    [{c.marker}] {c.source} — p.{c.page}
                    <div className="muted">{c.snippet}…</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="row">
        <input
          type="text"
          value={input}
          placeholder="Ask a question…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button onClick={ask} disabled={busy}>Ask</button>
      </div>
    </div>
  );
}
