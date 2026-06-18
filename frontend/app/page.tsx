"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = { marker: number; source: string; page: number; snippet: string };
type Message = { role: "user" | "assistant"; text: string; citations?: Citation[] };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [indexed, setIndexed] = useState(0);
  const [sources, setSources] = useState<string[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      const h = await fetch(`${API}/health`).then((r) => r.json());
      setIndexed(h.indexed_chunks ?? 0);
      const s = await fetch(`${API}/sources`).then((r) => r.json());
      setSources(s.sources ?? []);
    } catch {
      /* backend not up yet */
    }
  }
  useEffect(() => {
    refresh();
  }, []);

  async function upload() {
    const files = fileRef.current?.files;
    if (!files || files.length === 0) return;
    setBusy(true);
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    try {
      await fetch(`${API}/upload`, { method: "POST", body: fd });
      await refresh();
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function ask() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }, { role: "assistant", text: "" }]);
    setBusy(true);

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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

      <div className="card">
        <div className="row">
          <input ref={fileRef} type="file" accept="application/pdf" multiple />
          <button onClick={upload} disabled={busy}>Upload &amp; index</button>
          <span className="pill">{indexed} chunks indexed</span>
        </div>
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
