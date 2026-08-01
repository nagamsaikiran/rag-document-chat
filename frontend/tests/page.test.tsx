/**
 * Component tests for the chat UI. All network calls are mocked, so these
 * verify the frontend's real behaviors: upload validation, SSE stream
 * parsing (including malformed events), citations rendering, per-source
 * delete, and that conversation history is sent with follow-up questions.
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Home from "../app/page";

// ---- helpers ---------------------------------------------------------------

function sseResponse(events: string[]) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      for (const e of events) controller.enqueue(encoder.encode(e));
      controller.close();
    },
  });
  return { body } as unknown as Response;
}

type FetchCall = { url: string; init?: RequestInit };
let calls: FetchCall[] = [];

function mockFetch(routes: Record<string, (init?: RequestInit) => unknown>) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    for (const [path, handler] of Object.entries(routes)) {
      if (url.includes(path)) return handler(init) as Response;
    }
    return { json: async () => ({}) } as Response;
  });
}

const jsonRes = (data: unknown) => ({ json: async () => data }) as Response;

beforeEach(() => {
  calls = [];
  localStorage.clear();
  vi.stubGlobal("confirm", vi.fn(() => true));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---- tests -----------------------------------------------------------------

describe("Home", () => {
  it("renders and shows indexed chunk count from /health", async () => {
    vi.stubGlobal("fetch", mockFetch({
      "/health": () => jsonRes({ indexed_chunks: 7 }),
      "/sources": () => jsonRes({ sources: ["a.pdf"] }),
    }));
    render(<Home />);
    expect(await screen.findByText("7 chunks indexed")).toBeInTheDocument();
    expect(await screen.findByText("a.pdf")).toBeInTheDocument();
  });

  it("sends a stable X-Session-Id header on every request", async () => {
    vi.stubGlobal("fetch", mockFetch({
      "/health": () => jsonRes({ indexed_chunks: 0 }),
      "/sources": () => jsonRes({ sources: [] }),
    }));
    render(<Home />);
    await waitFor(() => expect(calls.length).toBeGreaterThanOrEqual(2));
    const sids = calls.map(
      (c) => (c.init?.headers as Record<string, string>)["X-Session-Id"]
    );
    expect(sids[0]).toBeTruthy();
    expect(new Set(sids).size).toBe(1); // same id everywhere
    expect(sids[0]).toMatch(/^[A-Za-z0-9_-]{1,64}$/); // backend contract
  });

  it("refuses to upload when no file is chosen", async () => {
    vi.stubGlobal("fetch", mockFetch({
      "/health": () => jsonRes({ indexed_chunks: 0 }),
      "/sources": () => jsonRes({ sources: [] }),
    }));
    render(<Home />);
    await userEvent.click(screen.getByRole("button", { name: /upload/i }));
    expect(await screen.findByText(/pick a file first/i)).toBeInTheDocument();
  });

  it("streams an answer, tolerates malformed SSE lines, renders citations", async () => {
    vi.stubGlobal("fetch", mockFetch({
      "/health": () => jsonRes({ indexed_chunks: 3 }),
      "/sources": () => jsonRes({ sources: ["doc.pdf"] }),
      "/chat/stream": () => sseResponse([
        'data: {"type": "token", "data": "The answer"}\n\n',
        "data: {not-json!!}\n\n", // must be skipped, not crash the stream
        'data: {"type": "token", "data": " is 42 [1]."}\n\n',
        'data: {"type": "citations", "data": [{"marker": 1, "source": "doc.pdf", "page": 3, "snippet": "the answer is 42"}]}\n\n',
        'data: {"type": "done"}\n\n',
      ]),
    }));
    render(<Home />);
    const box = screen.getByPlaceholderText(/ask a question/i);
    await userEvent.type(box, "what is the answer?{Enter}");
    expect(await screen.findByText("The answer is 42 [1].")).toBeInTheDocument();
    expect(await screen.findByText(/doc\.pdf — p\.3/)).toBeInTheDocument();
    expect(screen.getByText(/the answer is 42/)).toBeInTheDocument();
  });

  it("sends prior turns as history with follow-up questions", async () => {
    const stream = () => sseResponse([
      'data: {"type": "token", "data": "answer"}\n\n',
      'data: {"type": "citations", "data": []}\n\n',
      'data: {"type": "done"}\n\n',
    ]);
    vi.stubGlobal("fetch", mockFetch({
      "/health": () => jsonRes({ indexed_chunks: 3 }),
      "/sources": () => jsonRes({ sources: [] }),
      "/chat/stream": stream,
    }));
    render(<Home />);
    const box = screen.getByPlaceholderText(/ask a question/i);
    await userEvent.type(box, "first question{Enter}");
    await screen.findByText("answer");
    await userEvent.type(box, "and a follow-up?{Enter}");
    await waitFor(() => {
      const chatCalls = calls.filter((c) => c.url.includes("/chat/stream"));
      expect(chatCalls.length).toBe(2);
      const body = JSON.parse(String(chatCalls[1].init?.body));
      expect(body.question).toBe("and a follow-up?");
      expect(body.history.map((m: any) => m.role)).toEqual(["user", "assistant"]);
      expect(body.history[0].content).toBe("first question");
    });
  });

  it("deletes a single source via /sources/delete", async () => {
    vi.stubGlobal("fetch", mockFetch({
      "/health": () => jsonRes({ indexed_chunks: 5 }),
      "/sources/delete": () => jsonRes({ status: "deleted" }),
      "/sources": () => jsonRes({ sources: ["a.pdf", "b.pdf"] }),
    }));
    render(<Home />);
    await screen.findByText("a.pdf");
    await userEvent.click(screen.getByTitle("Remove a.pdf"));
    await waitFor(() => {
      const del = calls.find((c) => c.url.includes("/sources/delete"));
      expect(del).toBeTruthy();
      expect(JSON.parse(String(del!.init?.body))).toEqual({ source: "a.pdf" });
    });
  });

  it("shows the backend-down banner when /health fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("down"); }));
    render(<Home />);
    expect(await screen.findByText(/backend not reachable/i)).toBeInTheDocument();
  });
});
