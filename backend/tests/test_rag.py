"""RAG pipeline: grounding guardrail, citations, streaming events (mocked)."""
import app.rag as rag


class _Store:
    def __init__(self, distance):
        self._d = distance

    def query(self, q, session_id="public", top_k=None):
        return [{"text": "the answer text", "source": "f.pdf", "page": 2, "distance": self._d}]


class _LLM:
    def complete(self, system, user):
        return "Answer [1]."

    def stream(self, system, user):
        yield "Ans"
        yield "wer [1]."


def test_guardrail_refuses_when_not_relevant(monkeypatch):
    # Distance well above the default threshold (0.55) -> should refuse.
    monkeypatch.setattr(rag, "get_store", lambda: _Store(1.5))
    out = rag.answer("q")
    assert out["grounded"] is False
    assert out["citations"] == []


def test_answer_grounded_returns_citations(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _Store(0.1))
    monkeypatch.setattr(rag, "get_llm", lambda: _LLM())
    out = rag.answer("q")
    assert out["grounded"] is True
    assert out["citations"][0]["source"] == "f.pdf"
    assert out["citations"][0]["page"] == 2


def test_answer_stream_emits_tokens_then_citations(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _Store(0.1))
    monkeypatch.setattr(rag, "get_llm", lambda: _LLM())
    events = list(rag.answer_stream("q"))
    assert any(e["type"] == "token" for e in events)
    assert events[-1]["type"] == "citations"
