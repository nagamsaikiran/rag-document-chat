"""New RAG behaviors: citation filtering, history clipping, query rewriting."""
import app.rag as rag


class _Store:
    def count(self, session_id="public"):
        return 3  # non-empty: exercise retrieval, not the "no docs" guard

    def query(self, q, session_id="public", top_k=None):
        return [
            {"text": "first chunk", "source": "a.pdf", "page": 1, "distance": 0.1},
            {"text": "second chunk", "source": "a.pdf", "page": 2, "distance": 0.2},
            {"text": "third chunk", "source": "b.pdf", "page": 5, "distance": 0.3},
        ]


class _EmptyStore:
    def count(self, session_id="public"):
        return 0

    def query(self, q, session_id="public", top_k=None):
        raise AssertionError("query() should not be called when the index is empty")


def test_empty_index_says_upload_first_and_no_suggestions(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _EmptyStore())
    # A real (non-greeting) question with nothing indexed -> "upload first".
    events = list(rag.answer_stream("what is the revenue?"))
    kinds = [e["type"] for e in events]
    assert kinds == ["token", "citations", "suggestions"]
    assert events[0]["data"] == rag.NO_DOCS
    assert events[1]["data"] == []
    assert events[2]["data"] == []  # no follow-ups about a document that doesn't exist


class _NoQueryStore:
    """Non-empty index whose query() must never be called (smalltalk path)."""
    def count(self, session_id="public"):
        return 5

    def query(self, q, session_id="public", top_k=None):
        raise AssertionError("retrieval should be skipped for small talk")


def test_smalltalk_gets_help_reply_not_document_search(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _NoQueryStore())
    for q in ["hi", "how are you?", "what can you do", "who are you?", "thanks"]:
        events = list(rag.answer_stream(q))
        assert events[0]["type"] == "token"
        assert "DocChat" in events[0]["data"]          # helpful reply, not NO_ANSWER
        assert events[0]["data"] != rag.NO_ANSWER
        assert events[-1] == {"type": "suggestions", "data": []}


def test_real_question_is_not_treated_as_smalltalk():
    assert rag._is_smalltalk("hi") is True
    assert rag._is_smalltalk("what is this about?") is False
    assert rag._is_smalltalk("what does the document say about pricing?") is False


class _LLM:
    def __init__(self, reply="Blue is the answer [1][3]."):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply

    def stream(self, system, user):
        self.calls.append((system, user))
        for tok in self.reply.split(" "):
            yield tok + " "


def test_citations_filtered_to_used_markers(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _Store())
    monkeypatch.setattr(rag, "get_llm", lambda: _LLM("Answer citing [1] and [3]."))
    out = rag.answer("q")
    markers = [c["marker"] for c in out["citations"]]
    assert markers == [1, 3]  # [2] was retrieved but not cited


def test_citations_kept_when_none_cited(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _Store())
    monkeypatch.setattr(rag, "get_llm", lambda: _LLM("An answer with no markers."))
    out = rag.answer("q")
    assert len(out["citations"]) == 3  # fall back to all retrieved


def test_stream_filters_citations_from_full_text(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _Store())
    monkeypatch.setattr(rag, "get_llm", lambda: _LLM("Streamed [2] only."))
    events = list(rag.answer_stream("q"))
    # Citations are no longer the final event (a suggestions event now follows),
    # so locate the citations event by type instead of assuming it is last.
    cites = next(e for e in events if e["type"] == "citations")
    assert cites["type"] == "citations"
    assert [c["marker"] for c in cites["data"]] == [2]


def test_history_clipped_and_sanitized():
    history = [{"role": "user", "content": "x" * 99999}] * 100
    clipped = rag._clip_history(history)
    assert len(clipped) <= rag.get_settings().max_history_turns * 2
    assert all(len(m["content"]) <= 1500 for m in clipped)


def test_rewrite_used_for_retrieval_but_not_shown(monkeypatch):
    """With history, the model is asked to rewrite; retrieval uses the rewrite."""
    seen_queries = []

    class _SpyStore(_Store):
        def query(self, q, session_id="public", top_k=None):
            seen_queries.append(q)
            return super().query(q, session_id, top_k)

    llm = _LLM("standalone question about vacation days")
    monkeypatch.setattr(rag, "get_store", lambda: _SpyStore())
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    rag.answer("what about them?", history=[
        {"role": "user", "content": "how many vacation days do we get?"},
        {"role": "assistant", "content": "21 days [1]."},
    ])
    assert seen_queries and "vacation" in seen_queries[0]


def test_rewrite_failure_falls_back_to_original(monkeypatch):
    class _BrokenLLM(_LLM):
        def complete(self, system, user):
            raise RuntimeError("provider down")

    monkeypatch.setattr(rag, "get_llm", lambda: _BrokenLLM())
    q = rag._rewrite_question("original question", [{"role": "user", "content": "hi"}])
    assert q == "original question"


def test_system_prompt_hardened_against_injection():
    assert "UNTRUSTED" in rag.SYSTEM_PROMPT
    ctx = rag._format_context([{"text": "t", "source": "s", "page": 1}])
    assert ctx.startswith("<context>") and ctx.endswith("</context>")
