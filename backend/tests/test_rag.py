"""RAG pipeline: grounding guardrail, citations, streaming events (mocked)."""
import app.rag as rag


class _Store:
    def __init__(self, distance):
        self._d = distance

    def count(self, session_id="public"):
        return 1  # non-empty: exercise the retrieval path, not the "no docs" guard

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


def test_answer_stream_emits_tokens_citations_then_suggestions(monkeypatch):
    monkeypatch.setattr(rag, "get_store", lambda: _Store(0.1))
    monkeypatch.setattr(rag, "get_llm", lambda: _LLM())
    events = list(rag.answer_stream("q"))
    types = [e["type"] for e in events]
    assert "token" in types
    assert "citations" in types
    # Follow-up suggestions are emitted as the final event of the stream.
    assert types[-1] == "suggestions"


def test_is_overview_detects_whole_document_questions():
    assert rag._is_overview("what is this file about")
    assert rag._is_overview("Summarize the document")
    assert rag._is_overview("give me an overview")
    assert not rag._is_overview("what is the candidate's phone number")
    assert not rag._is_overview("which languages does he know")


def test_clean_snippet_drops_midword_lead_and_trims():
    # Leading partial word ("tems,") is dropped; whitespace collapsed.
    out = rag._clean_snippet("tems, developing optimized SQL queries and more")
    assert out.startswith("developing")
    # A clean, capitalized start is preserved.
    assert rag._clean_snippet("Sai Kiran Nagam").startswith("Sai")


def test_parse_suggestions_handles_json_and_fences():
    raw = '```json\n[{"question":"What next?","scope":"general"},' \
          '{"question":"More on role?","scope":"in_document"}]\n```'
    out = rag._parse_suggestions(raw)
    assert len(out) == 2
    assert out[0]["scope"] == "general"
    assert out[1]["question"] == "More on role?"
    # Garbage in -> empty list, never an exception.
    assert rag._parse_suggestions("not json at all") == []


def test_parse_suggestions_reclassifies_self_referential_general():
    # A 'general' question that refers back to the document can't be answered by
    # a web search, so it must be kept in the app (reclassified to in_document).
    raw = ('[{"question":"What does the candidate know about Angular?","scope":"general"},'
           '{"question":"What is Angular used for?","scope":"general"}]')
    out = rag._parse_suggestions(raw)
    by_q = {o["question"]: o["scope"] for o in out}
    assert by_q["What does the candidate know about Angular?"] == "in_document"
    assert by_q["What is Angular used for?"] == "general"  # genuinely standalone


def test_stream_deltas_passes_clean_deltas_through():
    # Ordinary delta stream: joined output must be identical, nothing dropped.
    parts = ["This ", "is ", "a ", "clean ", "answer ", "with the the end."]
    assert "".join(rag._stream_deltas(iter(parts))) == "".join(parts)


def test_stream_deltas_repairs_overlapping_chunks():
    # Reproduces the google-genai duplication: chunks that repeat/overlap already
    # sent text (a prefix re-send, a tail re-send, and a mid re-anchor).
    s = ("This file is a resume for Sai Kiran Nagam, a Full Stack .NET Developer "
         "with six years of experience [4]. It details his web application work "
         "and Agile delivery [1, 2, 3, 4]. It also showcases his technical skills "
         "on Azure and Entity Framework Core.")
    p1 = s.index(".NET") + 4
    p2 = s.index("[4].") + 4
    p3 = s.index("[1, 2, 3, 4].") + len("[1, 2, 3, 4].")
    chunks = [s[0:p1], s[0:p2], s[p1:p2], s[p2:p3], s[p2:len(s)]]
    assert "".join(rag._stream_deltas(iter(chunks))) == s


def test_citation_cap_limits_visible_sources(monkeypatch):
    # Six hits, the answer cites all six, but the cap keeps at most max_citations.
    hits = [{"text": f"chunk {i}", "source": "f.pdf", "page": i, "distance": 0.1}
            for i in range(1, 7)]
    text = "".join(f"[{i}]" for i in range(1, 7))
    cap = rag.get_settings().max_citations
    assert len(rag._citations_payload(hits, text)) == cap
