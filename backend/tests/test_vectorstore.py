"""VectorStore: per-session isolation against a real (temp) Chroma, fake embedder."""
import app.vectorstore as vs
from app.config import get_settings
from app.ingestion import Chunk


class _FakeEmb:
    # Deterministic 2-D vectors; identical text => identical vector (cosine 0).
    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_one(self, text):
        return [float(len(text)), 1.0]


def _store(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "get_embedder", lambda: _FakeEmb())
    monkeypatch.setattr(get_settings(), "chroma_dir", str(tmp_path / "chroma"))
    vs._store = None  # reset singleton so it picks up the temp dir
    return vs.get_store()


def test_add_query_clear_scoped_to_session(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    store.add([Chunk("hello world", "f.pdf", 1, "f.pdf::p1::c0")], session_id="alice")
    assert store.count("alice") == 1
    assert "f.pdf" in store.sources("alice")

    hits = store.query("hello world", session_id="alice")
    assert hits and hits[0]["source"] == "f.pdf"

    store.clear("alice")
    assert store.count("alice") == 0
    vs._store = None


def test_sessions_are_isolated(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    store.add([Chunk("alice secret", "a.pdf", 1, "a.pdf::p1::c0")], session_id="alice")
    store.add([Chunk("bob secret", "b.pdf", 1, "b.pdf::p1::c0")], session_id="bob")

    # Each session only sees its own docs.
    assert store.sources("alice") == ["a.pdf"]
    assert store.sources("bob") == ["b.pdf"]
    assert store.query("alice secret", session_id="bob") == [] or \
        all(h["source"] == "b.pdf" for h in store.query("alice secret", session_id="bob"))

    # Clearing one session leaves the other intact.
    store.clear("alice")
    assert store.count("alice") == 0
    assert store.count("bob") == 1
    assert store.sources("bob") == ["b.pdf"]
    vs._store = None
