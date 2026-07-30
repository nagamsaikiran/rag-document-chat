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

    def embed_query(self, text):
        return self.embed_one(text)


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


def test_delete_single_source(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    store.add([Chunk("doc one text", "one.pdf", 1, "one.pdf::p1::c0")], session_id="s1")
    store.add([Chunk("doc two text", "two.pdf", 1, "two.pdf::p1::c0")], session_id="s1")
    store.delete_source("s1", "one.pdf")
    assert store.sources("s1") == ["two.pdf"]
    assert store.count("s1") == 1
    vs._store = None


def test_ttl_cleanup_removes_old_chunks(monkeypatch, tmp_path):
    store = _store(monkeypatch, tmp_path)
    store.add([Chunk("old text", "old.pdf", 1, "old.pdf::p1::c0")], session_id="s2")
    # Backdate the chunk 10 days.
    got = store._collection.get(where={"session_id": "s2"})
    metas = got["metadatas"]
    for m in metas:
        m["uploaded_at"] = m["uploaded_at"] - 10 * 86400
    store._collection.update(ids=got["ids"], metadatas=metas)
    removed = store.cleanup_expired(ttl_days=7)
    assert removed == 1
    assert store.count("s2") == 0
    vs._store = None


def test_hybrid_finds_keyword_matches(monkeypatch, tmp_path):
    """BM25 should surface an exact-keyword chunk even if dense ranking is poor
    (our fake embedder only encodes text length, so dense is uninformative)."""
    store = _store(monkeypatch, tmp_path)
    chunks = [
        Chunk(f"filler text about nothing relevant number {i}", "f.pdf", 1, f"f.pdf::p1::c{i}")
        for i in range(8)
    ]
    chunks.append(Chunk("the escalation code is AX-7741", "f.pdf", 2, "f.pdf::p2::c0"))
    store.add(chunks, session_id="kw")
    hits = store.query("What is escalation code AX-7741?", session_id="kw")
    assert any("AX-7741" in h["text"] for h in hits)
    vs._store = None
