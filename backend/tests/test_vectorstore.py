"""VectorStore add / query / clear against a real (temp) Chroma, fake embedder."""
import app.vectorstore as vs
from app.config import get_settings
from app.ingestion import Chunk


class _FakeEmb:
    # Deterministic 2-D vectors; identical text => identical vector (cosine 0).
    def embed(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_one(self, text):
        return [float(len(text)), 1.0]


def test_add_query_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "get_embedder", lambda: _FakeEmb())
    monkeypatch.setattr(get_settings(), "chroma_dir", str(tmp_path / "chroma"))
    vs._store = None  # reset the module singleton so it picks up the temp dir

    store = vs.get_store()
    added = store.add([Chunk("hello", "f.pdf", 1, "f.pdf::p1::c0")])
    assert added == 1
    assert store.count() == 1
    assert "f.pdf" in store.sources()

    hits = store.query("hello")
    assert hits and hits[0]["source"] == "f.pdf" and hits[0]["page"] == 1

    store.clear()
    assert store.count() == 0
    vs._store = None  # leave the singleton clean for other tests
