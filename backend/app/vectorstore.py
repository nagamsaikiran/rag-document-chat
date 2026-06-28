"""Thin wrapper around a persistent Chroma collection.

We compute embeddings ourselves (via the provider layer) and hand the vectors
to Chroma, rather than letting Chroma own embedding. That keeps the embedding
model swappable through the same provider abstraction as the LLM.

Multi-tenancy: every chunk is tagged with the caller's `session_id`, and all
reads/writes are scoped to it via a metadata filter. This gives each visitor
their own private set of documents on the shared deployment — no login needed.
"""
from typing import List

import chromadb

from app.config import get_settings
from app.ingestion import Chunk
from app.llm.factory import get_embedder


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        # Cosine space so distances are comparable to our relevance threshold.
        self._collection = self._client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        self._embedder = get_embedder()

    def add(self, chunks: List[Chunk], session_id: str) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed([c.text for c in chunks])
        # Prefix ids with the session so the same filename in different sessions
        # never collides; upsert so re-uploading the same file is idempotent.
        self._collection.upsert(
            ids=[f"{session_id}::{c.chunk_id}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "page": c.page, "session_id": session_id}
                for c in chunks
            ],
        )
        return len(chunks)

    def query(self, question: str, session_id: str, top_k: int | None = None) -> List[dict]:
        settings = get_settings()
        k = top_k or settings.top_k
        if self.count(session_id) == 0:
            return []
        q_vec = self._embedder.embed_one(question)
        res = self._collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            where={"session_id": session_id},  # only this user's chunks
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "distance": dist,
                }
            )
        return hits

    def clear(self, session_id: str) -> None:
        """Remove only the given session's documents (others are untouched)."""
        self._collection.delete(where={"session_id": session_id})

    def count(self, session_id: str | None = None) -> int:
        if session_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"session_id": session_id})["ids"])

    def sources(self, session_id: str) -> List[str]:
        res = self._collection.get(where={"session_id": session_id})
        metas = res["metadatas"] or []
        return sorted({m["source"] for m in metas})


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
