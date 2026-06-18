"""Thin wrapper around a persistent Chroma collection.

We compute embeddings ourselves (via the provider layer) and hand the vectors
to Chroma, rather than letting Chroma own embedding. That keeps the embedding
model swappable through the same provider abstraction as the LLM.
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

    def add(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed([c.text for c in chunks])
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> List[dict]:
        settings = get_settings()
        k = top_k or settings.top_k
        if self._collection.count() == 0:
            return []
        q_vec = self._embedder.embed_one(question)
        res = self._collection.query(query_embeddings=[q_vec], n_results=k)
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

    def count(self) -> int:
        return self._collection.count()

    def sources(self) -> List[str]:
        if self._collection.count() == 0:
            return []
        metas = self._collection.get()["metadatas"]
        return sorted({m["source"] for m in metas})


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
