"""Thin wrapper around a persistent Chroma collection, plus hybrid retrieval.

We compute embeddings ourselves (via the provider layer) and hand the vectors
to Chroma, rather than letting Chroma own embedding. That keeps the embedding
model swappable through the same provider abstraction as the LLM.

Multi-tenancy: every chunk is tagged with the caller's `session_id`, and all
reads/writes are scoped to it via a metadata filter. This gives each visitor
their own private set of documents on the shared deployment — no login needed.
Sessions expire after SESSION_TTL_DAYS (see cleanup_expired).

Retrieval is hybrid: dense vector similarity fused with BM25 keyword scores
via Reciprocal Rank Fusion (RRF). Dense-only retrieval misses exact-match
queries (IDs, part numbers, names); BM25 catches those.
"""
import logging
import re
import time
from typing import List

import chromadb
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion import Chunk
from app.llm.factory import get_embedder

logger = logging.getLogger("docchat.vectorstore")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _rrf(rankings: List[List[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank + 1)
    return scores


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
        now = int(time.time())
        # Prefix ids with the session so the same filename in different sessions
        # never collides; upsert so re-uploading the same file is idempotent.
        self._collection.upsert(
            ids=[f"{session_id}::{c.chunk_id}" for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page,
                    "session_id": session_id,
                    "uploaded_at": now,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    # ------------------------------------------------------------- retrieval

    def query(self, question: str, session_id: str, top_k: int | None = None) -> List[dict]:
        """Hybrid (vector + BM25 via RRF) retrieval scoped to one session.

        Each hit carries the *vector* cosine distance — the grounding guardrail
        keys off the best dense distance regardless of fusion order, so BM25
        can re-rank but never fake relevance for the "I don't know" check.
        """
        settings = get_settings()
        k = top_k or settings.top_k
        if self.count(session_id) == 0:
            return []

        q_vec = self._embedder.embed_query(question)
        # Over-fetch for fusion headroom; fusion then keeps the best k.
        fetch_k = max(k * 3, 10)
        res = self._collection.query(
            query_embeddings=[q_vec],
            n_results=fetch_k,
            where={"session_id": session_id},  # only this user's chunks
        )
        by_id: dict[str, dict] = {}
        vector_ranking: List[str] = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            vector_ranking.append(id_)
            by_id[id_] = {
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "distance": dist,
            }

        if not settings.hybrid_search:
            return [by_id[i] for i in vector_ranking[:k]]

        bm25_ranking = self._bm25_ranking(question, session_id, fetch_k, by_id)
        fused = _rrf([vector_ranking, bm25_ranking])
        order = sorted(fused, key=fused.get, reverse=True)
        hits = [by_id[i] for i in order if i in by_id][:k]
        # Guardrail contract: hits[0]["distance"] must be the best dense
        # distance available, whatever fusion decided the order should be.
        if hits:
            best = min(h["distance"] for h in hits)
            hits[0] = {**hits[0], "distance": min(hits[0]["distance"], best)}
        return hits

    def _bm25_ranking(
        self, question: str, session_id: str, k: int, by_id: dict[str, dict]
    ) -> List[str]:
        """BM25 ranking over this session's chunks (docs already fetched for the
        session; fine at per-visitor scale — swap for a real index at corpus scale)."""
        try:
            res = self._collection.get(
                where={"session_id": session_id}, include=["documents"]
            )
            ids, docs = res["ids"], res["documents"]
            if not ids:
                return []
            bm25 = BM25Okapi([_tokenize(d) for d in docs])
            scores = bm25.get_scores(_tokenize(question))
            ranked = sorted(zip(ids, scores), key=lambda p: p[1], reverse=True)
            # Chunks BM25 likes but the vector fetch missed need their payload
            # loaded so fusion can actually return them.
            top = [id_ for id_, s in ranked[:k] if s > 0]
            missing = [i for i in top if i not in by_id]
            if missing:
                got = self._collection.get(ids=missing, include=["documents", "metadatas"])
                for id_, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                    by_id[id_] = {
                        "text": doc,
                        "source": meta["source"],
                        "page": meta["page"],
                        # No dense distance measured for BM25-only hits; use a
                        # sentinel above any threshold so they can support an
                        # answer but never satisfy the grounding check alone.
                        "distance": 1.0,
                    }
            return top
        except Exception:
            logger.exception("BM25 ranking failed; falling back to dense-only")
            return []

    # ------------------------------------------------------------ management

    def clear(self, session_id: str) -> None:
        """Remove only the given session's documents (others are untouched)."""
        self._collection.delete(where={"session_id": session_id})

    def delete_source(self, session_id: str, source: str) -> None:
        """Remove a single uploaded file from one session."""
        self._collection.delete(
            where={"$and": [{"session_id": session_id}, {"source": source}]}
        )

    def cleanup_expired(self, ttl_days: int) -> int:
        """Delete chunks older than ttl_days across all sessions. Returns count."""
        if ttl_days <= 0:
            return 0
        cutoff = int(time.time()) - ttl_days * 86400
        expired = self._collection.get(
            where={"uploaded_at": {"$lt": cutoff}}, include=[]
        )["ids"]
        if expired:
            self._collection.delete(ids=expired)
            logger.info("TTL cleanup removed %d expired chunks", len(expired))
        return len(expired)

    def count(self, session_id: str | None = None) -> int:
        if session_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"session_id": session_id}, include=[])["ids"])

    def sources(self, session_id: str) -> List[str]:
        res = self._collection.get(where={"session_id": session_id})
        metas = res["metadatas"] or []
        return sorted({m["source"] for m in metas})

    def all_chunks(self, session_id: str) -> List[dict]:
        """Every chunk for a session, in document order (source, page, id). Used
        by the "whole-document" answer path when the content is small enough to
        fit the model's context, so retrieval never has to guess which pieces
        matter (fixes counting / listing questions that top-k retrieval misses)."""
        res = self._collection.get(
            where={"session_id": session_id}, include=["documents", "metadatas"]
        )
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        items = [
            {"id": i, "text": d, "source": (m or {}).get("source", ""),
             "page": (m or {}).get("page", 1)}
            for i, d, m in zip(ids, docs, metas)
        ]
        items.sort(key=lambda x: (x["source"] or "", x["page"] or 0, x["id"]))
        return items


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
