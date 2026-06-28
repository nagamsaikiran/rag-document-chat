"""The RAG pipeline: retrieve -> build grounded prompt -> generate with citations.

Two things here are what separate this from a toy demo:
  1. Grounding guardrail: if the best retrieved chunk is below the relevance
     threshold, we short-circuit and say we don't know instead of letting the
     model hallucinate from parametric memory.
  2. Citations: context is numbered [1], [2], ... and the model is instructed to
     cite those markers, so every claim is traceable to a source + page.
"""
from typing import Iterator, List, Tuple

from app.config import get_settings
from app.llm.factory import get_llm
from app.vectorstore import get_store

SYSTEM_PROMPT = (
    "You are a careful assistant that answers ONLY from the provided context. "
    "Each context block is numbered. Cite the blocks you use with inline markers "
    "like [1] or [2]. If the context does not contain the answer, say you don't "
    "know based on the provided documents. Never invent facts or citations. "
    "Answer concisely and state the answer only once — do not repeat sentences."
)


def _format_context(hits: List[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] (source: {h['source']}, page {h['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def _build_user_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline citations like [1]."
    )


def _retrieve(question: str, session_id: str) -> Tuple[List[dict], bool]:
    """Return (hits, is_grounded). is_grounded is False when nothing is relevant."""
    settings = get_settings()
    hits = get_store().query(question, session_id)
    grounded = bool(hits) and hits[0]["distance"] <= settings.relevance_distance_threshold
    return hits, grounded


def _citations_payload(hits: List[dict]) -> List[dict]:
    return [
        {"marker": i, "source": h["source"], "page": h["page"],
         "snippet": h["text"][:240]}
        for i, h in enumerate(hits, start=1)
    ]


NO_ANSWER = (
    "I couldn't find anything relevant to that in the uploaded documents."
)


def answer(question: str, session_id: str = "public") -> dict:
    """Non-streaming answer (used by the eval harness and /chat)."""
    hits, grounded = _retrieve(question, session_id)
    if not grounded:
        return {"answer": NO_ANSWER, "citations": [], "grounded": False}
    context = _format_context(hits)
    text = get_llm().complete(SYSTEM_PROMPT, _build_user_prompt(question, context))
    return {
        "answer": text,
        "citations": _citations_payload(hits),
        "grounded": True,
    }


def answer_stream(question: str, session_id: str = "public") -> Iterator[dict]:
    """Yield events for SSE-style streaming: token deltas then a final citations event."""
    hits, grounded = _retrieve(question, session_id)
    if not grounded:
        yield {"type": "token", "data": NO_ANSWER}
        yield {"type": "citations", "data": []}
        return
    context = _format_context(hits)
    for delta in get_llm().stream(SYSTEM_PROMPT, _build_user_prompt(question, context)):
        yield {"type": "token", "data": delta}
    yield {"type": "citations", "data": _citations_payload(hits)}
