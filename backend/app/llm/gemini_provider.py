"""Google Gemini implementation of the LLM and embedding interfaces.

Gemini has a genuinely free tier, so this lets the whole app run at $0.
It plugs into the exact same interfaces as the OpenAI provider — proof that
the abstraction in base.py actually pays off.

Uses the current `google-genai` SDK (the older `google-generativeai` package
is deprecated).
"""
from functools import lru_cache
from typing import Iterator, List

from google import genai
from google.genai import types

from app.config import get_settings
from app.llm.base import EmbeddingProvider, LLMProvider


@lru_cache
def _client() -> "genai.Client":
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey, "
            "then add it to backend/.env."
        )
    return genai.Client(api_key=settings.gemini_api_key)


class GeminiLLM(LLMProvider):
    def __init__(self) -> None:
        self.model_name = get_settings().gemini_chat_model

    def _config(self, system: str) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(system_instruction=system, temperature=0.1)

    def complete(self, system: str, user: str) -> str:
        resp = _client().models.generate_content(
            model=self.model_name, contents=user, config=self._config(system)
        )
        return resp.text or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        for chunk in _client().models.generate_content_stream(
            model=self.model_name, contents=user, config=self._config(system)
        ):
            if getattr(chunk, "text", None):
                yield chunk.text


class GeminiEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = get_settings().gemini_embedding_model

    # Gemini's embed endpoint rejects requests with more than ~100 texts, so we
    # batch. A multi-page PDF easily exceeds this in a single upload.
    _BATCH = 100

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        client = _client()
        for i in range(0, len(texts), self._BATCH):
            batch = texts[i:i + self._BATCH]
            resp = client.models.embed_content(model=self.model, contents=batch)
            out.extend(e.values for e in resp.embeddings)
        return out

    def embed_one(self, text: str) -> List[float]:
        resp = _client().models.embed_content(model=self.model, contents=text)
        return resp.embeddings[0].values
