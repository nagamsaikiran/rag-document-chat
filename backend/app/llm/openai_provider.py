"""OpenAI implementation of the LLM and embedding interfaces."""
from typing import Iterator, List

from openai import OpenAI

from app.config import get_settings
from app.llm.base import EmbeddingProvider, LLMProvider


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy backend/.env.example to .env and add your key."
        )
    return OpenAI(api_key=settings.openai_api_key)


class OpenAILLM(LLMProvider):
    def __init__(self) -> None:
        self.model = get_settings().openai_chat_model

    def complete(self, system: str, user: str) -> str:
        resp = _client().chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        stream = _client().chat.completions.create(
            model=self.model,
            temperature=0.1,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        self.model = get_settings().openai_embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = _client().embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
