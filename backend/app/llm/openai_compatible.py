"""Chat providers that speak the OpenAI Chat Completions API.

Groq, Cerebras, and Mistral all expose OpenAI-compatible endpoints, so a single
thin implementation — just a different base URL, key, and model — covers all
three. We reuse the already-installed `openai` SDK (no new dependency).

These are used for **answer generation only**. They do not provide the
embeddings the RAG index needs; embeddings always come from EMBEDDING_PROVIDER
(see factory.py / config.py), because vectors are not portable across models.
"""
from typing import Iterator, List

from openai import OpenAI

from app.llm.base import LLMProvider


class OpenAICompatibleLLM(LLMProvider):
    """An OpenAI-Chat-API provider parameterized by endpoint, key, and model."""

    def __init__(self, provider: str, base_url: str, api_key: str, model: str) -> None:
        self.provider = provider
        self.model = model
        if not api_key:
            raise RuntimeError(
                f"{provider.upper()}_API_KEY is not set. Add it to backend/.env — "
                f"'{provider}' is listed in your LLM_PROVIDERS failover chain."
            )
        # base_url points the OpenAI client at Groq/Cerebras/Mistral instead.
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _messages(self, system: str, user: str) -> List[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0.1, messages=self._messages(system, user)
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, user: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model, temperature=0.1, stream=True,
            messages=self._messages(system, user),
        )
        for chunk in stream:
            # Some compatible providers emit trailing chunks with no choices
            # (usage/stats) — skip those defensively.
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # transcribe_image is intentionally NOT implemented — these providers/models
    # aren't guaranteed multimodal. The base class raises NotImplementedError,
    # and FailoverLLM skips past that to a vision-capable provider (Gemini).
