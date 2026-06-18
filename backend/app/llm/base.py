"""Provider-agnostic interfaces.

The rest of the app depends ONLY on these abstract types, never on a concrete
SDK. Swapping OpenAI for Anthropic, Ollama, or a local model means adding one
file and one line in factory.py — no changes to the RAG logic, API, or tests.
This is the seam that keeps the system portable.
"""
from abc import ABC, abstractmethod
from typing import Iterator, List


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return a full completion as a single string."""

    @abstractmethod
    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield completion tokens/chunks as they arrive."""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:
        """Convenience for a single text."""
