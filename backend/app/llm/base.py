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

    # Optional vision capability. Providers backed by a multimodal model override
    # this; others inherit the explicit "not supported" default. Used by the
    # multimodal ingestion path to transcribe a page image into markdown.
    def transcribe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support image transcription. "
            "Use a multimodal provider/model, or disable MULTIMODAL."
        )


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""

    @abstractmethod
    def embed_one(self, text: str) -> List[float]:
        """Convenience for a single text."""
