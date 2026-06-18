"""Wire the configured provider strings to concrete implementations.

To add a provider (e.g. Anthropic or a local Ollama model): implement the
base interfaces in a new file, then register it in the dicts below.
"""
from functools import lru_cache

from app.config import get_settings
from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.gemini_provider import GeminiEmbeddings, GeminiLLM
from app.llm.openai_provider import OpenAIEmbeddings, OpenAILLM

_LLMS = {
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
    # "anthropic": AnthropicLLM,
    # "ollama": OllamaLLM,
}

_EMBEDDERS = {
    "openai": OpenAIEmbeddings,
    "gemini": GeminiEmbeddings,
    # "ollama": OllamaEmbeddings,
}


@lru_cache
def get_llm() -> LLMProvider:
    name = get_settings().llm_provider
    if name not in _LLMS:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {list(_LLMS)}")
    return _LLMS[name]()


@lru_cache
def get_embedder() -> EmbeddingProvider:
    name = get_settings().embedding_provider
    if name not in _EMBEDDERS:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{name}'. Options: {list(_EMBEDDERS)}"
        )
    return _EMBEDDERS[name]()
