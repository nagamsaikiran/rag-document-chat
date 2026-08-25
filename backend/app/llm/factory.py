"""Wire the configured provider strings to concrete implementations.

`get_llm()` returns either a single provider (LLM_PROVIDER) or, when
LLM_PROVIDERS lists more than one, a FailoverLLM that tries them in order and
falls through when one hits its rate/quota limit. `get_embedder()` is always a
single provider — embeddings never fail over (vectors aren't portable across
models). To add a provider: implement the base interfaces and register it below.
"""
import logging
from functools import lru_cache

from app.config import Settings, get_settings
from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.failover import FailoverLLM
from app.llm.gemini_provider import GeminiEmbeddings, GeminiLLM
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.openai_provider import OpenAIEmbeddings, OpenAILLM

logger = logging.getLogger("docchat.factory")

# OpenAI-compatible free-tier providers: name -> (base_url, key attr, model attr)
_OPENAI_COMPAT = {
    "groq":     ("https://api.groq.com/openai/v1", "groq_api_key", "groq_chat_model"),
    "cerebras": ("https://api.cerebras.ai/v1",     "cerebras_api_key", "cerebras_chat_model"),
    "mistral":  ("https://api.mistral.ai/v1",      "mistral_api_key", "mistral_chat_model"),
}

_EMBEDDERS = {
    "openai": OpenAIEmbeddings,
    "gemini": GeminiEmbeddings,
}


def _build_llm(name: str, settings: Settings) -> LLMProvider:
    """Construct one chat provider. Raises if its API key is missing."""
    if name == "openai":
        return OpenAILLM()
    if name == "gemini":
        return GeminiLLM()
    if name in _OPENAI_COMPAT:
        base_url, key_attr, model_attr = _OPENAI_COMPAT[name]
        return OpenAICompatibleLLM(
            name, base_url, getattr(settings, key_attr), getattr(settings, model_attr)
        )
    raise ValueError(
        f"Unknown LLM provider '{name}'. Options: "
        f"{['openai', 'gemini', *_OPENAI_COMPAT]}"
    )


@lru_cache
def get_llm() -> LLMProvider:
    settings = get_settings()
    chain = [n.strip() for n in settings.llm_providers.split(",") if n.strip()]
    if not chain:
        chain = [settings.llm_provider]

    providers: list[tuple[str, LLMProvider]] = []
    for name in chain:
        try:
            providers.append((name, _build_llm(name, settings)))
        except Exception as e:
            # A listed provider with no key (or an unknown name) is skipped so it
            # doesn't break the chain — the others still work.
            logger.warning("LLM provider '%s' not available, skipping: %s", name, e)

    if not providers:
        raise ValueError(
            f"No usable LLM providers from {chain}. Set an API key for at least "
            "one (e.g. GEMINI_API_KEY) and list it in LLM_PROVIDERS."
        )
    if len(providers) == 1:
        return providers[0][1]
    logger.info("LLM failover chain: %s", " -> ".join(n for n, _ in providers))
    return FailoverLLM(providers)


@lru_cache
def get_embedder() -> EmbeddingProvider:
    name = get_settings().embedding_provider
    if name not in _EMBEDDERS:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{name}'. Options: {list(_EMBEDDERS)}"
        )
    return _EMBEDDERS[name]()
