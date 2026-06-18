"""Central configuration, loaded from environment / .env.

Keeping all tunable knobs in one typed settings object (instead of scattering
os.getenv calls) makes the system reproducible and easy to sweep during eval.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider selection
    llm_provider: str = "openai"
    embedding_provider: str = "openai"

    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Retrieval / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    relevance_distance_threshold: float = 0.55

    chroma_dir: str = "./.chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
