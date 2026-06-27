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

    # Google Gemini (free tier)
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Multimodal ingestion: render each PDF page to an image and have a vision
    # model transcribe it (reads tables, charts, figures, scanned text). Slower
    # and uses one vision call per page. Falls back to fast text extraction when
    # False, or automatically if the provider has no vision support.
    multimodal: bool = True
    multimodal_dpi: int = 150  # render resolution; higher = clearer but bigger

    # Security / resource limits
    cors_origins: str = "*"   # comma-separated allowed origins; lock down in prod
    max_upload_mb: int = 25   # reject larger PDFs (DoS / cost guard)
    max_pages: int = 50       # cap pages per PDF (esp. for per-page vision cost)

    # Retrieval / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    relevance_distance_threshold: float = 0.55

    chroma_dir: str = "./.chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
