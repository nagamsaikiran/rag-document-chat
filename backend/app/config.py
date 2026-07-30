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
    # Default OFF: vision costs more tokens/quota (one image call per page).
    # Enable per-upload from the UI, or set MULTIMODAL=true to default it on.
    multimodal: bool = False
    multimodal_dpi: int = 150  # render resolution; higher = clearer but bigger

    # Security / resource limits
    cors_origins: str = "*"        # comma-separated allowed origins; lock down in prod
    max_upload_mb: int = 25        # reject larger files (DoS / cost guard)
    max_files_per_upload: int = 5  # files per /upload request (quota guard)
    max_pages: int = 50            # cap pages per PDF
    max_pages_vision: int = 30     # tighter cap when vision mode is on (per-page cost)
    max_question_chars: int = 2000 # cap question length (embedding/LLM cost guard)
    max_session_chunks: int = 5000 # per-visitor total indexed chunks (storage guard)
    rate_limit_per_min: int = 20   # per-IP requests/min on upload+chat+clear (0 = off)
    session_ttl_days: int = 7      # auto-delete a session's documents after N days (0 = keep)
    # Show real exception details in API error responses. Keep True for local
    # dev (actionable messages: bad key, rate limit); the Dockerfile sets it to
    # false so deployed instances never leak internals to strangers.
    debug_errors: bool = True

    # Path to a built frontend (the Next.js `out/` dir) to serve from the same
    # origin. Set in the container; unset in local dev (frontend runs separately).
    static_dir: str = ""

    # Retrieval / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    # Hybrid retrieval: fuse vector similarity with BM25 keyword scores (RRF).
    # Dense-only misses exact keywords (IDs, names, numbers); BM25 catches them.
    hybrid_search: bool = True
    # Cosine distance above which the best hit is considered irrelevant (drives
    # the "I don't know" guardrail). Distances are NOT comparable across
    # embedding models, so when unset we resolve a per-provider default
    # (see rag.relevance_threshold). Calibrate with the eval harness.
    relevance_distance_threshold: float | None = None

    # Conversation memory: how many previous turns to use for follow-up
    # question rewriting and answer context.
    max_history_turns: int = 6

    chroma_dir: str = "./.chroma"


# Per-embedding-provider guardrail defaults (cosine distance). These are
# starting points — run `python -m eval.run_eval` against your own docs to
# calibrate, then pin RELEVANCE_DISTANCE_THRESHOLD in .env.
DEFAULT_RELEVANCE_THRESHOLDS = {
    "openai": 0.55,
    "gemini": 0.60,
}


def relevance_threshold(settings: "Settings") -> float:
    if settings.relevance_distance_threshold is not None:
        return settings.relevance_distance_threshold
    return DEFAULT_RELEVANCE_THRESHOLDS.get(settings.embedding_provider, 0.55)


@lru_cache
def get_settings() -> Settings:
    return Settings()
