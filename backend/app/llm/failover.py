"""Chain multiple chat providers so one takes over when another runs out.

Only the answer-generation LLM fails over. Embeddings stay on a single provider
(vectors aren't comparable across models, so switching would mean re-indexing).

On a transient failure — a rate-limit / quota error (HTTP 429,
RESOURCE_EXHAUSTED) or a provider outage (5xx / overloaded) — we move to the
next provider and put the failed one on a short cooldown so we don't hammer it
on every request. If everything is cooling down we try them all anyway (a stale
cooldown shouldn't take the whole app down).
"""
import logging
import time
from typing import Iterator, List, Tuple

from app.llm.base import LLMProvider

logger = logging.getLogger("docchat.failover")


def is_transient(e: Exception) -> bool:
    """True for 'this provider can't serve this now — try the next one'.

    Covers both temporary conditions (rate/quota limits, provider outages) and
    per-provider misconfiguration that the *other* providers may not share (a
    model that was retired from a free tier, an auth problem). Genuine request
    bugs that would fail everywhere (e.g. a malformed prompt) are NOT matched, so
    they surface instead of silently burning through every provider."""
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    text = f"{type(e).__name__} {code} {e}".lower()
    return any(s in text for s in (
        # temporary: rate/quota limits and outages
        "429", "quota", "resource_exhausted", "rate limit", "rate-limit",
        "insufficient_quota", "overloaded", "unavailable", "try again",
        "500", "502", "503", "529",
        # per-provider config problems -> skip this provider, try the next
        "model_not_found", "does not exist", "not found", "404",
        "deprecated", "decommission", "401", "403",
        "invalid api key", "authentication",
        "402", "payment required", "billing",
    ))


class FailoverLLM(LLMProvider):
    def __init__(self, providers: List[Tuple[str, LLMProvider]], cooldown_s: float = 60.0):
        # providers: ordered [(name, provider), ...]; first is primary.
        self._providers = providers
        self._cooldown = cooldown_s
        self._blocked: dict[str, float] = {}  # name -> unix time it becomes usable

    def _order(self) -> List[Tuple[str, LLMProvider]]:
        now = time.time()
        ready = [(n, p) for (n, p) in self._providers if self._blocked.get(n, 0) <= now]
        return ready or self._providers  # never leave the app with zero options

    def _trip(self, name: str, e: Exception) -> None:
        self._blocked[name] = time.time() + self._cooldown
        logger.warning("LLM provider '%s' unavailable (%s: %s); failing over",
                       name, type(e).__name__, str(e)[:120])

    def complete(self, system: str, user: str) -> str:
        last: Exception | None = None
        for name, prov in self._order():
            try:
                return prov.complete(system, user)
            except Exception as e:  # noqa: BLE001 — decide by error, then re-raise
                if is_transient(e):
                    self._trip(name, e)
                    last = e
                    continue
                raise
        raise last or RuntimeError("No LLM providers are currently available.")

    def stream(self, system: str, user: str) -> Iterator[str]:
        # Quota errors surface when the request is made, i.e. before any token.
        # So we pull the FIRST chunk inside the try: if a provider is going to
        # fail, it fails here and we switch cleanly. Once the first token is out
        # we're committed to that provider (can't swap mid-answer).
        last: Exception | None = None
        for name, prov in self._order():
            try:
                it = prov.stream(system, user)
                first = next(it, None)
            except Exception as e:  # noqa: BLE001
                if is_transient(e):
                    self._trip(name, e)
                    last = e
                    continue
                raise
            if first is not None:
                yield first
            yield from it
            return
        raise last or RuntimeError("No LLM providers are currently available.")

    def transcribe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        last: Exception | None = None
        for name, prov in self._order():
            try:
                return prov.transcribe_image(image_bytes, mime_type, prompt)
            except NotImplementedError as e:
                last = e  # this provider isn't multimodal; try the next
                continue
            except Exception as e:  # noqa: BLE001
                if is_transient(e):
                    self._trip(name, e)
                    last = e
                    continue
                raise
        if isinstance(last, NotImplementedError):
            raise NotImplementedError(
                "No configured provider supports image transcription. Use a "
                "multimodal provider (e.g. Gemini) or disable MULTIMODAL."
            )
        raise last or RuntimeError("No LLM providers are currently available.")
