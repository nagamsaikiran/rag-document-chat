"""Failover across chat providers: quota errors fall through, others don't."""
import pytest

from app.llm.failover import FailoverLLM, is_transient


class _Quota(Exception):
    status_code = 429
    def __str__(self):
        return "429 RESOURCE_EXHAUSTED: quota exceeded"


class _FakeLLM:
    def __init__(self, name, fail=False, transient=True):
        self.name, self.fail, self.transient = name, fail, transient
        self.calls = 0

    def _err(self):
        return _Quota() if self.transient else ValueError("bad request")

    def complete(self, system, user):
        self.calls += 1
        if self.fail:
            raise self._err()
        return f"answer from {self.name}"

    def stream(self, system, user):
        self.calls += 1
        if self.fail:
            raise self._err()
        yield f"[{self.name}] "
        yield "hello"


def _chain(*llms):
    return FailoverLLM([(l.name, l) for l in llms], cooldown_s=0)


def test_is_transient_classifies_quota_vs_bugs():
    assert is_transient(_Quota())
    assert not is_transient(ValueError("malformed request"))


def test_is_transient_skips_retired_model_and_auth():
    # A retired free-tier model (Groq) or a bad key should skip that provider,
    # not kill the whole chain.
    assert is_transient(Exception(
        "NotFoundError model_not_found: The model `llama-3.3-70b-versatile` "
        "does not exist or you do not have access to it."
    ))
    assert is_transient(Exception("401 invalid api key"))


def test_complete_falls_through_on_quota():
    a, b = _FakeLLM("a", fail=True), _FakeLLM("b")
    assert _chain(a, b).complete("s", "u") == "answer from b"
    assert a.calls == 1 and b.calls == 1


def test_complete_raises_when_all_exhausted():
    with pytest.raises(_Quota):
        _chain(_FakeLLM("a", fail=True), _FakeLLM("b", fail=True)).complete("s", "u")


def test_non_transient_error_does_not_fail_over():
    a, b = _FakeLLM("a", fail=True, transient=False), _FakeLLM("b")
    with pytest.raises(ValueError):
        _chain(a, b).complete("s", "u")
    assert b.calls == 0  # a real bug shouldn't silently hit the next provider


def test_stream_switches_before_first_token():
    a, b = _FakeLLM("a", fail=True), _FakeLLM("b")
    out = "".join(_chain(a, b).stream("s", "u"))
    assert out == "[b] hello"


def test_factory_skips_provider_without_key(monkeypatch):
    from app.config import Settings
    import app.llm.factory as f
    from app.llm.gemini_provider import GeminiLLM
    s = Settings(llm_providers="gemini,groq", gemini_api_key="g")  # no groq key
    monkeypatch.setattr(f, "get_settings", lambda: s)
    f.get_llm.cache_clear()
    try:
        assert isinstance(f.get_llm(), GeminiLLM)  # keyless groq dropped -> single
    finally:
        f.get_llm.cache_clear()


def test_factory_builds_failover_for_multiple(monkeypatch):
    from app.config import Settings
    import app.llm.factory as f
    s = Settings(llm_providers="groq,cerebras", groq_api_key="x", cerebras_api_key="y")
    monkeypatch.setattr(f, "get_settings", lambda: s)
    f.get_llm.cache_clear()
    try:
        assert isinstance(f.get_llm(), FailoverLLM)
    finally:
        f.get_llm.cache_clear()
