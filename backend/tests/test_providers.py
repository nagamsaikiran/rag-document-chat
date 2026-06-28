"""Provider abstraction: registration + vision default behavior + error unwrap."""
import pytest

import app.main as main
from app.llm.base import LLMProvider
from app.llm.factory import _EMBEDDERS, _LLMS


def test_providers_registered():
    assert {"gemini", "openai"} <= set(_LLMS)
    assert {"gemini", "openai"} <= set(_EMBEDDERS)


def test_transcribe_image_default_raises():
    class Bare(LLMProvider):
        def complete(self, system, user):
            return ""

        def stream(self, system, user):
            yield ""

    with pytest.raises(NotImplementedError):
        Bare().transcribe_image(b"x", "image/png", "prompt")


def test_readable_error_unwraps_code_and_message():
    class ClientError(Exception):
        code = 404
        message = "model not found"

    msg = main.readable_error(ClientError())
    assert "404" in msg and "model not found" in msg
