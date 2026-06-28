"""Chunking + ingestion guards (no PDF/network needed)."""
import pytest

import app.ingestion as ing
from app.config import get_settings
from app.ingestion import Chunk, _check_page_limit, _recursive_split, chunk_pdf


def test_recursive_split_respects_size_bound():
    text = "word " * 500
    chunks = _recursive_split(text, 200, 40)
    assert chunks
    # Each chunk stays within size + a small overlap slack.
    assert all(len(c) <= 200 + 40 for c in chunks)


def test_recursive_split_short_and_empty():
    assert _recursive_split("hi there", 100, 10) == ["hi there"]
    assert _recursive_split("    ", 100, 10) == []


def test_page_limit_guard(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "max_pages", 2)
    _check_page_limit(2)  # at the limit: allowed
    with pytest.raises(ValueError):
        _check_page_limit(3)  # over the limit: rejected


def test_chunk_pdf_builds_chunks(monkeypatch):
    # Avoid real PDF parsing: stub the page loader.
    monkeypatch.setattr(
        ing, "load_pdf_pages",
        lambda path, src, mm=None: [(1, "hello world"), (2, "second page text")],
    )
    chunks = chunk_pdf("x.pdf", "doc.pdf")
    assert chunks and all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "doc.pdf" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.chunk_id.startswith("doc.pdf::p") for c in chunks)
