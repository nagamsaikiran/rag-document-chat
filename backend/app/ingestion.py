"""PDF loading and chunking.

Design note (see README): we use a *recursive* splitter that prefers to break
on paragraph -> line -> sentence -> word boundaries before resorting to a hard
character cut. Naive fixed-size slicing splits mid-sentence and hurts retrieval
quality; this keeps chunks semantically coherent while bounding their size.
"""
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader

from app.config import get_settings


@dataclass
class Chunk:
    text: str
    source: str  # filename
    page: int    # 1-indexed page number
    chunk_id: str = field(default="")


def load_pdf_pages(path: str, source_name: str) -> List[tuple[int, str]]:
    """Return [(page_number, page_text)] for a PDF, skipping empty pages."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, overlap: int) -> List[str]:
    """Greedy recursive split that respects natural boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    # Find the best separator that exists in the window.
    for sep in _SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            chunks: List[str] = []
            current = ""
            for part in parts:
                candidate = part if not current else current + sep + part
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # Carry overlap from the tail of the previous chunk.
                    if overlap and chunks:
                        tail = chunks[-1][-overlap:]
                        current = tail + sep + part
                    else:
                        current = part
                    # A single part bigger than size: hard-split it.
                    while len(current) > size:
                        chunks.append(current[:size])
                        current = current[size - overlap:]
            if current:
                chunks.append(current)
            return [c.strip() for c in chunks if c.strip()]

    # No separator found: hard slice with overlap.
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_pdf(path: str, source_name: str) -> List[Chunk]:
    settings = get_settings()
    chunks: List[Chunk] = []
    for page_no, page_text in load_pdf_pages(path, source_name):
        for j, piece in enumerate(
            _recursive_split(page_text, settings.chunk_size, settings.chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source_name,
                    page=page_no,
                    chunk_id=f"{source_name}::p{page_no}::c{j}",
                )
            )
    return chunks
