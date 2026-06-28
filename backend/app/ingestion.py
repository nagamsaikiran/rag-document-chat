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


# Instruction given to the vision model for each page image.
_VISION_PROMPT = (
    "Transcribe this document page into clean Markdown. Rules:\n"
    "- Preserve reading order.\n"
    "- Render any tables as proper Markdown tables, keeping rows and columns.\n"
    "- For charts, graphs, photos, logos or diagrams, insert a concise description "
    "in the form: [Figure: <what it shows, including any numbers/labels>].\n"
    "- Do not add commentary or invent content. Output only the page content."
)


def _check_page_limit(n_pages: int) -> None:
    limit = get_settings().max_pages
    if n_pages > limit:
        raise ValueError(
            f"PDF has {n_pages} pages; the limit is {limit}. "
            "Split the document or raise MAX_PAGES."
        )


def load_pdf_pages_text(path: str) -> List[tuple[int, str]]:
    """Fast path: extract the embedded text layer with pypdf."""
    reader = PdfReader(path)
    _check_page_limit(len(reader.pages))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def load_pdf_pages_vision(path: str) -> List[tuple[int, str]]:
    """Multimodal path: render each page to an image and have a vision model
    transcribe it into Markdown — captures tables, charts, figures, and scanned
    text that the plain text layer misses."""
    import fitz  # PyMuPDF, imported lazily so text-only mode needs no dep

    from app.llm.factory import get_llm

    llm = get_llm()
    dpi = get_settings().multimodal_dpi
    pages = []
    doc = fitz.open(path)
    try:
        _check_page_limit(doc.page_count)
        for i in range(len(doc)):
            png = doc[i].get_pixmap(dpi=dpi).tobytes("png")
            text = llm.transcribe_image(png, "image/png", _VISION_PROMPT).strip()
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()
    return pages


def load_pdf_pages(
    path: str, source_name: str, multimodal: bool | None = None
) -> List[tuple[int, str]]:
    """Return [(page_number, page_text)] for a PDF, skipping empty pages.

    Vision mode (tables/charts/figures/scanned) is used when enabled, otherwise
    the fast pypdf text path. `multimodal` overrides the global default per call
    (so the UI can toggle it per upload). If the provider has no vision support
    we cleanly fall back to text; real vision errors propagate so they surface.
    """
    use_vision = get_settings().multimodal if multimodal is None else multimodal
    if use_vision:
        try:
            return load_pdf_pages_vision(path)
        except NotImplementedError:
            pass  # provider has no vision; use the text path instead
    return load_pdf_pages_text(path)


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


def chunk_pdf(
    path: str, source_name: str, multimodal: bool | None = None
) -> List[Chunk]:
    settings = get_settings()
    chunks: List[Chunk] = []
    for page_no, page_text in load_pdf_pages(path, source_name, multimodal):
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
