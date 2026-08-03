"""New ingestion behaviors: true recursion, header/footer stripping, formats."""
from app.ingestion import (
    _recursive_split,
    _strip_repeated_lines,
    chunk_file,
    load_document_pages,
)


def test_recursive_split_recurses_instead_of_hard_cutting():
    # One paragraph break at the top, then a long sentence-only paragraph.
    # A single-pass splitter hard-cuts the long paragraph; a recursive one
    # falls through to sentence boundaries.
    long_para = ("This is a sentence about topic alpha. " * 20).strip()
    text = "Short intro.\n\n" + long_para
    chunks = _recursive_split(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    # Sentence-boundary splits end at whole words/sentences, never mid-word.
    # (". " is consumed as the separator, so chunks may end without the dot.)
    assert all(c.rstrip(".").endswith(("intro", "alpha")) for c in chunks)


def test_strip_repeated_headers_and_footers():
    pages = [
        (i, f"ACME Corp Confidential\nReal content for page {i} here.\n{i}")
        for i in range(1, 6)
    ]
    cleaned = _strip_repeated_lines(pages)
    for _, text in cleaned:
        assert "Confidential" not in text
        assert "Real content" in text


def test_strip_keeps_small_docs_untouched():
    pages = [(1, "Header\nBody"), (2, "Header\nOther body")]
    assert _strip_repeated_lines(pages) == pages


def test_txt_md_html_docx_loaders(tmp_path):
    (tmp_path / "a.txt").write_text("plain text body", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nmarkdown body", encoding="utf-8")
    (tmp_path / "c.html").write_text(
        "<html><head><script>evil()</script></head>"
        "<body><h1>Doc</h1><p>html body</p></body></html>",
        encoding="utf-8",
    )
    assert load_document_pages(str(tmp_path / "a.txt"), "a.txt")[0][1] == "plain text body"
    assert "markdown body" in load_document_pages(str(tmp_path / "b.md"), "b.md")[0][1]
    html_text = load_document_pages(str(tmp_path / "c.html"), "c.html")[0][1]
    assert "html body" in html_text and "evil" not in html_text


def test_chunk_file_builds_chunks_for_txt(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello world " * 10, encoding="utf-8")
    chunks = chunk_file(str(p), "notes.txt")
    assert chunks and chunks[0].source == "notes.txt" and chunks[0].page == 1


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ")
    try:
        load_document_pages(str(p), "x.exe")
        assert False, "should have raised"
    except ValueError as e:
        assert "Unsupported" in str(e)
