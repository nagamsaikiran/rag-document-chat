"""API security guards: session validation, upload caps, body-size limits."""
import io

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import get_settings


class _FakeStore:
    def __init__(self):
        self.added = []

    def count(self, session_id=None):
        return 0

    def sources(self, session_id):
        return []

    def add(self, chunks, session_id):
        self.added.extend(chunks)
        return len(chunks)

    def clear(self, session_id):
        pass

    def delete_source(self, session_id, source):
        pass

    def cleanup_expired(self, ttl_days):
        return 0


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "get_store", lambda: _FakeStore())
    return TestClient(main.app)


def test_invalid_session_id_rejected(client):
    r = client.get("/health", headers={"X-Session-Id": "evil session!! " + "x" * 100})
    assert r.status_code == 400


def test_valid_session_id_accepted(client):
    r = client.get("/health", headers={"X-Session-Id": "abc-123_XYZ"})
    assert r.status_code == 200


def test_upload_rejects_too_many_files(client):
    limit = get_settings().max_files_per_upload
    files = [
        ("files", (f"f{i}.txt", io.BytesIO(b"hello world"), "text/plain"))
        for i in range(limit + 1)
    ]
    r = client.post("/upload", files=files)
    assert r.status_code == 422
    assert "Too many files" in r.json()["error"]


def test_upload_rejects_unsupported_extension(client):
    files = [("files", ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream"))]
    r = client.post("/upload", files=files)
    assert r.status_code == 200
    assert "Unsupported type" in r.json()["results"][0]["error"]


def test_upload_indexes_supported_text_file(client):
    files = [("files", ("notes.txt", io.BytesIO(b"the sky is blue"), "text/plain"))]
    r = client.post("/upload", files=files)
    body = r.json()
    assert body["results"][0].get("chunks_indexed", 0) >= 1


def test_oversized_body_rejected_early(client):
    huge = str(main._upload_body_limit() + 1)
    r = client.post(
        "/upload",
        content=b"x",
        headers={"Content-Length": huge, "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 413


def test_question_length_capped(client):
    q = "x" * (get_settings().max_question_chars + 1)
    r = client.post("/chat", json={"question": q})
    assert r.status_code == 422


def test_security_headers_present(client):
    r = client.get("/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Request-Id")
