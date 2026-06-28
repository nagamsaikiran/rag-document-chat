"""FastAPI app: upload + ingest PDFs, then chat over them with streaming.

Endpoints:
  GET  /health          -> liveness + indexed doc count
  POST /upload          -> accept PDF(s), chunk, embed, index
  POST /chat            -> non-streaming answer (JSON)
  POST /chat/stream     -> token streaming via Server-Sent Events
  GET  /sources         -> list indexed source filenames
"""
import json
import os
import tempfile
import time
import traceback
from collections import defaultdict

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import get_settings
from app.ingestion import chunk_pdf
from app.rag import answer, answer_stream
from app.vectorstore import get_store

def readable_error(e: Exception) -> str:
    """Unwrap nested provider errors into a human-readable message.

    Google's SDK retries internally and raises tenacity RetryError, which hides
    the real cause (e.g. invalid API key, unknown model, batch-size limit).
    """
    cause = e
    # tenacity RetryError -> underlying exception from the last attempt
    if type(e).__name__ == "RetryError" and hasattr(e, "last_attempt"):
        try:
            cause = e.last_attempt.exception() or e
        except Exception:
            cause = e
    code = getattr(cause, "code", None) or getattr(cause, "status_code", None)
    msg = getattr(cause, "message", None) or str(cause)
    return f"{type(cause).__name__} {code}: {msg}" if code else f"{type(cause).__name__}: {msg}"


app = FastAPI(title="DocChat RAG", version="1.0.0")

# CORS origins are configurable; defaults to "*" for local dev. Set CORS_ORIGINS
# (comma-separated) to your real frontend origin in production.
_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple in-memory per-IP rate limit on the expensive endpoints ---------
# Protects a public deploy from quota abuse. In-memory (per process), which is
# fine for a single free instance; use a shared store (Redis) for multi-instance.
_HITS: dict[str, list[float]] = defaultdict(list)
_LIMITED_PATHS = ("/upload", "/chat", "/chat/stream")


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    limit = get_settings().rate_limit_per_min
    if limit and request.url.path in _LIMITED_PATHS:
        ip = request.client.host if request.client else "?"
        now = time.time()
        recent = [t for t in _HITS[ip] if now - t < 60]
        if len(recent) >= limit:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Please wait a minute and try again."},
            )
        recent.append(now)
        _HITS[ip] = recent
    return await call_next(request)


class ChatRequest(BaseModel):
    question: str


# Each browser sends a unique X-Session-Id (see frontend); it scopes all
# documents to that visitor. Defaults to "public" for header-less API calls.
@app.get("/health")
def health(session_id: str = Header(default="public", alias="X-Session-Id")):
    from app.config import get_settings
    return {
        "status": "ok",
        "indexed_chunks": get_store().count(session_id),
        "multimodal": get_settings().multimodal,
    }


@app.get("/sources")
def sources(session_id: str = Header(default="public", alias="X-Session-Id")):
    return {"sources": get_store().sources(session_id)}


@app.post("/clear")
def clear(session_id: str = Header(default="public", alias="X-Session-Id")):
    get_store().clear(session_id)
    return {"status": "cleared", "indexed_chunks": get_store().count(session_id)}


@app.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    multimodal: bool | None = Form(default=None),
    session_id: str = Header(default="public", alias="X-Session-Id"),
):
    """Index PDFs. `multimodal` (form field) overrides the server default for
    this upload: true = vision (reads tables/figures, more quota), false = fast
    text. Omitted = use the server's MULTIMODAL setting."""
    store = get_store()
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    summary = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            summary.append({"file": f.filename, "error": "only .pdf supported"})
            continue
        data = await f.read()
        if len(data) > max_bytes:
            summary.append({
                "file": f.filename,
                "error": f"File too large ({len(data) // (1024*1024)} MB). "
                         f"Limit is {settings.max_upload_mb} MB.",
            })
            continue
        # Persist to a temp path because the PDF readers want a file path/handle.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            chunks = chunk_pdf(tmp_path, f.filename, multimodal=multimodal)
            if not chunks:
                summary.append({
                    "file": f.filename,
                    "error": "No extractable text found. If this is a scanned/image "
                             "PDF, enable MULTIMODAL to read it with the vision model.",
                })
                continue
            added = store.add(chunks, session_id)
            summary.append({"file": f.filename, "chunks_indexed": added})
        except Exception as e:
            # Most likely the embedding provider call failed (bad/missing API key,
            # wrong model name, or rate limit). Surface it instead of a blank 500.
            traceback.print_exc()
            summary.append({"file": f.filename, "error": readable_error(e)})
        finally:
            os.unlink(tmp_path)
    return {"results": summary, "total_indexed_chunks": store.count(session_id)}


@app.post("/chat")
def chat(req: ChatRequest, session_id: str = Header(default="public", alias="X-Session-Id")):
    try:
        return answer(req.question, session_id)
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"[Error] {type(e).__name__}: {e}", "citations": [], "grounded": False}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, session_id: str = Header(default="public", alias="X-Session-Id")):
    def event_gen():
        try:
            for event in answer_stream(req.question, session_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            traceback.print_exc()
            err = {"type": "token", "data": f"\n[Error] {type(e).__name__}: {e}"}
            yield f"data: {json.dumps(err)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Serve the built frontend from the same origin (single-container deploy).
# Mounted LAST so it never shadows the API routes above. No-op in local dev,
# where STATIC_DIR is unset and the frontend runs separately on :3000.
_static_dir = get_settings().static_dir
if _static_dir and os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
