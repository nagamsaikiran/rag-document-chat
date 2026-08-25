"""FastAPI app: upload + ingest documents, then chat over them with streaming.

Endpoints:
  GET  /health          -> liveness + indexed doc count
  POST /upload          -> accept PDF/DOCX/TXT/MD/HTML, chunk, embed, index
  POST /chat            -> non-streaming answer (JSON)
  POST /chat/stream     -> token streaming via Server-Sent Events
  GET  /sources         -> list indexed source filenames
  POST /sources/delete  -> remove one uploaded file from this session
  POST /clear           -> remove all of this session's documents

Security posture (see docs/AUDIT-2026-07-09.md):
  - per-IP rate limiting on expensive endpoints (proxy-aware when uvicorn runs
    with --proxy-headers)
  - upload size enforced BEFORE buffering (Content-Length + chunked reads)
  - caps: files/request, question length, pages/PDF, chunks/session
  - session ids validated; session data expires after SESSION_TTL_DAYS
  - generic error responses in production (DEBUG_ERRORS=false) with a request
    id that correlates to the full server-side traceback
  - security headers on every response
"""
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import summaries
from app.config import get_settings
from app.ingestion import SUPPORTED_EXTENSIONS, chunk_file
from app.rag import answer, answer_stream, summarize_document
from app.vectorstore import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("docchat.api")


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


def friendly_error(e: Exception) -> str | None:
    """Map expected, user-facing conditions (quota, bad key) to a plain message.

    These aren't bugs — the raw provider stack trace would only confuse a user —
    so we return a clean sentence regardless of DEBUG_ERRORS. Returns None for
    everything else (which then falls through to the normal handling)."""
    t = readable_error(e).lower()
    if any(s in t for s in ("429", "quota", "resource_exhausted", "rate limit", "rate-limit")):
        return ("All available AI providers are at their free-tier limit right now. "
                "Please wait a minute and try again.")
    if any(s in t for s in ("api key", "api_key", "unauthenticated", "permission_denied",
                            " 401", " 403")):
        return ("The AI service rejected the request — the API key may be missing, invalid, "
                "or lacking access. Check GEMINI_API_KEY.")
    return None


def safe_error(e: Exception, request_id: str, context: str) -> str:
    """Log the full exception server-side; return a client-safe message.

    Known user-facing conditions (rate limit, bad key) get a friendly sentence.
    Otherwise: with DEBUG_ERRORS=true (local dev) the real cause is included
    ('invalid API key' beats 'something went wrong' when setting up); deployed
    containers set DEBUG_ERRORS=false and clients only get the ref id.
    """
    logger.exception("[%s] error during %s", request_id, context)
    friendly = friendly_error(e)
    if friendly:
        return friendly
    if get_settings().debug_errors:
        return f"{readable_error(e)} (ref: {request_id})"
    return f"Something went wrong while {context}. Please try again. (ref: {request_id})"


app = FastAPI(title="DocChat RAG", version="2.0.0")

# CORS origins are configurable; defaults to "*" for local dev. Set CORS_ORIGINS
# (comma-separated) to your real frontend origin in production.
_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
if (_origins == ["*"] or not _origins) and get_settings().static_dir:
    logger.warning(
        "CORS_ORIGINS is '*' on a production-shaped deploy (STATIC_DIR set). "
        "Set CORS_ORIGINS to your frontend origin."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session id validation --------------------------------------------------
# Each browser sends a unique X-Session-Id (see frontend); it scopes all
# documents to that visitor. Only sane, bounded ids are accepted — anything
# else would flow into vector-store ids/filters unchecked.
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def valid_session(session_id: str) -> str:
    if not _SID_RE.match(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid X-Session-Id: use 1-64 chars of letters, digits, '-' or '_'.",
        )
    return session_id


# --- Simple in-memory per-IP rate limit on the expensive endpoints ----------
# Protects a public deploy from quota abuse. In-memory (per process), which is
# fine for a single free instance; use a shared store (Redis) for multi-instance.
# NOTE: behind a proxy (Render/Railway/etc.) run uvicorn with --proxy-headers
# so request.client.host is the real visitor IP, not the proxy's.
_HITS: dict[str, list[float]] = defaultdict(list)
_LIMITED_PATHS = ("/upload", "/chat", "/chat/stream", "/clear", "/sources/delete")
_MAX_TRACKED_IPS = 10_000

# Request-body ceilings enforced from Content-Length before any buffering (S1).
_CHAT_BODY_LIMIT = 256 * 1024  # question + clipped history, generous


def _upload_body_limit() -> int:
    s = get_settings()
    # All files together + multipart framing slack.
    return s.max_upload_mb * 1024 * 1024 * s.max_files_per_upload + 1024 * 1024


def _prune_hits(now: float) -> None:
    if len(_HITS) <= _MAX_TRACKED_IPS:
        return
    for ip in [ip for ip, ts in _HITS.items() if not ts or now - ts[-1] > 60]:
        _HITS.pop(ip, None)


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path
    # 1) Reject oversized bodies before reading them (memory DoS guard).
    if path in _LIMITED_PATHS and request.method == "POST":
        limit = _upload_body_limit() if path == "/upload" else _CHAT_BODY_LIMIT
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > limit:
            return JSONResponse(
                status_code=413,
                content={"error": f"Request body too large (limit {limit // (1024*1024)} MB)."},
            )
    # 2) Per-IP rate limit.
    limit = get_settings().rate_limit_per_min
    if limit and path in _LIMITED_PATHS:
        ip = request.client.host if request.client else "?"
        now = time.time()
        _prune_hits(now)
        recent = [t for t in _HITS[ip] if now - t < 60]
        if len(recent) >= limit:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Please wait a minute and try again."},
            )
        recent.append(now)
        _HITS[ip] = recent
    # 3) Request id for error correlation + security headers on the way out.
    request.state.request_id = uuid.uuid4().hex[:8]
    response = await call_next(request)
    response.headers.setdefault("X-Request-Id", request.state.request_id)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline' "
            "https://www.googletagmanager.com; connect-src 'self' "
            "https://www.google-analytics.com https://*.google-analytics.com; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: "
            "https://www.google-analytics.com; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'",
        )
    return response


# --- Session TTL cleanup -----------------------------------------------------
_last_cleanup = 0.0
_CLEANUP_INTERVAL = 3600  # seconds


def _maybe_cleanup() -> None:
    """Purge expired sessions at most once per hour (piggybacks on uploads)."""
    global _last_cleanup
    ttl = get_settings().session_ttl_days
    now = time.time()
    if ttl and now - _last_cleanup > _CLEANUP_INTERVAL:
        _last_cleanup = now
        try:
            get_store().cleanup_expired(ttl)
        except Exception:
            logger.exception("session TTL cleanup failed")


class HistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=24)
    # Whether to generate follow-up suggestions (an extra LLM call). The UI lets
    # visitors turn this off to conserve free-tier quota.
    suggest: bool = True


class DeleteSourceRequest(BaseModel):
    source: str = Field(min_length=1, max_length=512)


def _check_question_len(q: str) -> None:
    limit = get_settings().max_question_chars
    if len(q) > limit:
        raise HTTPException(
            status_code=422,
            detail=f"Question too long ({len(q)} chars). Limit is {limit}.",
        )


@app.get("/health")
def health(session_id: str = Header(default="public", alias="X-Session-Id")):
    sid = valid_session(session_id)
    return {
        "status": "ok",
        "indexed_chunks": get_store().count(sid),
        "multimodal": get_settings().multimodal,
    }


@app.get("/sources")
def sources(session_id: str = Header(default="public", alias="X-Session-Id")):
    sid = valid_session(session_id)
    return {"sources": get_store().sources(sid)}


@app.post("/sources/delete")
def delete_source(req: DeleteSourceRequest,
                  session_id: str = Header(default="public", alias="X-Session-Id")):
    sid = valid_session(session_id)
    get_store().delete_source(sid, req.source)
    summaries.delete_source(sid, req.source)
    return {
        "status": "deleted",
        "source": req.source,
        "indexed_chunks": get_store().count(sid),
    }


@app.post("/clear")
def clear(session_id: str = Header(default="public", alias="X-Session-Id")):
    sid = valid_session(session_id)
    get_store().clear(sid)
    summaries.clear(sid)
    return {"status": "cleared", "indexed_chunks": get_store().count(sid)}


async def _spool_to_tempfile(f: UploadFile, max_bytes: int, suffix: str) -> str | None:
    """Stream the upload to disk in chunks, aborting at the size limit.

    Returns the temp path, or None if the file exceeded max_bytes. Never holds
    the whole file in memory (S1: the old `await f.read()` did).
    """
    size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
        while True:
            chunk = await f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                tmp.close()
                os.unlink(path)
                return None
            tmp.write(chunk)
    return path


@app.post("/upload")
async def upload(
    request: Request,
    files: list[UploadFile] = File(...),
    multimodal: bool | None = Form(default=None),
    session_id: str = Header(default="public", alias="X-Session-Id"),
):
    """Index documents. `multimodal` (form field) overrides the server default
    for this upload: true = vision (reads tables/figures, more quota), false =
    fast text. Omitted = use the server's MULTIMODAL setting. PDF only for
    vision; DOCX/TXT/MD/HTML always use text extraction."""
    sid = valid_session(session_id)
    store = get_store()
    settings = get_settings()
    rid = getattr(request.state, "request_id", "-")
    _maybe_cleanup()

    if len(files) > settings.max_files_per_upload:
        return JSONResponse(
            status_code=422,
            content={"error": f"Too many files ({len(files)}). "
                              f"Limit is {settings.max_files_per_upload} per upload."},
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    summary = []
    for f in files:
        name = (f.filename or "").strip()
        ext = os.path.splitext(name.lower())[1]
        if ext not in SUPPORTED_EXTENSIONS:
            summary.append({
                "file": name,
                "error": f"Unsupported type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            })
            continue
        # Per-session storage quota (S2): stop strangers from filling the disk.
        if store.count(sid) >= settings.max_session_chunks:
            summary.append({
                "file": name,
                "error": f"Session storage limit reached ({settings.max_session_chunks} "
                         "chunks). Delete some documents first.",
            })
            break
        tmp_path = await _spool_to_tempfile(f, max_bytes, suffix=ext)
        if tmp_path is None:
            summary.append({
                "file": name,
                "error": f"File too large. Limit is {settings.max_upload_mb} MB.",
            })
            continue
        try:
            chunks = chunk_file(tmp_path, name, multimodal=multimodal)
            if not chunks:
                summary.append({
                    "file": name,
                    "error": "No extractable text found. If this is a scanned/image "
                             "PDF, enable 'Read tables & images' to use the vision model.",
                })
                continue
            added = store.add(chunks, sid)
            # Summarize the whole document once, so later "what is this about?"
            # questions are answered from a complete summary rather than a few
            # retrieved chunks. Best-effort: never fail the upload over this.
            if settings.enable_doc_summary:
                try:
                    doc_summary = summarize_document([c.text for c in chunks])
                    if doc_summary:
                        summaries.set_summary(sid, name, doc_summary)
                except Exception:
                    logger.exception("[%s] summary generation failed for %s", rid, name)
            summary.append({"file": name, "chunks_indexed": added})
        except ValueError as e:
            # Our own guards (page limit, unsupported type): always safe to show.
            summary.append({"file": name, "error": str(e)})
        except Exception as e:
            summary.append({"file": name, "error": safe_error(e, rid, "indexing the file")})
        finally:
            os.unlink(tmp_path)
    return {"results": summary, "total_indexed_chunks": store.count(sid)}


@app.post("/chat")
def chat(req: ChatRequest, request: Request,
         session_id: str = Header(default="public", alias="X-Session-Id")):
    sid = valid_session(session_id)
    _check_question_len(req.question)
    rid = getattr(request.state, "request_id", "-")
    try:
        history = [m.model_dump() for m in req.history]
        return answer(req.question, sid, history)
    except Exception as e:
        return {"answer": f"[Error] {safe_error(e, rid, 'answering the question')}",
                "citations": [], "grounded": False}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request,
                session_id: str = Header(default="public", alias="X-Session-Id")):
    sid = valid_session(session_id)
    _check_question_len(req.question)
    rid = getattr(request.state, "request_id", "-")
    history = [m.model_dump() for m in req.history]

    def event_gen():
        try:
            for event in answer_stream(req.question, sid, history, include_suggestions=req.suggest):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            err = {"type": "token",
                   "data": f"\n[Error] {safe_error(e, rid, 'answering the question')}"}
            yield f"data: {json.dumps(err)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Serve the built frontend from the same origin (single-container deploy).
# Mounted LAST so it never shadows the API routes above. No-op in local dev,
# where STATIC_DIR is unset and the frontend runs separately on :3000.
_static_dir = get_settings().static_dir
if _static_dir and os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
