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
import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

# Open CORS for local dev; tighten to your frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": get_store().count()}


@app.get("/sources")
def sources():
    return {"sources": get_store().sources()}


@app.post("/clear")
def clear():
    get_store().clear()
    return {"status": "cleared", "indexed_chunks": get_store().count()}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    store = get_store()
    summary = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            summary.append({"file": f.filename, "error": "only .pdf supported"})
            continue
        # Persist to a temp path because pypdf wants a file path/handle.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await f.read())
            tmp_path = tmp.name
        try:
            chunks = chunk_pdf(tmp_path, f.filename)
            if not chunks:
                summary.append({
                    "file": f.filename,
                    "error": "No extractable text found. Is this a scanned/image PDF? "
                             "(OCR is not supported.)",
                })
                continue
            added = store.add(chunks)
            summary.append({"file": f.filename, "chunks_indexed": added})
        except Exception as e:
            # Most likely the embedding provider call failed (bad/missing API key,
            # wrong model name, or rate limit). Surface it instead of a blank 500.
            traceback.print_exc()
            summary.append({"file": f.filename, "error": readable_error(e)})
        finally:
            os.unlink(tmp_path)
    return {"results": summary, "total_indexed_chunks": store.count()}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return answer(req.question)
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"[Error] {type(e).__name__}: {e}", "citations": [], "grounded": False}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def event_gen():
        try:
            for event in answer_stream(req.question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            traceback.print_exc()
            err = {"type": "token", "data": f"\n[Error] {type(e).__name__}: {e}"}
            yield f"data: {json.dumps(err)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
