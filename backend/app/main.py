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

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion import chunk_pdf
from app.rag import answer, answer_stream
from app.vectorstore import get_store

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
            added = store.add(chunks)
            summary.append({"file": f.filename, "chunks_indexed": added})
        finally:
            os.unlink(tmp_path)
    return {"results": summary, "total_indexed_chunks": store.count()}


@app.post("/chat")
def chat(req: ChatRequest):
    return answer(req.question)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def event_gen():
        for event in answer_stream(req.question):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
