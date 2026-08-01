# Single-container build: compiles the Next.js UI to static files and serves
# them from the FastAPI backend, so the whole app runs as ONE service / ONE URL.

# ---- Stage 1: build the frontend to static files ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Same-origin API (served by FastAPI); optional GA id baked at build time.
ENV NEXT_PUBLIC_API_URL=""
ARG NEXT_PUBLIC_GA_ID=""
ENV NEXT_PUBLIC_GA_ID=$NEXT_PUBLIC_GA_ID
RUN npm run build      # produces /fe/out

# ---- Stage 2: backend + the built frontend ----
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend /fe/out ./static

# Serve the UI from the same origin; vision off by default (protect quota).
ENV STATIC_DIR=/app/static
ENV MULTIMODAL=false
# Never leak internals (tracebacks, key fragments) to strangers on a deploy;
# errors carry a request id that correlates to the server logs instead.
ENV DEBUG_ERRORS=false
# Chroma index lives here; owned by the app user below.
ENV CHROMA_DIR=/app/.chroma
# Set at deploy time (platform secrets), NOT here:
#   GEMINI_API_KEY=...   (required)
#   CORS_ORIGINS=https://your-frontend-origin   (recommended)

# Run as a non-root user: a compromised process can't touch the rest of the
# container, and platforms increasingly refuse root images.
RUN useradd --create-home appuser \
    && mkdir -p /app/.chroma \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health')" || exit 1

# Hosts (Render/HF/Railway) inject $PORT; fall back to 8000 locally.
# --proxy-headers so request.client.host is the real visitor IP behind the
# platform's proxy (rate limiting keys off it).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
