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
# Set at deploy time (platform secrets), NOT here:
#   GEMINI_API_KEY=...   (required)

EXPOSE 8000
# Hosts (Render/HF/Railway) inject $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
