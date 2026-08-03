# Security

This document records the security posture of DocChat RAG: what was checked,
what was fixed, and the residual risks with their rationale. A full audit with
findings and fixes lives in [docs/AUDIT-2026-07-09.md](docs/AUDIT-2026-07-09.md).

## Dependency vulnerabilities — fixed

All Python dependencies are pinned and audited with `pip-audit` (also runs in
CI on every push). The initial pinned set carried 46 known CVEs, concentrated
in the security-sensitive parsers (`pypdf`, `python-multipart`, `starlette`,
`python-dotenv`); all were upgraded and `pip-audit` now reports **no known
vulnerabilities** that apply to this architecture. The frontend runs Next.js
15.x (past the 2024–2025 advisory wave — cache poisoning, middleware bypass —
with no high/critical advisories at pin time). Re-run with:

```bash
pip install pip-audit
pip-audit -r backend/requirements.txt
```

One advisory is explicitly assessed and suppressed rather than fixed:
**PYSEC-2026-311 / CVE-2026-45829** (chromadb ≥1.0.0, no patched release yet)
is a pre-auth code injection in the *standalone Chroma HTTP server's*
`/api/v2/...` collections endpoint. This app embeds Chroma in-process via
`PersistentClient` and never runs `chroma run`, so the vulnerable endpoint is
not reachable. Re-assess if the deployment ever switches to a Chroma server.

## Application hardening

| Area | Risk | Mitigation |
|---|---|---|
| **Secrets** | API key leakage | Keys live only in `backend/.env`, which is git-ignored. Only `.env.example` (placeholders) is committed. |
| **Memory DoS** | Huge request bodies buffered in RAM | `Content-Length` checked *before* reading; files are streamed to disk in 1 MB chunks and aborted at `MAX_UPLOAD_MB` (default 25). |
| **Quota abuse** | Many files / giant questions / vision spam | Caps: `MAX_FILES_PER_UPLOAD` (5), `MAX_QUESTION_CHARS` (2000), `MAX_PAGES` (50), `MAX_PAGES_VISION` (30), `MAX_SESSION_CHUNKS` (5000), per-IP `RATE_LIMIT_PER_MIN` (20) on upload/chat/clear/delete. |
| **Rate limiting behind proxies** | All visitors share the proxy's IP | Container runs uvicorn with `--proxy-headers` so limits key off the real client IP. |
| **Session isolation** | Cross-visitor data access | Every chunk is tagged with a validated `X-Session-Id` (`[A-Za-z0-9_-]{1,64}`); all reads/writes/deletes are metadata-filtered to it. |
| **Data retention** | Strangers' documents stored forever | Sessions auto-expire after `SESSION_TTL_DAYS` (default 7); hourly cleanup. |
| **Error leakage** | Tracebacks / key fragments shown to clients | `DEBUG_ERRORS=false` in the Docker image: clients get a generic message + request id; the full traceback is logged server-side under the same id. |
| **File type** | Arbitrary payloads | Extension allowlist (`.pdf .docx .txt .md .html .htm`); parsing errors are caught per-file, not fatal. |
| **Path traversal** | Malicious filenames | Uploaded bytes are streamed to a random `tempfile`; the original filename is never used as a filesystem path. |
| **Prompt injection** | Instructions embedded in uploaded documents | Retrieved text is wrapped in `<context>` delimiters and the system prompt declares it untrusted data. Reduces, does not eliminate — see residual risks. |
| **CORS** | Any origin calling the API | Configurable via `CORS_ORIGINS`; the server logs a warning when deployed with `*`. |
| **XSS** | Malicious text in a document rendered in the UI | React renders all content as text (no `dangerouslySetInnerHTML`); security headers (`nosniff`, `frame-ancestors 'none'`, CSP) on every response. |
| **Container** | Root-owned process | Image runs as a non-root `appuser`, with a `HEALTHCHECK`. |

## Residual risks (accepted, with rationale)

- **No authentication.** Session ids isolate visitors but are not credentials; anyone
  with a session id can access that session. For production, add real auth (OAuth /
  API keys) and per-user namespaces.
- **Prompt injection is mitigated, not solved.** No prompt-level defense is complete;
  a sufficiently crafted document may still steer the model. Treat answers over
  untrusted documents accordingly.
- **In-memory rate limiting.** Per-process only; a multi-instance deploy needs a
  shared store (Redis).
- **At-rest data.** The Chroma index persists extracted document text unencrypted in
  `CHROMA_DIR`. Use disk encryption / per-user isolation for sensitive corpora.
- **Outbound calls to the LLM provider.** Document text is sent to OpenAI/Google for
  embedding and generation. Use a local provider (e.g. Ollama) for fully on-prem.

## Reporting

This is a personal/portfolio project. For real deployments, review the residual
risks above before exposing it to untrusted users or networks.
