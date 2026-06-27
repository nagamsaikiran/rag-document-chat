# Security

This document records the security review of DocChat RAG: what was checked, what
was fixed, and the residual risks with their rationale. It reflects a deliberate
threat-model for a self-hostable RAG demo, not a hardened multi-tenant SaaS.

## Dependency vulnerabilities — fixed

All Python dependencies were audited with `pip-audit`. The initial pinned set
carried 46 known CVEs, concentrated in the security-sensitive parsers:

- `pypdf` (parses untrusted PDFs) — 30+ CVEs → upgraded to **6.14.2**
- `python-multipart` (parses uploaded form data) — 7 CVEs → upgraded to **0.0.32**
- `starlette` (web core, via FastAPI) — multiple CVEs → upgraded to **starlette 1.3.1** (FastAPI **0.138.1**)
- `python-dotenv` — 1 CVE → upgraded to **1.2.2**

After upgrading, `pip-audit` reports **no known vulnerabilities**. Re-run with:

```bash
pip install pip-audit
pip-audit -r backend/requirements.txt
```

## Application hardening — fixed

| Area | Risk | Mitigation |
|---|---|---|
| **Secrets** | API key leakage | Keys live only in `backend/.env`, which is git-ignored. Only `.env.example` (placeholders) is committed. |
| **Upload size** | DoS via huge files | Uploads over `MAX_UPLOAD_MB` (default 25) are rejected. |
| **Page count** | Cost/DoS via huge PDFs (one vision call per page) | PDFs over `MAX_PAGES` (default 50) are rejected. |
| **File type** | Non-PDF payloads | Only `.pdf` accepted; parsing errors are caught, not fatal. |
| **Path traversal** | Malicious filenames | Uploaded bytes are written to a random `tempfile`; the original filename is never used as a filesystem path. |
| **CORS** | Any origin calling the API | Origins are configurable via `CORS_ORIGINS`; lock to your frontend origin in production. |
| **XSS** | Malicious text in a PDF rendered in the UI | The React frontend renders all content as text (no `dangerouslySetInnerHTML`), so document content cannot inject markup. |

## Residual risks (accepted, with rationale)

These are documented rather than fixed, because the fix would change the project's
nature (a single-user, self-hosted demo). Each notes how you'd address it for prod.

- **No authentication / rate limiting.** Anyone who can reach the API can upload and
  query. *Fine for local/single-user use.* For a public deployment, put it behind
  auth and a rate limiter (or a gateway), and set `CORS_ORIGINS`.
- **Prompt injection.** A crafted document could contain instructions ("ignore the
  above…"). The system prompt constrains the model to the provided context and to
  refuse when unsupported, which reduces but does not eliminate this. Untrusted
  documents should be treated as untrusted input.
- **Verbose error messages.** Upload/chat errors are surfaced to the client to aid
  debugging (e.g. "model not found", "key invalid"). Convenient for a demo, but in
  production you'd return generic messages to clients and keep details server-side.
- **At-rest data.** The Chroma index persists extracted document text unencrypted in
  `backend/.chroma`. Use disk encryption / per-user isolation for sensitive corpora.
- **Outbound calls to the LLM provider.** Document text is sent to OpenAI/Google for
  embedding and generation. Use a local provider (e.g. Ollama) for fully on-prem.

## Reporting

This is a personal/portfolio project. For real deployments, review the residual
risks above before exposing it to untrusted users or networks.
