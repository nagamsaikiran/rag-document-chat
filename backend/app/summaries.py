"""Per-session document summaries, persisted as JSON next to the vector index.

Whole-document questions ("what is this file about?") are answered from a
summary generated once at upload time, not from top-k retrieval — so nothing
important is missed. Each session's summaries live in a small JSON file under
the Chroma directory, so they share the same persistence/lifecycle as the
vectors (created on upload, dropped on clear/delete, reset if the disk resets).
"""
import json
import logging
import os
import re
from threading import Lock

from app.config import get_settings

logger = logging.getLogger("docchat.summaries")

_lock = Lock()
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _dir() -> str:
    d = os.path.join(get_settings().chroma_dir, "summaries")
    os.makedirs(d, exist_ok=True)
    return d


def _path(session_id: str) -> str:
    # Session ids are validated at the API boundary; guard again for path safety
    # so a bad id can never escape the summaries directory.
    safe = session_id if _SID_RE.match(session_id or "") else "public"
    return os.path.join(_dir(), f"{safe}.json")


def _load(session_id: str) -> dict:
    try:
        with open(_path(session_id), encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    except Exception:
        logger.exception("failed to read summaries")
        return {}


def _save(session_id: str, data: dict) -> None:
    try:
        with open(_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        logger.exception("failed to write summaries")


def set_summary(session_id: str, source: str, summary: str) -> None:
    """Store (or replace) the summary for one source document in a session."""
    if not summary:
        return
    with _lock:
        data = _load(session_id)
        data[source] = summary
        _save(session_id, data)


def get_summaries(session_id: str) -> dict:
    """Return {source_filename: summary_text} for the session (may be empty)."""
    with _lock:
        return _load(session_id)


def delete_source(session_id: str, source: str) -> None:
    with _lock:
        data = _load(session_id)
        if source in data:
            data.pop(source, None)
            _save(session_id, data)


def clear(session_id: str) -> None:
    with _lock:
        try:
            os.remove(_path(session_id))
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("failed to clear summaries")
