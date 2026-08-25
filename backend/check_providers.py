"""Health check for every configured LLM provider (and the embedder).

Run from the backend directory with your virtualenv active:

    .venv\\Scripts\\python.exe check_providers.py      (Windows)
    python check_providers.py                          (Mac/Linux)

It makes ONE tiny call to each provider in your LLM_PROVIDERS chain and prints
OK / FAIL with the model name, latency, and any error. This is the only way to
confirm the failover providers (Groq, Cerebras, Mistral) actually work — in
normal use you never see them unless the primary runs out.

Reading the results:
  [ OK ]   the key and model name are valid and answered.
  [FAIL]   see the message: 'model_not_found' = wrong/retired model name (fix
           the *_CHAT_MODEL in .env); '401/invalid api key' = bad key; '429/
           quota' = the key works but you're rate-limited right now (still fine).
  [SKIP]   no API key set for that provider.
"""
import sys
import time

from app.config import get_settings
from app.llm.factory import _OPENAI_COMPAT, _build_llm, get_embedder

PING_SYS = "You are a connectivity test. Reply with exactly one short word."
PING_USER = "Reply with the single word: ok"


def _chain(settings):
    names = [n.strip() for n in settings.llm_providers.split(",") if n.strip()]
    names = names or [settings.llm_provider]
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _key_for(name, settings):
    if name in _OPENAI_COMPAT:
        return getattr(settings, _OPENAI_COMPAT[name][1], "")
    return getattr(settings, f"{name}_api_key", "")


def list_models(settings):
    """`python check_providers.py models` — list the model IDs each key can use.

    Free catalogs change often; run this to find a current model name to put in
    the matching *_CHAT_MODEL in .env. (Groq/Cerebras/Mistral expose /models;
    Gemini's list works differently and is skipped here.)"""
    from openai import OpenAI

    print("\nModels available to your keys:")
    for name, (base_url, key_attr, _model_attr) in _OPENAI_COMPAT.items():
        key = getattr(settings, key_attr, "")
        if not key:
            print(f"\n{name}: (no API key set)")
            continue
        try:
            ids = sorted(m.id for m in OpenAI(api_key=key, base_url=base_url).models.list().data)
            print(f"\n{name} ({len(ids)}):")
            for i in ids:
                print(f"    {i}")
        except Exception as e:
            print(f"\n{name}: could not list — {type(e).__name__}: {str(e)[:110]}")
    print()


def main():
    settings = get_settings()
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        list_models(settings)
        return
    names = _chain(settings)
    print()
    print(f"Chat providers to check: {', '.join(names)}")
    print("-" * 66)

    working = 0
    for name in names:
        if not _key_for(name, settings):
            print(f"[SKIP] {name:10}  no API key set")
            continue
        try:
            llm = _build_llm(name, settings)
            model = getattr(llm, "model", getattr(llm, "model_name", "?"))
            t0 = time.perf_counter()
            reply = llm.complete(PING_SYS, PING_USER).strip().replace("\n", " ")
            dt = time.perf_counter() - t0
            print(f"[ OK ] {name:10}  {dt:4.1f}s  model={model}  reply={reply[:24]!r}")
            working += 1
        except Exception as e:
            print(f"[FAIL] {name:10}  {type(e).__name__}: {str(e)[:110]}")

    print("-" * 66)
    try:
        dim = len(get_embedder().embed_query("hello world"))
        print(f"[ OK ] embeddings  provider={settings.embedding_provider}  dim={dim}")
    except Exception as e:
        print(f"[FAIL] embeddings  {type(e).__name__}: {str(e)[:110]}")

    print("-" * 66)
    print(f"{working}/{len(names)} chat providers working.\n")


if __name__ == "__main__":
    main()
