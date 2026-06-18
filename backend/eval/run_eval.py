"""Lightweight RAG evaluation harness.

Measures three things that hiring managers actually ask about:

  1. Retrieval hit-rate   -- did the retriever surface a chunk containing the
                             expected substring(s)? (retrieval quality)
  2. Answer correctness   -- does the final answer contain the required strings?
                             (end-to-end quality)
  3. Faithfulness (LLM-as-judge) -- is the answer supported by the retrieved
                             context, with no invented claims? Scored 1-5 by the
                             model itself. (hallucination check)

Negative tests (expected == "OUT_OF_SCOPE") verify the grounding guardrail:
the system should refuse rather than answer from parametric memory.

Usage:
    python -m eval.run_eval --questions eval/questions.json
"""
import argparse
import json
from statistics import mean

from app.llm.factory import get_llm
from app.rag import answer
from app.vectorstore import get_store

JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a QUESTION, an ANSWER, and the CONTEXT the "
    "answer was supposed to be grounded in, rate how faithful the answer is to the "
    "context on a 1-5 scale (5 = every claim is supported by the context, 1 = "
    "fabricated). Respond with ONLY the integer."
)


def judge_faithfulness(question: str, ans: str, citations: list[dict]) -> int:
    if not citations:
        return 5  # nothing claimed / correct refusal -> trivially faithful
    context = "\n\n".join(f"- {c['snippet']}" for c in citations)
    prompt = f"QUESTION: {question}\n\nANSWER: {ans}\n\nCONTEXT:\n{context}"
    raw = get_llm().complete(JUDGE_SYSTEM, prompt).strip()
    for ch in raw:
        if ch.isdigit():
            return int(ch)
    return 0


def run(path: str) -> None:
    with open(path) as f:
        cases = json.load(f)

    if get_store().count() == 0:
        raise SystemExit(
            "No documents indexed. Upload PDFs first (POST /upload) so retrieval "
            "has something to find, then re-run the eval."
        )

    rows = []
    for case in cases:
        result = answer(case["question"])
        ans = result["answer"]
        cites = result["citations"]
        out_of_scope = case.get("expected") == "OUT_OF_SCOPE"

        # Retrieval hit-rate (skip for negative tests).
        retrieved_text = " ".join(c["snippet"].lower() for c in cites)
        must = [s.lower() for s in case.get("must_include", [])]
        if out_of_scope:
            retrieval_hit = None
            correct = not result["grounded"]  # should have refused
        else:
            retrieval_hit = all(m in retrieved_text for m in must) if must else None
            correct = all(m in ans.lower() for m in must) if must else None

        faithfulness = judge_faithfulness(case["question"], ans, cites)

        rows.append(
            {
                "q": case["question"],
                "retrieval_hit": retrieval_hit,
                "correct": correct,
                "faithfulness": faithfulness,
                "grounded": result["grounded"],
            }
        )

    print("\n=== RAG Eval Results ===")
    for r in rows:
        print(
            f"- {r['q'][:60]:60s} | retrieval={r['retrieval_hit']} "
            f"correct={r['correct']} faithfulness={r['faithfulness']}/5 "
            f"grounded={r['grounded']}"
        )

    retr = [r["retrieval_hit"] for r in rows if r["retrieval_hit"] is not None]
    corr = [r["correct"] for r in rows if r["correct"] is not None]
    faith = [r["faithfulness"] for r in rows]
    print("\n--- Aggregate ---")
    if retr:
        print(f"Retrieval hit-rate : {mean(retr) * 100:.0f}%  ({sum(retr)}/{len(retr)})")
    if corr:
        print(f"Answer correctness : {mean(corr) * 100:.0f}%  ({sum(corr)}/{len(corr)})")
    print(f"Mean faithfulness  : {mean(faith):.2f}/5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    args = ap.parse_args()
    run(args.questions)
