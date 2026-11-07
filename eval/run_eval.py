"""Score the live pipeline against the eval set and store the run in Postgres.

Usage:
    python eval/run_eval.py --validate                 # check the set only (free)
    python eval/run_eval.py --limit 3 --no-ragas       # cheap smoke run
    python eval/run_eval.py --label baseline-1         # full run

Calls ``retrieve()`` and ``generate_answer()`` directly, never ``/query``, so
eval traffic stays out of ``query_logs`` and ``/history``. Every run spends
OpenAI and Cohere credits except ``--validate``.

Exit codes: 0 all thresholds met · 1 a threshold missed or the run is not
comparable (Cohere fallbacks, errors) · 2 the eval set is invalid.
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from eval.dataset import (
    EVAL_SET_PATH,
    THRESHOLDS_PATH,
    category_counts,
    index_pages,
    load_json,
    reference_of,
    validate_eval_set,
)
from eval.metrics import (
    check_thresholds,
    cited_gold,
    gold_pages,
    hit_at_k,
    is_refusal,
    mrr,
    page_recall,
    summarize,
)

# Questions scored by RAGAS at the same time (each fires ~4 judge calls)
RAGAS_CONCURRENCY = 4


def git_sha() -> str | None:
    """Current commit, suffixed ``+dirty`` when the working tree has changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}+dirty" if dirty else sha


def run_config(top_k: int, use_ragas: bool, set_version: str) -> dict[str, Any]:
    """Everything that can change a score, stored on ``eval_runs.config``."""
    from app.generator import load_prompt

    return {
        "eval_set_version": set_version,
        "top_k": top_k,
        "candidate_k": settings.candidate_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "rerank_model": settings.rerank_model,
        "prompt_version": load_prompt()["version"],
        "ragas": use_ragas,
        "judge_model": settings.eval_judge_model if use_ragas else None,
    }


def run_question(question: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Retrieve, generate and compute the custom metrics for one question."""
    from app.generator import generate_answer
    from app.retriever import retrieve

    row: dict[str, Any] = {
        "id": question["id"],
        "question": question["question"],
        "answerable": question["answerable"],
        "reference": reference_of(question),
        "error": None,
        "ragas": {},
    }
    started = time.perf_counter()
    try:
        chunks = retrieve(question["question"], top_k)
        result = generate_answer(question["question"], chunks)
    except Exception as error:  # noqa: BLE001 - one bad question must not lose the whole paid run
        row.update(error=f"{type(error).__name__}: {error}", latency_ms=None)
        return row

    report = result["citation_report"]
    gold = gold_pages(question)
    row.update(
        latency_ms=int((time.perf_counter() - started) * 1000),
        answer=result["answer"],
        chunks=chunks,
        retrieved_pages=[f"{c['source']} p.{c['page']}" for c in chunks],
        reranked=bool(chunks) and all("rerank_score" in c for c in chunks),
        refused=is_refusal(result["answer"]),
        has_citation=report["has_citation"],
        citation_accuracy=report["citation_accuracy"],
    )
    if question["answerable"]:
        row.update(
            hit=hit_at_k(chunks, gold),
            mrr=mrr(chunks, gold),
            page_recall=page_recall(chunks, gold),
            cited_gold=cited_gold(result["sources"], gold),
        )
    return row


def build_ragas_metrics() -> dict[str, Any]:
    """RAGAS 0.4 judge metrics, sharing one async OpenAI client."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.eval_judge_model, client=client)
    embeddings = OpenAIEmbeddings(client=client, model=settings.embedding_model)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def _score_row(row: dict[str, Any], metrics: dict[str, Any],
                     gate: asyncio.Semaphore) -> None:
    question, answer, reference = row["question"], row["answer"], row["reference"]
    contexts = [c["text"] for c in row["chunks"]]
    calls = {
        "faithfulness": lambda m: m.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts),
        "answer_relevancy": lambda m: m.ascore(user_input=question, response=answer),
        "context_precision": lambda m: m.ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts),
        "context_recall": lambda m: m.ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference),
    }
    async with gate:
        results = await asyncio.gather(
            *(calls[name](metric) for name, metric in metrics.items()),
            return_exceptions=True,
        )
    for name, result in zip(metrics, results):
        if isinstance(result, Exception):
            row["ragas"][name] = None
            row.setdefault("ragas_errors", {})[name] = f"{type(result).__name__}: {result}"
        else:
            row["ragas"][name] = float(result.value)


async def score_ragas(rows: list[dict[str, Any]]) -> None:
    """Add RAGAS scores to answerable, non-refused, non-errored rows (in place).

    Refusals are scored by the custom refusal metrics instead: judging the
    faithfulness of "I don't have enough information" says nothing.
    """
    metrics = build_ragas_metrics()
    gate = asyncio.Semaphore(RAGAS_CONCURRENCY)
    targets = [r for r in rows if r["answerable"] and not r["error"] and not r["refused"]]
    await asyncio.gather(*(_score_row(r, metrics, gate) for r in targets))


def to_db_result(row: dict[str, Any]) -> dict[str, Any]:
    """Map a scored row onto ``EvalResult`` columns; the rest goes in ``metrics``."""
    chunks = row.get("chunks", [])
    extra = {
        key: row[key]
        for key in ("mrr", "page_recall", "cited_gold", "refused", "has_citation",
                    "retrieved_pages", "ragas_errors")
        if key in row
    }
    return {
        "question_id": row["id"],
        "question": row["question"],
        "answer": row.get("answer"),
        "retrieved_ids": [c["id"] for c in chunks],
        "contexts": [c["text"] for c in chunks],
        "reranked": row.get("reranked"),
        "gold_hit": row.get("hit"),
        "citation_accuracy": row.get("citation_accuracy"),
        "metrics": {**row["ragas"], **extra},
        "latency_ms": row.get("latency_ms"),
        "error": row["error"],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_report(rows: list[dict[str, Any]], summary: dict[str, Any],
                 thresholds: dict[str, Any]) -> None:
    """Scorecard, then the questions worth reading (misses, wrong refusals, errors)."""
    print("\n=== Scorecard ===")
    for key, value in summary.items():
        limit = thresholds.get(key)
        upper = thresholds.get(f"{key}_max")
        bound = f"  (min {limit})" if limit is not None else (
            f"  (max {upper})" if upper is not None else "")
        print(f"  {key:<24}{_fmt(value):>10}{bound}")

    print("\n=== Review these ===")
    flagged = 0
    for r in rows:
        reasons = []
        if r["error"]:
            reasons.append(f"error: {r['error']}")
        elif r["answerable"]:
            if not r["hit"]:
                reasons.append("no gold page retrieved")
            if r["refused"]:
                reasons.append("refused an answerable question")
            elif not r["cited_gold"]:
                reasons.append("did not cite a gold page")
        elif not r["refused"]:
            reasons.append("answered an unanswerable question")
        if reasons:
            flagged += 1
            print(f"- {r['id']}: {'; '.join(reasons)}")
            print(f"    Q: {r['question']}")
            if r.get("retrieved_pages"):
                print(f"    retrieved: {', '.join(r['retrieved_pages'])}")
            if r.get("answer"):
                print(f"    A: {' '.join(r['answer'].split())[:240]}")
    if not flagged:
        print("  nothing flagged")


def validate_or_exit(data: dict[str, Any]) -> None:
    """Print validation results; exit 2 if the set is unusable."""
    errors, warnings = validate_eval_set(data, index_pages())
    counts = ", ".join(f"{k} {v}" for k, v in sorted(category_counts(data).items()))
    print(f"Eval set v{data.get('version')}: {len(data['questions'])} questions ({counts})")
    for warning in warnings:
        print(f"  warning: {warning}")
    for error in errors:
        print(f"  ERROR: {error}")
    if errors:
        sys.exit(2)
    print("  valid")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Score the RAG pipeline on the eval set.")
    parser.add_argument("--validate", action="store_true", help="check the set only; free")
    parser.add_argument("--limit", type=int, help="run only the first N questions")
    parser.add_argument("--label", help="name for this run, e.g. baseline-1")
    parser.add_argument("--no-ragas", action="store_true", help="skip the LLM judge metrics")
    parser.add_argument("--top-k", type=int, default=settings.eval_top_k)
    args = parser.parse_args()

    data = load_json(EVAL_SET_PATH)
    validate_or_exit(data)
    if args.validate:
        return

    from app.db import add_eval_results, create_eval_run, finish_eval_run

    questions = data["questions"][: args.limit] if args.limit else data["questions"]
    use_ragas = not args.no_ragas
    thresholds = load_json(THRESHOLDS_PATH) if THRESHOLDS_PATH.exists() else {}

    # Open the run first: if the database is down, fail before spending credits
    run_id = create_eval_run(
        args.label, git_sha(), run_config(args.top_k, use_ragas, data["version"]),
        len(questions),
    )
    print(f"Run {run_id}: {len(questions)} questions, top_k={args.top_k}, "
          f"ragas={'on' if use_ragas else 'off'}")

    rows = []
    for i, question in enumerate(questions, start=1):
        if i > 1 and settings.eval_rerank_delay_s > 0:
            time.sleep(settings.eval_rerank_delay_s)  # keep a trial Cohere key under its limit
        row = run_question(question, args.top_k)
        rows.append(row)
        status = "ERROR" if row["error"] else ("refused" if row["refused"] else "answered")
        print(f"  [{i}/{len(questions)}] {question['id']}: {status}")

    if use_ragas:
        print("Scoring with RAGAS...")
        asyncio.run(score_ragas(rows))

    summary = summarize(rows)
    add_eval_results(run_id, [to_db_result(r) for r in rows])
    finish_eval_run(run_id, summary)
    print_report(rows, summary, thresholds)

    failures, skipped = check_thresholds(summary, thresholds)
    problems = [f"threshold: {f}" for f in failures]
    if summary["rerank_fallbacks"]:
        problems.append(f"{summary['rerank_fallbacks']} questions used the RRF fallback "
                        "(Cohere failed): run is INVALID for comparison")
    if summary["errors"]:
        problems.append(f"{summary['errors']} questions errored")
    if skipped:
        print(f"\nNot measured this run, threshold skipped: {', '.join(skipped)}")
    if not any(v is not None for v in thresholds.values()):
        print("\nNo thresholds set yet (eval/thresholds.json): nothing enforced.")

    print(f"\nStored as eval run {run_id}.")
    if problems:
        print("FAIL")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
