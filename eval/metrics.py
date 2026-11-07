"""Custom eval metrics: free, deterministic, computed from page-level gold labels.

A "gold chunk" is a retrieved chunk whose ``(source, page)`` is listed in the
question's ``gold``. Chunk dicts are what ``retrieve()`` returns and source
dicts are ``generate_answer()["sources"]``; both carry ``source`` and ``page``.
"""

from statistics import mean, quantiles
from typing import Any

# The prompt's exact refusal phrase (prompts/rag_prompt.yaml)
REFUSAL_PHRASE = "i don't have enough information"

Page = tuple[str, int]


def gold_pages(question: dict[str, Any]) -> set[Page]:
    """Flatten a question's ``gold`` list into a set of ``(source, page)`` pairs."""
    return {(g["source"], page) for g in question["gold"] for page in g["pages"]}


def _pages(chunks: list[dict[str, Any]]) -> list[Page]:
    return [(c["source"], c["page"]) for c in chunks]


def hit_at_k(chunks: list[dict[str, Any]], gold: set[Page]) -> bool:
    """True if any retrieved chunk lies on a gold page."""
    return any(page in gold for page in _pages(chunks))


def mrr(chunks: list[dict[str, Any]], gold: set[Page]) -> float:
    """Reciprocal rank of the first gold chunk (1.0 = top), 0.0 if none."""
    for rank, page in enumerate(_pages(chunks), start=1):
        if page in gold:
            return 1 / rank
    return 0.0


def page_recall(chunks: list[dict[str, Any]], gold: set[Page]) -> float:
    """Share of gold pages that at least one retrieved chunk lies on."""
    if not gold:
        return 0.0
    return len(gold & set(_pages(chunks))) / len(gold)


def cited_gold(sources: list[dict[str, Any]], gold: set[Page]) -> bool:
    """True if the answer cites at least one chunk on a gold page."""
    return hit_at_k(sources, gold)


def is_refusal(answer: str) -> bool:
    """True if the answer uses the prompt's refusal phrase (curly or straight apostrophe)."""
    return REFUSAL_PHRASE in answer.lower().replace("’", "'")


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    # 'inclusive' keeps the result within the observed range
    return quantiles(values, n=100, method="inclusive")[pct - 1]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question rows into the run summary.

    Each row has ``answerable``, ``error``, ``reranked``, ``refused``,
    ``has_citation``, ``citation_accuracy``, ``latency_ms`` and, for answerable
    questions, ``hit``, ``mrr``, ``page_recall``, ``cited_gold`` plus a
    ``ragas`` dict of scores (``None`` = not scored). Errored rows are excluded
    from every mean and only counted. Means are ``None`` when nothing qualifies.
    """
    ok = [r for r in rows if not r["error"]]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    cited = [r for r in ok if r["has_citation"]]

    summary: dict[str, Any] = {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "rerank_fallbacks": sum(1 for r in ok if not r["reranked"]),
        "hit@k": _mean([float(r["hit"]) for r in answerable]),
        "mrr": _mean([r["mrr"] for r in answerable]),
        "page_recall@k": _mean([r["page_recall"] for r in answerable]),
        "cited_gold": _mean([float(r["cited_gold"]) for r in answerable]),
        # Refusals score 0.0 by design, so only answers that cited something count
        "citation_accuracy": _mean([r["citation_accuracy"] for r in cited]),
        "false_refusal_rate": _mean([float(r["refused"]) for r in answerable]),
        "correct_refusal_rate": _mean(
            [float(r["refused"] and not r["has_citation"]) for r in unanswerable]
        ),
        "latency_p50_ms": _percentile([r["latency_ms"] for r in ok], 50),
        "latency_p95_ms": _percentile([r["latency_ms"] for r in ok], 95),
    }

    names = sorted({name for r in answerable for name in r.get("ragas", {})})
    for name in names:
        scores = [r["ragas"][name] for r in answerable if r.get("ragas", {}).get(name) is not None]
        summary[name] = _mean(scores)
    return summary


def check_thresholds(
    summary: dict[str, Any], thresholds: dict[str, float | None]
) -> tuple[list[str], list[str]]:
    """Compare the summary to thresholds; return ``(failures, skipped)``.

    A threshold of ``None`` is not enforced. A key ending in ``_max`` is an upper
    bound on the metric without the suffix (``false_refusal_rate_max``); every
    other key is a lower bound. A metric that was not measured in this run
    (e.g. RAGAS with ``--no-ragas``) is reported as skipped, not failed.
    """
    failures: list[str] = []
    skipped: list[str] = []
    for key, limit in thresholds.items():
        if limit is None:
            continue
        is_max = key.endswith("_max")
        metric = key.removesuffix("_max") if is_max else key
        value = summary.get(metric)
        if value is None:
            skipped.append(metric)
        elif is_max and value > limit:
            failures.append(f"{metric} = {value:.3f} > max {limit}")
        elif not is_max and value < limit:
            failures.append(f"{metric} = {value:.3f} < min {limit}")
    return failures, skipped
