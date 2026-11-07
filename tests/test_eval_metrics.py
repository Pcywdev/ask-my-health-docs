import pytest

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

GOLD = {("who.pdf", 11), ("pag.pdf", 56)}


def chunk(source: str, page: int) -> dict:
    return {"id": f"{source}::p{page}", "source": source, "page": page, "text": ""}


def test_gold_pages_flattens_every_source() -> None:
    question = {"gold": [{"source": "who.pdf", "pages": [11, 12]}, {"source": "pag.pdf", "pages": [56]}]}
    assert gold_pages(question) == {("who.pdf", 11), ("who.pdf", 12), ("pag.pdf", 56)}


def test_hit_and_mrr_use_the_first_gold_chunk() -> None:
    chunks = [chunk("who.pdf", 3), chunk("pag.pdf", 56), chunk("who.pdf", 11)]
    assert hit_at_k(chunks, GOLD) is True
    assert mrr(chunks, GOLD) == pytest.approx(1 / 2)


def test_same_page_wrong_source_is_a_miss() -> None:
    chunks = [chunk("pag.pdf", 11)]
    assert hit_at_k(chunks, GOLD) is False
    assert mrr(chunks, GOLD) == 0.0


def test_page_recall_counts_distinct_gold_pages() -> None:
    chunks = [chunk("who.pdf", 11), chunk("who.pdf", 11), chunk("who.pdf", 4)]
    assert page_recall(chunks, GOLD) == 0.5
    assert page_recall(chunks, set()) == 0.0


def test_cited_gold_checks_cited_sources_only() -> None:
    assert cited_gold([chunk("pag.pdf", 56)], GOLD) is True
    assert cited_gold([chunk("pag.pdf", 57)], GOLD) is False
    assert cited_gold([], GOLD) is False


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have enough information to answer that.",
        "i don’t have enough information to answer that.",  # curly apostrophe
    ],
)
def test_refusal_detection(answer: str) -> None:
    assert is_refusal(answer)


def test_normal_answer_is_not_a_refusal() -> None:
    assert not is_refusal("Adults need 150 minutes a week [Source 1].")


def row(answerable: bool = True, **overrides) -> dict:
    base = {
        "answerable": answerable,
        "error": None,
        "reranked": True,
        "refused": False,
        "has_citation": True,
        "citation_accuracy": 1.0,
        "latency_ms": 1000,
        "ragas": {},
    }
    if answerable:
        base.update(hit=True, mrr=1.0, page_recall=1.0, cited_gold=True)
    return {**base, **overrides}


def test_summarize_splits_answerable_and_unanswerable() -> None:
    rows = [
        row(mrr=1.0, ragas={"faithfulness": 0.8}),
        row(hit=False, mrr=0.0, page_recall=0.0, cited_gold=False, ragas={"faithfulness": 0.6}),
        row(refused=True, has_citation=False, citation_accuracy=0.0, cited_gold=False,
            ragas={"faithfulness": None}),
        row(answerable=False, refused=True, has_citation=False, citation_accuracy=0.0),
        row(answerable=False, refused=False, has_citation=True, citation_accuracy=1.0),
    ]
    s = summarize(rows)
    assert (s["answerable"], s["unanswerable"]) == (3, 2)
    assert s["hit@k"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx(2 / 3)
    assert s["false_refusal_rate"] == pytest.approx(1 / 3)
    assert s["correct_refusal_rate"] == pytest.approx(0.5)
    # refusals (accuracy 0.0) are left out of citation accuracy
    assert s["citation_accuracy"] == 1.0
    # None = not scored, so it is not averaged in as zero
    assert s["faithfulness"] == pytest.approx(0.7)


def test_summarize_excludes_errors_and_counts_fallbacks() -> None:
    rows = [
        row(),
        row(reranked=False, hit=False, mrr=0.0),
        {"answerable": True, "error": "ApiError: boom", "ragas": {}},
    ]
    s = summarize(rows)
    assert s["errors"] == 1
    assert s["answerable"] == 2
    assert s["rerank_fallbacks"] == 1
    assert s["hit@k"] == 0.5


def test_summarize_latency_percentiles() -> None:
    rows = [row(latency_ms=ms) for ms in (100, 200, 300, 400, 500)]
    s = summarize(rows)
    assert s["latency_p50_ms"] == pytest.approx(300)
    assert 400 <= s["latency_p95_ms"] <= 500


def test_thresholds_min_max_and_skipped() -> None:
    summary = {"hit@k": 0.7, "false_refusal_rate": 0.2, "mrr": 0.9, "faithfulness": None}
    failures, skipped = check_thresholds(
        summary,
        {"hit@k": 0.8, "false_refusal_rate_max": 0.1, "mrr": 0.5,
         "faithfulness": 0.9, "context_recall": None},
    )
    assert failures == ["hit@k = 0.700 < min 0.8", "false_refusal_rate = 0.200 > max 0.1"]
    assert skipped == ["faithfulness"]


def test_null_thresholds_enforce_nothing() -> None:
    assert check_thresholds({"hit@k": 0.0}, {"hit@k": None}) == ([], [])
