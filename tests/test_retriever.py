import logging
from types import SimpleNamespace

import pytest
from cohere.core.api_error import ApiError

from app import retriever
from app.retriever import reciprocal_rank_fusion, retrieve

# --- reciprocal_rank_fusion --------------------------------------------------


def test_single_list_scores_by_rank(make_chunk) -> None:
    fused = reciprocal_rank_fusion([[make_chunk("a"), make_chunk("b")]])
    assert [c["id"] for c in fused] == ["a", "b"]
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61)
    assert fused[1]["rrf_score"] == pytest.approx(1 / 62)
    assert all(c["found_in"] == 1 for c in fused)


def test_chunk_in_both_lists_sums_and_ranks_first(make_chunk) -> None:
    vector = [make_chunk("a"), make_chunk("shared")]
    bm25 = [make_chunk("b"), make_chunk("shared")]
    fused = reciprocal_rank_fusion([vector, bm25])
    assert fused[0]["id"] == "shared"
    assert fused[0]["found_in"] == 2
    assert fused[0]["rrf_score"] == pytest.approx(2 / 62)


def test_ties_keep_first_seen_order(make_chunk) -> None:
    # Rank 1 in each list -> identical scores; "a" was seen first
    fused = reciprocal_rank_fusion([[make_chunk("a")], [make_chunk("b")]])
    assert [c["id"] for c in fused] == ["a", "b"]
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]


def test_method_specific_keys_are_dropped(make_chunk) -> None:
    fused = reciprocal_rank_fusion(
        [[make_chunk("a", distance=0.1)], [make_chunk("a", score=3.2, score_source="bm25")]]
    )
    assert set(fused[0]) == {"id", "text", "source", "page", "rrf_score", "found_in"}


def test_custom_k_changes_scores(make_chunk) -> None:
    fused = reciprocal_rank_fusion([[make_chunk("a")]], k=10)
    assert fused[0]["rrf_score"] == pytest.approx(1 / 11)


def test_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --- retrieve(): rerank and fallback -----------------------------------------


@pytest.fixture
def stub_searches(monkeypatch: pytest.MonkeyPatch, make_chunk) -> None:
    """Replace both searches with fixed results so no index is needed."""
    monkeypatch.setattr(
        retriever, "vector_search", lambda query, k: [make_chunk("a"), make_chunk("b")]
    )
    monkeypatch.setattr(
        retriever, "bm25_search", lambda query, k: [make_chunk("b"), make_chunk("c")]
    )


def test_retrieve_returns_reranked_chunks(
    monkeypatch: pytest.MonkeyPatch, stub_searches: None
) -> None:
    def fake_rerank(*, query, documents, model, top_n):
        assert top_n == 2
        # Cohere says the 2nd candidate is best
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=1, relevance_score=0.9),
                SimpleNamespace(index=0, relevance_score=0.4),
            ]
        )

    monkeypatch.setattr(retriever.cohere_client, "rerank", fake_rerank)
    chunks = retrieve("how much exercise", k=2)
    assert [c["rerank_score"] for c in chunks] == [0.9, 0.4]
    assert len(chunks) == 2
    assert "rrf_score" in chunks[0]  # earlier-stage keys survive reranking


def test_retrieve_falls_back_to_rrf_order_when_cohere_fails(
    monkeypatch: pytest.MonkeyPatch, stub_searches: None, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_rerank(**kwargs):
        raise ApiError(status_code=500, body="boom")

    monkeypatch.setattr(retriever.cohere_client, "rerank", failing_rerank)
    with caplog.at_level(logging.WARNING, logger="app.retriever"):
        chunks = retrieve("how much exercise", k=2)

    # "b" is in both lists, so RRF puts it first; no rerank_score on fallback
    assert [c["id"] for c in chunks] == ["b", "a"]
    assert all("rerank_score" not in c for c in chunks)
    assert "falling back to RRF order" in caplog.text


def test_retrieve_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        retrieve("   ")
    with pytest.raises(ValueError):
        retrieve("question", k=0)
