from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import (
    QueryLog,
    citation_stats,
    fabricated_queries,
    log_query,
    recent_queries,
)


def add_row(
    factory: sessionmaker,
    *,
    total: int,
    valid: int,
    num_sources: int,
    age_days: float = 0,
    question: str = "q",
) -> None:
    """Insert one log row; accuracy follows the app's rule (0.0 when nothing cited)."""
    with factory() as session:
        session.add(
            QueryLog(
                question=question,
                answer="a",
                num_sources=num_sources,
                total_citations=total,
                valid_citations=valid,
                citation_accuracy=valid / total if total else 0.0,
                has_citations=total > 0,
                created_at=datetime.now(UTC) - timedelta(days=age_days),
            )
        )
        session.commit()


@pytest.fixture
def seeded(sqlite_db: sessionmaker) -> sessionmaker:
    """Clean, fabricated, refusal, clean (this week) plus one clean row 30 days old."""
    add_row(sqlite_db, total=2, valid=2, num_sources=5, age_days=0.4, question="clean-1")
    add_row(sqlite_db, total=2, valid=1, num_sources=5, age_days=0.3, question="fabricated")
    add_row(sqlite_db, total=0, valid=0, num_sources=4, age_days=0.2, question="refusal")
    add_row(sqlite_db, total=1, valid=1, num_sources=4, age_days=0.1, question="clean-2")
    add_row(sqlite_db, total=1, valid=1, num_sources=5, age_days=30, question="old")
    return sqlite_db


# --- citation_stats ----------------------------------------------------------


def test_stats_over_all_time(seeded: sessionmaker) -> None:
    stats = citation_stats()
    assert stats["total_queries"] == 5
    assert stats["answers_with_citations"] == 4
    assert stats["refusals"] == 1
    assert stats["refusal_rate"] == pytest.approx(0.2)
    assert stats["queries_with_fabrication"] == 1
    assert stats["total_fabricated_citations"] == 1
    assert stats["avg_sources_retrieved"] == pytest.approx(4.6)
    assert stats["window_days"] is None


def test_refusals_drag_down_only_the_all_rows_mean(seeded: sessionmaker) -> None:
    stats = citation_stats()
    # (1.0 + 0.5 + 1.0 + 1.0) / 4 answers that cited something
    assert stats["mean_citation_accuracy_cited"] == pytest.approx(0.875)
    # (1.0 + 0.5 + 0.0 + 1.0 + 1.0) / 5 rows, the refusal counted as 0.0
    assert stats["mean_citation_accuracy_all"] == pytest.approx(0.7)


def test_stats_window_excludes_old_rows(seeded: sessionmaker) -> None:
    stats = citation_stats(days=7)
    assert stats["total_queries"] == 4
    assert stats["mean_citation_accuracy_cited"] == pytest.approx(2.5 / 3)
    assert stats["mean_citation_accuracy_all"] == pytest.approx(0.625)
    assert stats["refusal_rate"] == pytest.approx(0.25)
    assert stats["avg_sources_retrieved"] == pytest.approx(4.5)
    assert stats["window_days"] == 7


def test_stats_on_empty_table(sqlite_db: sessionmaker) -> None:
    stats = citation_stats()
    assert stats["total_queries"] == 0
    assert stats["mean_citation_accuracy_cited"] is None
    assert stats["mean_citation_accuracy_all"] is None
    assert stats["refusal_rate"] is None
    assert stats["avg_sources_retrieved"] is None
    assert stats["total_fabricated_citations"] == 0  # sum() over no rows is NULL


# --- readers -----------------------------------------------------------------


def test_fabricated_queries_returns_only_fabricating_rows(seeded: sessionmaker) -> None:
    assert [r.question for r in fabricated_queries()] == ["fabricated"]


def test_recent_queries_newest_first_and_limited(seeded: sessionmaker) -> None:
    assert [r.question for r in recent_queries(limit=3)] == ["clean-2", "refusal", "fabricated"]
    assert "old" not in [r.question for r in recent_queries(days=7)]


# --- log_query ---------------------------------------------------------------


def result_with(total: int, valid: int) -> dict:
    return {
        "answer": "An answer [Source 1].",
        "citation_report": {
            "total_citations": total,
            "valid_citations": valid,
            "fabricated_citations": total - valid,
            "citation_accuracy": valid / total if total else 0.0,
            "has_citation": total > 0,
        },
    }


def test_log_query_stores_trace_for_reranked_chunks(sqlite_db: sessionmaker) -> None:
    chunks = [
        {"id": "a", "rerank_score": 0.9},
        {"id": "b", "rerank_score": 0.4},
    ]
    row_id = log_query("question?", result_with(1, 1), chunks, 5, 120, "1.0")
    with sqlite_db() as session:
        row = session.get(QueryLog, row_id)
    assert row.retrieved_ids == ["a", "b"]
    assert row.rerank_scores == [0.9, 0.4]
    assert row.reranked is True
    assert row.num_sources == 2
    assert row.has_citations is True
    assert row.prompt_version == "1.0"
    assert row.latency_ms == 120


def test_log_query_marks_rrf_fallback_as_not_reranked(sqlite_db: sessionmaker) -> None:
    chunks = [{"id": "a"}, {"id": "b"}]
    row_id = log_query("question?", result_with(0, 0), chunks, 5, 80, "1.0")
    with sqlite_db() as session:
        row = session.get(QueryLog, row_id)
    assert row.reranked is False
    assert row.rerank_scores == [None, None]
    assert row.has_citations is False
