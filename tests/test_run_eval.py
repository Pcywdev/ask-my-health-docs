"""The runner's per-question step and its database round trip, with the pipeline faked."""

import pytest
from sqlalchemy.orm import sessionmaker

from app import generator, retriever
from app.db import (
    EvalResult,
    EvalRun,
    add_eval_results,
    create_eval_run,
    finish_eval_run,
)
from eval.run_eval import run_question, to_db_result

QUESTION = {
    "id": "q001",
    "question": "How much activity should children do?",
    "category": "numeric",
    "answerable": True,
    "answer": "60 minutes a day.",
    "gold": [{"source": "who.pdf", "pages": [11]}],
}


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch, make_chunk):
    chunks = [
        make_chunk("who.pdf::p3::c0", page=3, source="who.pdf", rerank_score=0.9),
        make_chunk("who.pdf::p11::c0", page=11, source="who.pdf", rerank_score=0.8),
    ]
    answer = {
        "answer": "At least 60 minutes daily [Source 2].",
        "sources": [{"label": 2, **chunks[1]}],
        "citation_report": {
            "total_citations": 1, "valid_citations": 1, "fabricated_citations": 0,
            "citation_accuracy": 1.0, "has_citation": True,
        },
    }
    monkeypatch.setattr(retriever, "retrieve", lambda question, k: chunks)
    monkeypatch.setattr(generator, "generate_answer", lambda question, c: answer)
    return chunks


def test_run_question_scores_against_gold(fake_pipeline) -> None:
    row = run_question(QUESTION, top_k=2)
    assert row["error"] is None
    assert row["hit"] is True
    assert row["mrr"] == 0.5
    assert row["cited_gold"] is True
    assert row["reranked"] is True
    assert row["refused"] is False
    assert row["retrieved_pages"] == ["who.pdf p.3", "who.pdf p.11"]


def test_run_question_records_errors_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(question, k):
        raise RuntimeError("index missing")

    monkeypatch.setattr(retriever, "retrieve", boom)
    row = run_question(QUESTION, top_k=2)
    assert row["error"] == "RuntimeError: index missing"


def test_eval_run_round_trip(sqlite_db: sessionmaker, fake_pipeline) -> None:
    row = run_question(QUESTION, top_k=2)
    row["ragas"] = {"faithfulness": 0.9}

    run_id = create_eval_run("pilot", "abc123", {"top_k": 2}, 1)
    add_eval_results(run_id, [to_db_result(row)])
    finish_eval_run(run_id, {"hit@k": 1.0})

    with sqlite_db() as session:
        run = session.get(EvalRun, run_id)
        result = session.query(EvalResult).filter_by(run_id=run_id).one()
    assert run.summary == {"hit@k": 1.0}
    assert run.label == "pilot"
    assert result.question_id == "q001"
    assert result.gold_hit is True
    assert result.retrieved_ids == ["who.pdf::p3::c0", "who.pdf::p11::c0"]
    assert result.metrics["faithfulness"] == 0.9
    assert result.metrics["mrr"] == 0.5


def test_finish_unknown_run_raises(sqlite_db: sessionmaker) -> None:
    with pytest.raises(ValueError):
        finish_eval_run(999, {})
