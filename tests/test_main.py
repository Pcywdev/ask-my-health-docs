import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import main

ANSWER = {
    "answer": "Adults need 150 minutes weekly [Source 1].",
    "sources": [{"label": 1, "id": "a", "source": "doc.pdf", "page": 3, "text": "t"}],
    "citation_report": {
        "total_citations": 1,
        "valid_citations": 1,
        "fabricated_citations": 0,
        "citation_accuracy": 1.0,
        "has_citation": True,
    },
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose pipeline is faked, so no index or API is touched."""
    monkeypatch.setattr(main, "retrieve", lambda question, k: [{"id": "a"}])
    monkeypatch.setattr(main, "generate_answer", lambda question, chunks: ANSWER)
    monkeypatch.setattr(main, "load_prompt", lambda: {"version": "1.0"})
    return TestClient(main.app)


def test_query_survives_a_logging_failure(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database outage must not cost the user an answer that is already paid for."""

    def broken_log(*args, **kwargs):
        raise SQLAlchemyError("database down")

    monkeypatch.setattr(main, "log_query", broken_log)
    response = client.post("/query", json={"question": "How much exercise?"})
    assert response.status_code == 200
    assert response.json()["answer"] == ANSWER["answer"]
    assert response.json()["sources"][0]["index"] == 1


def test_query_logs_on_the_happy_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(main, "log_query", lambda *args: calls.append(args) or 1)
    response = client.post("/query", json={"question": "How much exercise?", "top_k": 3})
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][0] == "How much exercise?"
    assert calls[0][3] == 3  # top_k
    assert calls[0][5] == "1.0"  # prompt version


def test_query_rejects_an_empty_question(client: TestClient) -> None:
    assert client.post("/query", json={"question": "   "}).status_code == 422
