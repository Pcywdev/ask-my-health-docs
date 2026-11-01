import pytest

from app.generator import build_context, extract_cited_sources, validate_citation


def chunks(n: int) -> list[dict]:
    return [
        {"id": f"c{i}", "text": f"text {i}", "source": "doc.pdf", "page": i}
        for i in range(1, n + 1)
    ]


def test_all_citations_valid() -> None:
    report = validate_citation("Walk daily [Source 1]. Lift twice [Source 2].", chunks(3))
    assert report == {
        "total_citations": 2,
        "valid_citations": 2,
        "fabricated_citations": 0,
        "citation_accuracy": 1.0,
        "has_citation": True,
    }


def test_out_of_range_citation_is_fabricated() -> None:
    report = validate_citation("Claim [Source 1] and claim [Source 9].", chunks(3))
    assert report["total_citations"] == 2
    assert report["valid_citations"] == 1
    assert report["fabricated_citations"] == 1
    assert report["citation_accuracy"] == 0.5


def test_source_zero_is_fabricated() -> None:
    assert validate_citation("Claim [Source 0].", chunks(3))["fabricated_citations"] == 1


def test_repeated_citation_counts_once() -> None:
    report = validate_citation("A [Source 1]. B [Source 1]. C [Source 1].", chunks(3))
    assert report["total_citations"] == 1


@pytest.mark.parametrize(
    "answer",
    ["Both [Source 1, 3].", "Both [Source 1][Source 3]."],
)
def test_multi_source_formats_parse(answer: str) -> None:
    assert validate_citation(answer, chunks(3))["valid_citations"] == 2


def test_no_citations_scores_zero_not_perfect() -> None:
    report = validate_citation("The sources do not contain the answer.", chunks(3))
    assert report["citation_accuracy"] == 0.0
    assert report["has_citation"] is False
    assert report["total_citations"] == 0


def test_any_citation_with_no_chunks_is_fabricated() -> None:
    report = validate_citation("Claim [Source 1].", [])
    assert report["valid_citations"] == 0
    assert report["fabricated_citations"] == 1


def test_extract_cited_sources_labels_and_order() -> None:
    sources = extract_cited_sources("B [Source 3] then A [Source 1].", chunks(3))
    assert [s["label"] for s in sources] == [1, 3]
    assert [s["id"] for s in sources] == ["c1", "c3"]
    assert sources[0]["page"] == 1


def test_extract_cited_sources_ignores_out_of_range() -> None:
    sources = extract_cited_sources("Real [Source 2], fake [Source 7].", chunks(3))
    assert [s["label"] for s in sources] == [2]


def test_build_context_numbers_chunks_from_one() -> None:
    context = build_context(chunks(2))
    assert context.startswith("[Source 1] doc.pdf, p.1\ntext 1")
    assert "\n\n[Source 2] doc.pdf, p.2\ntext 2" in context


@pytest.mark.xfail(
    strict=True,
    reason="Known bug (ROADMAP open issues): '[Source 1-3]' parses as {1, 3}, not {1, 2, 3}",
)
def test_source_range_expands() -> None:
    report = validate_citation("Claim [Source 1-3].", chunks(3))
    assert report["total_citations"] == 3
