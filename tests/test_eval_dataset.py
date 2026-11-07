import copy

import pytest

from eval.dataset import EVAL_SET_PATH, load_json, reference_of, validate_eval_set

PAGES = {"who.pdf": {10, 11, 12}, "pag.pdf": {56}}


def answerable(qid: str, **overrides) -> dict:
    return {
        "id": qid,
        "question": "How much activity?",
        "category": "numeric",
        "answerable": True,
        "answer": "150 minutes.",
        "gold": [{"source": "who.pdf", "pages": [11]}],
        **overrides,
    }


def unanswerable(qid: str) -> dict:
    return {
        "id": qid, "question": "Which smartwatch?", "category": "unanswerable",
        "answerable": False, "answer": None, "gold": [],
    }


@pytest.fixture
def valid_set() -> dict:
    # 4 answerable + 1 unanswerable = 20%
    return {
        "version": "1.0",
        "questions": [answerable(f"q{i}") for i in range(4)] + [unanswerable("q9")],
    }


def test_valid_set_passes(valid_set: dict) -> None:
    assert validate_eval_set(valid_set, PAGES) == ([], [])


def test_reference_accepts_answer_or_reference() -> None:
    assert reference_of({"answer": "a"}) == "a"
    assert reference_of({"reference": "r", "answer": "a"}) == "r"
    assert reference_of({"answer": None}) is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda q: q.update(gold=[{"source": "who.pdf", "pages": [99]}]), "p.99 is not in the index"),
        (lambda q: q.update(gold=[{"source": "nope.pdf", "pages": [1]}]), "unknown source"),
        (lambda q: q.update(gold=[]), "need at least one gold page"),
        (lambda q: q.update(answer=""), "need a reference"),
        (lambda q: q.update(answerable="yes"), "'answerable' must be true or false"),
        (lambda q: q.update(gold=[{"source": "who.pdf", "pages": ["11"]}]), "must be an integer"),
        (lambda q: q.update(question=" "), "'question' must be a non-empty string"),
    ],
)
def test_bad_answerable_question_is_an_error(valid_set: dict, change, message: str) -> None:
    change(valid_set["questions"][0])
    errors, _ = validate_eval_set(valid_set, PAGES)
    assert any(message in e for e in errors), errors


def test_unanswerable_with_gold_is_an_error(valid_set: dict) -> None:
    valid_set["questions"][-1]["gold"] = [{"source": "who.pdf", "pages": [11]}]
    errors, _ = validate_eval_set(valid_set, PAGES)
    assert any("must have empty 'gold'" in e for e in errors)


def test_duplicate_ids_are_an_error(valid_set: dict) -> None:
    valid_set["questions"][1]["id"] = "q0"
    errors, _ = validate_eval_set(valid_set, PAGES)
    assert any("duplicate id" in e for e in errors)


def test_unbalanced_set_is_a_warning_not_an_error(valid_set: dict) -> None:
    unbalanced = copy.deepcopy(valid_set)
    unbalanced["questions"][-1] = answerable("q9")
    errors, warnings = validate_eval_set(unbalanced, PAGES)
    assert errors == []
    assert "0/5 unanswerable" in warnings[0]


def test_real_eval_set_has_a_valid_shape() -> None:
    """The committed set parses and every field is well-formed (pages are checked live)."""
    data = load_json(EVAL_SET_PATH)
    pages = {
        g["source"]: set(range(1, 1000))
        for q in data["questions"] for g in q["gold"]
    }
    errors, _ = validate_eval_set(data, pages)
    assert errors == []
