"""Load and validate the eval set (eval/eval_set.json)."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.json"

# Target share of unanswerable questions, with some slack for small sets
UNANSWERABLE_SHARE = (0.15, 0.25)


def load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def reference_of(question: dict[str, Any]) -> str | None:
    """The question's reference answer; the set may call it ``answer`` or ``reference``."""
    return question.get("reference") or question.get("answer")


def index_pages() -> dict[str, set[int]]:
    """Map each indexed source file to the set of its pages in Chroma (no API calls)."""
    from app.config import settings
    from app.ingest import chroma_client

    collection = chroma_client.get_collection(settings.collection_name)
    pages: dict[str, set[int]] = {}
    for meta in collection.get(include=["metadatas"])["metadatas"]:
        pages.setdefault(meta["source"], set()).add(meta["page"])
    return pages


def _check_question(
    q: dict[str, Any], pages: dict[str, set[int]], errors: list[str]
) -> None:
    qid = q.get("id", "<no id>")
    if not isinstance(q.get("question"), str) or not q["question"].strip():
        errors.append(f"{qid}: 'question' must be a non-empty string")
    if not isinstance(q.get("category"), str) or not q["category"].strip():
        errors.append(f"{qid}: 'category' must be a non-empty string")
    if not isinstance(q.get("answerable"), bool):
        errors.append(f"{qid}: 'answerable' must be true or false")
        return
    gold = q.get("gold")
    if not isinstance(gold, list):
        errors.append(f"{qid}: 'gold' must be a list")
        return

    if not q["answerable"]:
        if gold:
            errors.append(f"{qid}: unanswerable questions must have empty 'gold'")
        return

    if not gold:
        errors.append(f"{qid}: answerable questions need at least one gold page")
    reference = reference_of(q)
    if not isinstance(reference, str) or not reference.strip():
        errors.append(f"{qid}: answerable questions need a reference 'answer'")

    for entry in gold:
        source = entry.get("source")
        if source not in pages:
            errors.append(f"{qid}: unknown source {source!r} (not in the index)")
            continue
        entry_pages = entry.get("pages")
        if not isinstance(entry_pages, list) or not entry_pages:
            errors.append(f"{qid}: 'pages' for {source} must be a non-empty list")
            continue
        for page in entry_pages:
            if not isinstance(page, int) or isinstance(page, bool):
                errors.append(f"{qid}: page {page!r} must be an integer")
            elif page not in pages[source]:
                errors.append(
                    f"{qid}: {source} p.{page} is not in the index "
                    "(use the PDF viewer's page number, check with eval/find_pages.py)"
                )


def validate_eval_set(
    data: dict[str, Any], pages: dict[str, set[int]]
) -> tuple[list[str], list[str]]:
    """Check an eval set against the indexed pages; return ``(errors, warnings)``.

    Errors make the set unusable (bad schema, duplicate ids, gold pages that do
    not exist in the index). Warnings flag balance issues, such as the share of
    unanswerable questions drifting from ~20%.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data.get("version"), str):
        errors.append("top level: 'version' must be a string")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("top level: 'questions' must be a non-empty list")
        return errors, warnings

    ids = Counter(q.get("id") for q in questions)
    for qid, count in ids.items():
        if not isinstance(qid, str) or not qid:
            errors.append("every question needs a string 'id'")
        elif count > 1:
            errors.append(f"{qid}: duplicate id ({count} times)")

    for q in questions:
        _check_question(q, pages, errors)

    unanswerable = sum(1 for q in questions if q.get("answerable") is False)
    share = unanswerable / len(questions)
    low, high = UNANSWERABLE_SHARE
    if not low <= share <= high:
        warnings.append(
            f"{unanswerable}/{len(questions)} unanswerable ({share:.0%}); target is ~20%"
        )
    return errors, warnings


def category_counts(data: dict[str, Any]) -> Counter:
    """Count questions per category."""
    return Counter(q.get("category") for q in data["questions"])
