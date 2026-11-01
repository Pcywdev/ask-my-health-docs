"""Generation: build a cited context from retrieved chunks and ask the LLM."""
import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from app.config import settings
from app.ingest import openai_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CITATION_PATTERN = re.compile(r"\[Source[^\]]*\]")


@cache
def load_prompt() -> dict[str, str]:
    """Load the versioned prompt (``system`` and ``template``) from YAML."""
    prompt_file = PROJECT_ROOT / settings.prompt_path
    return yaml.safe_load(prompt_file.read_text(encoding="utf-8"))


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Joins just the chunks into a context string, each tagged ``[Source N] file, p.X``.

    ``N`` is the chunk's 1-based position in ``chunks``, which is what lets a
    ``[Source N]`` citation in the answer be mapped back to a chunk.
    """
    blocks = [
        f"[Source {n}] {chunk['source']}, p.{chunk['page']}\n{chunk['text']}"
        for n, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def _cited_numbers(answer: str) -> set[int]:
    """Return the distinct source numbers the answer cites.

    Shared by ``extract_cited_sources`` and ``validate_citation`` so both parse
    citations identically.
    """
    cited: set[int] = set()
    # findall on the answer returns every full citation as a string -> ["[Source 1, 3]", "[Source 2]", "[Source 9]"]
    for citation in CITATION_PATTERN.findall(answer):
        # For each one, a second regex, \d+ pulls out every run of digits
        # "[Source 1, 3]" becomes ["1", "3"]
        # A set removes duplicates, so citing Source 1 three times still yields one entry.
        cited.update(int(number) for number in re.findall(r"\d+", citation))
    return cited


def extract_cited_sources(
    answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the chunks the answer cites, labelled with their ``Source N`` number.

    Handles ``[Source 1]``, ``[Source 1, 2]`` and ``[Source 1][Source 3]``.
    Numbers that do not match a chunk are ignored.
    """
    return [
        {
            "label": n,
            "id": chunks[n - 1]["id"],
            "source": chunks[n - 1]["source"],
            "page": chunks[n - 1]["page"],
            "text": chunks[n - 1]["text"],
        }
        for n in sorted(_cited_numbers(answer)) if 1 <= n <= len(chunks)
        # safety guard. Drops numbers that don't match a chunk
    ]


# Checks the model's [Source N] tags against the chunks it was actually given
def validate_citation(answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Report how many of the answer's citations point at a real chunk.

    A citation is valid when its number falls in ``1..len(chunks)``; anything
    else was fabricated by the model. Counts distinct source numbers, so citing
    ``[Source 1]`` twice counts once. ``citation_accuracy`` is 0.0 when nothing
    was cited (undefined rather than perfect), so use ``has_citation`` to tell a
    refusal apart from a badly cited answer.
    """
    cited_numbers = _cited_numbers(answer)
    valid = {n for n in cited_numbers if 1 <= n <= len(chunks)}
    return {
        "total_citations": len(cited_numbers),
        "valid_citations": len(valid),
        "fabricated_citations": len(cited_numbers) - len(valid),
        "citation_accuracy": len(valid) / len(cited_numbers) if cited_numbers else 0.0,
        "has_citation": len(cited_numbers) > 0,
    }

# Generates an answer to a query -> cites chunks that are actually relevant
def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer ``query`` from ``chunks`` and return the answer plus cited sources.

    Returns ``{"answer": str, "sources": list[dict], "citation_report": dict}``.
    ``sources`` holds only the chunks the model cited, and is empty if it cited
    none (for example when it says the sources do not contain the answer).
    ``citation_report`` is ``validate_citation``'s verdict on those citations.
    """
    prompt = load_prompt()
    # Building the user message
    user_message = prompt["template"].format(context=build_context(chunks), question=query)
    response = openai_client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_message},
        ],
    )
    # answer: Ordinary prose with [Source N] tags inline
    answer = response.choices[0].message.content or ""
    return {
        "answer": answer,
        "sources": extract_cited_sources(answer, chunks),
        "citation_report": validate_citation(answer, chunks),
    }

"""
if __name__ == "__main__":
    from app.retriever import retrieve
    test_query = "How much aerobic activity do adults need each week?"
    result = generate_answer(test_query, retrieve(test_query, k=5))
    print(f"Question: {test_query}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for source in result["sources"]:
        print(f"  [Source {source['label']}] {source['source']}, p.{source['page']}")
"""