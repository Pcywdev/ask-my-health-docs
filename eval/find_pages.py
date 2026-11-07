"""Find which indexed pages contain a phrase. Reads Chroma only: no API calls, no cost.

Usage:
    python eval/find_pages.py "150 minutes"
    python eval/find_pages.py "balance" --source Physical_Activity_Guidelines_2nd_edition.pdf

Prints ``source p.N`` with a snippet for each matching chunk. The page is the
PDF viewer's page number, which is what gold labels must use. No match is
evidence (not proof) that a question is unanswerable; try a few phrasings.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingest import chroma_client

SNIPPET_CHARS = 90


def find(phrase: str, source: str | None = None) -> list[tuple[str, int, str, str]]:
    """Return ``(source, page, chunk_id, snippet)`` for chunks containing ``phrase``.

    Matching is case-insensitive and ignores whitespace entirely, because PDF
    text extraction inserts stray spaces ("150– 300", "balance tr aining").
    """
    chars = [re.escape(ch) for ch in phrase if not ch.isspace()]
    pattern = re.compile(r"\s*".join(chars), re.IGNORECASE)
    collection = chroma_client.get_collection(settings.collection_name)
    data = collection.get(include=["documents", "metadatas"])

    matches = []
    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if source and meta["source"] != source:
            continue
        found = pattern.search(text)
        if found:
            start = max(found.start() - SNIPPET_CHARS, 0)
            snippet = " ".join(text[start : found.end() + SNIPPET_CHARS].split())
            matches.append((meta["source"], meta["page"], chunk_id, snippet))
    return sorted(matches, key=lambda m: (m[0], m[1], m[2]))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phrase", help="text to look for (case-insensitive)")
    parser.add_argument("--source", help="only search this PDF file name")
    args = parser.parse_args()

    matches = find(args.phrase, args.source)
    if not matches:
        print(f"No chunk contains {args.phrase!r}.")
        return
    pages = sorted({(m[0], m[1]) for m in matches})
    print(f"{len(matches)} chunks on {len(pages)} pages:\n")
    for source, page, chunk_id, snippet in matches:
        print(f"{source} p.{page}  ({chunk_id})")
        print(f"    ...{snippet}...\n")


if __name__ == "__main__":
    main()
