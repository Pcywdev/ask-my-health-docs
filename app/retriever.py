"""Retrieval: hybrid vector + BM25 search, merged by RRF and reranked by Cohere."""

import logging
import re
from functools import lru_cache
from typing import Any

import cohere
from cohere.core.api_error import ApiError
from rank_bm25 import BM25Okapi

from app.config import settings
from app.ingest import chroma_client, embed_texts

logger = logging.getLogger(__name__)

cohere_client = cohere.Client(api_key = settings.cohere_api_key)
# Regular Semantic Search
def vector_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return the ``k`` stored chunks most similar to ``query``.

    Context: Each chunk is a dict with ``id``, ``text``, ``source``, ``page`` and cosine
    ``distance`` (lower = more similar). Raises if
    the collection does not exist, i.e. ingestion has not been run.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    query_embedding = embed_texts([query])[0]
    collection = chroma_client.get_collection(settings.collection_name)
    # semantic vector search
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    # chunks: each member = the selected chunks
    chunks: list[dict[str, Any]] = []
    for chunk_id, text, metadata, distance in zip(results["ids"][0], results["documents"][0], 
    results["metadatas"][0], results["distances"][0]):
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance,
            }
        )
    return chunks


def _tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into word tokens.

    Used for both the corpus and the query; they must match or BM25 misses.
    """
    # Keeps runs of letters, digits, and underscores.
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1) # Keys the result on chunk_count. Later calls take about only 2ms. A new chunk count triggers a rebuild. Exists as only a cache key
def _load_bm25_index(
    chunk_count: int,
) -> tuple[BM25Okapi, list[str], list[str], list[dict[str, Any]]]:
    """Build a BM25 index over every stored chunk.

    ``chunk_count`` is the cache key: the index is built once and rebuilt
    whenever ingestion changes the number of chunks. (Re-ingesting changed text
    with an identical count would serve a stale index until restart.)
    """
    collection = chroma_client.get_collection(settings.collection_name)
    data = collection.get(include=["documents", "metadatas"])
    # Tokenize each chunk and build the index
    # Takes a list of token lists, one per chunk. Precomputes three things with no scoring yet.
    '''
    - Term counts for each chunk (how many time each word appears in it)
    - The document frequency of each word (how many chunks contain it)
    - The average chunk length, it also precomputes an IDF weight per worde
    '''
    # Each chunk gets tokenized
    bm25 = BM25Okapi([_tokenize(document) for document in data["documents"]])
    return bm25, data["ids"], data["documents"], data["metadatas"]

# BM25 keyword search
def bm25_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return the ``k`` stored chunks with the best BM25 keyword match to ``query``.

    Each chunk is a dict with ``id``, ``text``, ``source``, ``page``, BM25
    ``score`` (higher = better match) and ``score_source`` (always ``"bm25"``),
    best first. Chunks sharing no words with the query are dropped, so fewer
    than ``k`` results (or none) can come back. Raises if the collection does
    not exist, i.e. ingestion has not been run.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    collection = chroma_client.get_collection(settings.collection_name)
    chunk_count = collection.count()
    if chunk_count == 0:
        return []
    # Takes a 
    bm25, ids, documents, metadatas = _load_bm25_index(chunk_count)
    # Returns a numpy array with 1 score/chunks <-> keyword search w tokenized query
    scores = bm25.get_scores(_tokenize(query))
    # Sorts the score and selects the top 5
    top_indices = scores.argsort()[::-1][:k]

    chunks: list[dict[str, Any]] = []
    for i in top_indices:
        if scores[i] <= 0:
            break  # sorted descending: everything after has no keyword overlap
        chunks.append(
            {
                "id": ids[i],
                "text": documents[i],
                "source": metadatas[i]["source"],
                "page": metadatas[i]["page"],
                "score": float(scores[i]),
                "score_source": "bm25",
            }
        )
    return chunks


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]], k: int = 60
) -> list[dict[str, Any]]:
    """Merge ranked chunk lists into one ranking with reciprocal rank fusion.

    Each chunk earns ``1 / (k + rank)`` from every list it appears in (rank is
    1-based position), so chunks found by several methods float to the top. Only
    rank is used, which sidesteps BM25 scores and cosine distances being on
    incomparable scales. Chunks are matched across lists by ``id``.

    ``k`` here is RRF's damping constant (60 is the standard value). Returns every merged chunk, best
    first, as dicts with ``id``, ``text``, ``source``, ``page``, ``rrf_score``
    and ``found_in`` (number of lists containing it). Method-specific keys
    (``distance``, ``score``, ``score_source``) are dropped. Ties keep the order
    in which chunks were first seen.
    """
    # Stores the RRF score for each chunk
    scores: dict[str, float] = {}
    # Stores the number of lists containing a chunk
    found_in: dict[str, int] = {}
    merged: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["id"]
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "id": chunk_id,
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                }
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            found_in[chunk_id] = found_in.get(chunk_id, 0) + 1

    fused = [
        {**chunk, "rrf_score": scores[chunk_id], "found_in": found_in[chunk_id]}
        for chunk_id, chunk in merged.items()
    ]
    # sorted() is stable, so equal scores keep first-seen order
    return sorted(fused, key=lambda chunk: chunk["rrf_score"], reverse=True)


def rerank(
    query: str, chunks: list[dict[str, Any]], top_n: int = 5
) -> list[dict[str, Any]]:
    """Rerank candidate ``chunks`` with Cohere and keep the best ``top_n``.

    A cross-encoder reads the query together with each chunk's ``text`` and
    scores how well it answers the query. Returns at most ``top_n`` chunks, best
    first, each a copy of its input dict plus ``rerank_score`` (0-1, higher =
    more relevant). Input keys such as ``rrf_score`` are preserved.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if not chunks:
        return []

    response = cohere_client.rerank(
        query=query,
        documents=[chunk["text"] for chunk in chunks],
        model=settings.rerank_model,
        top_n=min(top_n, len(chunks)),
    )
    # result.index is the chunk's position in the input list
    return [
        {**chunks[result.index], "rerank_score": result.relevance_score}
        for result in response.results
    ]


# Hybrid retrieval: Entry point caller uses
def retrieve(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return the ``k`` chunks most relevant to ``query`` via hybrid retrieval.

    Runs four stages: vector search and BM25 keyword search over a wide
    candidate pool (``settings.candidate_k`` each), a reciprocal rank fusion
    merge, then a Cohere rerank that cuts the pool to the best ``k``.

    Each chunk is a dict with ``id``, ``text``, ``source``, ``page``,
    ``rrf_score``, ``found_in`` and ``rerank_score`` (0-1, higher = more
    relevant), best first. If the Cohere call fails, the RRF order is returned
    instead and ``rerank_score`` is absent. Raises if the collection does not
    exist, i.e. ingestion has not been run.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    # Cast a wide net before reranking, but never fewer than the caller wants
    pool = max(settings.candidate_k, k)
    fused = reciprocal_rank_fusion(
        [vector_search(query, pool), bm25_search(query, pool)]
    )

    try:
        return rerank(query, fused, top_n=k)
    except ApiError:
        # Degrade to the merge order rather than failing the whole query
        logger.warning(
            # Reranking fails
            "Cohere rerank failed; falling back to RRF order", exc_info=True
        )
        return fused[:k]


"""
Note: this demo prints chunk['distance'], which hybrid retrieve() no longer
returns (RRF drops method-specific scores). Use vector_search() for distances.

if __name__ == "__main__":
    test_query = "How much exercise should adults get per week?"
    print(f"Query: {test_query}\n")
    for rank, chunk in enumerate(retrieve(test_query, k=3), start=1):
        print(f"[{rank}] {chunk['id']}")
        print(f"    {chunk['source']} p.{chunk['page']}  distance={chunk['distance']:.3f}")
        print(f"    {chunk['text'][:150].strip()!r}\n")
"""
