"""Retrieval: embed a query and fetch top-k chunks from ChromaDB (as a list)."""

from typing import Any

from app.config import settings
from app.ingest import chroma_client, embed_texts


def retrieve(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return the ``k`` stored chunks most similar to ``query``.

    Context: Each chunk is a dict with ``id``, ``text``, ``source``, ``page`` and cosine
    ``distance`` (lower = more similar). Raises if
    the collection does not exist, i.e. ingestion has not been run.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    query_embedding = embed_texts([query])[0]
    collection = chroma_client.get_collection(settings.collection_name)
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

"""
if __name__ == "__main__":
    test_query = "How much exercise should adults get per week?"
    print(f"Query: {test_query}\n")
    for rank, chunk in enumerate(retrieve(test_query, k=3), start=1):
        print(f"[{rank}] {chunk['id']}")
        print(f"    {chunk['source']} p.{chunk['page']}  distance={chunk['distance']:.3f}")
        print(f"    {chunk['text'][:150].strip()!r}\n")
"""
