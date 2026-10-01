"""Ingestion pipeline: PDFs -> page docs -> chunks -> embeddings -> ChromaDB."""

import logging
from pathlib import Path
import chromadb
from chromadb.api.models.Collection import Collection
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Clients
openai_client = OpenAI(api_key=settings.openai_api_key, max_retries=5)
chroma_client = chromadb.PersistentClient(path=settings.chroma_path)


def get_collection() -> Collection:
    """Return the shared ChromaDB collection, creating it if needed."""
    return chroma_client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def load_pdfs(directory: Path) -> list[Document]:
    """Load every PDF in ``directory`` as one Document per page.

    Each Document's metadata is normalised to ``source`` (filename only) and
    ``page`` (1-indexed). Pages with no extractable text are skipped.
    """
    # Deterministic list returned
    pdf_paths = sorted(directory.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {directory}")

    # pages store all the documents (metadata of each page)
    pages: list[Document] = [] 
    skipped = 0
    # Loops through each pdf in the path
    for path in pdf_paths:
        # doc holds one Document/page of each path
        # doc is raw metadata {'source', 'page', etc}
        for doc in PyPDFLoader(str(path)).load():
            # counts and skips pages with no extractable text
            if not doc.page_content.strip():
                skipped += 1
                continue
            # Rebuild the Document
            pages.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": path.name, "page": doc.metadata["page"] + 1},
                )
            )
    logger.info(
        "Loaded %d pages from %d PDFs (%d pages skipped (empty))",
        len(pages),
        len(pdf_paths),
        skipped,
    )
    return pages


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split page-level Documents into smaller chunks with recursive chunking.

    Chunks inherit ``source`` and ``page`` and gain a per-page ``chunk_index``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # A dict keyed by (source, page)
    counters: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        # Each chunks automatilly copies its parent page's source and page
        key = (chunk.metadata["source"], chunk.metadata["page"])
        chunk.metadata["chunk_index"] = counters.get(key, 0)
        # The dictionary stores chunk_index (0, 1, 2, ...) and bumps the counter.
        # The index restarts at 0 on every page
        counters[key] = chunk.metadata["chunk_index"] + 1
    logger.info("Split %d pages into %d chunks", len(docs), len(chunks))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed ``texts`` with OpenAI in batches, preserving input order."""
    embeddings: list[list[float]] = []
    batch_size = settings.embed_batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = openai_client.embeddings.create(
            model=settings.embedding_model, input=batch
        )
        # Appens each item's vector. Each vecotr is a list of 1536 floats for text-embedding-3-small
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def store_chunks(chunks: list[Document], embeddings: list[list[float]]) -> int:
    """Upsert chunks, embeddings and metadata into ChromaDB.

    IDs are deterministic (source, page, chunk index), so re-ingesting the same
    files overwrites existing entries instead of duplicating them.
    """
    # Chunks count and embedding count differs -> Raises Value Error.
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunks but {len(embeddings)} embeddings"
        )
    # returns the collection
    collection = get_collection()
    # returns the most records Chroma accepts in one write
    max_batch = chroma_client.get_max_batch_size()
    for start in range(0, len(chunks), max_batch):
        batch = chunks[start : start + max_batch]
        # upsert as IDs are deterministic. Ingestion is safe to repeat
        '''
        Re-ingesting after chaging chunking settings (size and overlap):
        Old IDs that no longer exist stay in Chromadb
        Best practice: Delete the collection first before
        '''
        collection.upsert(
            ids=[
                f"{c.metadata['source']}::p{c.metadata['page']}::c{c.metadata['chunk_index']}"
                for c in batch
            ],
            documents=[c.page_content for c in batch],
            embeddings=embeddings[start : start + max_batch],
            metadatas=[
                {
                    "source": c.metadata["source"],
                    "page": c.metadata["page"],
                    "chunk_index": c.metadata["chunk_index"],
                }
                for c in batch
            ],
        )
    # Returns the # of chunks written
    return len(chunks)


def ingest(directory: Path | None = None) -> int:
    """Run the full pipeline and return the number of chunks stored."""
    directory = directory or Path(settings.raw_data_dir)
    pages = load_pdfs(directory)
    chunks = chunk_documents(pages)
    embeddings = embed_texts([c.page_content for c in chunks])
    stored = store_chunks(chunks, embeddings)
    logger.info("Stored %d chunks in collection '%s'", stored, settings.collection_name)
    return stored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
