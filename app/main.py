"""FastAPI backend: HTTP routes over the ingest -> retrieve -> generate pipeline."""

import logging
import openai
from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from app.generator import generate_answer
from app.ingest import ingest
from app.retriever import retrieve

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ask My Health Docs",
    description="RAG system for exercise science and public health guidelines.",
)


# --- Request/response models -------------------------------------------------


class QueryRequest(BaseModel):
    """A question to answer from the ingested documents."""
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=2000, description="The user's question.")
    top_k: int = Field(5, ge=1, le=20, description="Required # of retrieved chunks.")


class SourceItem(BaseModel):
    """A chunk the answer cites."""
    index: int = Field(description="The N in the answer's [Source N] citation.")
    source: str = Field(description="Name of the source document.")
    page: int = Field(description="1-indexed page number in the source document.")
    text: str = Field(description="The cited chunk's text.")


class QueryResponse(BaseModel):
    """The generated answer and the sources it cites."""
    answer: str
    sources: list[SourceItem] = Field(
        description="Only the cited chunks -> empty if the answer cites none."
    )


class IngestResponse(BaseModel):
    """Result of an ingestion run."""
    chunks_stored: int


# --- Routes ------------------------------------------------------------------
# Handlers are plain `def`: the pipeline is blocking, so FastAPI runs them in a
# thread pool instead of stalling the event loop.


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check"""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve the top-k chunks for the question and answer with citations."""
    try:
        chunks = retrieve(request.question, request.top_k)
        result = generate_answer(request.question, chunks)
    except NotFoundError:
        raise HTTPException(
            status_code=503, detail="Index not built. Run POST /ingest first."
        )
    except openai.APIError:
        logger.exception("OpenAI call failed during /query")
        raise HTTPException(
            status_code=502, detail="Upstream LLM/embedding service error."
        )

    return QueryResponse(
        answer=result["answer"],
        sources=[
            SourceItem(
                index=source["label"],
                source=source["source"],
                page=source["page"],
                text=source["text"],
            )
            for source in result["sources"]
        ],
    )


@app.post("/ingest", response_model=IngestResponse)
def run_ingest() -> IngestResponse:
    """Ingest every PDF in the raw data directory into ChromaDB."""
    try:
        return IngestResponse(chunks_stored=ingest())
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except openai.APIError:
        logger.exception("OpenAI call failed during /ingest")
        raise HTTPException(
            status_code=502, detail="Upstream embedding service error."
        )
