"""FastAPI backend: HTTP routes over the ingest -> retrieve -> generate pipeline."""

import logging
import time
from datetime import datetime

import openai
from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from app.db import (
    QueryLog,
    citation_stats,
    fabricated_queries,
    log_query,
    recent_queries,
)
from app.generator import generate_answer, load_prompt
from app.ingest import ingest
from app.retriever import retrieve

logger = logging.getLogger(__name__)

# How much of a logged answer /history shows per row
PREVIEW_CHARS = 200

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


class CitationReport(BaseModel):
    """How well the answer's [Source N] citations match the retrieved chunks."""
    total_citations: int = Field(description="Distinct source numbers the answer cites.")
    valid_citations: int = Field(description="Citations that point at a real retrieved chunk.")
    fabricated_citations: int = Field(description="Citations outside the retrieved chunk range.")
    citation_accuracy: float = Field(
        description="valid / total, 0.0 when nothing is cited (check has_citation)."
    )
    has_citation: bool = Field(description="Whether the answer cites anything at all.")


class QueryResponse(BaseModel):
    """The generated answer, the sources it cites, and a citation quality report."""
    answer: str
    sources: list[SourceItem] = Field(
        description="Only the cited chunks -> empty if the answer cites none."
    )
    citation_report: CitationReport


class IngestResponse(BaseModel):
    """Result of an ingestion run."""
    chunks_stored: int


class HistoryStats(BaseModel):
    """Citation quality aggregated all the logged queries."""
    total_queries: int = Field(description="Queries logged in the window.")
    mean_citation_accuracy_cited: float | None = Field(
        description="Mean accuracy over answers that cited something; null if none did."
    )
    mean_citation_accuracy_all: float | None = Field(
        description="Mean accuracy over every query, including refusals (which score 0.0)."
    )
    answers_with_citations: int = Field(description="Answers that cited at least one source.")
    refusals: int = Field(description="Answers that cited nothing.")
    refusal_rate: float | None = Field(description="refusals / total_queries.")
    queries_with_fabrication: int = Field(
        description="Answers with at least one citation outside the retrieved chunks."
    )
    total_fabricated_citations: int = Field(description="Fabricated citations summed over answers.")
    avg_sources_retrieved: float | None = Field(description="Mean chunks retrieved per query.")
    window_days: int | None = Field(description="The ?days filter applied, or null for all time.")


class HistoryItem(BaseModel):
    """One logged query and how well it cited its sources."""
    id: int
    question: str
    answer_preview: str = Field(description=f"First {PREVIEW_CHARS} characters of the answer.")
    created_at: datetime | None
    num_sources: int | None = Field(description="Chunks retrieved, not chunks cited.")
    total_citations: int | None
    valid_citations: int | None
    fabricated_citations: int | None = Field(description="Derived as total minus valid.")
    citation_accuracy: float | None
    has_citations: bool | None


class HistoryResponse(BaseModel):
    """Citation quality dashboard over the logged queries."""
    stats: HistoryStats
    recent: list[HistoryItem] = Field(description="Most recent queries, newest first.")
    # Queries whose answers cited a chunk that was not retrieved
    fabricated: list[HistoryItem] = Field(
        description="Queries whose answers cited a chunk that was not retrieved."
    )


def _to_history_item(row: QueryLog) -> HistoryItem:
    """Map a stored row to its API shape, truncating the answer."""
    answer = row.answer or ""
    preview = answer[:PREVIEW_CHARS] + "…" if len(answer) > PREVIEW_CHARS else answer
    fabricated = (
        row.total_citations - row.valid_citations
        if row.total_citations is not None and row.valid_citations is not None
        else None
    )
    return HistoryItem(
        id=row.id,
        question=row.question,
        answer_preview=preview,
        created_at=row.created_at,
        num_sources=row.num_sources,
        total_citations=row.total_citations,
        valid_citations=row.valid_citations,
        fabricated_citations=fabricated,
        citation_accuracy=row.citation_accuracy,
        has_citations=row.has_citations,
    )


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
        started = time.perf_counter()
        chunks = retrieve(request.question, request.top_k)
        result = generate_answer(request.question, chunks)
        latency_ms = int((time.perf_counter() - started) * 1000)
    except NotFoundError:
        raise HTTPException(
            status_code=503, detail="Index not built. Run POST /ingest first."
        )
    except openai.APIError:
        logger.exception("OpenAI call failed during /query")
        raise HTTPException(
            status_code=502, detail="Upstream LLM/embedding service error."
        )

    # A database outage must not cost the user an answer that is already paid for
    try:
        log_query(request.question, result, chunks, request.top_k, latency_ms, load_prompt()["version"])
    except SQLAlchemyError:
        logger.exception("Failed to log query to the database")

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
        citation_report=CitationReport(**result["citation_report"]),
    )


@app.get("/history", response_model=HistoryResponse)
def history(
    days: int | None = Query(
        None, ge=1, le=365, description="Only queries from the last N days; omit for all time."
    ),
    limit: int = Query(10, ge=1, le=100, description="Rows per list."),
) -> HistoryResponse:
    """Citation quality stats plus the recent and fabricating queries behind them."""
    try:
        stats = citation_stats(days)
        recent = recent_queries(limit, days)
        fabricated = fabricated_queries(limit, days)
    except SQLAlchemyError:
        logger.exception("Failed to read query history")
        raise HTTPException(status_code=503, detail="History database unavailable.")

    return HistoryResponse(
        stats=HistoryStats(**stats),
        recent=[_to_history_item(row) for row in recent],
        fabricated=[_to_history_item(row) for row in fabricated],
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
