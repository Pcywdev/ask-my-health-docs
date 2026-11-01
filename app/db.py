"""Persistence: log every answered query with its citation quality metrics."""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Select,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

JsonType = JSON().with_variant(JSONB(), "postgresql")

# create_engine does not connect, so importing this module works with the
# database down; pool_pre_ping discards connections that died while idle.
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for the ORM models."""


class QueryLog(Base):
    """One answered question and how well it cited its sources."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    question = Column(String, nullable=False)
    answer = Column(Text, nullable=False)
    num_sources = Column(Integer)  # chunks retrieved, not chunks cited
    total_citations = Column(Integer)
    valid_citations = Column(Integer)
    citation_accuracy = Column(Float)
    has_citations = Column(Boolean)
    retrieved_ids = Column(JsonType)     # chunk ids, final rank order
    rerank_scores = Column(JsonType)     # [null, ...] when the Cohere fallback fired
    top_k = Column(Integer)
    reranked = Column(Boolean)           # False = served in RRF order
    prompt_version = Column(String)
    llm_model = Column(String)
    embedding_model = Column(String)
    latency_ms = Column(Integer)
    
    # timezone=True keeps the UTC offset; a naive column would silently drop it
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),       # non-ORM inserts no longer get NULL
        index=True,                      # /history orders and filters on this
    )
    

class EvalRun(Base):
    """One execution of the eval runner."""
    __tablename__ = "eval_runs"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    label = Column(String)
    git_sha = Column(String)
    config = Column(JsonType, nullable=False)   # chunk_size, overlap, models, candidate_k, top_k, prompt_version
    eval_set_size = Column(Integer)
    summary = Column(JsonType)                  # aggregate scores


class EvalResult(Base):
    """One question's outcome within an eval run."""
    __tablename__ = "eval_results"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"),
                    nullable=False, index=True)
    question_id = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(Text)
    retrieved_ids = Column(JsonType)
    contexts = Column(JsonType)        # chunk texts the judge scored against
    reranked = Column(Boolean)
    gold_hit = Column(Boolean)         # custom: gold page in the top-k
    citation_accuracy = Column(Float)  # custom
    metrics = Column(JsonType)         # RAGAS scores; names change between versions
    latency_ms = Column(Integer)
    error = Column(Text)


def log_query(question: str, result: dict[str, Any], chunks: list[dict[str, Any]],
              top_k: int, latency_ms: int, prompt_version: str) -> int:
    """Store one answered query, its citation metrics and retrieval trace; return the row id.

    ``result`` is a whole ``generate_answer`` return value, so the metric fields
    cannot be mismatched by the caller. ``chunks`` is what ``retrieve`` returned:
    its ids, rerank scores and count are stored, and ``reranked`` is False when
    the Cohere fallback served RRF order. ``latency_ms`` covers retrieval plus
    generation; ``prompt_version`` comes from the prompt YAML.
    """
    report = result["citation_report"]
    row = QueryLog(
        question=question, answer=result["answer"],
        num_sources=len(chunks),
        total_citations=report["total_citations"],
        valid_citations=report["valid_citations"],
        citation_accuracy=report["citation_accuracy"],
        has_citations=report["has_citation"],
        retrieved_ids=[c["id"] for c in chunks],
        rerank_scores=[c.get("rerank_score") for c in chunks],
        top_k=top_k,
        reranked=bool(chunks) and all("rerank_score" in c for c in chunks),
        prompt_version=prompt_version,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        latency_ms=latency_ms,
    )
    with SessionLocal() as session:
        try:
            session.add(row)
            session.commit()
        except Exception:
            session.rollback()
            raise
    return row.id # Post-gres generated ID, SQLAlchemy reads it back after the inset and fills it in on the object
    # expire_of_commit = False on the session maker lets you still read row.id after the with block closes


# --- Reading the log: citation quality stats ---------------------------------

# fabricated citations are not stored; they are total minus valid
FABRICATED = QueryLog.total_citations - QueryLog.valid_citations


# stmt -> Query statement
# _window -> selects query logs created before a cutoff
"""
select(QueryLog)                      →  SELECT * FROM query_logs
  .where(created_at >= cutoff)        →  SELECT * FROM query_logs WHERE created_at >= '...'
  .order_by(created_at.desc())        →  ... ORDER BY created_at DESC
  .limit(3)                           →  ... LIMIT 3
                                          ↑ still just a string being assembled
"""

def _window(stmt: Select, days: int | None) -> Select:
    """Limit ``stmt`` to rows created in the last ``days``; all rows if None."""
    if days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        stmt = stmt.where(QueryLog.created_at >= cutoff)
    # days = None -> Hands back query untouched
    return stmt



def citation_stats(days: int | None = None) -> dict[str, Any]:
    """Aggregate citation quality over the logged queries.

    Two means are reported: 
        -one over answers that cited something, 
        -one over every row. 
    Refusals store an accuracy of 0.0, so the second is dragged down
    by answers that were correct to cite nothing. Averages are ``None`` when the
    window holds no queries.
    """
    cited = QueryLog.has_citations.is_(True)
    stmt = _window(
        select(
            func.count().label("total_queries"),
            func.avg(QueryLog.citation_accuracy).filter(cited),
            func.avg(QueryLog.citation_accuracy),
            func.count().filter(cited),
            func.count().filter(QueryLog.has_citations.is_(False)),
            func.count().filter(FABRICATED > 0),
            func.sum(FABRICATED),
            func.avg(QueryLog.num_sources),
        ),
        days,
    )
    with SessionLocal() as session:
        (
            total,
            mean_cited,
            mean_all,
            with_citations,
            refusals,
            with_fabrication,
            fabricated_total,
            avg_sources,
        ) = session.execute(stmt).one()
        '''
        one() demandas exactly one row back. The eight results come back as a tuple in the same order
        they were listed in select(), and get unpacked into eight variables
        '''
        '''
                             ONE query to Postgres
     4 rows  ──────────────────────────────────>  (4, 0.833, 0.625, 3, 1, 1, 1, 4.5)
                                                   ↑ one tuple, eight numbers
        '''

        

    # Postgres avg() returns Decimal, and sum() over no rows returns NULL
    return {
        "total_queries": total,
        "mean_citation_accuracy_cited": float(mean_cited) if mean_cited is not None else None,
        "mean_citation_accuracy_all": float(mean_all) if mean_all is not None else None,
        "answers_with_citations": with_citations,
        "refusals": refusals,
        "refusal_rate": refusals / total if total else None,
        "queries_with_fabrication": with_fabrication,
        "total_fabricated_citations": int(fabricated_total or 0),
        "avg_sources_retrieved": float(avg_sources) if avg_sources is not None else None,
        "window_days": days,
    }


def recent_queries(limit: int = 10, days: int | None = None) -> list[QueryLog]:
    """Return the most recently logged queries, newest first."""
    stmt = _window(select(QueryLog), days)
    with SessionLocal() as session:
        return list(
            session.scalars(stmt.order_by(QueryLog.created_at.desc()).limit(limit))
        )
    """
            all 4 rows                 ORDER BY created_at DESC       LIMIT 3
  ┌──────────────────────┐        ┌──────────────────────┐    ┌──────────────────────┐
  │ 1  today 09:00       │        │ 3  today 11:00       │    │ 3  today 11:00       │
  │ 2  today 10:00       │  ───>  │ 2  today 10:00       │──> │ 2  today 10:00       │
  │ 3  today 11:00       │        │ 1  today 09:00       │    │ 1  today 09:00       │
  │ 4  30 days ago       │        │ 4  30 days ago       │    └──────────────────────┘
  └──────────────────────┘        └──────────────────────┘      row 4 never sent

    """
    


def fabricated_queries(limit: int = 10, days: int | None = None) -> list[QueryLog]:
    """Return logged queries whose answers cited a chunk that was not retrieved."""
    stmt = _window(select(QueryLog).where(FABRICATED > 0), days)
    with SessionLocal() as session:
        return list(
            session.scalars(stmt.order_by(QueryLog.created_at.desc()).limit(limit))
        )
    '''
        ┌────┬───────┬───────┬─────────────┐
        │ id │ total │ valid │ FABRICATED  │
        ├────┼───────┼───────┼─────────────┤
        │  1 │     2 │     2 │      0      │ ✗ clean
        │  2 │     2 │     1 │      1      │ ✓ KEPT — cited a source that wasn't retrieved
        │  3 │     0 │     0 │      0      │ ✗ refusal
        │  4 │     1 │     1 │      0      │ ✗ clean
        └────┴───────┴───────┴─────────────┘
    '''
    
