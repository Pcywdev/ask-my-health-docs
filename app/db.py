"""Persistence: log every answered query with its citation quality metrics."""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    Select,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

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
    # timezone=True keeps the UTC offset; a naive column would silently drop it
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def init_db() -> None:
    """Create the ``query_logs`` table if it does not exist."""
    Base.metadata.create_all(engine)


def log_query(question: str, result: dict[str, Any], num_sources: int) -> int:
    """Store one answered query and its citation metrics; return the row id.

    ``result`` is a whole ``generate_answer`` return value, so the metric fields
    cannot be mismatched by the caller. ``num_sources`` is the number of chunks
    retrieved, which is not part of that result.
    """
    report = result["citation_report"]
    row = QueryLog(
        question=question,
        answer=result["answer"],
        num_sources=num_sources,
        total_citations=report["total_citations"],
        valid_citations=report["valid_citations"],
        citation_accuracy=report["citation_accuracy"],
        has_citations=report["has_citation"],
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
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
    
