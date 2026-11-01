"""Shared test setup: everything runs offline, with no real keys or databases.

The environment is set before ``app`` is imported anywhere, because
``app.config`` builds ``settings`` and several modules create API clients at
import time. Variables are assigned (not ``setdefault``) so a real ``.env`` key
can never leak into a test run.
"""

import os
import tempfile

os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["COHERE_API_KEY"] = "test-cohere-key"
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["CHROMA_PATH"] = tempfile.mkdtemp(prefix="askdocs-test-chroma-")

from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import db


@pytest.fixture
def sqlite_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker:
    """Point ``app.db`` at a fresh in-memory SQLite database.

    ``StaticPool`` keeps one shared connection; without it every connection
    would get its own empty in-memory database and the tables would vanish.
    """
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    db.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, "SessionLocal", factory)
    return factory


@pytest.fixture
def make_chunk() -> Callable[..., dict[str, Any]]:
    """Build a retrieved-chunk dict; ``make_chunk("a")`` is enough."""

    def _make(chunk_id: str, page: int = 1, **extra: Any) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "text": f"text of {chunk_id}",
            "source": "doc.pdf",
            "page": page,
            **extra,
        }

    return _make
