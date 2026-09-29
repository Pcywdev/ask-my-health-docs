# Project Name

One-sentence description of what this project is and does.

## Tech Stack
- Backend: FastAPI, Python 3.11
- Vector store: ChromaDB
- Database: PostgreSQL + SQLAlchemy
- Task queue: Celery + Redis
- Eval: RAGAS
- Frontend: Streamlit
- Containerization: Docker Compose
- CI: GitHub Actions

## Commands
- Dev server: `uvicorn app.main:app --reload`
- Run tests: `pytest`
- Run eval: `python eval/run_eval.py`
- Docker: `docker compose up --build`
- Migrations: `alembic upgrade head`
- Lint: `ruff check .`

## Architecture
- `app/` — FastAPI application (routes, retriever, generator, config)
- `prompts/` — versioned prompt templates in YAML
- `eval/` — evaluation set + eval runner
- `data/raw/` — source PDFs (not committed to git)

## Conventions
- All config via pydantic-settings, loaded from .env
- Prompts stored in YAML, never hardcoded in Python
- Every retrieval change must pass eval thresholds before merge
- Type hints on all function signatures
- Docstrings on public functions

## Off-limits
- Never modify .env directly — use .env.example as the template
- Never commit API keys or secrets
- Don't change eval thresholds without explicit instruction