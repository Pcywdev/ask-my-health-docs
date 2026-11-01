# Roadmap

Where the project stands. Update after each milestone (see CLAUDE.md, Documentation).
Last updated: 2026-11-01 · branch `test/unit-tests` · Alembic head `a9d742709640`

Legend: [x] finished · [~] in progress · [ ] not started

## Now / Next up
- **Now:** [~] Step 8, tests (`test/unit-tests`), awaiting commit and PR
- **Next up:** step 9: write the eval set (you), then `run_eval.py`
- **Then:** Docker + CI

## Done
- [x] Ingestion: PDFs → chunks → embeddings → Chroma (843 chunks, 2 documents) · tag `phase-1-complete`
- [x] Hybrid retrieval: vector + BM25 → RRF → Cohere rerank, with RRF fallback · tag `phase-1-complete`
- [x] Generation with `[Source N]` citations, YAML prompt v1.0, `validate_citation` · tag `phase-2-complete`
- [x] Query logging, `GET /history` dashboard · tag `phase-2-complete`
- [x] Cleanup: key revoked and purged from git history, `.env` restored, `.gitignore` fixed (reported done by you)
- [x] Remote: private GitHub repo, phase tags, old branches removed
- [x] Dependencies: hand-written `requirements.txt` / `requirements-dev.txt` · PR #1
- [x] Schema: retrieval trace on `query_logs`, `eval_runs` + `eval_results`, first real Alembic migration, verified live · PR #2
- [x] `ragas` import fix: `langchain-community` pinned `<0.4` · PR #3
- [x] Docs: `docs/architecture.md`, `docs/ROADMAP.md`, CLAUDE.md update · PR #4
- [x] Study notes: `docs/record_notes.md` + keep-updated rule in CLAUDE.md · PR #5
- [x] Lint cleanup: `ruff check .` clean · PR #6

## Remaining (in order)
- [~] **Step 8, tests** — `pytest.ini`, `tests/conftest.py` (dummy env, in-memory SQLite); 32 offline tests + 1 strict xfail for the `[Source 1-3]` range bug: RRF, `retrieve()` Cohere fallback, citation parsing, stats aggregate, `/query` survives a logging failure; `app/test.py` deleted (branch `test/unit-tests`)
- [ ] **Step 9, eval** — `eval/eval_set.json` (30–50 questions, page-level gold labels, ~20% unanswerable), `eval/thresholds.json` (set by you), `eval/run_eval.py` (custom metrics + RAGAS, writes `eval_runs` / `eval_results`)
- [ ] **Step 10, Docker + CI** — `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` (ruff, `alembic upgrade head`, `alembic check`, pytest)
- [ ] **Later** — Celery + Redis, Streamlit frontend

## Open issues
- `eval/eval_set.json` and `eval/run_eval.py` are empty, so "retrieval changes must pass eval" cannot be enforced yet.
- CI cannot run the eval (PDFs and Chroma are not in git; needs paid keys). Run it locally and paste scores in the PR.
- `/history` and `/ingest` have no authentication.
- `CITATION_PATTERN` mishandles ranges: `[Source 1-3]` parses as {1, 3}.
- BM25 cache is keyed on chunk count; `load_prompt()` is cached: restart after re-ingesting or editing the prompt.
- `temperature=0` is not deterministic: average several eval runs.

## How to update
After a milestone: move the item to Done with its PR or tag, refresh Now / Next up, update the date line.
