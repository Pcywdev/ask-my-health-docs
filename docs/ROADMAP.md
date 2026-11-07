# Roadmap

Where the project stands. Update after each milestone (see CLAUDE.md, Documentation).
Last updated: 2026-11-08 · branch `feat/eval-runner` · Alembic head `a9d742709640`

Legend: [x] finished · [~] in progress · [ ] not started

## Now / Next up
- **Now:** [~] Step 9, eval: done, awaiting commit and PR (`feat/eval-runner`)
- **Next up:** step 10, Docker + CI
- **Then:** fix the eval findings in Open issues (cross-document retrieval, q027), each with a before/after eval

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
- [x] Step 8, offline unit tests (RRF, citations, stats, Cohere fallback, `/query` resilience) · PR #7

## Remaining (in order)
- [~] **Step 9, eval** — `eval/run_eval.py` (custom metrics + RAGAS, threshold gate), `metrics.py`, `dataset.py`, `find_pages.py`, `README.md`; eval set v1.3 (40 questions, 8 unanswerable); baseline × 3 (eval runs 3–5, retrieval metrics identical across runs, judge metrics within ±0.01); thresholds set in `eval/thresholds.json`; 30 new offline tests (branch `feat/eval-runner`)
- [ ] **Step 10, Docker + CI** — `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` (ruff, `alembic upgrade head`, `alembic check`, pytest)
- [ ] **Later** — Celery + Redis, Streamlit frontend

## Open issues
- Comparison questions that need both documents can retrieve only one (q035 false refusal, q033 faithfulness 0.50): nothing in retrieval guarantees both sources are represented.
- q027 answers with the outdated 2008 rule (10-minute bouts) before the current one; faithfulness 0.75.
- `answer_relevancy` scores correct, concise answers as low as 0.56, so it is measured but not gated (`null`).
- CI cannot run the eval (PDFs and Chroma are not in git; needs paid keys). Run it locally and paste scores in the PR.
- `/history` and `/ingest` have no authentication.
- `CITATION_PATTERN` mishandles ranges: `[Source 1-3]` parses as {1, 3}.
- BM25 cache is keyed on chunk count; `load_prompt()` is cached: restart after re-ingesting or editing the prompt.
- `temperature=0` is not deterministic, but only the RAGAS judge scores varied across baseline runs (±0.01); retrieval and citation metrics were identical.

## How to update
After a milestone: move the item to Done with its PR or tag, refresh Now / Next up, update the date line.
