# eval/ — file-by-file reference

This is a function-level map of the evaluation module: what each file does,
what every function takes in and gives back, and how one question's data
flows through all of them. For the concepts behind the metrics (what MRR
*means*, why page-level gold labels), see `docs/record_notes.md` section 8.
For the system-level design, see `docs/architecture.md`.

## The module at a glance

| File | Role |
|---|---|
| `eval_set.json` | the questions you write, with page-level gold labels |
| `dataset.py` | loads the set and checks it's well-formed before anything runs |
| `find_pages.py` | free lookup tool: which indexed pages contain a phrase |
| `metrics.py` | pure scoring functions — no I/O, no network, no cost |
| `run_eval.py` | the CLI that retrieves, generates, scores and stores one run |
| `thresholds.json` | pass/fail bounds the runner checks the summary against |
| `app/db.py` (3 functions) | writes the run and its results to Postgres |

Everything in `dataset.py` and `metrics.py` is free and offline. Everything
in `run_eval.py` except `--validate` calls OpenAI and Cohere and spends money.

---

## `eval_set.json`

Not code — the data file you maintain. Shape:

```json
{
  "version": "1.1",
  "questions": [
    {
      "id": "q001",
      "question": "...",
      "category": "numeric",
      "answerable": true,
      "answer": "a short reference answer, 1-2 sentences",
      "gold": [
        {"source": "9789240015128-eng.pdf", "pages": [11, 35, 37]}
      ]
    }
  ]
}
```

Rules enforced by `dataset.py` (see below):
- `answerable: false` ⇒ `gold` must be `[]`.
- `answerable: true` ⇒ `gold` must have at least one entry, and a reference
  answer is required (either an `answer` key or a `reference` key — both are
  accepted; `reference_of()` reads whichever is present).
- Every `(source, page)` in `gold` must actually exist in the Chroma index.
  The page number is the **PDF viewer's page position**, not the number
  printed in the document's footer.
- `version` is a free-text string; bump it whenever you edit questions.

---

## `eval/dataset.py` — load and validate

| Function | Takes in | Gives back | What it does |
|---|---|---|---|
| `load_json(path)` | a `Path` | the parsed dict | Reads any UTF-8 JSON file. Used for both `eval_set.json` and `thresholds.json`. |
| `reference_of(question)` | one question dict | `str \| None` | Returns the question's reference answer, preferring `reference` over `answer` if both are present. |
| `index_pages()` | nothing | `{source_filename: {page numbers}}` | Reads Chroma directly (no API call) and builds the "ground truth" of what pages actually exist in the index. Every gold page is checked against this. |
| `_check_question(q, pages, errors)` | one question, the page map, a list to append to | nothing (mutates `errors`) | Validates one question: question/category are non-empty strings, `answerable` is a bool, `gold` matches the answerable/unanswerable rule, every gold source is known and every gold page exists, answerable questions have a reference. Each problem becomes one readable string like `"q003: 9789240015128-eng.pdf p.99 is not in the index..."`. |
| `validate_eval_set(data, pages)` | the loaded set, the page map from `index_pages()` | `(errors, warnings)` | Checks `version`/`questions` exist, checks for duplicate or missing ids, runs `_check_question` on every question, and warns if the unanswerable share drifts outside ~15–25%. **Errors** mean the set can't be used. **Warnings** are printed but don't block anything. |
| `category_counts(data)` | the loaded set | a `Counter` | Counts questions per `category`, used only for the printed report. |

**Constants:** `EVAL_SET_PATH`, `THRESHOLDS_PATH` (absolute paths to the two
JSON files), `UNANSWERABLE_SHARE = (0.15, 0.25)`.

---

## `eval/find_pages.py` — phrase lookup (free, offline)

| Function | Takes in | Gives back | What it does |
|---|---|---|---|
| `find(phrase, source=None)` | a phrase to search for, optionally one PDF filename to restrict to | a list of `(source, page, chunk_id, snippet)` tuples, sorted | Searches every chunk's stored text for `phrase`. Matching ignores case and **all whitespace** in the phrase, because PDF extraction inserts stray spaces into words (e.g. "150– 300", "balance tr aining") — without this, a label-correct phrase can fail to match its own source text. |
| `main()` | command-line args | prints to stdout | The CLI entry point: `python eval/find_pages.py "150 minutes" [--source FILE.pdf]`. Prints every matching page with a short snippet of surrounding text. |

**How you actually use this:** before writing a gold label, run `find_pages.py`
with a phrase from the fact you're asking about. It tells you every page that
states it — which is exactly what should go in `gold`. It's also how you
confirm an "unanswerable" question really has no match anywhere (no match is
evidence, not proof — try a couple of phrasings).

---

## `eval/metrics.py` — the scoring logic (free, offline, pure functions)

A **gold chunk** is a retrieved chunk whose `(source, page)` appears in the
question's `gold`. `Page` is just a type alias for `tuple[str, int]`.

| Function | Takes in | Gives back | What it does |
|---|---|---|---|
| `gold_pages(question)` | one question dict | a `set[Page]` | Flattens the question's `gold` list (which groups pages by source) into one flat set of `(source, page)` pairs. |
| `hit_at_k(chunks, gold)` | the chunks `retrieve()` returned, the gold set | `bool` | True if *any* retrieved chunk sits on a gold page. Answers: did retrieval find the answer at all? |
| `mrr(chunks, gold)` | same | `float` (0.0–1.0) | `1 / rank` of the *first* gold chunk in the list (rank 1 = top result), or `0.0` if none match. Unlike `hit_at_k`, this rewards finding the right chunk near the top, not just somewhere in the top-k. |
| `page_recall(chunks, gold)` | same | `float` (0.0–1.0) | Share of the *distinct* gold pages that were retrieved. Matters for multi-page answers: finding 1 of 3 needed pages scores 0.33 here even though `hit_at_k` would say `True`. |
| `cited_gold(sources, gold)` | the chunks the answer actually **cited** (`generate_answer()["sources"]`), the gold set | `bool` | Same check as `hit_at_k`, but over cited chunks instead of all retrieved chunks. Catches the case where the right page was retrieved but the model cited a different, wrong one — something `validate_citation()` in `app/generator.py` can't see, because it only checks that a cited number is in range. |
| `is_refusal(answer)` | the answer text | `bool` | True if the answer contains the prompt's exact refusal phrase ("I don't have enough information..."), case-insensitive, tolerant of curly vs. straight apostrophes. |
| `_mean(values)` | a list of floats | `float \| None` | Plain mean, or `None` for an empty list (so an unmeasured metric shows as "not measured", never as a misleading 0). |
| `_percentile(values, pct)` | a list of floats, a percentile 0–100 | `float \| None` | The p-th percentile, used for latency. |
| `summarize(rows)` | the list of per-question result dicts from one run | one summary dict | Builds the run-level "Scorecard": splits rows into answerable/unanswerable, drops errored rows from every average (but still counts them), and computes `hit@k`, `mrr`, `page_recall@k`, `cited_gold`, `citation_accuracy` (only over answers that cited something — refusals are 0.0 by design and would drag this down otherwise), `false_refusal_rate`, `correct_refusal_rate`, latency percentiles, plus the mean of every RAGAS score that's present. |
| `check_thresholds(summary, thresholds)` | the summary, the dict loaded from `thresholds.json` | `(failures, skipped)` | Compares each non-`null` threshold to the matching summary value. A key ending in `_max` is an upper bound (e.g. `false_refusal_rate_max`); every other key is a lower bound. If a metric wasn't measured this run (e.g. you ran `--no-ragas`), it's reported as **skipped**, not failed. |

---

## `eval/run_eval.py` — the CLI that runs everything

```
python eval/run_eval.py --validate                 # check the set only, free
python eval/run_eval.py --limit 3 --no-ragas        # cheap smoke run
python eval/run_eval.py --label baseline-1          # full paid run
```

| Function | Takes in | Gives back | What it does |
|---|---|---|---|
| `git_sha()` | nothing | `str \| None` | The current commit hash, with `+dirty` appended if there are uncommitted changes. Stored on the run so you always know exactly what code produced a given score. |
| `run_config(top_k, use_ragas, set_version)` | the top-k setting, whether RAGAS is on, the eval set's version | a dict | Snapshots every setting that could change a score — chunk size/overlap, embedding/LLM/rerank/judge models, `candidate_k`, prompt version, eval-set version — into one dict stored as `eval_runs.config`. |
| `run_question(question, top_k)` | one question dict, the top-k to retrieve | one result-row dict | The per-question engine. Calls `retrieve()` then `generate_answer()` **directly** — never the `/query` HTTP route, so eval traffic never touches `query_logs` or `/history`. Times both calls. If anything raises, the exception is caught and recorded in `error` instead of crashing the whole run (so one bad question can't lose a paid-for batch). On success it computes `reranked`, `refused`, citation stats, and — only for answerable questions — `hit`, `mrr`, `page_recall`, `cited_gold` via the `metrics.py` functions. |
| `build_ragas_metrics()` | nothing | a dict of 4 RAGAS metric objects | Constructs `Faithfulness`, `AnswerRelevancy`, `ContextPrecisionWithReference`, `ContextRecall`, all wired to one shared async OpenAI client and the judge model (`settings.eval_judge_model`, default `gpt-4o-mini`). |
| `_score_row(row, metrics, gate)` | one result row, the metric objects, a concurrency-limiting semaphore | nothing (mutates `row["ragas"]`) | Sends that question's text/answer/retrieved-text/reference to all four RAGAS metrics concurrently. If a metric call fails, its score is set to `None` and the error text is kept in `row["ragas_errors"]` rather than crashing the run. |
| `score_ragas(rows)` | the full list of result rows | nothing (mutates rows in place) | Runs `_score_row` over every answerable, non-refused, non-errored row, with concurrency capped by `RAGAS_CONCURRENCY` (4). Refusals are deliberately skipped — judging the "faithfulness" of a refusal sentence against the source text is meaningless; refusals are already scored by `correct_refusal_rate` instead. |
| `to_db_result(row)` | one scored row | a dict matching `EvalResult`'s columns | Reshapes a row: the free metrics not covered by their own column (`mrr`, `page_recall`, `cited_gold`, `refused`, `has_citation`, `retrieved_pages`, any `ragas_errors`) and every RAGAS score are merged into one `metrics` JSON blob. |
| `_fmt(value)` | any scorecard value | a display string | `None` → `"-"`; floats → 3 decimal places; everything else → `str()`. |
| `print_report(rows, summary, thresholds)` | the rows, the summary, the thresholds dict | prints to stdout | Prints the "Scorecard" (every summary metric next to its threshold, if any) and a "Review these" section listing specific questions worth reading: errors, answerable questions that missed their gold page or weren't cited correctly or were wrongly refused, and unanswerable questions that got an answer instead of a refusal. |
| `validate_or_exit(data)` | the loaded set | prints; exits 2 on error | Runs `validate_eval_set` + `index_pages`, prints the category breakdown and any warnings/errors, and exits with code 2 if the set is unusable. This is exactly what `--validate` calls — it never retrieves or generates anything. |
| `main()` | CLI args | process exit code | The entry point: parses flags → loads and validates the set (stops here if `--validate`) → opens the `eval_runs` row *before* spending any credits (so a database outage fails cheaply) → loops over questions, sleeping `settings.eval_rerank_delay_s` between them to stay under a trial Cohere key's rate limit → optionally runs RAGAS → aggregates, stores, prints → exits 1 if a threshold failed, any question errored, or Cohere ever fell back to RRF order (which the summary flags as `rerank_fallbacks`, since a fallback silently changes ranking and makes the run not comparable to others). |

**Module constant:** `RAGAS_CONCURRENCY = 4` — how many questions' RAGAS
scoring runs at once (each one fires up to 4 judge calls).

---

## `eval/thresholds.json` — the pass/fail gate

Not code. One key per metric name from the summary, each either a number or
`null`. `null` means "measured, but not enforced yet" — that's the state of
every key right now, before the baseline exists. A key suffixed `_max` (e.g.
`false_refusal_rate_max`) is read as an upper bound by `check_thresholds`;
every other key is a lower bound.

---

## `app/db.py` — the 3 functions that write eval results to Postgres

(These live in `app/db.py`, not `eval/`, because all SQL stays in that
module — but the runner calls them directly, so they're part of this flow.)

| Function | Takes in | Gives back | What it does |
|---|---|---|---|
| `create_eval_run(label, git_sha, config, eval_set_size)` | the CLI `--label`, the commit sha, the config dict, the question count | the new row's `id` (`int`) | Opens one `eval_runs` row with `summary` left `NULL`. Called *before* any question runs, so a crashed run still leaves a visible row with no summary — proof it started but didn't finish. |
| `add_eval_results(run_id, results)` | the run's id, a list of dicts from `to_db_result` | nothing | Inserts one `eval_results` row per question, all in a single transaction (rolled back entirely if any insert fails). |
| `finish_eval_run(run_id, summary)` | the run's id, the summary dict from `summarize()` | nothing (raises `ValueError` if the run doesn't exist) | Attaches the final aggregate scores to the run once every question is done. |

**Where results land — `EvalRun` columns:** `id`, `created_at`, `label`,
`git_sha`, `config` (JSON), `eval_set_size`, `summary` (JSON, `NULL` until
`finish_eval_run`).

**`EvalResult` columns:** `id`, `run_id` (FK, cascade-deletes with its run),
`question_id`, `question`, `answer`, `retrieved_ids` (JSON), `contexts` (JSON
— the chunk texts the judge scored against), `reranked`, `gold_hit`,
`citation_accuracy`, `metrics` (JSON — everything else: RAGAS scores plus the
free metrics), `latency_ms`.

---

## Process map

### `--validate` (free, no retrieval, no generation)

```
run_eval.py main()
  → dataset.load_json(EVAL_SET_PATH)
  → run_eval.validate_or_exit(data)
      → dataset.index_pages()          # read Chroma: what pages exist
      → dataset.validate_eval_set(data, pages)
          → dataset._check_question()  # per question
      → dataset.category_counts(data)  # for the printed line
      → print results; sys.exit(2) if any error
```

### A real run — one question's full path

```
run_eval.py main()
  → dataset.load_json + validate_or_exit        (as above, must pass)
  → app.db.create_eval_run(...)                  → eval_runs row opened, id returned
  → for each question:
      → run_eval.run_question(question, top_k)
          → app.retriever.retrieve(question, top_k)        → chunks
          → app.generator.generate_answer(question, chunks) → answer + citation_report
          → metrics.gold_pages(question)
          → metrics.hit_at_k / mrr / page_recall / cited_gold / is_refusal
          → returns one result row
  → if RAGAS enabled:
      → run_eval.score_ragas(rows)
          → run_eval.build_ragas_metrics()          → the 4 judge metric objects
          → run_eval._score_row(row, metrics, gate)  → per row, 4 concurrent judge calls
  → metrics.summarize(rows)                          → the run-level summary dict
  → app.db.add_eval_results(run_id, [run_eval.to_db_result(r) for r in rows])
  → app.db.finish_eval_run(run_id, summary)
  → run_eval.print_report(rows, summary, thresholds)
  → metrics.check_thresholds(summary, thresholds)    → failures, skipped
  → sys.exit(1) if any failure, error, or rerank fallback; else exit 0
```

This is the same flow as the "Data flow: an eval run" diagram in
`docs/architecture.md`, at function-call detail instead of a high-level
sequence diagram.
