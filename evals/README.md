# Evals

The 100-Q&A golden set, eval-run scripts, and captured run artifacts.

## Layout

```
evals/
  golden/
    qa.jsonl              # hand-authored Q&As (100 at spec target)
  runs/
    <run_id>/             # one dir per run, gitignored except .gitkeep
      manifest.json       # git SHA, prompt versions, config, totals, metrics
      results.jsonl       # per-Q&A: answer, citations, judge verdict, cost
      llm_calls.jsonl     # snapshot of logs/llm_calls.jsonl for this run
      summary.md          # human-readable metrics table + skipped Q&As
  metrics.py              # pure-function metric helpers + aggregate()
  run.py                  # CLI entrypoint: `python -m evals.run --pipeline p1`
  validate_golden.py      # `make validate` — golden-set linter
```

## How to run

Prerequisites:

```
make db-up          # start local Postgres + pgvector
make ingest         # populate the P1 chunks table (idempotent)
make validate       # golden-set integrity check (exit 0 on OK)
```

Then:

```
make eval           # runs `uv run python -m evals.run --pipeline p1`
```

Artifacts land under `evals/runs/eval_<UTC-timestamp>/`. Each run captures
the git SHA, prompt versions, config, totals, and metrics into
`manifest.json`; the same numbers render as tables in `summary.md`.

## Exit-code contract

`evals.run.main()` returns **0 on any successful invocation**, including
runs where individual Q&As were skipped. Skips are reported in the
"Skipped Q&As" section of `summary.md` and counted in
`manifest.json:totals.n_skipped`; the operator decides whether to
investigate. The loop returns **1 only** when it refuses to start —
today the only such condition is a malformed `qa.jsonl` (parse errors
listed to stdout). This matches the `make validate` contract: fix the
golden file before evaluating against it.

## Golden set

- 100 Q&As, hand-authored. **LLM generation is prohibited** — the golden set
  is the load-bearing artifact and its integrity IS the credibility of the
  eval. See CLAUDE.md.
- Distribution: 40 factual / 25 cross-source synthesis / 20 paraphrase /
  10 out-of-corpus / 5 adversarial.
- Each Q&A: `question`, `gold_answer` (2–4 sentences), `gold_citations`
  (source + page/section), `type` (one of the five above), `sources` (list
  of source IDs referenced).

## Run reproducibility

Every eval run captures:

- `git_sha` — the commit the pipeline ran at
- `prompt_versions` — `{answer: "v1", judge: "v1", ...}`
- `pipeline` — one of `p1_baseline`, `p2_semantic`, `p3_hybrid`, `p4_rerank`
- `config` — chunk size, top-k, rerank-k, model IDs
- `cost_usd` — sum from llm_calls.jsonl
- `metrics` — accuracy, citation precision, retrieval recall@5, p95 latency

Numbers without provenance don't count.
