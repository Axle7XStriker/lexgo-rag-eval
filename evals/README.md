# Evals

The 100-Q&A golden set, eval-run scripts, and captured run artifacts.

## Layout (planned)

```
evals/
  golden/
    qa.jsonl              # 100 hand-authored Q&As (W1–W2 deliverable)
  runs/
    <run_id>/             # one dir per run, gitignored except .gitkeep
      manifest.json       # git SHA, prompt versions, config, variant
      results.jsonl       # per-Q&A: model answer, citations, judge verdict
      llm_calls.jsonl     # snapshot of logs/llm_calls.jsonl for this run
      summary.md          # aggregate metrics, generated
  score.py                # load golden + run → aggregate metrics (W3)
  judge.py                # LLM-as-judge wrapper (W3)
```

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
- `variant` — one of `v1_baseline`, `v2_semantic`, `v3_hybrid`, `v4_rerank`
- `config` — chunk size, top-k, rerank-k, model IDs
- `cost_usd` — sum from llm_calls.jsonl
- `metrics` — accuracy, citation precision, retrieval recall@5, p95 latency

Numbers without provenance don't count.
