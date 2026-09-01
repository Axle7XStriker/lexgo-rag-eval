# lexgo-rag-eval

> Rigorously evaluating multi-source RAG over MIT 6.006 (Algorithms) + MIT 6.830 (Database Systems) course materials. Comparing dense / semantic-chunked / hybrid / hybrid+rerank retrieval strategies against 100 hand-curated golden Q&As.

**Status:** 🚧 In progress

---

## Results

*Numbers land as evals run.*

| Pipeline | Chunking | Retrieval | Rerank | Accuracy | Citation precision | Cost/query | P95 latency |
|---|---|---|---|---|---|---|---|
| P1 baseline | Fixed 500/50 | Dense (Voyage) top-10 | — | — | — | — | — |
| P2 semantic | Semantic | Dense top-10 | — | — | — | — | — |
| P3 hybrid | Fixed 500/50 | BM25 + dense (RRF), top-10 | — | — | — | — | — |
| P4 hybrid+rerank | Fixed 500/50 | Hybrid top-20 | Cohere Rerank 3 → top-5 | — | — | — | — |

---

## System

*Diagram + component walkthrough land as the pipeline stabilizes.*

---

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Postgres+pgvector is only needed once the retrieval pipeline lands in W3.

```bash
git clone https://github.com/Axle7XStriker/lexgo-rag-eval.git
cd lexgo-rag-eval

cp .env.example .env         # then fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY

make setup                   # uv sync — creates .venv, installs pinned deps
make corpus                  # download MIT OCW PDFs into corpus/ (idempotent, ~2 min)
make app                     # streamlit run app.py — Demo + Author pages in the sidebar
make validate                # lint the golden Q&A set (evals/golden/qa.jsonl)
make eval                    # run the eval loop (W3)
```

The Streamlit app ships two pages:

- **Demo** — RAG query UI (pipeline lands W3).
- **Author** — hand-curation UI for the 100-Q&A golden set with live progress
  against the 40/25/20/10/5 distribution targets. Reads/writes
  `evals/golden/qa.jsonl` with atomic file-rewrite semantics.

`make help` lists all targets.

---

## Blog post

*Link added once the writeup is published.*

---

## Project context

Full scoping, corpus breakdown, working conventions, and schedule live in [`CLAUDE.md`](./CLAUDE.md).

---

**Author:** [Aman Bansal](https://github.com/Axle7XStriker)
