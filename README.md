# lexgo-rag-eval

> Rigorously evaluating multi-source RAG over MIT 6.006 (Algorithms) + MIT 6.830 (Database Systems) course materials. Comparing dense / semantic-chunked / hybrid / hybrid+rerank retrieval strategies against 100 hand-curated golden Q&As.

**Status:** 🚧 In progress

---

## Results

*Numbers land as evals run.*

| Variant | Chunking | Retrieval | Rerank | Accuracy | Citation precision | Cost/query | P95 latency |
|---|---|---|---|---|---|---|---|
| V1 baseline | Fixed 500/50 | Dense (Voyage) top-10 | — | — | — | — | — |
| V2 semantic | Semantic | Dense top-10 | — | — | — | — | — |
| V3 hybrid | Fixed 500/50 | BM25 + dense (RRF), top-10 | — | — | — | — | — |
| V4 hybrid+rerank | Fixed 500/50 | Hybrid top-20 | Cohere Rerank 3 → top-5 | — | — | — | — |

---

## System

*Diagram + component walkthrough land as the pipeline stabilizes.*

---

## Reproduce

*Setup + run instructions land as the pipeline stabilizes.*

---

## Blog post

*Link added once the writeup is published.*

---

## Project context

Full scoping, corpus breakdown, working conventions, and schedule live in [`CLAUDE.md`](./CLAUDE.md).

---

**Author:** [Aman Bansal](https://github.com/Axle7XStriker)
