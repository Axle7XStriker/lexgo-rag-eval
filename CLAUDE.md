# lexgo-rag-eval — Project Context

**What this file is:** project-level context for Claude Code sessions in this repo. Auto-loaded when Claude works here.

**Last updated:** 2026-10-15

---

## What this project is

A rigorously-evaluated multi-source RAG system over academic course materials, with measured tradeoffs across chunking, retrieval, and reranking strategies.

---

## Scope

### In

- Corpus: MIT 6.006 (Algorithms) A1..A4 and MIT 6.830 (Databases) B1..B5
- 4 retrieval pipelines (matrix below), single answer-generation prompt held constant
- 100 hand-curated Q&As with source citations
- Metrics: answer accuracy, citation precision, retrieval recall@5
- Streamlit demo — query → answer + citations + retrieved chunks
- 1500-2500 word blog post on eval design + numbers

### Out (protect ruthlessly)

- Frontend polish beyond default Streamlit
- Authentication / user accounts
- Multi-turn chat / chat history
- Streaming (nice-to-have; cut if it eats budget)
- Fine-tuning
- More than one answer-generation prompt or 4 retrieval pipelines
- Concept extraction / concept map
- More than 100 Q&As, or per-course accuracy claims (would need 50+ per course)

### Scope-creep tripwires

If any of the following comes up, STOP and flag before proceeding:

- "Let me also add [feature not in scope]"
- Adding auth, user accounts, or personalization
- Extending to multi-turn conversation
- Building a more elaborate frontend
- Adding more pipelines beyond the 4 named
- Suggesting LLM-generated Q&As "to save time"

---

## Corpus

**Course A — MIT 6.006 (Intro Algorithms):**
- **A1:** Lecture notes — full F11 set, lectures 1-24
- **A2:** Recitation notes — full F11 set, recitations 1-24
- **A3:** Problem sets with solutions (PS1-PS7)
- **A4:** CLRS 3rd ed textbook. Copyrighted; not distributed from this repo. Place a legal PDF at `corpus/6.006/textbook/A4_clrs_3ed.pdf` to participate in retrieval.

**Course B — MIT 6.830 (Database Systems):**
- **B1:** Lecture notes (bundled — they synthesize the papers)
- **B2:** Quizzes + solutions (OCW-hosted Fall 2010 quiz 1 & quiz 2, both with solutions)
- **B3:** Papers bundle — 5 papers spanning query processing, transactions, concurrency, and column stores (Selinger, Franklin, Kung/Robinson, Gray, C-Store)
- **B4:** Red Book 4th ed (Hellerstein/Stonebraker). Copyrighted; not distributed. Place a legal PDF at `corpus/6.830/textbook/B4_red_book_4ed.pdf` to participate in retrieval.
- **B5:** Ramakrishnan/Gehrke DMS 3rd ed — fetched from a user-supplied URL.

---

## Retrieval pipeline matrix

All 4 pipelines use the **same 100 Q&As** and the **same generation prompt**. Only the retrieval pipeline changes.

| Pipeline | Chunking | Retrieval | Rerank | Purpose |
|---|---|---|---|---|
| **P1 baseline** | Fixed | Dense (Voyage) | — | Establish floor |
| **P2 semantic chunks** | Semantic split | Dense | — | Isolate chunking effect |
| **P3 hybrid** | Fixed | BM25 (Postgres FTS) + dense (RRF fusion) | — | Isolate retrieval fusion effect |
| **P4 hybrid + rerank** | Fixed | Hybrid | Cohere Rerank 3 | Best case |

---

## Golden set (100 Q&As) — distribution

- **40** factual recall (answer explicit in one source)
- **25** cross-source synthesis (answer requires 2+ sources)
- **20** semantic paraphrase (question wording ≠ source wording)
- **10** out-of-corpus (correct answer is "not in the materials" — measures over-answering)
- **5** edge / adversarial (multi-part, ambiguous phrasing, low-recall)

Each Q&A: **question**, **gold answer** (2-4 sentences), **gold citation** (source + page/section reference).

**Budget: 8-10 hrs — distinct sub-task.** LLM-generated Q&As are **prohibited** — the golden set is a load-bearing artifact and its integrity IS the credibility of the whole eval.

---

## Target numbers [anchors — revise after P1 baseline]

| Metric | Baseline (P1) | Best (P4) | Delta target |
|---|---|---|---|
| Answer accuracy | 55-65% | 80-90% | **+20pp minimum** |
| Citation precision | ~60% | ~85% | +25pp |
| Cost per query | <$0.02 | <$0.02 | constraint |
| P95 latency | <3s | <3s | constraint |

**Sanity gates:**
- If P1 baseline > 75% → golden set is too easy → rebuild with harder cross-source questions
- If P4 < 70% → pipeline is broken → debug before writing

---

## Definition of accuracy

LLM-as-judge using Claude Haiku. Given (question, gold answer, gold citations, model answer, model citations) → binary correct/incorrect per Q&A. Reported as % correct.

---

## Tech stack (assume unless a decision says otherwise)

- **Python 3.12** — `uv` for env + package management
- **Embeddings:** Voyage `voyage-3-large`
- **Chat model:** Anthropic Claude Haiku 5.5
- **Judge model:** Claude Haiku 5.5 (same model, different prompt)
- **Vector store:** Postgres + pgvector
- **Full-text search:** Postgres FTS (for BM25-alike in hybrid)
- **Reranker:** Cohere Rerank 3
- **Frontend:** Streamlit only
- **Deploy target:** HuggingFace Spaces or Railway for the demo

---

## Definition of done

- [ ] Repo public with README leading with numbers table
- [ ] Blog post published (public URL, linked from README)
- [ ] Live Streamlit demo URL works and is linked from README
- [ ] Retro notes (2-3 lines) on what worked / didn't

---

## Working conventions

- **Every commit small and reviewable.**
- **Every LLM call logged.** model, input tokens, output tokens, estimated cost — to disk. Cost visibility is a story element for the blog post.
- **Prompts under version control** — `prompts/` directory with dated versions once code lands.
- **Eval reproducibility** — every run captures git SHA + prompt version + config. Numbers are meaningless without provenance.
- **No mocked LLM calls in eval runs.** Mocks are OK for pipeline structure tests only; numbers are only real if they came from real API calls.
- **Kill early.** If a direction isn't producing signal within 2 weeks, escalate — don't burn remaining time hoping.

---

## Behavior expectations for Claude sessions

- If asked to add something outside "In scope," flag scope creep and confirm before doing.
- If asked to fluff up the frontend (Next.js, styling, auth), refer back to the "Streamlit only" constraint.
- If asked to speed up golden-set curation via LLM generation, refuse — the golden set's integrity is load-bearing.
- Prefer running actual evals over theorizing about them.
