# lexgo-rag-eval — Project Context

**What this file is:** project-level context for Claude Code sessions in this repo. Auto-loaded when Claude works here.

**Last updated:** 2026-08-21

---

## What this project is

A rigorously-evaluated multi-source RAG system over academic course materials, with measured tradeoffs across chunking, retrieval, and reranking strategies.

**Timebox:** ~26 hrs of build time over 4 weeks. Non-negotiable.

**Primary success metric:** answer accuracy on a 100-Q&A golden set — the **delta between baseline (V1) and best pipeline (V4) is the story.**

---

## Scope

### In

- Corpus: MIT 6.006 (Algorithms) = A1..A4 (A4 = CLRS textbook, reference-only). MIT 6.830 (Databases) = B1..B5 (B4 = Red Book, reference-only; B5 = DMS textbook, user-supplied URL, optional).
- 4 retrieval variants (matrix below), single answer-generation prompt held constant
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
- More than one answer-generation prompt or 4 retrieval variants
- Concept extraction / concept map
- More than 100 Q&As, or per-course accuracy claims (would need 50+ per course)

### Scope-creep tripwires

If any of the following comes up, STOP and flag before proceeding:

- "Let me also add [feature not in scope]"
- Adding auth, user accounts, or personalization
- Extending to multi-turn conversation
- Building a more elaborate frontend
- Adding more variants beyond the 4 named
- Suggesting LLM-generated Q&As "to save time"

---

## Corpus

**Course A — MIT 6.006 (Intro Algorithms):**
- **A1:** Lecture notes — full F11 set, lectures 1-24
- **A2:** Recitation notes — full F11 set, recitations 1-24 (rec03/rec04 + rec13-24 optional pending first fetch)
- **A3:** Problem sets with solutions (PS1-PS4)
- **A4:** CLRS 3rd ed textbook. Copyrighted MIT Press title, not distributed from this repo. Manifest entry is `optional=True` — the fetcher attempts the publisher URL, fails the `%PDF` magic check, and reports `missing_optional` without breaking CI. Manually place a legal PDF at `corpus/6.006/textbook/A4_clrs_3ed.pdf` to participate in retrieval.

**Course B — MIT 6.830 (Database Systems):**
- **B1:** Lecture notes (bundled — they synthesize the papers)
- **B2:** Quizzes + solutions (OCW-hosted Fall 2010 quiz 1 & quiz 2, both with solutions)
- **B3:** Papers bundle — 5 papers spanning query processing, transactions, concurrency, and column stores (Selinger, Franklin, Kung/Robinson, Gray, C-Store)
- **B4:** Red Book 4th ed (Hellerstein/Stonebraker). Copyrighted MIT Press title, not distributed. Same `optional=True` treatment and manual-placement contract as A4.
- **B5:** Ramakrishnan/Gehrke DMS 3rd ed — fetched from a user-supplied URL (see manifest for legal-provenance note). Marked `optional` so a takedown never breaks CI.

**Note:** OCW 6.830 has no recitations — no B-bucket for them. (Do not invent one.)

**Textbook semantics:** A4 and B4 are `optional=True` entries pointing to publisher pages that don't serve PDF bytes. The fetcher attempts them, fast-fails the `%PDF` magic check, and marks them `missing_optional`. Downstream chunking/retrieval skips missing files. If a human legally obtains a copy and places the PDF at `dest_path`, the standard `dest.exists()` short-circuit picks it up on the next run — no code changes needed for retrieval participation.

**Q&A authoring strategy for 6.830:**
- **Lectures first (B1)** — they simplify the papers and anchor most factual/paraphrase Q&As.
- **Quizzes (B2)** — mine for factual-recall Q&As grounded in the *solutions* (never treat the quiz question as a golden Q&A verbatim — it must be re-authored so the golden Q wording differs from the corpus wording, or it becomes a trivial retrieval test). Great for cross-source synthesis: "the quiz tests X; how does the lecture explain it / the paper deepen it."
- **Papers (B3)** — primarily for cross-source synthesis ("how does this paper's approach to X compare to the lecture recommendation?").

Design 6.006 Q&As around concepts and complexity claims, not "what does line 47 do."

**Format verified 2026-08-15:** OCW PDFs copy-paste cleanly into text editor.

---

## Retrieval variant matrix

All 4 variants use the **same 100 Q&As** and the **same generation prompt**. Only the retrieval pipeline changes.

| Variant | Chunking | Retrieval | Rerank | Purpose |
|---|---|---|---|---|
| **V1 baseline** | Fixed 500 tokens, 50 overlap | Dense (Voyage) top-10 | — | Establish floor |
| **V2 semantic chunks** | Semantic split | Dense top-10 | — | Isolate chunking effect |
| **V3 hybrid** | Fixed 500/50 | BM25 (Postgres FTS) + dense (RRF fusion), top-10 | — | Isolate retrieval fusion effect |
| **V4 hybrid + rerank** | Fixed 500/50 | Hybrid top-20 | Cohere Rerank 3 → top-5 | Best case |

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

## Target numbers [anchors — revise after V1 baseline]

| Metric | Baseline (V1) | Best (V4) | Delta target |
|---|---|---|---|
| Answer accuracy | 55-65% | 80-90% | **+20pp minimum** |
| Citation precision | ~60% | ~85% | +25pp |
| Cost per query | <$0.02 | <$0.02 | constraint |
| P95 latency | <3s | <3s | constraint |

**Sanity gates:**
- If V1 baseline > 75% → golden set is too easy → rebuild with harder cross-source questions
- If V4 < 70% → pipeline is broken → debug before writing

---

## Definition of accuracy

LLM-as-judge using Claude Sonnet. Given (question, gold answer, gold citations, model answer, model citations) → binary correct/incorrect per Q&A. Reported as % correct.

---

## Tech stack (assume unless a decision says otherwise)

- **Python 3.12** — `uv` for env + package management
- **Embeddings:** Voyage `voyage-3-large`
- **Chat model:** Anthropic Claude Sonnet 4.6
- **Judge model:** Claude Sonnet 4.6 (same model, different prompt)
- **Vector store:** Postgres + pgvector
- **Full-text search:** Postgres FTS (for BM25-alike in hybrid)
- **Reranker:** Cohere Rerank 3
- **Frontend:** Streamlit only
- **Deploy target:** HuggingFace Spaces or Railway for the demo

---

## 4-week schedule (26 hrs total)

| Week | Hrs | Work |
|---|---|---|
| **1** | 6.5 | Scoping locked. README skeleton + blog outline (2 hrs) ← **BEFORE any code**. Repo + Streamlit skeleton + Voyage/Cohere API keys (1 hr). Start golden set — target 20-30 Q&As (3-3.5 hrs). |
| **2** | 6.5 | Finish golden set — 70-80 more Q&As (5-6 hrs). **Hard checkpoint:** if incomplete by end of W2, cut V2 or V3 from the variants matrix. |
| **3** | 6.5 | V1 baseline pipeline + eval loop (3 hrs). V2/V3/V4 variants (3.5 hrs — each ~1 hr once infra reused). Sanity gate check after V1. |
| **4** | 6.5 | Streamlit demo polish (1 hr). Deploy demo (1 hr). Blog post draft (3 hrs). README final + repo cleanup (1.5 hrs). |

---

## Risks & mitigations

1. **Golden set curation blows the budget** *(most likely failure mode)*. W1 checkpoint = 20-30 done; W2 hard checkpoint = 100 done. Cut a variant before cutting golden-set quality.
2. **Baseline too good or too bad kills the story.** Sanity-check numbers after V1 baseline before building V2-V4.
3. **Writeup debt.** README skeleton + blog outline land W1 BEFORE any code. Budget 8 of 26 hrs on writing, front-loaded.
4. **Cohere Rerank cost.** ~2000 rerank calls × $0.001 ≈ $2. Not a real risk, noted for completeness.
5. **6.830 papers dense — Q&A authoring slower than lectures.** Get most Q&As from lectures; use papers only for cross-source synthesis.

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
- **Kill early.** If a direction isn't producing signal by end of W2, escalate — don't burn W3-4 hoping.

---

## Behavior expectations for Claude sessions

- If asked to add something outside "In scope," flag scope creep and confirm before doing.
- If asked to fluff up the frontend (Next.js, styling, auth), refer back to the "Streamlit only" constraint.
- If asked to speed up golden-set curation via LLM generation, refuse — the golden set's integrity is load-bearing.
- Prefer running actual evals over theorizing about them.
- When scoping variants, remember the 26-hr timebox. If a proposed addition costs >2 hrs, name what gets cut to accommodate.
