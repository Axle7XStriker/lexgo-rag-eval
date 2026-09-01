# Blog post outline — lexgo-rag-eval

**Working title (pick at draft time):**
- *Does RAG plumbing actually matter? A 100-Q&A eval across 4 pipelines*
- *I built 4 RAG pipelines and graded them against 100 hand-written Q&As*
- *How much does chunking, hybrid search, and reranking each move the needle?*

**Target length:** 1500–2500 words.
**Audience:** layered — skimmers (hiring managers, general tech readers) exit after §1; engineers read through §7.
**Voice:** first person, plainspoken. Show the thinking, not just the result. No hedging, no throat-clearing.
**Anchor thesis (one line to earn early):** *The P1→P4 accuracy delta is the story — and I built the eval so I could trust the number.*

**Do NOT include (scope guards for the writing itself):**
- Per-course accuracy claims (n=50 per course is too small for that)
- Tutorial-style setup walkthroughs — link the repo, don't retype it
- LLM-generation of Q&As anywhere in the narrative (it's prohibited; don't muddy the story)
- Speculative "future work" beyond one paragraph
- Any pipeline beyond the 4 named

---

## §1 — TL;DR + results (150–200 words) [skim exit]

**Job:** deliver the whole story to a 30-second reader.

- 2-sentence thesis: what I built, what I measured, headline delta.
- Full results table (accuracy, citation precision, cost/query, P95 latency) — same shape as README.
- One-line "here's what surprised me" hook to pull down-page readers.
- Links: repo, live demo.

---

## §2 — The question this eval answers (150–200 words)

**Job:** frame the problem so the rest of the post is inevitable.

- The RAG discourse is full of "add reranking, use hybrid search" advice with no numbers behind it.
- I wanted to know, on my own corpus, how much each layer *actually* moves accuracy — and whether the effort of building each layer pays back.
- Framed as 4 pipelines that each isolate one variable. That's the whole design.

---

## §3 — Corpus + why these two courses (200–250 words)

**Job:** justify the corpus so the results generalize (or don't) honestly.

- MIT 6.006 (Algorithms) + MIT 6.830 (Databases) — 3 sources per course, 6 total.
- Why academic materials: dense, unambiguous ground truth; questions with defensible right answers.
- Why *two* courses: cross-source synthesis questions become possible; single-course RAG evals miss the hardest failure mode.
- Why *these* two: one algorithmic (proofs, complexity), one systems (papers + lectures) — different reasoning styles under the same eval harness.
- Honest caveat: this is not a benchmark for general-domain RAG. It's a benchmark for RAG over technical educational content.

---

## §4 — The 4 pipelines and what each isolates (300–400 words)

**Job:** make the experimental design legible. This is the "did you actually run a real experiment" section.

- Small matrix table (chunking / retrieval / rerank / purpose) — same as CLAUDE.md.
- One paragraph per pipeline explaining *what hypothesis it tests*, not just *what it does*:
  - **P1 baseline** — floor. Fixed chunks + dense retrieval. If this is already good enough, everything else is theater.
  - **P2 semantic chunks** — does chunk boundary quality matter more than retrieval math?
  - **P3 hybrid** — does adding BM25/FTS to dense buy you meaningful recall on questions where wording ≠ source?
  - **P4 hybrid + rerank** — does a rerank stage on top of hybrid recover precision without giving back the recall gain?
- Emphasize: **same 100 Q&As, same generation prompt, only retrieval changes.** That's the whole point.

---

## §5 — The golden set: the load-bearing artifact (300–400 words)

**Job:** convince the reader the numbers mean something. This is the section that separates this post from every "I built a RAG demo" post.

- 100 Q&As, hand-authored (no LLM generation — and *why* that matters).
- Distribution rationale: 40 factual / 25 cross-source synthesis / 20 semantic paraphrase / 10 out-of-corpus / 5 adversarial.
- Especially call out the **10 out-of-corpus** — measures over-answering, which almost no public RAG eval bothers with.
- Every Q&A has a gold citation, not just a gold answer. That's what makes citation precision measurable.
- Time cost: honest number (8–10 hrs of the 26-hr budget). This section earns respect by admitting it.
- One-line lesson: *the eval is only as good as the golden set, and the golden set is only as good as the person willing to sit and write it.*

---

## §6 — Judge design + reproducibility (200–300 words)

**Job:** close the "how do I know you're not cooking the numbers" loop.

- LLM-as-judge with Claude Sonnet. Fixed prompt, versioned in the repo.
- Judge gets: question, gold answer, gold citations, model answer, model citations → binary correct/incorrect.
- Why binary (not 1–5): rubric drift is the enemy of run-to-run comparability.
- What I did to check the judge: [spot-check N Q&As manually, report agreement rate].
- Every eval run captures: git SHA + prompt version + config → results are re-runnable.
- Every LLM call logged with tokens + cost.

---

## §7 — What the numbers say (and what they don't) (300–400 words) [payoff]

**Job:** deliver the actual insight. This is why anyone kept reading.

- Restate the results table with commentary per pipeline:
  - P1→P2 delta: what did semantic chunking buy? [surprise or non-surprise]
  - P1→P3 delta: was hybrid worth the plumbing? [named answer]
  - P3→P4 delta: how much did reranking recover?
  - Headline: P1→P4 total delta, cost delta, latency delta.
- Break-out by question type: where does the pipeline win, where does it lose? (Especially: does P4 correctly refuse the 10 out-of-corpus questions, or does it hallucinate?)
- The 3 sharpest specific findings (fill after eval; e.g. "reranking bought +8pp on synthesis but 0 on factual", "hybrid helped paraphrase questions most").
- What the numbers **don't** say: no per-course claim (n too small), no generalization beyond this corpus type.

---

## §8 — What I'd change with more time (150–200 words)

**Job:** show calibration. What I *chose not to do* is as informative as what I did.

- Bigger golden set (300+) for per-question-type confidence intervals.
- Add a fine-tuned reranker to compare against Cohere Rerank 3.
- Multi-turn / follow-up questions — currently single-shot only.
- Chunk-size sweep (I fixed 500/50 as a constraint; didn't sweep it as a variable).
- One-sentence honest cost of building this: 26 hrs, ~$X in API spend.

---

## §9 — Repo + demo (50–100 words)

**Job:** convert the reader to a repo visitor.

- Repo link (README leads with the same numbers table).
- Live demo link (Streamlit).
- Note: everything reproducible — clone, add API keys, `make eval`.
- Invite: eval methodology feedback welcome.

---

## Post-draft checklist (before publish)

- [ ] Numbers in §1 and §7 match the README table exactly
- [ ] Every claim in §7 traceable to a run captured in the repo
- [ ] Judge prompt version cited in §6 matches the one in `prompts/`
- [ ] No LLM-generated Q&As mentioned anywhere as a shortcut
- [ ] Word count 1500–2500
- [ ] Demo URL live at time of publish
- [ ] Retro notes (2–3 lines) added to README
