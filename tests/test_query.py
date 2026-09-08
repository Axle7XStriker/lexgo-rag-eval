"""P1 query pipeline tests. Fully offline — fake embedder, store, generator.

Covers: end-to-end shape, citation dedup + first-mention order, out-of-range
marker handling, out-of-corpus prompt path, empty-retrieval short-circuit,
context block format.

Tests create a small prompt file rather than use the production prompt, so
they exercise `_load_prompt` without coupling pipeline behaviour to any
production prompt's wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.pipeline import query as query_module
from src.pipeline.chunk import PIPELINE_TAG
from src.pipeline.generate import GenerateResult
from src.pipeline.query import (
    OUT_OF_CORPUS_SENTINEL,
    PROMPT_VERSION,
    _format_context,
    _parse_citations,
    answer_question,
)
from src.pipeline.store import RetrievedChunk


@pytest.fixture(autouse=True)
def prompt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a minimal prompt file and point the real loader at its directory."""
    prompt_path = tmp_path / "prompts" / "answer" / f"{PROMPT_VERSION}.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        """---
name: test-answer
version: 1
---

# System

SYSTEM: sentinel is '{out_of_corpus_sentinel}'.

# User template

Q: {question}
CTX:
{context}
""",
        encoding="utf-8",
    )
    query_module._load_prompt.cache_clear()
    monkeypatch.setattr(query_module, "PROMPTS_DIR", tmp_path / "prompts")
    yield
    query_module._load_prompt.cache_clear()


# ── Fake dependencies ────────────────────────────────────────────────


@dataclass
class _FakeEmbedder:
    """Duck-types VoyageEmbedder — only .embed_query is called by the pipeline."""

    vector: list[float] = field(default_factory=lambda: [0.1] * 4)
    calls: list[str] = field(default_factory=list)

    def embed_query(self, text: str, *, run_id: str | None = None) -> list[float]:
        self.calls.append(text)
        return self.vector


@dataclass
class _FakeStore:
    """Duck-types VectorStore — only .dense_search is called by the pipeline."""

    to_return: list[RetrievedChunk] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def dense_search(
        self, pipeline: str, query_embedding: list[float], k: int
    ) -> list[RetrievedChunk]:
        self.calls.append({"pipeline": pipeline, "k": k, "embedding_len": len(query_embedding)})
        return list(self.to_return)


@dataclass
class _FakeGenerator:
    """Duck-types ClaudeGenerator — only .generate is called."""

    reply_text: str = "generic answer"
    input_tokens: int = 100
    output_tokens: int = 20
    cost_usd: float = 0.001
    calls: list[dict] = field(default_factory=list)

    def generate(
        self,
        *,
        system: str,
        user: str,
        prompt_version: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        run_id: str | None = None,
    ) -> GenerateResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "prompt_version": prompt_version,
                "run_id": run_id,
            }
        )
        return GenerateResult(
            text=self.reply_text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )


def _chunk(marker: int, doc_path: str, source_id: str = "A1") -> RetrievedChunk:
    """Build a RetrievedChunk with unique-per-marker text so we can spot it in prompts.

    `doc_path` here is a synthetic fixture, NOT validated against the real
    corpus manifest — pipeline tests are decoupled from corpus content by
    design (same principle as the prompt fixture above). Manifest
    integrity is tested by test_ingest.py + test_fetch_corpus.py.
    """
    return RetrievedChunk(
        chunk_id=1000 + marker,
        document_id=1,
        doc_path=doc_path,
        source_id=source_id,
        pipeline=PIPELINE_TAG,
        chunk_index=marker - 1,
        text=f"chunk-{marker}-body",
        page_start=marker,
        page_end=marker,
        score=0.9 - 0.01 * marker,
    )


# ── End-to-end ────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_happy_path(self, tmp_path: Path) -> None:
        """Returns model's answer verbatim; cited chunks parsed in first-mention order."""
        chunks = [
            _chunk(1, "fixture/A1_doc03.pdf", "A1"),
            _chunk(2, "fixture/A2_doc03.pdf", "A2"),
            _chunk(3, "fixture/A3_doc01.pdf", "A3"),
        ]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        # Generator cites [1] then [3] — assert both make it into citations,
        # in that order, with the right doc_path provenance.
        generator = _FakeGenerator(
            reply_text="The answer is X [1] because of Y [3].",
            input_tokens=500,
            output_tokens=25,
            cost_usd=0.0018,
        )

        result = answer_question(
            query="what is merge sort?",
            embedder=embedder,
            store=store,
            generator=generator,
            run_id="run_test",
        )

        # Answer text is returned verbatim.
        assert result.answer == "The answer is X [1] because of Y [3]."
        assert result.tokens_input == 500
        assert result.tokens_output == 25
        assert result.cost_usd == 0.0018
        assert result.prompt_version == PROMPT_VERSION
        assert result.query == "what is merge sort?"

        # Full retrieval preserved for UI + eval.
        assert result.retrieved_chunks == chunks

        # Citations parsed, deduped, first-mention order.
        assert [c.marker for c in result.citations] == [1, 3]
        assert result.citations[0].doc_path == "fixture/A1_doc03.pdf"
        assert result.citations[1].doc_path == "fixture/A3_doc01.pdf"
        assert result.citations[0].source_id == "A1"

        # Dependencies invoked with the query + run_id threaded through.
        assert embedder.calls == ["what is merge sort?"]
        assert len(store.calls) == 1
        assert store.calls[0]["pipeline"] == PIPELINE_TAG
        assert len(generator.calls) == 1
        assert generator.calls[0]["prompt_version"] == PROMPT_VERSION
        assert generator.calls[0]["run_id"] == "run_test"

    def test_generator_receives_substituted_prompt(self, tmp_path: Path) -> None:
        """{question} and {context} placeholders are filled before the generator sees the prompt."""
        chunks = [
            _chunk(1, "fixture/A1_doc01.pdf", "A1"),
            _chunk(2, "fixture/B1_doc02.pdf", "B1"),
        ]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="ok [1]")

        answer_question(
            query="what is merge sort?",
            embedder=embedder,
            store=store,
            generator=generator,
        )

        call = generator.calls[0]
        # Canned template is passed through verbatim as system, and the
        # out-of-corpus sentinel must be reachable to Claude via the system
        # prompt — otherwise a P1 "not in corpus" answer would be impossible.
        assert call["system"] == f"SYSTEM: sentinel is '{OUT_OF_CORPUS_SENTINEL}'."
        assert OUT_OF_CORPUS_SENTINEL in call["system"]
        # Question text made it into the substituted user prompt.
        assert "what is merge sort?" in call["user"]
        # Both enumerated chunk headers appear, in order.
        assert "[1] A1 fixture/A1_doc01.pdf" in call["user"]
        assert "[2] B1 fixture/B1_doc02.pdf" in call["user"]
        # And the chunk bodies.
        assert "chunk-1-body" in call["user"]
        assert "chunk-2-body" in call["user"]


# ── Citation parsing ──────────────────────────────────────────────────


class TestCitationParsing:
    def test_dedup_first_mention_order(self) -> None:
        """Repeated `[N]` markers dedup; overall order = first-appearance order."""
        chunks = [_chunk(i, f"fixture/A1_doc{i:02d}.pdf") for i in (1, 2, 3)]
        cits = _parse_citations("Foo [3] bar [1] baz [3][1][2] end.", chunks)
        assert [c.marker for c in cits] == [3, 1, 2]

    def test_out_of_range_dropped(self) -> None:
        """`[N]` past top-k is dropped (not raised) — no invented citations in the output."""
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        cits = _parse_citations("Ok [1] and also [9].", chunks)
        assert [c.marker for c in cits] == [1]

    def test_zero_and_negative_ignored(self) -> None:
        """[0] out of range (1-indexed), [-N] doesn't match \\d+ — both ignored."""
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        cits = _parse_citations("Bad [0] good [1] weird [-2].", chunks)
        assert [c.marker for c in cits] == [1]

    def test_no_markers_returns_empty(self) -> None:
        """An answer with no `[N]` brackets produces zero citations, not an error."""
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        cits = _parse_citations("No brackets here at all.", chunks)
        assert cits == []

    def test_all_out_of_range_returns_empty(self, tmp_path: Path) -> None:
        """End-to-end: only-invented markers → empty citations; answer text round-trips."""
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="Something [9] and [42].")
        result = answer_question(query="q", embedder=embedder, store=store, generator=generator)
        assert result.answer == "Something [9] and [42]."
        assert result.citations == []


# ── Out-of-corpus paths ───────────────────────────────────────────────


class TestOutOfCorpus:
    def test_sentinel_text_preserved(self, tmp_path: Path) -> None:
        """When Claude returns the sentinel, pipeline passes it through (generator ran)."""
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text=OUT_OF_CORPUS_SENTINEL)
        result = answer_question(query="q", embedder=embedder, store=store, generator=generator)
        assert result.answer == OUT_OF_CORPUS_SENTINEL
        assert result.citations == []
        assert len(generator.calls) == 1

    def test_empty_retrieval_short_circuits(self, tmp_path: Path) -> None:
        """No chunks retrieved → skip generator entirely, return sentinel + zero cost."""
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator(reply_text="should not be called")

        result = answer_question(query="q", embedder=embedder, store=store, generator=generator)
        assert result.answer == OUT_OF_CORPUS_SENTINEL
        assert result.citations == []
        assert result.retrieved_chunks == []
        assert result.tokens_input == 0
        assert result.tokens_output == 0
        assert result.cost_usd == 0.0
        # The generator MUST NOT be called — that's the whole point.
        assert generator.calls == []
        # Embedder + store both ran (retrieval was attempted).
        assert embedder.calls == ["q"]
        assert len(store.calls) == 1


# ── Context formatter ────────────────────────────────────────────────


class TestFormatContext:
    def test_shape(self) -> None:
        """Each chunk renders `[N] source_id doc_path (pages P-Q ...)` header + body, blank line."""
        chunks = [
            _chunk(1, "fixture/A1_doc01.pdf", "A1"),
            _chunk(2, "fixture/B1_doc02.pdf", "B1"),
        ]
        out = _format_context(chunks)
        assert "[1] A1 fixture/A1_doc01.pdf" in out
        assert "[2] B1 fixture/B1_doc02.pdf" in out
        # Page range rendered in the header (chunk 1 → page_start=1, page_end=1).
        assert "pages 1–1" in out
        assert "chunk-1-body" in out
        assert "chunk-2-body" in out
        # Blank line between chunks.
        assert "\n\n" in out
