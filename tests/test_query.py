"""P1 query pipeline tests. Fully offline — fake embedder, store, generator.

Covers: end-to-end shape, citation dedup + first-mention order, out-of-range
marker handling, out-of-corpus prompt path, empty-retrieval short-circuit,
context block format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline.chunk import PIPELINE_TAG
from src.pipeline.generate import GenerateResult
from src.pipeline.query import (
    DEFAULT_TOP_K,
    OUT_OF_CORPUS_SENTINEL,
    PROMPT_VERSION,
    _format_context,
    _parse_citations,
    answer_question,
)
from src.pipeline.store import RetrievedChunk

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
    """Build a RetrievedChunk with unique-per-marker text so we can spot it in prompts."""
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
        chunks = [
            _chunk(1, "6.006/lectures/A1_lec03.pdf", "A1"),
            _chunk(2, "6.006/recitations/A2_rec03.pdf", "A2"),
            _chunk(3, "6.006/psets/A3_pset1.pdf", "A3"),
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
        assert result.citations[0].doc_path == "6.006/lectures/A1_lec03.pdf"
        assert result.citations[1].doc_path == "6.006/psets/A3_pset1.pdf"
        assert result.citations[0].source_id == "A1"

        # Dependencies invoked as expected.
        assert embedder.calls == ["what is merge sort?"]
        assert len(store.calls) == 1
        assert store.calls[0]["pipeline"] == PIPELINE_TAG
        assert store.calls[0]["k"] == DEFAULT_TOP_K
        assert len(generator.calls) == 1
        assert generator.calls[0]["prompt_version"] == PROMPT_VERSION
        assert generator.calls[0]["run_id"] == "run_test"

    def test_prompt_includes_enumerated_context(self, tmp_path: Path) -> None:
        chunks = [
            _chunk(1, "6.006/lectures/A1_lec01.pdf", "A1"),
            _chunk(2, "6.830/lectures/B1_lec02.pdf", "B1"),
        ]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="ok [1]")

        answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
        )

        # The generator saw a system prompt + a user prompt containing the
        # enumerated chunks and the question.
        call = generator.calls[0]
        assert "answer strictly" in call["system"].lower()
        assert "Question:" in call["user"]
        assert "q" in call["user"]
        # Both bracketed chunk headers must appear, in order.
        assert "[1] A1 6.006/lectures/A1_lec01.pdf" in call["user"]
        assert "[2] B1 6.830/lectures/B1_lec02.pdf" in call["user"]
        # And the chunk bodies too.
        assert "chunk-1-body" in call["user"]
        assert "chunk-2-body" in call["user"]

    def test_top_k_override(self, tmp_path: Path) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[_chunk(1, "6.006/lectures/A1_lec01.pdf")])
        generator = _FakeGenerator(reply_text="hi [1]")
        answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            top_k=5,
        )
        assert store.calls[0]["k"] == 5


# ── Citation parsing ──────────────────────────────────────────────────


class TestCitationParsing:
    def test_dedup_first_mention_order(self) -> None:
        chunks = [_chunk(i, f"6.006/lectures/A1_lec{i:02d}.pdf") for i in (1, 2, 3)]
        cits = _parse_citations("Foo [3] bar [1] baz [3][1][2] end.", chunks)
        assert [c.marker for c in cits] == [3, 1, 2]

    def test_out_of_range_dropped(self) -> None:
        chunks = [_chunk(1, "6.006/lectures/A1_lec01.pdf")]
        # [9] doesn't exist — dropped. [1] kept.
        cits = _parse_citations("Ok [1] and also [9].", chunks)
        assert [c.marker for c in cits] == [1]

    def test_zero_and_negative_ignored(self) -> None:
        # `[0]` is out of range (1-indexed). `[-1]` won't even match the
        # `\d+` regex — sanity check both.
        chunks = [_chunk(1, "6.006/lectures/A1_lec01.pdf")]
        cits = _parse_citations("Bad [0] good [1] weird [-2].", chunks)
        assert [c.marker for c in cits] == [1]

    def test_no_markers_returns_empty(self) -> None:
        chunks = [_chunk(1, "6.006/lectures/A1_lec01.pdf")]
        cits = _parse_citations("No brackets here at all.", chunks)
        assert cits == []

    def test_all_out_of_range_returns_empty(self, tmp_path: Path) -> None:
        # End-to-end version — an answer that only cites nonexistent chunks
        # yields an empty citations list but preserves the answer text.
        chunks = [_chunk(1, "6.006/lectures/A1_lec01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="Something [9] and [42].")
        result = answer_question(query="q", embedder=embedder, store=store, generator=generator)
        assert result.answer == "Something [9] and [42]."
        assert result.citations == []


# ── Out-of-corpus paths ───────────────────────────────────────────────


class TestOutOfCorpus:
    def test_sentinel_text_preserved(self, tmp_path: Path) -> None:
        # When Claude produces the exact out-of-corpus sentence, the pipeline
        # doesn't do anything special — the answer round-trips, citations are
        # empty (no brackets in the reply), and the generator DID run.
        chunks = [_chunk(1, "6.006/lectures/A1_lec01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text=OUT_OF_CORPUS_SENTINEL)
        result = answer_question(query="q", embedder=embedder, store=store, generator=generator)
        assert result.answer == OUT_OF_CORPUS_SENTINEL
        assert result.citations == []
        assert len(generator.calls) == 1

    def test_empty_retrieval_short_circuits(self, tmp_path: Path) -> None:
        # No chunks retrieved → skip the generator call entirely, return the
        # sentinel with zero cost. Saves a token spend and keeps eval cost
        # accounting honest for degenerate cases (wrong pipeline_tag,
        # empty DB, over-filtering).
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
        chunks = [
            _chunk(1, "6.006/lectures/A1_lec01.pdf", "A1"),
            _chunk(2, "6.830/lectures/B1_lec02.pdf", "B1"),
        ]
        out = _format_context(chunks)
        assert "[1] A1 6.006/lectures/A1_lec01.pdf" in out
        assert "[2] B1 6.830/lectures/B1_lec02.pdf" in out
        assert "chunk-1-body" in out
        assert "chunk-2-body" in out
        # Blank line between chunks.
        assert "\n\n" in out

    def test_page_range_and_score(self) -> None:
        c = _chunk(1, "6.006/lectures/A1_lec01.pdf", "A1")
        out = _format_context([c])
        # page_start=1, page_end=1 → "pages 1–1"; score renders with 2 dp.
        assert "pages 1–1" in out
        # score for marker 1 = 0.9 - 0.01 = 0.89 → "0.89"
        assert "0.89" in out

    def test_empty_input(self) -> None:
        assert _format_context([]) == ""


class TestPipelineTagOverride:
    def test_custom_pipeline_tag_forwarded(self, tmp_path: Path) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])  # empty → short-circuit, but store IS called
        generator = _FakeGenerator()
        answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            pipeline_tag="p2_semantic",
        )
        assert store.calls[0]["pipeline"] == "p2_semantic"
