"""Query pipeline tests.

Covers: end-to-end shape, citation dedup + first-mention order, out-of-range
marker handling, out-of-corpus prompt path, empty-retrieval short-circuit,
context block format, and dense/hybrid retriever dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.pipeline import prompts as prompts_module
from src.pipeline.generate import GenerateResult
from src.pipeline.pipeline_config import RerankerConfig, RetrieverConfig
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL
from src.pipeline.query import (
    PROMPT_VERSION,
    QueryExecutionError,
    _format_context,
    _parse_citations,
    answer_question,
)
from src.pipeline.rerank import RerankExecutionError, RerankResult
from src.pipeline.store import RetrievedChunk
from tests.test_pipeline_config import TEST_PIPELINE_CFG


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
    prompts_module.load_prompt.cache_clear()
    monkeypatch.setattr(prompts_module, "PROMPTS_DIR", tmp_path / "prompts")
    yield
    prompts_module.load_prompt.cache_clear()


@dataclass
class _FakeEmbedder:
    """Duck-types VoyageEmbedder — only what is called by the pipeline."""

    vector: list[float] = field(default_factory=lambda: [0.1] * 4)
    calls: list[str] = field(default_factory=list)

    def embed_query(self, text: str, *, run_id: str | None = None) -> list[float]:
        self.calls.append(text)
        return self.vector


@dataclass
class _FakeStore:
    """Duck-types VectorStore — implements both dense_search and hybrid_search."""

    to_return: list[RetrievedChunk] = field(default_factory=list)
    dense_calls: list[dict] = field(default_factory=list)
    hybrid_calls: list[dict] = field(default_factory=list)

    def dense_search(
        self, chunk_set_id: str, query_embedding: list[float], k: int
    ) -> list[RetrievedChunk]:
        self.dense_calls.append(
            {"chunk_set_id": chunk_set_id, "k": k, "embedding_len": len(query_embedding)}
        )
        return list(self.to_return)

    def hybrid_search(
        self,
        chunk_set_id: str,
        query_embedding: list[float],
        query_text: str,
        k: int,
        rrf_k: int,
    ) -> list[RetrievedChunk]:
        self.hybrid_calls.append(
            {
                "chunk_set_id": chunk_set_id,
                "k": k,
                "rrf_k": rrf_k,
                "query_text": query_text,
                "embedding_len": len(query_embedding),
            }
        )
        return list(self.to_return)


@dataclass
class _FakeReranker:
    """Duck-types CohereReranker — only what is called."""

    reorder: list[int] | None = None
    cost_usd: float = 0.0042
    calls: list[dict] = field(default_factory=list)

    def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
        run_id: str | None = None,
    ) -> RerankResult:
        self.calls.append(
            {"query": query, "top_n": top_n, "run_id": run_id, "n_candidates": len(chunks)}
        )
        order = self.reorder if self.reorder is not None else list(reversed(range(len(chunks))))
        order = order[: min(top_n, len(chunks))]
        reranked = [
            RetrievedChunk(
                chunk_id=chunks[i].chunk_id,
                document_id=chunks[i].document_id,
                doc_path=chunks[i].doc_path,
                source_id=chunks[i].source_id,
                chunk_set_id=chunks[i].chunk_set_id,
                chunk_index=chunks[i].chunk_index,
                text=chunks[i].text,
                page_start=chunks[i].page_start,
                page_end=chunks[i].page_end,
                score=0.99 - 0.01 * rank,
            )
            for rank, i in enumerate(order)
        ]
        return RerankResult(chunks=reranked, cost_usd=self.cost_usd, latency_ms=5.0)


@dataclass
class _FakeGenerator:
    """Duck-types ClaudeGenerator — only what is called."""

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
        chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
        chunk_index=marker - 1,
        text=f"chunk-{marker}-body",
        page_start=marker,
        page_end=marker,
        score=0.9 - 0.01 * marker,
    )


class TestEndToEnd:
    def test_happy_path(self, tmp_path: Path) -> None:
        chunks = [
            _chunk(1, "fixture/A1_doc03.pdf", "A1"),
            _chunk(2, "fixture/A2_doc03.pdf", "A2"),
            _chunk(3, "fixture/A3_doc01.pdf", "A3"),
        ]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
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
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
            run_id="run_test",
        )

        assert result.answer == "The answer is X [1] because of Y [3]."
        assert result.tokens_input == 500
        assert result.tokens_output == 25
        assert result.cost_usd == 0.0018
        assert result.prompt_version == PROMPT_VERSION
        assert result.query == "what is merge sort?"

        assert result.retrieved_chunks == chunks

        assert [c.marker for c in result.citations] == [1, 3]
        assert result.citations[0].doc_path == "fixture/A1_doc03.pdf"
        assert result.citations[1].doc_path == "fixture/A3_doc01.pdf"
        assert result.citations[0].source_id == "A1"

        assert embedder.calls == ["what is merge sort?"]
        assert len(store.dense_calls) == 1
        assert store.dense_calls[0]["chunk_set_id"] == TEST_PIPELINE_CFG.chunk_set_id
        assert len(generator.calls) == 1
        assert generator.calls[0]["prompt_version"] == PROMPT_VERSION
        assert generator.calls[0]["run_id"] == "run_test"

    def test_generator_receives_substituted_prompt(self, tmp_path: Path) -> None:
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
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
        )

        call = generator.calls[0]
        assert call["system"] == f"SYSTEM: sentinel is '{OUT_OF_CORPUS_SENTINEL}'."
        assert OUT_OF_CORPUS_SENTINEL in call["system"]
        assert "what is merge sort?" in call["user"]
        assert "[1] A1 fixture/A1_doc01.pdf" in call["user"]
        assert "[2] B1 fixture/B1_doc02.pdf" in call["user"]
        assert "chunk-1-body" in call["user"]
        assert "chunk-2-body" in call["user"]


class TestCitationParsing:
    def test_dedup_first_mention_order(self) -> None:
        chunks = [_chunk(i, f"fixture/A1_doc{i:02d}.pdf") for i in (1, 2, 3)]
        cits = _parse_citations("Foo [3] bar [1] baz [3][1][2] end.", chunks)
        assert [c.marker for c in cits] == [3, 1, 2]

    def test_out_of_range_dropped(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        cits = _parse_citations("Ok [1] and also [9].", chunks)
        assert [c.marker for c in cits] == [1]

    def test_zero_and_negative_ignored(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        cits = _parse_citations("Bad [0] good [1] weird [-2].", chunks)
        assert [c.marker for c in cits] == [1]

    def test_no_markers_returns_empty(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        cits = _parse_citations("No brackets here at all.", chunks)
        assert cits == []

    def test_all_out_of_range_returns_empty(self, tmp_path: Path) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="Something [9] and [42].")
        result = answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
        )
        assert result.answer == "Something [9] and [42]."
        assert result.citations == []


class TestOutOfCorpus:
    def test_sentinel_text_preserved(self, tmp_path: Path) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text=OUT_OF_CORPUS_SENTINEL)
        result = answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
        )
        assert result.answer == OUT_OF_CORPUS_SENTINEL
        assert result.citations == []
        assert len(generator.calls) == 1

    def test_empty_retrieval_short_circuits(self, tmp_path: Path) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator(reply_text="should not be called")

        result = answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
        )
        assert result.answer == OUT_OF_CORPUS_SENTINEL
        assert result.citations == []
        assert result.retrieved_chunks == []
        assert result.tokens_input == 0
        assert result.tokens_output == 0
        assert result.cost_usd == 0.0
        assert generator.calls == []
        assert embedder.calls == ["q"]
        assert len(store.dense_calls) == 1


class TestFormatContext:
    def test_shape(self) -> None:
        chunks = [
            _chunk(1, "fixture/A1_doc01.pdf", "A1"),
            _chunk(2, "fixture/B1_doc02.pdf", "B1"),
        ]
        out = _format_context(chunks)
        assert "[1] A1 fixture/A1_doc01.pdf" in out
        assert "[2] B1 fixture/B1_doc02.pdf" in out
        assert "pages 1–1" in out
        assert "chunk-1-body" in out
        assert "chunk-2-body" in out
        assert "\n\n" in out

    def test_score_not_rendered_in_llm_prompt(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf", "A1")]
        out = _format_context(chunks)
        assert "score" not in out.lower()


class TestRetrieverDispatch:
    """`answer_question` picks the store method based on `retriever.kind`."""

    def test_dense_routes_to_dense_search(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="ok [1]")

        answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id="test_dense_tag",
            retriever=RetrieverConfig(kind="dense", top_k=5),
        )

        assert len(store.dense_calls) == 1
        assert store.dense_calls[0] == {
            "chunk_set_id": "test_dense_tag",
            "k": 5,
            "embedding_len": len(embedder.vector),
        }
        assert store.hybrid_calls == []

    def test_hybrid_routes_to_hybrid_search(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        generator = _FakeGenerator(reply_text="ok [1]")

        answer_question(
            query="what is quicksort?",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id="test_hybrid_tag",
            retriever=RetrieverConfig(kind="hybrid", top_k=7, rrf_k=60),
        )

        assert store.dense_calls == []
        assert len(store.hybrid_calls) == 1
        assert store.hybrid_calls[0] == {
            "chunk_set_id": "test_hybrid_tag",
            "k": 7,
            "rrf_k": 60,
            "query_text": "what is quicksort?",
            "embedding_len": len(embedder.vector),
        }

    def test_hybrid_without_rrf_k_raises(self) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator()

        with pytest.raises(ValueError, match=r"hybrid retriever requires rrf_k"):
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id="test_broken_hybrid",
                retriever=RetrieverConfig(kind="hybrid", top_k=5, rrf_k=None),
            )
        assert embedder.calls == []
        assert store.dense_calls == []
        assert store.hybrid_calls == []
        assert generator.calls == []

    @pytest.mark.parametrize("bad_rrf_k", [0, -1, -60])
    def test_hybrid_with_non_positive_rrf_k_raises_before_embed(self, bad_rrf_k: int) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator()

        with pytest.raises(ValueError, match=r"rrf_k must be positive"):
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id="test_bad_rrf_k",
                retriever=RetrieverConfig(kind="hybrid", top_k=5, rrf_k=bad_rrf_k),
            )
        assert embedder.calls == []
        assert store.dense_calls == []
        assert store.hybrid_calls == []
        assert generator.calls == []

    def test_unknown_kind_raises(self) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator()
        broken = RetrieverConfig(kind="mystery", top_k=5)

        with pytest.raises(ValueError, match=r"unknown retriever\.kind"):
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id="test_broken",
                retriever=broken,
            )
        assert embedder.calls == []
        assert store.dense_calls == []
        assert store.hybrid_calls == []
        assert generator.calls == []


_RERANK_CFG = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=2)


class TestRerankBranch:
    """`answer_question` optionally reranks between retrieval and generation."""

    def test_rerank_applied_when_configured(self) -> None:
        chunks = [
            _chunk(1, "fixture/A1_doc01.pdf", "A1"),
            _chunk(2, "fixture/A2_doc02.pdf", "A2"),
            _chunk(3, "fixture/A3_doc03.pdf", "A3"),
        ]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        reranker = _FakeReranker(reorder=[2, 0], cost_usd=0.001)
        generator = _FakeGenerator(reply_text="Answer uses [1] and [2].")

        result = answer_question(
            query="what is quicksort?",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
            reranker=reranker,
            reranker_config=_RERANK_CFG,
            run_id="eval_test_run",
        )

        assert len(reranker.calls) == 1
        assert reranker.calls[0]["top_n"] == 2
        assert reranker.calls[0]["run_id"] == "eval_test_run"
        assert reranker.calls[0]["n_candidates"] == 3

        assert len(result.retrieved_chunks) == 2
        assert [c.doc_path for c in result.retrieved_chunks] == [
            "fixture/A3_doc03.pdf",
            "fixture/A1_doc01.pdf",
        ]
        assert [c.doc_path for c in result.citations] == [
            "fixture/A3_doc03.pdf",
            "fixture/A1_doc01.pdf",
        ]
        assert result.rerank_cost_usd == 0.001

    def test_rerank_omitted_when_none(self) -> None:
        chunks = [_chunk(1, "fixture/A1_doc01.pdf")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        reranker = _FakeReranker()
        generator = _FakeGenerator(reply_text="ok [1]")

        result = answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
            reranker=None,
            reranker_config=None,
        )

        assert reranker.calls == []
        assert [c.doc_path for c in result.retrieved_chunks] == ["fixture/A1_doc01.pdf"]
        assert result.rerank_cost_usd == 0.0

    def test_reranker_without_config_raises(self) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator()
        reranker = _FakeReranker()
        with pytest.raises(ValueError, match=r"reranker and reranker_config"):
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id="tag",
                retriever=TEST_PIPELINE_CFG.retriever,
                reranker=reranker,
                reranker_config=None,
            )
        assert embedder.calls == []
        assert reranker.calls == []
        assert generator.calls == []

    def test_config_without_reranker_raises(self) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        generator = _FakeGenerator()
        with pytest.raises(ValueError, match=r"reranker and reranker_config"):
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id="tag",
                retriever=TEST_PIPELINE_CFG.retriever,
                reranker=None,
                reranker_config=_RERANK_CFG,
            )
        assert embedder.calls == []
        assert generator.calls == []

    def test_empty_retrieval_skips_rerank(self) -> None:
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=[])
        reranker = _FakeReranker()
        generator = _FakeGenerator(reply_text="should not be called")

        result = answer_question(
            query="q",
            embedder=embedder,
            store=store,
            generator=generator,
            chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
            retriever=TEST_PIPELINE_CFG.retriever,
            reranker=reranker,
            reranker_config=_RERANK_CFG,
        )

        assert result.answer == OUT_OF_CORPUS_SENTINEL
        assert result.rerank_cost_usd == 0.0
        assert reranker.calls == []
        assert generator.calls == []

    def test_rerank_parse_error_preserves_paid_cost(self) -> None:

        @dataclass
        class _BillingRaiseReranker:
            cost_usd: float
            calls: list[dict] = field(default_factory=list)

            def rerank(
                self,
                *,
                query: str,
                chunks: list[RetrievedChunk],
                top_n: int,
                run_id: str | None = None,
            ) -> RerankResult:
                self.calls.append({"query": query, "top_n": top_n, "run_id": run_id})
                raise RerankExecutionError(
                    IndexError("bad index from Cohere"),
                    cost_usd=self.cost_usd,
                )

        chunks = [_chunk(1, "fixture/A1_doc01.pdf", "A1")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        reranker = _BillingRaiseReranker(cost_usd=0.002)
        generator = _FakeGenerator(reply_text="should not be called")

        with pytest.raises(QueryExecutionError) as exc_info:
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
                retriever=TEST_PIPELINE_CFG.retriever,
                reranker=reranker,
                reranker_config=_RERANK_CFG,
            )

        err = exc_info.value
        assert err.rerank_cost_usd == pytest.approx(0.002)
        assert isinstance(err.cause, IndexError)
        assert generator.calls == []
        assert len(reranker.calls) == 1

    def test_rerank_prebilling_raise_leaves_cost_zero(self) -> None:

        @dataclass
        class _PreBillingRaiseReranker:
            calls: list[dict] = field(default_factory=list)

            def rerank(
                self,
                *,
                query: str,
                chunks: list[RetrievedChunk],
                top_n: int,
                run_id: str | None = None,
            ) -> RerankResult:
                self.calls.append({"query": query, "top_n": top_n})
                raise RuntimeError("transport failed after retries")

        chunks = [_chunk(1, "fixture/A1_doc01.pdf", "A1")]
        embedder = _FakeEmbedder()
        store = _FakeStore(to_return=chunks)
        reranker = _PreBillingRaiseReranker()
        generator = _FakeGenerator(reply_text="should not be called")

        with pytest.raises(QueryExecutionError) as exc_info:
            answer_question(
                query="q",
                embedder=embedder,
                store=store,
                generator=generator,
                chunk_set_id=TEST_PIPELINE_CFG.chunk_set_id,
                retriever=TEST_PIPELINE_CFG.retriever,
                reranker=reranker,
                reranker_config=_RERANK_CFG,
            )

        err = exc_info.value
        assert err.rerank_cost_usd == 0.0
        assert isinstance(err.cause, RuntimeError)
        assert generator.calls == []
