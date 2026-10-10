"""CohereReranker tests. Fully offline — fake Cohere client, no HTTP.

Covers: __init__ model gate, happy path (reorder + score overwrite),
empty-chunks short-circuit, top_n clamping, invalid top_n, run_id threaded
through to log, log_llm_call record shape (provider/operation/cost/extra),
and tenacity retry on transient failures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from cohere.errors import InternalServerError, TooManyRequestsError
from pydantic import SecretStr

from src.pipeline import rerank as rerank_module
from src.pipeline.rerank import CohereReranker, RerankResult
from src.pipeline.store import RetrievedChunk

# ── Fake Cohere client ────────────────────────────────────────────────


@dataclass
class _FakeResult:
    """Shape of V2RerankResponse.results[N] — only the two fields we read."""

    index: int
    relevance_score: float


@dataclass
class _FakeResponse:
    """Shape of V2RerankResponse — only `results` is read."""

    results: list[_FakeResult]


# One entry in `_FakeCohereClient.responses`: a canned response or a callable
# invoked with the request kwargs so retry tests can raise then succeed.
_ResponseItem = _FakeResponse | Callable[[dict], _FakeResponse]


@dataclass
class _FakeCohereClient:
    """Duck-types cohere.ClientV2. Only `.rerank(...)` is called."""

    responses: list[_ResponseItem] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def rerank(self, **kwargs) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            # Default: identity reorder (same order, equal relevance). Tests
            # that care about ordering program `responses` explicitly.
            documents = kwargs.get("documents", [])
            top_n = kwargs.get("top_n", len(documents))
            return _FakeResponse(
                results=[_FakeResult(index=i, relevance_score=0.5) for i in range(top_n)]
            )
        item = self.responses.pop(0)
        if callable(item):
            return item(kwargs)
        return item


# ── Fixtures ──────────────────────────────────────────────────────────


def _chunk(i: int, doc_path: str = "6.006/lectures/A1_lec01.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1000 + i,
        document_id=1,
        doc_path=doc_path,
        source_id="A1",
        pipeline="p4_fixed_500_50_test",
        chunk_index=i,
        text=f"chunk-{i}-body",
        page_start=i + 1,
        page_end=i + 1,
        score=0.9 - 0.01 * i,  # pre-rerank score — should be overwritten
    )


def _log(path: Path) -> list[dict]:
    """Read every non-blank line in `path` as a dict."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reranker(
    tmp_path: Path,
    *,
    model: str = "rerank-english-v3.0",
) -> tuple[CohereReranker, _FakeCohereClient, Path]:
    """Build a CohereReranker with a FakeCohereClient and tmp-path log."""
    client = _FakeCohereClient()
    log_path = tmp_path / "logs" / "llm_calls.jsonl"
    rr = CohereReranker(
        api_key=SecretStr("test-key"),
        model=model,
        log_path=log_path,
        client=client,
    )
    return rr, client, log_path


# ── Init gate ─────────────────────────────────────────────────────────


class TestInit:
    def test_unknown_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"unknown Cohere model"):
            CohereReranker(
                api_key=SecretStr("test-key"),
                model="rerank-made-up-v9",
                log_path=tmp_path / "logs" / "llm_calls.jsonl",
                client=_FakeCohereClient(),
            )

    def test_known_model_initializes(self, tmp_path: Path) -> None:
        rr, _, _ = _reranker(tmp_path)
        assert rr.model == "rerank-english-v3.0"


# ── Happy path ────────────────────────────────────────────────────────


class TestHappyPath:
    def test_reorders_and_overwrites_score(self, tmp_path: Path) -> None:
        """Cohere returns new order; CohereReranker builds chunks in that order
        with `score` overwritten by `relevance_score`."""
        rr, client, _ = _reranker(tmp_path)
        chunks = [_chunk(0), _chunk(1), _chunk(2)]
        # Cohere ranks chunk index 2 first, then 0, then 1 — different from input.
        client.responses.append(
            _FakeResponse(
                results=[
                    _FakeResult(index=2, relevance_score=0.95),
                    _FakeResult(index=0, relevance_score=0.72),
                    _FakeResult(index=1, relevance_score=0.11),
                ]
            )
        )

        result = rr.rerank(query="what is x?", chunks=chunks, top_n=3)

        assert isinstance(result, RerankResult)
        # Order is Cohere's; chunk_id proves provenance through the mapping.
        assert [c.chunk_id for c in result.chunks] == [1002, 1000, 1001]
        # Scores overwritten with Cohere's relevance scores.
        assert [c.score for c in result.chunks] == [0.95, 0.72, 0.11]
        # Non-score fields preserved from the source chunks.
        assert [c.text for c in result.chunks] == ["chunk-2-body", "chunk-0-body", "chunk-1-body"]
        assert all(c.pipeline == "p4_fixed_500_50_test" for c in result.chunks)

    def test_passes_query_and_documents_to_cohere(self, tmp_path: Path) -> None:
        rr, client, _ = _reranker(tmp_path)
        chunks = [_chunk(0), _chunk(1)]
        rr.rerank(query="my question", chunks=chunks, top_n=2)
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["model"] == "rerank-english-v3.0"
        assert call["query"] == "my question"
        assert call["documents"] == ["chunk-0-body", "chunk-1-body"]
        assert call["top_n"] == 2


# ── Short-circuit and clamping ────────────────────────────────────────


class TestShortCircuitAndClamping:
    def test_empty_chunks_short_circuits_no_api_call(self, tmp_path: Path) -> None:
        """Empty input → empty result, zero cost, zero latency, NO API call."""
        rr, client, log_path = _reranker(tmp_path)
        result = rr.rerank(query="q", chunks=[], top_n=5)
        assert result.chunks == []
        assert result.cost_usd == 0.0
        assert result.latency_ms == 0.0
        assert client.calls == []
        # Also: nothing landed in the LLM-call log — short-circuit means no spend.
        assert _log(log_path) == []

    def test_top_n_clamped_to_candidate_count(self, tmp_path: Path) -> None:
        """top_n > len(chunks) is clamped before the API call (Cohere would 400)."""
        rr, client, _ = _reranker(tmp_path)
        chunks = [_chunk(0), _chunk(1)]  # only 2 candidates
        rr.rerank(query="q", chunks=chunks, top_n=10)
        assert client.calls[0]["top_n"] == 2

    @pytest.mark.parametrize("bad_top_n", [0, -1, -10])
    def test_zero_or_negative_top_n_raises(self, tmp_path: Path, bad_top_n: int) -> None:
        rr, client, _ = _reranker(tmp_path)
        with pytest.raises(ValueError, match=r"top_n must be positive"):
            rr.rerank(query="q", chunks=[_chunk(0)], top_n=bad_top_n)
        # No API call attempted on a config-error.
        assert client.calls == []


# ── Logging ───────────────────────────────────────────────────────────


class TestLogging:
    def test_log_record_has_expected_fields(self, tmp_path: Path) -> None:
        """One log record per API call, with correct provider/operation/cost/extra."""
        rr, client, log_path = _reranker(tmp_path)
        chunks = [_chunk(i) for i in range(5)]
        client.responses.append(
            _FakeResponse(
                results=[_FakeResult(index=i, relevance_score=0.5 - 0.05 * i) for i in range(3)]
            )
        )

        rr.rerank(query="q", chunks=chunks, top_n=3, run_id="eval_abc")

        records = _log(log_path)
        assert len(records) == 1
        rec = records[0]
        assert rec["provider"] == "cohere"
        assert rec["model"] == "rerank-english-v3.0"
        assert rec["operation"] == "rerank_chunks"
        # Rerank bills per-call; token counts are explicitly zero.
        assert rec["input_tokens"] == 0
        assert rec["output_tokens"] == 0
        # Cost is $0.002 per search per src/pricing.py.
        assert rec["cost_usd"] == pytest.approx(0.002)
        assert rec["run_id"] == "eval_abc"
        # No prompt template → prompt_version stays None.
        assert rec["prompt_version"] is None
        # Latency recorded as a non-negative number (jitter-tolerant).
        assert rec["latency_ms"] >= 0
        # Extra fields match the three-key contract.
        assert rec["num_candidates"] == 5
        assert rec["top_n"] == 3
        assert rec["returned"] == 3

    def test_run_id_omitted_when_not_provided(self, tmp_path: Path) -> None:
        rr, _, log_path = _reranker(tmp_path)
        rr.rerank(query="q", chunks=[_chunk(0)], top_n=1)
        rec = _log(log_path)[0]
        # run_id defaults to None — same contract as log_llm_call's other callers.
        assert rec["run_id"] is None


# ── Retry on transient errors ─────────────────────────────────────────


class TestRetry:
    def test_retries_on_rate_limit_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tenacity retries TooManyRequestsError; second attempt returns normally."""
        # Zero the exponential wait so the retry fires immediately in the test.
        monkeypatch.setattr(rerank_module.CohereReranker._call.retry, "wait", lambda *a, **kw: 0)
        rr, client, log_path = _reranker(tmp_path)
        chunks = [_chunk(0), _chunk(1)]

        def raise_once(_kwargs: dict) -> _FakeResponse:
            # Cohere's typed subclasses pin status_code internally; the public
            # constructor is (body, headers=None). We just pass the body.
            raise TooManyRequestsError(body="rate limited")

        client.responses.extend(
            [
                raise_once,
                _FakeResponse(
                    results=[
                        _FakeResult(index=1, relevance_score=0.9),
                        _FakeResult(index=0, relevance_score=0.4),
                    ]
                ),
            ]
        )

        result = rr.rerank(query="q", chunks=chunks, top_n=2)
        # Succeeded on the retry; two client calls, one log record (the failed
        # call never logged — log happens AFTER the API returns).
        assert len(client.calls) == 2
        assert [c.chunk_id for c in result.chunks] == [1001, 1000]
        assert len(_log(log_path)) == 1

    def test_retries_on_5xx_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rerank_module.CohereReranker._call.retry, "wait", lambda *a, **kw: 0)
        rr, client, _ = _reranker(tmp_path)

        def raise_once(_kwargs: dict) -> _FakeResponse:
            raise InternalServerError(body="oops")

        client.responses.extend(
            [raise_once, _FakeResponse(results=[_FakeResult(index=0, relevance_score=0.5)])]
        )

        result = rr.rerank(query="q", chunks=[_chunk(0)], top_n=1)
        assert len(client.calls) == 2
        assert len(result.chunks) == 1
