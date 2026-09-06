"""VoyageEmbedder tests. Fully offline via a fake `voyageai.Client`.

Covers: batching, order preservation, per-call `log_llm_call` shape,
tenacity retry on transient errors, hard-fail on deterministic errors.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import SecretStr
from voyageai.error import APIConnectionError, RateLimitError, Timeout

from src.pipeline import embed as embed_module
from src.pipeline.embed import (
    DEFAULT_BATCH_SIZE,
    MAX_BATCH_SIZE,
    PRICING,
    VoyageEmbedder,
    _cost_for,
)


@dataclass
class _FakeResponse:
    """Shape of the object voyageai.Client.embed returns — only what we read."""

    embeddings: list[list[float]]
    total_tokens: int


# One entry in `_FakeClient.responses`: either a canned response or a
# callable that receives the batch texts and returns/raises. Callables are
# how retry tests simulate transient failures then a success.
_ResponseItem = _FakeResponse | Callable[[list[str]], _FakeResponse]


@dataclass
class _FakeClient:
    """Tracks calls and returns pre-programmed responses in order.

    Each call consumes one entry from `responses`; a callable entry is
    invoked (used for retry tests where the first N raise).
    """

    responses: list[_ResponseItem] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def embed(self, texts: list[str], model: str, input_type: str) -> _FakeResponse:
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})
        if not self.responses:
            # Sensible default so tests that don't program responses still work.
            return _FakeResponse(
                embeddings=[[0.0] * 4 for _ in texts],
                total_tokens=len(texts),
            )
        item = self.responses.pop(0)
        if callable(item):
            return item(texts)
        return item


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "llm_calls.jsonl"


def _make_embedder(
    *,
    client: _FakeClient,
    log_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = "voyage-3-large",
) -> VoyageEmbedder:
    return VoyageEmbedder(
        api_key=SecretStr("test-key"),
        model=model,
        log_path=log_path,
        batch_size=batch_size,
        client=client,
    )


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestBatching:
    """Batching honors batch_size and preserves input order."""

    def test_batch_size_ceiling(self, log_path: Path) -> None:
        # 129 inputs at batch_size=64 → 3 API calls (64 + 64 + 1).
        client = _FakeClient()
        e = _make_embedder(client=client, log_path=log_path, batch_size=64)
        out = e.embed_documents([f"t{i}" for i in range(129)])
        assert len(out) == 129
        assert len(client.calls) == 3
        assert [len(c["texts"]) for c in client.calls] == [64, 64, 1]

    def test_order_preserved(self, log_path: Path) -> None:
        # Program each batch response to reflect the batch inputs so we can
        # verify concatenation preserves order end-to-end.
        client = _FakeClient(
            responses=[
                _FakeResponse(embeddings=[[1.0]], total_tokens=1),
                _FakeResponse(embeddings=[[2.0]], total_tokens=1),
                _FakeResponse(embeddings=[[3.0]], total_tokens=1),
            ]
        )
        e = _make_embedder(client=client, log_path=log_path, batch_size=1)
        out = e.embed_documents(["a", "b", "c"])
        assert out == [[1.0], [2.0], [3.0]]

    def test_empty_input_no_calls(self, log_path: Path) -> None:
        client = _FakeClient()
        e = _make_embedder(client=client, log_path=log_path)
        assert e.embed_documents([]) == []
        assert client.calls == []


class TestLogging:
    """Every successful batch writes exactly one llm_calls.jsonl record."""

    def test_log_record_shape_document(self, log_path: Path) -> None:
        client = _FakeClient(responses=[_FakeResponse(embeddings=[[0.0]] * 3, total_tokens=42)])
        e = _make_embedder(client=client, log_path=log_path, batch_size=64)
        e.embed_documents(["x", "y", "z"])
        recs = _read_log(log_path)
        assert len(recs) == 1
        r = recs[0]
        assert r["provider"] == "voyage"
        assert r["model"] == "voyage-3-large"
        assert r["operation"] == "embed_documents"
        assert r["input_tokens"] == 42
        assert r["output_tokens"] == 0
        # log_llm_call rounds cost_usd to 6 dp; tolerate that quantization.
        assert r["cost_usd"] == pytest.approx(_cost_for("voyage-3-large", 42), abs=1e-6)
        assert r["batch_size"] == 3
        assert r["input_type"] == "document"

    def test_log_record_shape_query(self, log_path: Path) -> None:
        client = _FakeClient(responses=[_FakeResponse(embeddings=[[0.0]], total_tokens=7)])
        e = _make_embedder(client=client, log_path=log_path)
        e.embed_query("hello?")
        recs = _read_log(log_path)
        assert len(recs) == 1
        r = recs[0]
        assert r["provider"] == "voyage"
        assert r["model"] == "voyage-3-large"
        assert r["operation"] == "embed_query"
        assert r["input_tokens"] == 7
        assert r["output_tokens"] == 0
        assert r["cost_usd"] == pytest.approx(_cost_for("voyage-3-large", 7), abs=1e-6)
        assert r["batch_size"] == 1
        assert r["input_type"] == "query"

    def test_one_record_per_batch(self, log_path: Path) -> None:
        client = _FakeClient()
        e = _make_embedder(client=client, log_path=log_path, batch_size=2)
        e.embed_documents(["a", "b", "c", "d", "e"])  # 3 batches: 2+2+1
        assert len(_read_log(log_path)) == 3


class TestRetry:
    """Transient failures retry; deterministic ones do not."""

    def test_retries_on_rate_limit(self, log_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Zero wait to keep the test snappy. tenacity's wait is a callable on
        # the retry object; patching the attribute is the documented seam.
        monkeypatch.setattr(
            embed_module.VoyageEmbedder._embed_batch.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _flaky(texts: list[str]) -> _FakeResponse:
            state["n"] += 1
            if state["n"] < 3:
                raise RateLimitError("simulated 429")
            return _FakeResponse(embeddings=[[0.5]] * len(texts), total_tokens=len(texts))

        client = _FakeClient(responses=[_flaky, _flaky, _flaky])
        e = _make_embedder(client=client, log_path=log_path)
        out = e.embed_documents(["a", "b"])
        assert len(out) == 2
        # 2 failures + 1 success = 3 attempts on the client.
        assert state["n"] == 3
        # Only successful calls are logged.
        assert len(_read_log(log_path)) == 1

    def test_gives_up_after_max_attempts(
        self, log_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            embed_module.VoyageEmbedder._embed_batch.retry,
            "wait",
            lambda *a, **kw: 0,
        )

        def _always_fails(texts: list[str]) -> _FakeResponse:
            raise RateLimitError("still 429")

        client = _FakeClient(responses=[_always_fails] * 10)
        e = _make_embedder(client=client, log_path=log_path)
        with pytest.raises(RateLimitError):
            e.embed_documents(["a"])
        # tenacity is configured stop_after_attempt(5); no successful log line.
        assert _read_log(log_path) == []

    @pytest.mark.parametrize(
        "exc_factory",
        [
            # Voyage's OWN network-level exceptions — NOT Python built-in
            # TimeoutError / ConnectionError. If _RETRYABLE regresses to the
            # built-ins these tests fail immediately instead of silently
            # skipping retries in production.
            lambda: Timeout("simulated timeout"),
            lambda: APIConnectionError("simulated connection error"),
        ],
        ids=["voyage_timeout", "voyage_api_connection_error"],
    )
    def test_retries_on_voyage_network_exceptions(
        self,
        log_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        exc_factory,
    ) -> None:
        monkeypatch.setattr(
            embed_module.VoyageEmbedder._embed_batch.retry,
            "wait",
            lambda *a, **kw: 0,
        )
        state = {"n": 0}

        def _flaky(texts: list[str]) -> _FakeResponse:
            state["n"] += 1
            if state["n"] < 2:
                raise exc_factory()
            return _FakeResponse(embeddings=[[0.5]] * len(texts), total_tokens=len(texts))

        client = _FakeClient(responses=[_flaky, _flaky])
        e = _make_embedder(client=client, log_path=log_path)
        out = e.embed_documents(["a"])
        assert len(out) == 1
        assert state["n"] == 2  # 1 failure + 1 success


class TestValidation:
    """Constructor guards on batch_size + unknown model."""

    def test_batch_size_zero_rejected(self, log_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_embedder(client=_FakeClient(), log_path=log_path, batch_size=0)

    def test_batch_size_over_ceiling_rejected(self, log_path: Path) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _make_embedder(client=_FakeClient(), log_path=log_path, batch_size=MAX_BATCH_SIZE + 1)

    def test_unknown_model_rejected_at_init(self, log_path: Path) -> None:
        # Silent $0 cost from an unpriced model would poison the blog's cost
        # story with no visible signal — fail loud at construction instead.
        with pytest.raises(ValueError, match="unknown Voyage model"):
            _make_embedder(client=_FakeClient(), log_path=log_path, model="voyage-not-real")


class TestPricing:
    """Cost table stays honest; missing model degrades gracefully."""

    def test_voyage_3_large_present(self) -> None:
        assert "voyage-3-large" in PRICING
        assert PRICING["voyage-3-large"] > 0

    def test_unknown_model_zero_cost(self) -> None:
        assert _cost_for("model-that-does-not-exist", 1_000_000) == 0.0
