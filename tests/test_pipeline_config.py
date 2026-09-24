"""Pipeline-config registry tests.

Load-bearing invariants:
  - `get_pipeline` errors are actionable — listing valid keys.
  - `PipelineConfig.tag` is a hybrid `<readable>_<hash>` and is unique per
    config. The mutation-safety test enumerates every mutable field and
    asserts the tag changes when it changes — this is what makes the
    hash-suffix approach trustworthy against future field additions.
  - `pipeline_to_manifest_dict` round-trips through JSON so eval manifests
    always serialize cleanly.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields, replace
from typing import ClassVar

import pytest

from src.pipeline.pipeline_config import (
    PIPELINES,
    ChunkerConfig,
    PipelineConfig,
    RerankerConfig,
    RetrieverConfig,
    get_pipeline,
    pipeline_to_manifest_dict,
)

# ── Shared test fixture ───────────────────────────────────────────────

TEST_PIPELINE_CFG = PipelineConfig(
    key="test",
    chunker=ChunkerConfig(
        algorithm="fixed",
        target_tokens=400,
        overlap_tokens=40,
        percentile_threshold=95.0,
        min_tokens=200,
        max_tokens=750,
    ),
    retriever=RetrieverConfig(kind="dense", top_k=8),
    reranker=None,
)


class TestGetPipeline:
    def test_known_key(self) -> None:
        assert get_pipeline("p1") is PIPELINES["p1"]

    def test_unknown_key_lists_valid_keys(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_pipeline("p9")
        msg = str(exc_info.value)
        assert "'p9'" in msg
        assert "valid keys" in msg
        # Every registered key should appear in the error — the message is
        # the debugging surface, not just a "not found" signal.
        for key in PIPELINES:
            assert key in msg


class TestTagFormat:
    def test_tag_prefix(self) -> None:
        tag = TEST_PIPELINE_CFG.tag
        assert tag.startswith("test_fixed_400_40_")

    def test_hash_suffix_is_short_hex(self) -> None:
        tag = TEST_PIPELINE_CFG.tag
        suffix = tag.rsplit("_", 1)[-1]
        assert re.fullmatch(r"[0-9a-f]{8}", suffix) is not None

    def test_unknown_algorithm_raises(self) -> None:
        cfg = PipelineConfig(
            key="pX",
            # Bypass the Literal check with a cast-through-object so the test
            # exercises the runtime `raise` in `_readable_prefix` (which
            # exists precisely for the Literal drift case).
            chunker=ChunkerConfig(algorithm="fixed", target_tokens=1, overlap_tokens=0),
            retriever=RetrieverConfig(kind="dense", top_k=1),
            reranker=None,
        )
        broken = replace(cfg, chunker=replace(cfg.chunker, algorithm="mystery"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=r"unknown chunker\.algorithm"):
            _ = broken.tag


class TestTagChangesOnFieldMutation:
    """Load-bearing safety: mutating any single field must change the tag.

    This is what makes the hash-suffix design trustworthy against future
    field additions — a new `ChunkerConfig` field that isn't included in
    `_readable_prefix` still lands in the hash, so the tag still changes.
    If someone adds a field and BOTH omits it from `_readable_prefix` AND
    breaks this test's iteration, it's a genuine bug — the test forces the
    contributor to notice.
    """

    def _all_tags_distinct(self, cfgs: list[PipelineConfig]) -> None:
        tags = [c.tag for c in cfgs]
        assert len(set(tags)) == len(tags), f"expected distinct tags, got {tags}"

    def test_chunker_target_tokens(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, target_tokens=501)),
            replace(base, chunker=replace(base.chunker, target_tokens=1000)),
        ]
        self._all_tags_distinct(variants)

    def test_chunker_overlap_tokens(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, overlap_tokens=25)),
            replace(base, chunker=replace(base.chunker, overlap_tokens=100)),
        ]
        self._all_tags_distinct(variants)

    def test_chunker_encoding(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, encoding="o200k_base")),
        ]
        self._all_tags_distinct(variants)

    def test_retriever_top_k(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, retriever=replace(base.retriever, top_k=20)),
        ]
        self._all_tags_distinct(variants)

    def test_reranker_addition(self) -> None:
        base = TEST_PIPELINE_CFG
        with_reranker = replace(
            base,
            reranker=RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5),
        )
        self._all_tags_distinct([base, with_reranker])

    def test_reranker_top_n(self) -> None:
        rr_a = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5)
        rr_b = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=10)
        base = TEST_PIPELINE_CFG
        self._all_tags_distinct(
            [
                replace(base, reranker=rr_a),
                replace(base, reranker=rr_b),
            ]
        )

    def test_key_change_also_changes_tag(self) -> None:
        base = TEST_PIPELINE_CFG
        forked = replace(base, key="test_smoke")
        assert base.tag != forked.tag

    def test_chunker_algorithm(self) -> None:
        # Swapping algorithm implies swapping the surrounding fixed/semantic
        # knobs to keep the config coherent. The hash MUST catch the shift
        # even when both configs are otherwise well-formed for their algorithm.
        base = TEST_PIPELINE_CFG
        semantic = replace(base, chunker=replace(base.chunker, algorithm="semantic"))
        self._all_tags_distinct([base, semantic])

    def test_chunker_percentile_threshold(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, percentile_threshold=90.0)),
            replace(base, chunker=replace(base.chunker, percentile_threshold=95.5)),
        ]
        self._all_tags_distinct(variants)

    def test_chunker_min_tokens(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, min_tokens=150)),
        ]
        self._all_tags_distinct(variants)

    def test_chunker_max_tokens(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, max_tokens=1000)),
        ]
        self._all_tags_distinct(variants)

    def test_retriever_kind(self) -> None:
        # `kind` is a Literal["dense", "hybrid"]; both are valid runtime
        # values so no `type: ignore` is needed. P3 (hybrid) will lean on
        # this — a `dense`↔`hybrid` swap MUST land under a fresh DB tag.
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, retriever=replace(base.retriever, kind="hybrid")),
        ]
        self._all_tags_distinct(variants)

    def test_reranker_model(self) -> None:
        base = TEST_PIPELINE_CFG
        rr_a = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5)
        rr_b = RerankerConfig(provider="cohere", model="rerank-multilingual-v3.0", top_n=5)
        self._all_tags_distinct(
            [
                replace(base, reranker=rr_a),
                replace(base, reranker=rr_b),
            ]
        )

    def test_reranker_provider(self) -> None:
        # `provider` is currently Literal["cohere"] (single value). Cast
        # through `type: ignore` to prove the hash includes this field even
        # before a second provider joins the Literal — same trick as
        # `test_unknown_algorithm_raises` uses for algorithm drift.
        base = TEST_PIPELINE_CFG
        rr_a = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5)
        rr_b = replace(rr_a, provider="voyage")  # type: ignore[arg-type]
        self._all_tags_distinct(
            [
                replace(base, reranker=rr_a),
                replace(base, reranker=rr_b),
            ]
        )


class TestTagMutationCoverage:
    """Meta-guard: every field on every config dataclass has a mutation test
    above. Introspects `dataclasses.fields()` so adding a new field without
    a corresponding test fails CI loudly — the docstring's "safety net"
    claim is only true if we enforce it here.

    `PipelineConfig`'s sub-dataclass fields (`chunker`/`retriever`/`reranker`)
    are covered transitively: any nested-field mutation test also mutates
    the outer PipelineConfig field that holds the sub-config. Only the
    scalar `key` needs a direct top-level test (it has one).
    """

    _COVERED_CHUNKER_FIELDS: ClassVar[set[str]] = {
        "algorithm",
        "target_tokens",
        "overlap_tokens",
        "percentile_threshold",
        "min_tokens",
        "max_tokens",
        "encoding",
    }
    _COVERED_RETRIEVER_FIELDS: ClassVar[set[str]] = {"kind", "top_k"}
    _COVERED_RERANKER_FIELDS: ClassVar[set[str]] = {"provider", "model", "top_n"}
    _COVERED_PIPELINE_FIELDS: ClassVar[set[str]] = {"key", "chunker", "retriever", "reranker"}

    def _assert_covered(self, dataclass_type: type, covered: set[str]) -> None:
        declared = {f.name for f in fields(dataclass_type)}
        missing = declared - covered
        assert not missing, (
            f"{dataclass_type.__name__} has field(s) with no mutation test: "
            f"{sorted(missing)}. Add a case to TestTagChangesOnFieldMutation, "
            f"then update the covered set on TestTagMutationCoverage."
        )
        stale = covered - declared
        assert not stale, (
            f"{dataclass_type.__name__} covered set names field(s) that no "
            f"longer exist: {sorted(stale)}. Prune the covered set."
        )

    def test_chunker_fields_all_covered(self) -> None:
        self._assert_covered(ChunkerConfig, self._COVERED_CHUNKER_FIELDS)

    def test_retriever_fields_all_covered(self) -> None:
        self._assert_covered(RetrieverConfig, self._COVERED_RETRIEVER_FIELDS)

    def test_reranker_fields_all_covered(self) -> None:
        self._assert_covered(RerankerConfig, self._COVERED_RERANKER_FIELDS)

    def test_pipeline_fields_all_covered(self) -> None:
        self._assert_covered(PipelineConfig, self._COVERED_PIPELINE_FIELDS)


class TestManifestSerialization:
    def test_round_trips_through_json(self) -> None:
        # Manifest is written with `json.dumps`, so every field must be
        # JSON-serializable via `default=str` (which the module uses in its
        # hash payload too). Guards against a future dataclass field type
        # that would break both.
        raw = pipeline_to_manifest_dict(TEST_PIPELINE_CFG)
        payload = json.dumps(raw, default=str)
        parsed = json.loads(payload)
        assert parsed["tag"] == TEST_PIPELINE_CFG.tag
        assert parsed["key"] == "test"
        assert parsed["chunker"]["algorithm"] == "fixed"
        assert parsed["retriever"]["top_k"] == 8
        assert parsed["reranker"] is None

    def test_stores_tag_at_top_level(self) -> None:
        # The manifest reader shouldn't have to re-derive the tag from the
        # config — it's stored explicitly so a downstream consumer can
        # dedupe runs by tag without depending on this Python module.
        d = pipeline_to_manifest_dict(TEST_PIPELINE_CFG)
        assert d["tag"] == TEST_PIPELINE_CFG.tag
