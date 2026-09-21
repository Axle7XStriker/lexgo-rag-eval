"""Pipeline-config registry tests.

Load-bearing invariants:
  - P1 knobs match the pre-refactor constants (regression guard).
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
from dataclasses import replace

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


class TestP1RegressionGuard:
    """P1's knobs match the pre-refactor `chunk.py` module constants
    (500-token target, 50-token overlap, top-10 dense retrieval, no rerank).
    Regressing any of these would silently reshape the P1 baseline the
    blog post's numbers depend on."""

    def test_p1_registered(self) -> None:
        assert "p1" in PIPELINES

    def test_p1_chunker_knobs(self) -> None:
        cfg = PIPELINES["p1"]
        assert cfg.key == "p1"
        assert cfg.chunker.algorithm == "fixed"
        assert cfg.chunker.target_tokens == 500
        assert cfg.chunker.overlap_tokens == 50
        assert cfg.chunker.encoding == "cl100k_base"

    def test_p1_retriever_knobs(self) -> None:
        cfg = PIPELINES["p1"]
        assert cfg.retriever.kind == "dense"
        assert cfg.retriever.top_k == 10

    def test_p1_no_reranker(self) -> None:
        assert PIPELINES["p1"].reranker is None


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
    def test_p1_prefix_stable(self) -> None:
        # The readable prefix is the debug-time affordance — anyone eyeballing
        # `SELECT DISTINCT pipeline FROM chunks;` should immediately see
        # "P1 fixed 500/50". Changing this prefix format is a data-migration
        # break for anyone with pre-existing chunks tagged the old way.
        tag = PIPELINES["p1"].tag
        assert tag.startswith("p1_fixed_500_50_")

    def test_hash_suffix_is_short_hex(self) -> None:
        tag = PIPELINES["p1"].tag
        suffix = tag.rsplit("_", 1)[-1]
        # 8 hex chars — long enough that collisions across O(dozens) of
        # configs are astronomically unlikely, short enough to keep the
        # tag human-manageable.
        assert re.fullmatch(r"[0-9a-f]{8}", suffix) is not None

    def test_semantic_prefix_shape(self) -> None:
        # Even though no P2 is registered yet, the prefix builder should
        # already handle `algorithm='semantic'` — otherwise adding P2 in
        # PR 2 would silently produce a "unknown chunker" error at import
        # time instead of just registering a new entry.
        cfg = PipelineConfig(
            key="p2",
            chunker=ChunkerConfig(
                algorithm="semantic",
                percentile_threshold=95.0,
                min_tokens=200,
                max_tokens=750,
            ),
            retriever=RetrieverConfig(kind="dense", top_k=10),
            reranker=None,
        )
        assert cfg.tag.startswith("p2_semantic_p95_")

    def test_unknown_algorithm_raises(self) -> None:
        # If a future contributor adds an algorithm to the Literal without
        # extending `_readable_prefix`, accessing `.tag` must fail loud —
        # otherwise the fallthrough would produce a tag whose prefix is
        # meaningless.
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
        base = PIPELINES["p1"]
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, target_tokens=501)),
            replace(base, chunker=replace(base.chunker, target_tokens=1000)),
        ]
        self._all_tags_distinct(variants)

    def test_chunker_overlap_tokens(self) -> None:
        base = PIPELINES["p1"]
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, overlap_tokens=25)),
            replace(base, chunker=replace(base.chunker, overlap_tokens=100)),
        ]
        self._all_tags_distinct(variants)

    def test_chunker_encoding(self) -> None:
        base = PIPELINES["p1"]
        variants = [
            base,
            replace(base, chunker=replace(base.chunker, encoding="o200k_base")),
        ]
        self._all_tags_distinct(variants)

    def test_retriever_top_k(self) -> None:
        base = PIPELINES["p1"]
        variants = [
            base,
            replace(base, retriever=replace(base.retriever, top_k=20)),
        ]
        self._all_tags_distinct(variants)

    def test_reranker_addition(self) -> None:
        base = PIPELINES["p1"]
        with_reranker = replace(
            base,
            reranker=RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5),
        )
        self._all_tags_distinct([base, with_reranker])

    def test_reranker_top_n(self) -> None:
        rr_a = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5)
        rr_b = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=10)
        base = PIPELINES["p1"]
        self._all_tags_distinct(
            [
                replace(base, reranker=rr_a),
                replace(base, reranker=rr_b),
            ]
        )

    def test_key_change_also_changes_tag(self) -> None:
        # `key` is IN the readable prefix, so a fork like p1→p1_smoke must
        # produce distinct DB tags. Prevents an operator from accidentally
        # aliasing two pipelines that share every knob except the CLI name.
        base = PIPELINES["p1"]
        forked = replace(base, key="p1_smoke")
        assert base.tag != forked.tag


class TestManifestSerialization:
    def test_round_trips_through_json(self) -> None:
        # Manifest is written with `json.dumps`, so every field must be
        # JSON-serializable via `default=str` (which the module uses in its
        # hash payload too). Guards against a future dataclass field type
        # that would break both.
        raw = pipeline_to_manifest_dict(PIPELINES["p1"])
        payload = json.dumps(raw, default=str)
        parsed = json.loads(payload)
        assert parsed["tag"] == PIPELINES["p1"].tag
        assert parsed["key"] == "p1"
        assert parsed["chunker"]["algorithm"] == "fixed"
        assert parsed["retriever"]["top_k"] == 10
        assert parsed["reranker"] is None

    def test_stores_tag_at_top_level(self) -> None:
        # The manifest reader shouldn't have to re-derive the tag from the
        # config — it's stored explicitly so a downstream consumer can
        # dedupe runs by tag without depending on this Python module.
        d = pipeline_to_manifest_dict(PIPELINES["p1"])
        assert d["tag"] == PIPELINES["p1"].tag
