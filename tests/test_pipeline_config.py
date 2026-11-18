"""Pipeline-config registry tests.

Covers:
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
            chunker=ChunkerConfig(algorithm="fixed", target_tokens=1, overlap_tokens=0),
            retriever=RetrieverConfig(kind="dense", top_k=1),
            reranker=None,
        )
        broken = replace(cfg, chunker=replace(cfg.chunker, algorithm="mystery"))
        with pytest.raises(ValueError, match=r"unknown chunker\.algorithm"):
            _ = broken.tag


class TestTagChangesOnFieldMutation:
    """Load-bearing safety: mutating any single field must change the tag."""

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

    def test_retriever_rrf_k(self) -> None:
        base = TEST_PIPELINE_CFG
        variants = [
            base,
            replace(base, retriever=replace(base.retriever, rrf_k=60)),
            replace(base, retriever=replace(base.retriever, rrf_k=30)),
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
        base = TEST_PIPELINE_CFG
        rr_a = RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5)
        rr_b = replace(rr_a, provider="voyage")
        self._all_tags_distinct(
            [
                replace(base, reranker=rr_a),
                replace(base, reranker=rr_b),
            ]
        )


class TestTagMutationCoverage:
    _COVERED_CHUNKER_FIELDS: ClassVar[set[str]] = {
        "algorithm",
        "target_tokens",
        "overlap_tokens",
        "percentile_threshold",
        "min_tokens",
        "max_tokens",
        "encoding",
    }
    _COVERED_RETRIEVER_FIELDS: ClassVar[set[str]] = {"kind", "top_k", "rrf_k"}
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


class TestChunkSetId:
    """`chunk_set_id` is the DB identifier for a chunk set.

    Load-bearing properties:
      - Pipelines with equal `ChunkerConfig` derive the SAME id.
      - Pipelines with different `ChunkerConfig` derive DIFFERENT ids.
      - The chunk set is independent of retrieval choice.
      - Any `ChunkerConfig` field mutation rotates the id — a change to
        chunking must force a fresh ingest.
    """

    @pytest.mark.parametrize(
        ("chunker", "expected_prefix"),
        [
            (TEST_PIPELINE_CFG.chunker, "fixed_400_40_"),
            (replace(TEST_PIPELINE_CFG.chunker, algorithm="semantic"), "semantic_p95_"),
        ],
    )
    def test_prefix_matches_chunker_algorithm(
        self, chunker: ChunkerConfig, expected_prefix: str
    ) -> None:
        cfg = replace(TEST_PIPELINE_CFG, chunker=chunker)
        assert cfg.chunk_set_id.startswith(expected_prefix)

    def test_hash_suffix_is_short_hex(self) -> None:
        suffix = TEST_PIPELINE_CFG.chunk_set_id.rsplit("_", 1)[-1]
        assert re.fullmatch(r"[0-9a-f]{8}", suffix) is not None

    def test_shared_across_pipelines_with_equal_chunker(self) -> None:
        pipelines = [
            TEST_PIPELINE_CFG,
            replace(
                TEST_PIPELINE_CFG,
                key="forked",
                retriever=replace(TEST_PIPELINE_CFG.retriever, top_k=42),
            ),
            replace(
                TEST_PIPELINE_CFG,
                key="reranked",
                reranker=RerankerConfig(provider="cohere", model="rerank-english-v3.0", top_n=5),
            ),
        ]
        assert len({cfg.chunk_set_id for cfg in pipelines}) == 1

    @pytest.mark.parametrize(
        "chunker",
        [
            replace(TEST_PIPELINE_CFG.chunker, algorithm="semantic"),
            replace(TEST_PIPELINE_CFG.chunker, target_tokens=501),
            replace(TEST_PIPELINE_CFG.chunker, percentile_threshold=90.0),
        ],
    )
    def test_distinct_for_different_chunker(self, chunker: ChunkerConfig) -> None:
        mutated = replace(TEST_PIPELINE_CFG, chunker=chunker)
        assert TEST_PIPELINE_CFG.chunk_set_id != mutated.chunk_set_id


class TestManifestSerialization:
    def test_round_trips_through_json(self) -> None:
        raw = pipeline_to_manifest_dict(TEST_PIPELINE_CFG)
        payload = json.dumps(raw, default=str)
        parsed = json.loads(payload)
        assert parsed["tag"] == TEST_PIPELINE_CFG.tag
        assert parsed["chunk_set_id"] == TEST_PIPELINE_CFG.chunk_set_id
        assert parsed["key"] == "test"
        assert parsed["chunker"]["algorithm"] == "fixed"
        assert parsed["retriever"]["top_k"] == 8
        assert parsed["reranker"] is None

    def test_stores_derived_ids_at_top_level(self) -> None:
        d = pipeline_to_manifest_dict(TEST_PIPELINE_CFG)
        assert d["tag"] == TEST_PIPELINE_CFG.tag
        assert d["chunk_set_id"] == TEST_PIPELINE_CFG.chunk_set_id
