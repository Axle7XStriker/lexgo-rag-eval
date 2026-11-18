"""Pipeline configuration registry — single source of truth for P1..P4.

Every knob that defines a retrieval pipeline (chunker choice + parameters,
retriever choice + top-k, optional reranker) lives here as typed dataclasses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ChunkerConfig:
    """Chunker parameters.

    Only the fields relevant to `algorithm` are meaningful; the others
    are set to `None`.
    """

    algorithm: Literal["fixed", "semantic"]
    # fixed-chunker parameters
    target_tokens: int | None = None
    overlap_tokens: int | None = None
    # semantic-chunker parameters
    percentile_threshold: float | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    encoding: str = "cl100k_base"


@dataclass(frozen=True)
class RetrieverConfig:
    """Retriever parameters. Fields not used by `kind` are `None`."""

    kind: Literal["dense", "hybrid"]
    top_k: int
    rrf_k: int | None = None  # Set only for hybrid retrieval.


@dataclass(frozen=True)
class RerankerConfig:
    """Reranker parameters"""

    provider: Literal["cohere"]
    model: str
    top_n: int


@dataclass(frozen=True)
class PipelineConfig:
    """One retrieval pipeline's full config. Registered in `PIPELINES`.

    `tag` and `chunk_set_id` are computed properties. Short form:
      - `key` → CLI short name (`"p1"`)
      - `tag` → pipeline identity (`"p1_fixed_512_64_a3f2b1c8"`); eval only
      - `chunk_set_id` → DB chunk-set identity (`"fixed_512_64_1b2c3d4e"`);
        ingest + retrieval key off this
    """

    key: str
    chunker: ChunkerConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig | None

    @property
    def tag(self) -> str:
        """Hybrid pipeline tag: `<key>_<chunker_prefix>_<sha256(config)[:8]>`.

        Examples::

            p1_fixed_512_64_a3f2b1c8
            p2_semantic_p90_9e4d7f2a

        Any change to any field on this config (or its nested configs)
        changes the hash suffix, which changes the tag. Used in eval
        artifacts and logs.
        """
        return f"{self.key}_{_chunker_readable_prefix(self.chunker)}_{_short_hash(asdict(self))}"

    @property
    def chunk_set_id(self) -> str:
        """Chunk-set identifier: `<chunker_prefix>_<sha256(chunker)[:8]>`.

        Examples::

            fixed_512_64_1b2c3d4e
            semantic_p90_3f4a5b6c

        Hashes `ChunkerConfig` ONLY. Two pipelines whose `ChunkerConfig`
        values are equal derive the same `chunk_set_id` and share chunks
        in the DB. Changing a retriever or reranker knob does not rotate
        this; changing any chunker field does. Stored in `chunks.chunk_set_id`;
        the ingest idempotency check + the query-time retrieval filter both
        key off this.
        """
        return f"{_chunker_readable_prefix(self.chunker)}_{_short_hash(asdict(self.chunker))}"


def _chunker_readable_prefix(c: ChunkerConfig) -> str:
    """Human-legible prefix derived from a `ChunkerConfig`."""
    if c.algorithm == "fixed":
        return f"fixed_{c.target_tokens}_{c.overlap_tokens}"
    if c.algorithm == "semantic":
        p = c.percentile_threshold if c.percentile_threshold is not None else 0
        return f"semantic_p{int(p)}"
    raise ValueError(f"unknown chunker.algorithm: {c.algorithm!r}")


def _short_hash(payload: Any) -> str:
    """Short sha256 over a JSON-serializable payload. Every field participates."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


PIPELINES: dict[str, PipelineConfig] = {
    "p1": PipelineConfig(
        key="p1",
        chunker=ChunkerConfig(
            algorithm="fixed",
            target_tokens=512,
            overlap_tokens=64,
        ),
        retriever=RetrieverConfig(kind="dense", top_k=10),
        reranker=None,
    ),
    "p2": PipelineConfig(
        key="p2",
        chunker=ChunkerConfig(
            algorithm="semantic",
            target_tokens=512,  # Hashed for identity; the semantic cutter does not split on it.
            percentile_threshold=90.0,
            min_tokens=256,
            max_tokens=1024,
        ),
        retriever=RetrieverConfig(kind="dense", top_k=10),
        reranker=None,
    ),
    "p3": PipelineConfig(
        key="p3",
        chunker=ChunkerConfig(  # Same chunker as p1, so both share one stored chunk set.
            algorithm="fixed",
            target_tokens=512,
            overlap_tokens=64,
        ),
        retriever=RetrieverConfig(kind="hybrid", top_k=10, rrf_k=60),  # Same result count as dense.
        reranker=None,
    ),
    "p4": PipelineConfig(
        key="p4",
        chunker=ChunkerConfig(
            algorithm="fixed",
            target_tokens=512,
            overlap_tokens=64,
        ),
        retriever=RetrieverConfig(
            kind="hybrid",
            top_k=20,
            rrf_k=60,
        ),
        reranker=RerankerConfig(
            provider="cohere",
            model="rerank-english-v3.0",
            top_n=10,
        ),
    ),
}


def get_pipeline(key: str) -> PipelineConfig:
    """Lookup by CLI key. Raises `KeyError` with a listing of valid keys."""
    if key not in PIPELINES:
        raise KeyError(f"unknown pipeline {key!r}; valid keys: {sorted(PIPELINES)}")
    return PIPELINES[key]


def pipeline_to_manifest_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Serialize `cfg` for the eval run manifest.

    Stores both derived identifiers explicitly alongside the full config, so
    a run artifact records its pipeline identity (`tag`), physical retrieval
    set (`chunk_set_id`), and shape (nested config dicts). Downstream consumers
    need not re-run either derivation.
    """
    return {**asdict(cfg), "tag": cfg.tag, "chunk_set_id": cfg.chunk_set_id}
