"""Pipeline configuration registry — single source of truth for P1..P4.

Every knob that defines a retrieval pipeline (chunker choice + parameters,
retriever choice + top-k, optional reranker) lives here as typed dataclasses.

Two distinct derived identifiers hang off a `PipelineConfig`:

  - `tag` — PIPELINE identity. Hashes the full nested config (chunker +
    retriever + reranker). Used in eval artifacts (`manifest.json`,
    `summary.md`) and structured logs so operators can see which pipeline
    a given run was. Any config change rotates `tag`.
  - `chunk_set_id` — CHUNK-SET identity. Hashes `ChunkerConfig` only. The
    DB filter column (`chunks.chunk_set_id`) stores this; ingest + retrieval
    key off it exclusively. Two pipelines with equal `ChunkerConfig` values
    derive the SAME `chunk_set_id` and share one physical copy of chunks +
    embeddings.

Adding a new pipeline is one dict entry — no changes to any callsite
except the dispatch table in `scripts.ingest` (a new `elif algorithm ==
"..."` branch).

Design notes worth remembering:
  - Both `tag` and `chunk_set_id` are hybrid: readable prefix (e.g.
    `p1_fixed_500_50` for `tag`, `fixed_500_50` for `chunk_set_id`) +
    short sha256 suffix. Prefixes read cleanly in `SELECT DISTINCT ...
    FROM chunks;` output; the hash suffix catches fields not covered by
    the prefix so no config change can silently reuse an existing id.
  - Hashes run over the full `asdict(...)` of their scope, so every field
    participates.
  - Configs are frozen dataclasses so accidental mutation raises rather
    than corrupts the registry.
  - `key` is the CLI-facing short name (`"p1"`); `tag` is the pipeline-
    identity string; `chunk_set_id` is the DB chunk-set identifier.
    Three distinct concepts, don't conflate.
  - The embedder is deliberately a global in `src/config.py`, not a
    per-pipeline field — every pipeline uses the same Voyage model and
    dim, so `chunk_set_id` can safely ignore it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ChunkerConfig:
    """Chunker parameters. Populated per pipeline in `PIPELINES` below.

    Only the fields relevant to `algorithm` are meaningful; the others
    are set to `None`. Kept flat (rather than an algorithm-specific
    subclass hierarchy) so `asdict` serializes trivially and every field
    is greppable in one place.
    """

    algorithm: Literal["fixed", "semantic"]
    # Fixed-window knobs. `target_tokens` also documents the intended
    # band center for the semantic chunker even though it doesn't drive
    # its cut logic directly — NOTE: every field on this dataclass
    # participates in the derived hashes regardless of whether the current
    # algorithm reads it, so tweaking `target_tokens` on a semantic
    # config still regenerates `tag` and `chunk_set_id`. This is intentional:
    # "documentary" fields are still versioned so a future contributor
    # who reinterprets them as load-bearing gets fresh DB rows
    # automatically, not silent data-mixing under an old tag.
    target_tokens: int | None = None
    overlap_tokens: int | None = None  # fixed only
    # Semantic-chunker knobs. Only meaningful when algorithm == "semantic".
    percentile_threshold: float | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    # Tokenizer used for `num_tokens` accounting. Shared across algorithms
    # so P1 and P2 token counts are directly comparable.
    encoding: str = "cl100k_base"


@dataclass(frozen=True)
class RetrieverConfig:
    """Retriever parameters. Fields not used by `kind` are `None`."""

    kind: Literal["dense", "hybrid"]
    top_k: int
    # RRF (Reciprocal Rank Fusion) fusion constant. Only meaningful when
    # kind == "hybrid". Standard value is 60 (per Cormack et al. 2009);
    # larger values flatten rank-source weighting, smaller values sharpen it.
    rrf_k: int | None = None


@dataclass(frozen=True)
class RerankerConfig:
    """Reranker parameters. Populated for P4 (Cohere Rerank 3)."""

    provider: Literal["cohere"]
    model: str
    top_n: int  # post-rerank final result count


@dataclass(frozen=True)
class PipelineConfig:
    """One retrieval pipeline's full config. Registered in `PIPELINES`.

    `tag` and `chunk_set_id` are computed properties — see module docstring
    for the semantic split. Short form:
      - `key` → CLI short name (`"p1"`)
      - `tag` → pipeline identity (`"p1_fixed_500_50_a3f2b1c8"`); eval only
      - `chunk_set_id` → DB chunk-set identity (`"fixed_500_50_1b2c3d4e"`);
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

            p1_fixed_500_50_a3f2b1c8
            p2_semantic_p95_9e4d7f2a

        Any change to any field on this config (or its nested configs)
        changes the hash suffix, which changes the tag. Used in eval
        artifacts and logs.
        """
        return f"{self.key}_{_chunker_readable_prefix(self.chunker)}_{_short_hash(asdict(self))}"

    @property
    def chunk_set_id(self) -> str:
        """Chunk-set identifier: `<chunker_prefix>_<sha256(chunker)[:8]>`.

        Examples::

            fixed_500_50_1b2c3d4e
            semantic_p95_3f4a5b6c

        Hashes `ChunkerConfig` ONLY. Two pipelines whose `ChunkerConfig`
        values are equal derive the same `chunk_set_id` and share chunks
        in the DB. Changing a retriever or reranker knob does not rotate
        this; changing any chunker field does. Stored in `chunks.chunk_set_id`;
        the ingest idempotency check + the query-time retrieval filter both
        key off this.
        """
        return f"{_chunker_readable_prefix(self.chunker)}_{_short_hash(asdict(self.chunker))}"


def _chunker_readable_prefix(c: ChunkerConfig) -> str:
    """Human-legible prefix derived from a `ChunkerConfig`.

    Covers the fields most operators care about at `psql` inspection
    time; the hash suffix covers everything else.
    """
    if c.algorithm == "fixed":
        return f"fixed_{c.target_tokens}_{c.overlap_tokens}"
    if c.algorithm == "semantic":
        # `int(percentile_threshold)` keeps the prefix short — 95.0 → "p95".
        # The full float still lands in the hash, so 95.0 vs 95.5 still
        # produce distinct ids via the suffix.
        p = c.percentile_threshold if c.percentile_threshold is not None else 0
        return f"semantic_p{int(p)}"
    raise ValueError(f"unknown chunker.algorithm: {c.algorithm!r}")


def _short_hash(payload: Any) -> str:
    """Short sha256 over a JSON-serializable payload. Every field participates.

    `default=str` keeps the hash deterministic even if a future field
    holds something JSON doesn't natively serialize (Path, Enum, etc.).
    8 hex chars — long enough that collisions across O(dozens) of
    configs are astronomically unlikely, short enough to keep ids
    human-manageable.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


# ── Registry ──────────────────────────────────────────────────────────
#
# The single source of truth. Adding P2/P3/P4 = one dict entry each.
# CLI `--pipeline` args are exactly the keys here.

PIPELINES: dict[str, PipelineConfig] = {
    "p1": PipelineConfig(
        key="p1",
        chunker=ChunkerConfig(
            algorithm="fixed",
            target_tokens=500,
            overlap_tokens=50,
        ),
        retriever=RetrieverConfig(kind="dense", top_k=10),
        reranker=None,
    ),
    "p2": PipelineConfig(
        key="p2",
        chunker=ChunkerConfig(
            algorithm="semantic",
            # `target_tokens` documents the intended band center; the
            # semantic chunker doesn't cut ON it, but min/max are anchored
            # around it so P2 chunks are roughly comparable in size to P1.
            target_tokens=500,
            percentile_threshold=95.0,
            min_tokens=200,
            max_tokens=750,
        ),
        retriever=RetrieverConfig(kind="dense", top_k=10),
        reranker=None,
    ),
    "p3": PipelineConfig(
        key="p3",
        # Same chunker as P1 — the point of P3 is to isolate the retrieval
        # fusion effect, so the chunk set must be byte-identical to P1's.
        # Its pipeline tag still differs because the retriever config changes,
        # while its chunk_set_id matches P1 so both reuse the same DB rows.
        chunker=ChunkerConfig(
            algorithm="fixed",
            target_tokens=500,
            overlap_tokens=50,
        ),
        # Hybrid: BM25 (Postgres FTS ts_rank_cd) ∪ dense (pgvector cosine),
        # fused via Reciprocal Rank Fusion with k=60. top_k=10 = same
        # final size as others so eval accuracy is comparable head-to-head.
        retriever=RetrieverConfig(kind="hybrid", top_k=10, rrf_k=60),
        reranker=None,
    ),
    "p4": PipelineConfig(
        key="p4",
        # Same chunker as P1/P3 — P4 isolates the rerank effect on top of
        # hybrid retrieval, so the chunk set must be byte-identical. Its
        # pipeline tag differs because the retriever + reranker configs differ,
        # while its chunk_set_id matches P1/P3 so all three share DB rows.
        chunker=ChunkerConfig(
            algorithm="fixed",
            target_tokens=500,
            overlap_tokens=50,
        ),
        # Hybrid top-20 pre-rerank: a wider pool than P3's top-10 so the
        # reranker has more candidates to re-order. rrf_k=60 keeps the fusion
        # knob identical to P3 so a P3→P4 delta attributes cleanly to the
        # rerank stage, not fusion.
        retriever=RetrieverConfig(kind="hybrid", top_k=20, rrf_k=60),
        # Cohere Rerank 3 (English). top_n=10 matches P1-P3's retriever
        # top_k=10 so the LLM sees the same number of chunks across all
        # pipelines — the P3→P4 accuracy delta then reflects reorder-and-
        # prune quality (best 10 of 20, in Cohere's order), not a smaller
        # context window. The matrix originally spec'd top_n=5 but that
        # confounded the measurement with context size.
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
