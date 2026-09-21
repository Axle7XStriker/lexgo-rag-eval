"""Pipeline configuration registry — single source of truth for P1..P4.

Every knob that defines a retrieval pipeline (chunker choice + parameters,
retriever choice + top-k, optional reranker) lives here as typed dataclasses.
The DB tag under which a pipeline's chunks are stored is DERIVED from the
config, not hand-authored — any config change automatically produces a new
tag, so tuning a knob and re-ingesting yields fresh rows for side-by-side
comparison rather than silently overwriting the previous run's rows.

Callers:
  - `scripts.ingest` — reads `PIPELINES[args.pipeline]`, dispatches on
    `cfg.chunker.algorithm`, writes rows tagged `cfg.tag`.
  - `evals.run` — reads `PIPELINES[args.pipeline]`, uses `cfg.tag` to filter
    retrieval and `cfg.retriever.top_k` for retrieval fan-out. Serializes the
    whole `PipelineConfig` into the run manifest via
    `pipeline_to_manifest_dict` so the artifact is self-describing.

Adding a new pipeline (P2/P3/P4) is one dict entry — no changes to any
callsite except the dispatch table in `scripts.ingest` (a new
`elif algorithm == "..."` branch).

Design notes worth remembering:
  - `PipelineConfig.tag` is a hybrid: readable prefix (e.g.
    `p1_fixed_500_50`) + short sha256 suffix. The prefix reads cleanly in
    `SELECT DISTINCT pipeline FROM chunks;` output; the hash suffix catches
    fields not covered by `_readable_prefix` so no config change can silently
    reuse an existing tag. Adding a field to `ChunkerConfig` therefore never
    creates a stealth data-mixing bug.
  - `_config_hash` uses the full `asdict(self)`, so every field participates.
    `test_pipeline_config.py` mutates each field one at a time and asserts
    `.tag` changes each time — the guard that makes this trustworthy.
  - Configs are frozen dataclasses so accidental mutation raises rather than
    corrupts the registry. `dataclasses.replace(cfg, ...)` for controlled
    edits (used by tests).
  - `key` is the CLI-facing short name (`"p1"`); `tag` is the DB-facing
    unique identifier. Two different concepts, don't conflate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

# Chunker knob defaults ship as module constants inside `ChunkerConfig`
# instances (below). No top-level DEFAULT_* here — the registry IS the
# defaults.


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
    # its cut logic directly.
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
    """Retriever parameters. Extended when P3 (hybrid) lands."""

    kind: Literal["dense", "hybrid"]
    top_k: int
    # Reserved for P3: rrf_k, bm25_weight, dense_weight, etc. Absent
    # fields don't appear in the tag's readable prefix (dense-only P1/P2
    # tags stay short) but DO appear in the hash — so tuning a P3 knob
    # regenerates its tag automatically.


@dataclass(frozen=True)
class RerankerConfig:
    """Reranker parameters. Populated for P4 (Cohere Rerank 3)."""

    provider: Literal["cohere"]
    model: str
    top_n: int  # post-rerank final result count


@dataclass(frozen=True)
class PipelineConfig:
    """One retrieval pipeline's full config. Registered in `PIPELINES`.

    `tag` is a computed property, not a stored field — see the module
    docstring for why. `key` is the CLI-facing short name (`"p1"`);
    `tag` is the DB-facing unique identifier (`"p1_fixed_500_50_a3f2b1c8"`).
    """

    key: str
    chunker: ChunkerConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig | None

    @property
    def tag(self) -> str:
        """Hybrid DB tag: `<readable_prefix>_<sha256(config)[:8]>`.

        Examples::

            p1_fixed_500_50_a3f2b1c8
            p2_semantic_p95_9e4d7f2a

        Any change to any field on this config (or its nested configs)
        changes the hash suffix, which changes the tag, which lands new
        rows under a new key in the `chunks` table. Old rows sit inert.
        """
        return f"{self._readable_prefix()}_{self._config_hash()}"

    def _readable_prefix(self) -> str:
        """Human-legible prefix. Covers the fields most operators care about
        at `psql` inspection time; the hash suffix covers everything else."""
        c = self.chunker
        if c.algorithm == "fixed":
            return f"{self.key}_fixed_{c.target_tokens}_{c.overlap_tokens}"
        if c.algorithm == "semantic":
            # `int(percentile_threshold)` keeps the prefix short — 95.0 → "p95".
            # The full float still lands in the hash, so 95.0 vs 95.5 still
            # produce distinct tags via the suffix.
            return f"{self.key}_semantic_p{int(c.percentile_threshold or 0)}"
        raise ValueError(f"unknown chunker.algorithm: {c.algorithm!r}")

    def _config_hash(self) -> str:
        """Short sha256 over the full config. Every field participates."""
        # `default=str` keeps the hash deterministic even if a future field
        # holds something JSON doesn't natively serialize (Path, Enum, etc.).
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


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
    # P2/P3/P4 slot in here in subsequent PRs.
}


def get_pipeline(key: str) -> PipelineConfig:
    """Lookup by CLI key. Raises `KeyError` with a listing of valid keys."""
    if key not in PIPELINES:
        raise KeyError(f"unknown pipeline {key!r}; valid keys: {sorted(PIPELINES)}")
    return PIPELINES[key]


def pipeline_to_manifest_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Serialize `cfg` for the eval run manifest.

    Stores the DERIVED tag explicitly at the top level alongside the full
    config, so a run artifact self-describes both identity (`tag`) and
    shape (nested `chunker` / `retriever` / `reranker` dicts). Downstream
    consumers can dedupe runs by tag and reproduce a pipeline from the
    config fields without re-running the derivation.
    """
    return {"tag": cfg.tag, **asdict(cfg)}
