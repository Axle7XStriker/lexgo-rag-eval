"""End-to-end test for `evals.run.main` — fully offline via fake deps.

Verifies the eval loop's plumbing without any live API or DB:
  - Load qa.jsonl from tmp_path.
  - Inject fake VoyageEmbedder / VectorStore / ClaudeGenerator / ClaudeJudge.
  - Run `evals.run.main()`.
  - Assert `results.jsonl` shape, `manifest.json` contents, `summary.md`
    contains the expected metric strings, exit code is 0.

The fakes are constructed locally (not shared with test_query.py) to keep
each test file's readability high and because the eval-run wiring exercises
slightly different call surfaces (judge, run_id, etc.).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from evals import run as run_module
from src.pipeline import prompts as prompts_module
from src.pipeline.generate import GenerateResult
from src.pipeline.judge import JudgeResult
from src.pipeline.pipeline_config import get_pipeline
from src.pipeline.prompts import OUT_OF_CORPUS_SENTINEL
from src.pipeline.query import PROMPT_VERSION
from src.pipeline.store import RetrievedChunk

_P1_CFG = get_pipeline("p1")

# ── Prompt fixture ────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _prompt_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both the answer and judge prompt loaders at tmp_path fixtures."""
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "answer").mkdir(parents=True)
    (prompts_dir / "answer" / f"{PROMPT_VERSION}.md").write_text(
        """# System
SYS: sentinel='{out_of_corpus_sentinel}'.
# User template
Q: {question}
CTX:
{context}
""",
        encoding="utf-8",
    )
    (prompts_dir / "judge").mkdir(parents=True)
    (prompts_dir / "judge" / "v1.md").write_text(
        """# System
JUDGE. Sentinel: '{out_of_corpus_sentinel}'.
# User template
Q: {question}
GA: {gold_answer}
GC: {gold_citations_block}
MA: {model_answer}
MC: {model_citations_block}
""",
        encoding="utf-8",
    )
    prompts_module.load_prompt.cache_clear()
    monkeypatch.setattr(prompts_module, "PROMPTS_DIR", prompts_dir)
    yield
    prompts_module.load_prompt.cache_clear()


# ── Fakes ─────────────────────────────────────────────────────────────


@dataclass
class _FakeEmbedder:
    vector: list[float] = field(default_factory=lambda: [0.1] * 4)

    def embed_query(self, text: str, *, run_id: str | None = None) -> list[float]:
        return self.vector


@dataclass
class _FakeStore:
    """Duck-types VectorStore + context manager. Returns per-query chunks."""

    doc_paths_by_query: dict[str, list[str]] = field(default_factory=dict)
    default_doc_paths: list[str] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def __enter__(self) -> _FakeStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def dense_search(
        self, pipeline: str, query_embedding: list[float], k: int
    ) -> list[RetrievedChunk]:
        self.calls.append({"pipeline": pipeline, "k": k})
        doc_paths = self.doc_paths_by_query.get("<default>", self.default_doc_paths)
        return [
            RetrievedChunk(
                chunk_id=1000 + i,
                document_id=1,
                doc_path=dp,
                source_id=dp.split("/")[-1][:2],
                pipeline=pipeline,
                chunk_index=i,
                text=f"chunk-{i}-body",
                page_start=i + 1,
                page_end=i + 1,
                score=0.9 - 0.01 * i,
            )
            for i, dp in enumerate(doc_paths)
        ]


@dataclass
class _FakeGenerator:
    """Return answer text keyed by question — lets tests drive citation behavior."""

    replies_by_question: dict[str, str] = field(default_factory=dict)
    default_reply: str = "generic answer [1]"

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
        # Match on question substring embedded in the user body (Q: ...).
        for q, reply in self.replies_by_question.items():
            if q in user:
                return GenerateResult(
                    text=reply, input_tokens=150, output_tokens=25, cost_usd=0.002
                )
        return GenerateResult(
            text=self.default_reply, input_tokens=150, output_tokens=25, cost_usd=0.002
        )


@dataclass
class _FakeJudge:
    """Simple oracle judge: correct iff model answer contains the gold answer verbatim.

    For out-of-corpus Q&As (empty gold citations), correct iff model answer
    equals the sentinel — mirrors the real judge's specified behavior so
    the aggregate metric is meaningful.
    """

    def judge(
        self,
        *,
        question: str,
        gold_answer: str,
        gold_citation_doc_paths: list[str],
        model_answer: str,
        model_citation_doc_paths: list[str],
        run_id: str | None = None,
    ) -> JudgeResult:
        if not gold_citation_doc_paths:
            correct = model_answer.strip() == OUT_OF_CORPUS_SENTINEL
            cite_valid = True
        else:
            # Case-insensitive substring match — good enough for the fake.
            correct = gold_answer.lower()[:20] in model_answer.lower()
            cite_valid = bool(set(model_citation_doc_paths) & set(gold_citation_doc_paths))
        return JudgeResult(
            answer_correct=correct,
            citations_semantically_valid=cite_valid,
            rationale="fake verdict",
            input_tokens=60,
            output_tokens=15,
            cost_usd=0.0009,
        )


# ── Helpers ───────────────────────────────────────────────────────────


def _write_qa_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _valid_records() -> list[dict]:
    """Five hand-crafted records spanning FACTUAL / CROSS / PARA / OOC / ADV."""
    return [
        {
            "id": "f001",
            "type": "factual",
            "question": "What is the worst-case complexity of merge sort?",
            "gold_answer": (
                "Merge sort has worst-case time complexity O(n log n) because it always "
                "recursively splits the input in half and merges in linear time."
            ),
            "gold_citations": [
                {"doc_path": "6.006/lectures/A1_lec03.pdf", "page_or_section": "slide 12"}
            ],
        },
        {
            "id": "x001",
            "type": "cross_source_synthesis",
            "question": "How does the Selinger cost model relate to the quiz derivation?",
            "gold_answer": (
                "Both attribute the dominant cost of a join to the product of the input "
                "cardinalities times a per-tuple factor tied to the physical operator."
            ),
            "gold_citations": [
                {"doc_path": "6.830/lectures/B1_lec05.pdf", "page_or_section": "slide 7"},
                {"doc_path": "6.830/quizzes/B2_quiz01.pdf", "page_or_section": "Q3"},
            ],
        },
        {
            "id": "p001",
            "type": "semantic_paraphrase",
            "question": "Why do BSTs need balancing?",
            "gold_answer": (
                "Without a balance invariant a BST can degenerate into a linked list on "
                "sorted insertions, giving O(n) operations instead of O(log n)."
            ),
            "gold_citations": [
                {"doc_path": "6.006/lectures/A1_lec06.pdf", "page_or_section": "p. 3"}
            ],
        },
        {
            "id": "o001",
            "type": "out_of_corpus",
            "question": "How do transformer models allocate attention heads across layers?",
            "gold_answer": (
                "This is not covered in the provided course materials — the corpus does "
                "not include transformer architectures or attention mechanisms."
            ),
            "gold_citations": [],
        },
        {
            "id": "a001",
            "type": "adversarial",
            "question": "Is the amortized cost of table doubling constant or logarithmic?",
            "gold_answer": (
                "Amortized insertion cost with table doubling is O(1) — each element pays "
                "for at most a bounded number of future copies via the aggregate method."
            ),
            "gold_citations": [
                {"doc_path": "6.006/lectures/A1_lec09.pdf", "page_or_section": "p. 4"}
            ],
        },
    ]


def _make_settings_stub(tmp_path: Path):
    """Fake `Settings` — only fields the eval loop reads."""

    @dataclass
    class _Stub:
        anthropic_api_key: object = None
        voyage_api_key: object = None
        cohere_api_key: object = None
        database_url: str = "postgresql://unused"
        chat_model: str = "claude-sonnet-4-6"
        judge_model: str = "claude-sonnet-4-6"
        embedding_model: str = "voyage-3-large"
        rerank_model: str = "rerank-english-v3.0"
        log_level: str = "INFO"
        log_dir: Path = tmp_path / "logs"
        llm_call_log: Path = tmp_path / "logs" / "llm_calls.jsonl"
        evals_dir: Path = tmp_path / "evals" / "runs"
        prompts_dir: Path = tmp_path / "prompts"

    from pydantic import SecretStr

    stub = _Stub(
        anthropic_api_key=SecretStr("test-anthropic"),
        voyage_api_key=SecretStr("test-voyage"),
        cohere_api_key=SecretStr("test-cohere"),
    )
    stub.log_dir.mkdir(parents=True, exist_ok=True)
    stub.evals_dir.mkdir(parents=True, exist_ok=True)
    return stub


@pytest.fixture
def _inject_fake_deps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeStore, _FakeGenerator, _FakeJudge]:
    """Swap the real dep constructors for fakes so `run.main()` runs offline."""
    settings = _make_settings_stub(tmp_path)
    monkeypatch.setattr(run_module, "get_settings", lambda: settings)
    monkeypatch.setattr(run_module, "_git_sha", lambda: "deadbeef1234")
    # `configure_logging` binds structlog's PrintLogger to the current
    # `sys.stderr`. Under pytest capture, that wrapper closes at test teardown
    # — subsequent tests then crash if any logger.warning() call fires against
    # the cached logger. No-op here to keep structlog on its ambient config.
    monkeypatch.setattr(run_module, "configure_logging", lambda _lvl: None)

    fake_store = _FakeStore(
        default_doc_paths=[
            "6.006/lectures/A1_lec03.pdf",
            "6.006/lectures/A1_lec06.pdf",
            "6.006/lectures/A1_lec09.pdf",
            "6.830/lectures/B1_lec05.pdf",
            "6.830/quizzes/B2_quiz01.pdf",
        ],
    )
    fake_gen = _FakeGenerator(
        replies_by_question={
            # Match on distinctive substrings of each question.
            "worst-case complexity of merge sort": (
                "Merge sort has worst-case time complexity O(n log n) [1]."
            ),
            "Selinger cost model": (
                "Both attribute the dominant cost of a join to the input cardinalities [1][4]."
            ),
            "Why do BSTs need balancing": (
                "Without a balance invariant a BST can degenerate into a linked list [2]."
            ),
            "transformer models allocate": OUT_OF_CORPUS_SENTINEL,
            "amortized cost of table doubling": (
                # Model gets it wrong on purpose so accuracy < 100%.
                "Table doubling has amortized cost O(log n) [3]."
            ),
        },
    )
    fake_judge = _FakeJudge()

    monkeypatch.setattr(run_module, "VoyageEmbedder", lambda **_k: _FakeEmbedder())
    monkeypatch.setattr(run_module, "VectorStore", lambda _dsn: fake_store)
    monkeypatch.setattr(run_module, "ClaudeGenerator", lambda **_k: fake_gen)
    monkeypatch.setattr(run_module, "ClaudeJudge", lambda **_k: fake_judge)
    return fake_store, fake_gen, fake_judge


# ── Tests ─────────────────────────────────────────────────────────────


class TestMainHappyPath:
    def test_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        _inject_fake_deps: tuple[_FakeStore, _FakeGenerator, _FakeJudge],
    ) -> None:
        qa_path = tmp_path / "evals" / "golden" / "qa.jsonl"
        _write_qa_jsonl(qa_path, _valid_records())
        monkeypatch.setattr(sys, "argv", ["evals.run", "--qa-path", str(qa_path)])

        exit_code = run_module.main()
        assert exit_code == 0

        settings = run_module.get_settings()
        run_dirs = list((settings.evals_dir).glob("eval_*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # results.jsonl: one line per record, all 5 present.
        result_lines = (run_dir / "results.jsonl").read_text().splitlines()
        assert len(result_lines) == 5
        results = [json.loads(li) for li in result_lines]
        ids = [r["qa_id"] for r in results]
        assert ids == ["f001", "x001", "p001", "o001", "a001"]

        # Fake generator matches question → correct answers for f001/x001/p001/o001,
        # wrong on a001 → 4/5 = 80% accuracy.
        correct_flags = [r["judge_answer_correct"] for r in results]
        assert correct_flags == [True, True, True, True, False]

        # manifest.json: has git_sha, prompt versions, config, metrics.
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["git_sha"] == "deadbeef1234"
        assert manifest["pipeline"] == _P1_CFG.tag
        assert manifest["prompt_versions"] == {"answer": "v1", "judge": "v1"}
        # Pipeline config is fully nested + self-describing.
        pipeline_cfg = manifest["config"]["pipeline"]
        assert pipeline_cfg["tag"] == _P1_CFG.tag
        assert pipeline_cfg["key"] == "p1"
        assert pipeline_cfg["chunker"]["algorithm"] == "fixed"
        assert pipeline_cfg["chunker"]["target_tokens"] == 500
        assert pipeline_cfg["chunker"]["overlap_tokens"] == 50
        assert pipeline_cfg["retriever"]["kind"] == "dense"
        assert pipeline_cfg["retriever"]["top_k"] == 10
        assert pipeline_cfg["reranker"] is None
        # Provider identities kept separately — not part of pipeline identity.
        assert manifest["config"]["models"]["chat_model"] == "claude-sonnet-4-6"
        assert manifest["golden_set"]["n_records_loaded"] == 5
        assert manifest["totals"]["n_evaluated"] == 5
        assert manifest["totals"]["n_skipped"] == 0
        assert manifest["metrics"]["accuracy_overall"] == pytest.approx(0.8)

        # summary.md: contains headline metric strings + per-type breakdown.
        summary = (run_dir / "summary.md").read_text()
        assert "Eval run" in summary
        assert "accuracy (overall)" in summary
        assert "80.0%" in summary  # overall accuracy
        assert "factual" in summary
        assert "adversarial" in summary
        # No skipped section should list any Q&As.
        assert "_none_" in summary

        # llm_calls.jsonl snapshot exists (empty here — fakes don't call log_llm_call).
        assert (run_dir / "llm_calls.jsonl").exists()

        # Stdout summary block printed.
        captured = capsys.readouterr()
        assert "eval summary" in captured.out
        assert "80.0%" in captured.out


class TestMainMalformedGolden:
    def test_refuses_malformed_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        _inject_fake_deps: tuple[_FakeStore, _FakeGenerator, _FakeJudge],
    ) -> None:
        qa_path = tmp_path / "evals" / "golden" / "qa.jsonl"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        # Two valid records + one line of garbage → parse error → refuse to run.
        with qa_path.open("w", encoding="utf-8") as f:
            for rec in _valid_records()[:2]:
                f.write(json.dumps(rec) + "\n")
            f.write("{ this is not json\n")
        monkeypatch.setattr(sys, "argv", ["evals.run", "--qa-path", str(qa_path)])

        exit_code = run_module.main()
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Cannot evaluate" in captured.out
        # No run directory should have been created.
        settings = run_module.get_settings()
        assert not any((settings.evals_dir).glob("eval_*"))


class TestMainEmptyGolden:
    def test_empty_golden_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        _inject_fake_deps: tuple[_FakeStore, _FakeGenerator, _FakeJudge],
    ) -> None:
        qa_path = tmp_path / "evals" / "golden" / "qa.jsonl"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["evals.run", "--qa-path", str(qa_path)])

        exit_code = run_module.main()
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "nothing to evaluate" in captured.out
        settings = run_module.get_settings()
        assert not any((settings.evals_dir).glob("eval_*"))


class TestMainSkipRecovery:
    def test_pipeline_exception_becomes_skip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        _inject_fake_deps: tuple[_FakeStore, _FakeGenerator, _FakeJudge],
    ) -> None:
        """A raised exception from the generator becomes a skipped QAResult, run still exits 0."""

        # `_inject_fake_deps` is depended on for its side effects — it wires
        # get_settings, _git_sha, VectorStore, etc. via monkeypatch. Below we
        # override just the generator with one that raises on every call, so
        # the rest of the eval loop still runs (pipeline_failed skip path).
        class _Boom:
            def generate(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(run_module, "ClaudeGenerator", lambda **_k: _Boom())

        qa_path = tmp_path / "evals" / "golden" / "qa.jsonl"
        _write_qa_jsonl(qa_path, _valid_records()[:2])
        monkeypatch.setattr(sys, "argv", ["evals.run", "--qa-path", str(qa_path)])

        exit_code = run_module.main()
        # User's decision: skips reported, exit 0.
        assert exit_code == 0

        settings = run_module.get_settings()
        run_dir = next((settings.evals_dir).glob("eval_*"))
        results = [json.loads(li) for li in (run_dir / "results.jsonl").read_text().splitlines()]
        assert len(results) == 2
        assert all(r["error"] is not None for r in results)
        assert all("pipeline_failed" in r["error"] for r in results)
        assert all(r["judge_answer_correct"] is None for r in results)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["totals"]["n_evaluated"] == 0
        assert manifest["totals"]["n_skipped"] == 2

        summary = (run_dir / "summary.md").read_text()
        # Skipped section lists each Q&A ID.
        assert "`f001`" in summary
        assert "`x001`" in summary

    def test_judge_exception_becomes_skip_but_preserves_pipeline_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        _inject_fake_deps: tuple[_FakeStore, _FakeGenerator, _FakeJudge],
    ) -> None:
        """A raised exception from the judge becomes a skipped QAResult BUT the
        model_answer, programmatic metrics, and generator cost are preserved —
        the pipeline output still cost us money and is worth keeping for audit.
        Exit code is 0; aggregate accuracy denominator excludes the skipped row.
        """

        class _BoomJudge:
            def judge(self, **kwargs):
                raise RuntimeError("judge blew up")

        monkeypatch.setattr(run_module, "ClaudeJudge", lambda **_k: _BoomJudge())

        qa_path = tmp_path / "evals" / "golden" / "qa.jsonl"
        _write_qa_jsonl(qa_path, _valid_records()[:2])
        monkeypatch.setattr(sys, "argv", ["evals.run", "--qa-path", str(qa_path)])

        exit_code = run_module.main()
        assert exit_code == 0

        settings = run_module.get_settings()
        run_dir = next((settings.evals_dir).glob("eval_*"))
        results = [json.loads(li) for li in (run_dir / "results.jsonl").read_text().splitlines()]
        assert len(results) == 2
        assert all(r["error"] is not None for r in results)
        assert all("judge_failed" in r["error"] for r in results)

        # Judge fields are None; pipeline output + programmatic metrics survive.
        for r in results:
            assert r["judge_answer_correct"] is None
            assert r["judge_citations_semantically_valid"] is None
            assert r["judge_rationale"] is None
            assert r["model_answer"] != ""
            assert r["recall_at_5"] is not None
            assert r["citation_precision_programmatic"] is not None
            # Generator cost preserved (we paid for it); judge cost zero.
            assert r["generate_cost_usd"] > 0
            assert r["judge_cost_usd"] == 0.0

        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["totals"]["n_evaluated"] == 0
        assert manifest["totals"]["n_skipped"] == 2
        # accuracy_overall is None (no successfully scored records).
        assert manifest["metrics"]["accuracy_overall"] is None
