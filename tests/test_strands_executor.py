# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.execute.strands_executor — StrandsExecutor.

The positive replay tests drive the REAL ClaimsCrew Strands Graph through the committed cassette
(offline, deterministic) — the same crew the corpus froze — so a genuine owned Trace with the real
span grammar is produced, not a hand-shaped one. Error-path tests use focused stubs.
"""

from __future__ import annotations

import builtins
import json
import re
import sys
import typing
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.execute import (
    ExecutionError,
    ExecutorBackend,
    Mode,
    StrandsExecutor,
    StrandsTask,
)
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.schema.validate import validate

_REPO = Path(__file__).resolve().parents[1]

# The real ClaimsCrew replay tests drive the corpus reference crew (corpus/ tree, at the repo
# root — not under src/). Put the repo root on sys.path so `corpus.*` resolves, exactly as
# tests/corpus/conftest.py does for the corpus-integrity tests.
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# The executor's live-replay tests drive the marquee CLM-4471 ClaimsCrew (5-agent Graph) via its
# committed cassette (regenerated during so it replays a full run offline — the original was
# incomplete: its payout turn was unrecorded, a latent corpus defect this feature surfaced + fixed).
_CLM_CASSETTE = _REPO / "fixtures" / "raw" / "cassettes" / "claimscrew_strands.cassette.json"
_CLM_ORACLE = _REPO / "fixtures" / "normalized" / "CLM-4471.json"


def _claims_task(trace_id: str = "CLM-4471", **overrides: Any) -> StrandsTask:
    """Build a StrandsTask reconstructing the real ClaimsCrew Graph for the committed cassette."""
    from corpus.agents.claimscrew_strands import build_claims_graph
    from corpus.generate.scenarios import load_scenario

    scenario = load_scenario(trace_id)
    narrative = scenario.trace_input.get("narrative", "")
    claim_id = scenario.trace_input.get("claim_id", trace_id)
    amount = scenario.trace_input.get("amount_requested_usd", "")
    crew_input = f"Adjudicate claim {claim_id} for ${amount}. Narrative: {narrative}"

    def build(model: Any) -> Any:
        return build_claims_graph(model, scenario)

    kwargs: dict[str, Any] = {"mode": "legacy", "trace_id": trace_id}
    kwargs.update(overrides)
    return StrandsTask(build=build, input=crew_input, **kwargs)


def _replay_executor(**kwargs: Any) -> StrandsExecutor:
    return StrandsExecutor(cassette_path=_CLM_CASSETTE, cassette_mode=Mode.REPLAY, **kwargs)


# --- protocol conformance + vendor-type absence ------------------------------


class TestProtocolConformance:
    """StrandsExecutor structurally satisfies ExecutorBackend with no vendor types."""

    def test_is_structural_executor_backend(self) -> None:
        assert isinstance(StrandsExecutor(), ExecutorBackend)

    def test_run_returns_list_of_trace(self) -> None:
        hints = typing.get_type_hints(StrandsExecutor.run)
        assert hints["return"] == list[Trace]

    def test_no_vendor_type_in_public_signatures(self) -> None:
        # Contract: no strands/boto3 type appears in StrandsExecutor.run or StrandsTask signatures.
        for fn in (StrandsExecutor.run, StrandsExecutor.__init__, StrandsTask.__init__):
            rendered = str(typing.get_type_hints(fn))
            assert "strands" not in rendered.lower(), fn
            assert "boto3" not in rendered.lower(), fn
            assert "bedrock" not in rendered.lower(), fn


# --- the real ClaimsCrew replay path -----------------------


class TestClaimsCrewReplay:
    """Run the real ClaimsCrew Graph via the committed cassette → a valid owned Trace."""

    def test_claimscrew_replay_produces_owned_trace(self) -> None:
        with _replay_executor() as ex:
            traces = ex.run(_claims_task())
        assert len(traces) == 1
        trace = traces[0]
        assert trace.ownership == "owned"
        assert trace.spans, "the crew run must produce a populated span tree"
        validate(trace)
        # The real Strands grammar reconstructs multiple agents (supervisor + specialists).
        agents = {span.agent for span in trace.spans}
        assert len(agents) >= 2, f"expected a multi-agent crew tree, got agents={agents}"

    def test_replay_needs_no_aws_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """boto3 refuses a named profile that does not exist; replay must never build a session."""
        monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-no-such-profile")
        with _replay_executor() as ex:
            traces = ex.run(_claims_task())
        assert traces[0].ownership == "owned"

    def test_replay_accepts_a_string_cassette_path(self) -> None:
        with StrandsExecutor(cassette_path=str(_CLM_CASSETTE), cassette_mode=Mode.REPLAY) as ex:
            traces = ex.run(_claims_task())
        assert len(traces) == 1
        validate(traces[0])

    def test_replay_trace_matches_frozen_oracle_structure(self) -> None:
        with _replay_executor() as ex:
            trace = ex.run(_claims_task())[0]
        oracle = json.loads(_CLM_ORACLE.read_text())
        # Contract against the frozen oracle: same trace id, ownership, and agent set.
        assert trace.trace_id == oracle["trace_id"]
        assert trace.ownership == oracle["ownership"]
        oracle_agents = {s["agent"] for s in oracle["spans"]}
        assert {span.agent for span in trace.spans} == oracle_agents

    def test_otlp_json_export_persisted_and_reparses(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        with _replay_executor(log_dir=log_dir) as ex:
            ex.run(_claims_task())
        exports = list(log_dir.glob("*.otlp.json"))
        assert exports, "run() must persist the captured OTLP-JSON export to log_dir"
        payload = json.loads(exports[0].read_text())
        assert "resourceSpans" in payload, "the export must be canonical OTLP-JSON"

    def test_swarm_topology_replays_to_owned_trace(self) -> None:
        # claims Agent/Graph/Swarm. Beyond the Graph (above), prove the Swarm topology:
        # the claims-triage Swarm (a different Strands multiagent construct — invoke_swarm root +
        # a handoff_to_agent coordination tool) replays through the SAME executor to a valid Trace.
        from corpus.agents.claimscrew_swarm import _TASK, build_claims_swarm

        swarm_cassette = _REPO / "fixtures" / "raw" / "cassettes" / "claimscrew_swarm.cassette.json"
        task = StrandsTask(
            build=build_claims_swarm, input=_TASK, mode="legacy", trace_id="CLM-7720"
        )
        with StrandsExecutor(cassette_path=swarm_cassette, cassette_mode=Mode.REPLAY) as ex:
            trace = ex.run(task)[0]
        assert trace.ownership == "owned"
        assert trace.spans
        validate(trace)
        # The Swarm's two agents both appear in the reconstructed tree.
        agents = {span.agent for span in trace.spans}
        assert {"triage", "adjuster"} <= agents, f"expected both swarm agents, got {agents}"


# --- negative: validation failure refuses --------------------------------------------


class TestValidationFailure:
    """A normalized-but-invalid Trace makes run raise, returning nothing."""

    def test_invalid_trace_raises_execution_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        from calibrated_agent_evaluation.schema.errors import SchemaError

        def boom(_trace: Trace) -> None:
            raise SchemaError("duplicate span_id 'dup'")

        # `calibrated_agent_evaluation.schema.validate` the *name* is shadowed by the re-exported
        # function, so patch the real module object (which the executor imports `validate` from)
        # directly.
        validate_mod = importlib.import_module("calibrated_agent_evaluation.schema.validate")
        monkeypatch.setattr(validate_mod, "validate", boom)
        with _replay_executor() as ex:
            with pytest.raises(ExecutionError, match="failed validation"):
                ex.run(_claims_task())


# --- empty capture refuses ----------------------------------------------------------


class TestEmptyCapture:
    """A run that emits no spans raises, never a degraded/empty Trace."""

    def test_no_spans_raises(self) -> None:
        # A build whose runnable does nothing (emits no spans) → empty capture → refusal.
        task = StrandsTask(build=lambda _model: lambda _inp: None, input="x", trace_id="EMPTY")
        with _replay_executor() as ex:
            with pytest.raises(ExecutionError, match="no spans captured"):
                ex.run(task)


# --- offline replay is network-free; miss refuses ----------------------------


class TestReplayOffline:
    """Replay makes no network call; a miss is a typed ExecutionError."""

    def test_replay_makes_no_boto3_network_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Spy: if the real Bedrock stream were hit, boto3's client creation would fire. In REPLAY
        # the CassetteModel serves recorded events, so inner BedrockModel.stream is never awaited.
        import strands.models.bedrock as bedrock_mod

        calls: list[str] = []
        real_stream = bedrock_mod.BedrockModel.stream

        async def spy_stream(
            self: Any, *a: Any, **k: Any
        ) -> Any:  # pragma: no cover - must not run
            calls.append("stream")
            async for ev in real_stream(self, *a, **k):
                yield ev

        monkeypatch.setattr(bedrock_mod.BedrockModel, "stream", spy_stream)
        with _replay_executor() as ex:
            ex.run(_claims_task())
        assert calls == [], "REPLAY must serve recorded events — no live BedrockModel.stream call"

    def test_replay_miss_raises_execution_error(self) -> None:
        # A task whose input never appears in the committed cassette → cassette miss in REPLAY.
        task = _claims_task()
        task = StrandsTask(
            build=task.build,
            input="a totally unrecorded crew request",
            mode="legacy",
            trace_id="MISS",
        )
        with _replay_executor() as ex:
            with pytest.raises(ExecutionError, match="cassette"):
                ex.run(task)


# --- record mode (hermetic — inner stream stubbed, no live Bedrock) ----------


def _committed_events_by_fingerprint() -> dict[str, list[Any]]:
    """Load the committed CLM-4471 cassette's recorded stream events, keyed by request fingerprint.

    Lets a hermetic record-mode test replay the *real* recorded events as the stubbed inner
    BedrockModel.stream output — exercising the RECORD write path offline, deterministically.
    """
    doc = json.loads(_CLM_CASSETTE.read_text())
    return {e["fingerprint"]: e["response"]["events"] for e in doc["entries"]}


class TestRecordMode:
    """Record writes a replayable cassette; an unreachable model raises, bounded."""

    def test_record_writes_cassette_then_replays(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hermetic: stub the inner BedrockModel.stream to emit the committed events (no live call),
        # so the executor's CassetteModel in RECORD mode writes a fresh cassette we can replay.
        import strands.models.bedrock as bedrock_mod

        from calibrated_agent_evaluation.execute.cassette import fingerprint

        events_by_fp = _committed_events_by_fingerprint()

        async def stub_stream(
            self: Any,
            messages: Any,
            tool_specs: Any = None,
            system_prompt: str | None = None,
            **_k: Any,
        ) -> Any:
            from calibrated_agent_evaluation.execute.strands_cassette_model import _jsonable

            fp = fingerprint(
                {
                    "messages": _jsonable(messages),
                    "tool_specs": _jsonable(tool_specs),
                    "system_prompt": system_prompt,
                }
            )
            for ev in events_by_fp[fp]:
                yield ev

        monkeypatch.setattr(bedrock_mod.BedrockModel, "stream", stub_stream)

        fresh = tmp_path / "recorded.cassette.json"
        with StrandsExecutor(cassette_path=fresh, cassette_mode=Mode.RECORD) as ex:
            recorded = ex.run(_claims_task())[0]
        assert recorded.ownership == "owned"
        assert fresh.exists()
        entries = json.loads(fresh.read_text())["entries"]
        assert len(entries) >= 1, "record mode must persist at least one cassette entry"

        # The freshly-recorded cassette now replays offline (no stub) to the same structural trace.
        monkeypatch.undo()
        with StrandsExecutor(cassette_path=fresh, cassette_mode=Mode.REPLAY) as ex:
            replayed = ex.run(_claims_task())[0]
        assert {s.agent for s in replayed.spans} == {s.agent for s in recorded.spans}

    def test_unreachable_model_in_record_raises_chained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub the inner stream to raise a connection error (mirrors BedrockModel.stream's
        # synchronous raise on an unreachable peer) — the executor must wrap it, not hang/leak.
        import strands.models.bedrock as bedrock_mod

        async def unreachable_stream(self: Any, *a: Any, **k: Any) -> Any:
            raise ConnectionError("bedrock endpoint unreachable")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(bedrock_mod.BedrockModel, "stream", unreachable_stream)
        fresh = tmp_path / "rec.cassette.json"
        with StrandsExecutor(cassette_path=fresh, cassette_mode=Mode.RECORD) as ex:
            with pytest.raises(ExecutionError, match="strands run failed") as exc_info:
                ex.run(_claims_task())
        # The underlying connection error is preserved in the chain (never a hang / raw escape).
        assert exc_info.value.__cause__ is not None


# --- temp-dir default + export not discarded -----------------------------------------


class TestLogDir:
    """Default temp dir is created and the export is persisted (not silently discarded)."""

    def test_default_temp_dir_created_and_export_persisted(self) -> None:
        with _replay_executor() as ex:
            ex.run(_claims_task())
            log_dir = ex.log_dir
            assert log_dir.exists()
            assert "calibrated_agent_evaluation_strands_" in str(log_dir)
            exports = list(log_dir.glob("*.otlp.json"))
            assert exports
            assert exports[0].stat().st_size > 0

    def test_export_survives_run_return(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "persist"
        with _replay_executor(log_dir=log_dir) as ex:
            ex.run(_claims_task())
        # After run() returns, the export is still readable (no silent mid-run cleanup).
        exports = list(log_dir.glob("*.otlp.json"))
        assert exports
        assert json.loads(exports[0].read_text())


# --- corrupt cassette refuses --------------------------------------------------------


class TestCorruptCassette:
    """A corrupt cassette raises ExecutionError naming the path; no live fall-through."""

    def test_corrupt_cassette_raises_naming_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.cassette.json"
        bad.write_text("{ this is not valid json")
        ex = StrandsExecutor(cassette_path=bad, cassette_mode=Mode.REPLAY)
        with ex:
            with pytest.raises(ExecutionError, match=str(bad.name)):
                ex.run(_claims_task())


# --- headers pass through ------------------------------------------------------------


class TestHeaderPassthrough:
    """Mode / trace_id / taxonomy_cell reach the produced Trace."""

    def test_trace_id_and_taxonomy_reach_the_trace(self) -> None:
        cell = {"feature": "claim_adjudication", "persona": "P1_consumer"}
        with _replay_executor() as ex:
            trace = ex.run(_claims_task(taxonomy_cell=cell))[0]
        assert trace.trace_id == "CLM-4471"
        assert trace.taxonomy_cell == cell

    def test_capability_absent_no_bogus_header(self) -> None:
        # Default mode, no explicit taxonomy_cell → the Trace carries an empty cell (wire/default
        # fallback), nothing bogus injected.
        task = _claims_task(taxonomy_cell={})
        with _replay_executor() as ex:
            trace = ex.run(task)[0]
        assert trace.taxonomy_cell == {}


# --- lazy import + missing-extra refusal ------------------------------------


class TestLazyImportAndMissingExtra:
    """Import succeeds without [strands]; calling without it raises typed."""

    def test_import_execute_without_strands_and_names_resolve(self) -> None:
        import calibrated_agent_evaluation.execute as ex

        assert ex.StrandsExecutor.__name__ == "StrandsExecutor"
        assert ex.StrandsTask.__name__ == "StrandsTask"

    def test_importing_execute_does_not_eagerly_import_strands(self) -> None:
        import subprocess
        import sys

        program = (
            "import sys, calibrated_agent_evaluation.execute\n"
            "assert 'strands' not in sys.modules, 'strands eagerly imported'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_missing_strands_raises_naming_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def blocked(name: str, *a: object, **k: object) -> object:
            if name == "strands" or name.startswith("strands."):
                raise ImportError("No module named 'strands'")
            return real_import(name, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)
        with StrandsExecutor(cassette_path=_CLM_CASSETTE, cassette_mode=Mode.REPLAY) as ex:
            with pytest.raises(ExecutionError, match=re.escape(install_hint("strands"))):
                ex.run(_claims_task())


# --- run exceptions wrapped ----------------------------------------------------------


class TestRunExceptionWrapped:
    """Any Strands run exception is wrapped in ExecutionError (chained), not raw."""

    def test_build_raising_is_wrapped_and_chained(self) -> None:
        def build(_model: Any) -> Any:
            raise ValueError("graph build blew up")

        task = StrandsTask(build=build, input="x", trace_id="BOOM")
        with _replay_executor() as ex:
            with pytest.raises(ExecutionError, match="strands run failed") as exc_info:
                ex.run(task)
        assert isinstance(exc_info.value.__cause__, ValueError)


# --- drop-in: generate_and_run drives StrandsExecutor ----------------------------


class TestStrandsExecutorDropIn:
    """StrandsExecutor is a drop-in ExecutorBackend for the generate_and_run."""

    def test_generate_and_run_with_strands_executor(self) -> None:
        from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
        from calibrated_agent_evaluation.synth import generate_and_run

        class Renderer:
            def render(self, cell: dict[str, str]) -> str:
                return "Adjudicate a water-damage claim narrative."

        cell = {"feature": "claim_adjudication", "persona": "P1_consumer"}

        class StrandsTaskBuilder:
            # A real Strands TaskBuilder MUST thread the synthesized input's taxonomy cell
            # into the StrandsTask so the executor's Trace carries it — otherwise the frame
            # propagation check rejects the mismatch , the synthesis-to-executor composition
            # contract.
            def build(self, inputs: list[Any]) -> Any:
                return _claims_task(taxonomy_cell=inputs[0].cell)

        frame = FrameResult(
            rows=[FrameRow(cell=cell, frame="discovery")],
            frame="discovery",
        )
        with _replay_executor(frame="discovery") as ex:
            traces = generate_and_run(frame, Renderer(), ex, StrandsTaskBuilder())
        assert len(traces) == 1
        assert traces[0].frame == "discovery"
        assert traces[0].taxonomy_cell == cell
        assert traces[0].ownership == "owned"


# --- the decoding parameter is the caller's, not the executor's ------------------------


class TestTemperatureIsCallerControlled:
    """`temperature=0.0` used to be hard-coded, which silently restricted this executor to model
    generations that still accept an explicit temperature. The default is unchanged; `None` is the
    escape hatch, and it must omit the key rather than pass a null through."""

    @staticmethod
    def _captured_kwargs(**executor_kwargs: Any) -> dict[str, Any]:
        seen: dict[str, Any] = {}

        class FakeBedrockModel:
            def __init__(self, **kwargs: Any) -> None:
                seen.update(kwargs)

        executor = StrandsExecutor(**executor_kwargs)
        executor._build_model(FakeBedrockModel, None)  # the seam under test
        return seen

    def test_the_default_still_sends_zero(self) -> None:
        assert self._captured_kwargs()["temperature"] == 0.0

    def test_an_explicit_value_is_forwarded(self) -> None:
        assert self._captured_kwargs(temperature=0.7)["temperature"] == 0.7

    def test_none_omits_the_parameter_entirely(self) -> None:
        # Negative path: the models this exists for reject a null temperature as surely as a value,
        # so passing `None` through would not have fixed anything.
        assert "temperature" not in self._captured_kwargs(temperature=None)
