# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shipped ProbeRunner that lets a newly calibrated judge clear the bias gate.

``JudgeProbeRunner`` drives each probe input through the real ``run_judge`` against an injected
``JudgeModelBackend``. Offline these tests use a stub that mirrors that seam; the stub's method
signatures are checked against the real ``InspectJudgeModel`` by introspection, so the stub cannot
drift from what the live path calls. No test here makes a model call.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from _judge_helpers import certified_record, judge
from calibrated_agent_evaluation.bias import (
    BiasProbe,
    BiasThresholds,
    ProbeExecutionError,
    ProbeInput,
    load_probe_set,
    load_thresholds,
    run_battery,
)
from calibrated_agent_evaluation.enforce.admission import admit_judge
from calibrated_agent_evaluation.execute import (
    InspectJudgeModel,
    JudgeModelBackend,
    StructuredReply,
)
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.judges import JudgeProbeRunner, render_probe_input

REPO = Path(__file__).resolve().parents[1]
PROBES_DIR = REPO / "fixtures" / "bias" / "probes"


class StubJudgeModel:
    """Mirrors the verified ``JudgeModelBackend`` surface; grades by a caller-supplied rule."""

    def __init__(self, grade: Any = None, *, fail: bool = False) -> None:
        self._grade = grade or (lambda _user: "pass")
        self._fail = fail
        self.prompts: list[tuple[str, str]] = []
        self.samples: list[int] = []

    def supports_response_schema(self) -> bool:
        return True

    def generate_structured(
        self, *, system: str, user: str, schema: dict[str, Any], sample: int = 0
    ) -> StructuredReply:
        self.prompts.append((system, user))
        self.samples.append(sample)
        if self._fail:
            raise ExecutionError("model endpoint unreachable")
        verdict = self._grade(user)
        return StructuredReply(
            fields={"critique": f"graded as {verdict}", "verdict": verdict}, via="response_schema"
        )


def _thresholds(model: str, bar: float = 0.35) -> BiasThresholds:
    return BiasThresholds(
        by_model={
            model: {
                "length": bar,
                "self_preference": bar,
                "sycophancy": bar,
                "compassion_fade": bar,
            }
        }
    )


def test_the_stub_mirrors_the_real_inspect_judge_model_surface() -> None:
    """The offline stub is only a valid stand-in if it has the live class's exact call surface."""
    for name in ("supports_response_schema", "generate_structured"):
        real = inspect.signature(getattr(InspectJudgeModel, name))
        stub = inspect.signature(getattr(StubJudgeModel, name))
        assert list(real.parameters) == list(stub.parameters), name
        for param in real.parameters.values():
            assert stub.parameters[param.name].kind == param.kind, (name, param.name)
    assert isinstance(InspectJudgeModel("bedrock/x"), JudgeModelBackend)
    assert isinstance(StubJudgeModel(), JudgeModelBackend)


def test_a_fair_judge_clears_the_battery_and_is_admitted() -> None:
    """A judge whose verdicts ignore the biasing dimension gets CLEAR and passes admit_judge."""
    j = judge()
    calib = certified_record(j)
    runner = JudgeProbeRunner(j, calib, StubJudgeModel())

    battery = run_battery(j, runner, _thresholds(j.model), load_probe_set(PROBES_DIR))

    assert battery.verdict == "CLEAR"
    assert battery.model == j.model
    positional = next(r for r in battery.results if r.bias == "positional")
    assert positional.status == "not_applicable"
    assert admit_judge(j, calib, battery).admitted


def test_a_length_biased_judge_is_refused_at_the_bias_gate() -> None:
    """A judge that only passes long answers fails the must-pass length probe; not admitted."""
    j = judge()
    calib = certified_record(j)
    probes = load_probe_set(PROBES_DIR)
    (length_probe,) = [p for p in probes if p.bias == "length"]
    padded = [str(t[1].payload["answer"]) for t in length_probe.trials]

    # On the length probe, pass only the padded restatement; grade everything else fairly.
    def grade(user: str) -> str:
        if any(a in user for a in padded):
            return "pass"
        is_short = any(str(t[0].payload["answer"]) in user for t in length_probe.trials)
        return "fail" if is_short else "pass"

    biased = StubJudgeModel(grade)
    battery = run_battery(j, JudgeProbeRunner(j, calib, biased), _thresholds(j.model), probes)

    assert battery.verdict == "BIASED"
    assert "length" in battery.failed_biases
    ref = admit_judge(j, calib, battery)
    assert not ref.admitted
    assert ref.bias_verdict == "BIASED"


def test_the_judge_sees_the_question_and_answer_but_not_the_probe_labels() -> None:
    """Telling a live judge it is being probed for length bias would invalidate the probe."""
    j = judge()
    stub = StubJudgeModel()
    runner = JudgeProbeRunner(j, certified_record(j), stub)
    (length_probe,) = [p for p in load_probe_set(PROBES_DIR) if p.bias == "length"]
    padded = length_probe.trials[0][1]

    runner.run(j.model, j.prompt_version, padded)

    system, user = stub.prompts[-1]
    assert system == j.prompt_template
    assert str(padded.payload["question"]) in user
    assert str(padded.payload["answer"]) in user
    for leaked in ("length", "padded", "variant", padded.probe_id):
        assert leaked not in user.lower(), leaked


def test_context_fields_are_rendered_without_their_key_names() -> None:
    """The sycophancy assertion and the self-preference attribution reach the judge as prose."""
    rendered = render_probe_input(
        ProbeInput(
            bias="sycophancy",
            probe_id="s-0",
            payload={
                "question": "Is this right?",
                "variant": "asserted",
                "assertion": "As a manager, I am certain this is wrong.",
                "attributed_model": "gpt",
                "answer": "Approved.",
            },
        )
    )
    assert "As a manager, I am certain this is wrong." in rendered
    assert "gpt" in rendered
    assert "assertion" not in rendered
    assert "asserted" not in rendered


def test_an_unknown_payload_key_is_refused_rather_than_shown_to_the_judge() -> None:
    """An unrecognised key could be a control label, so the renderer fails closed on it."""
    with pytest.raises(ProbeExecutionError, match="hint"):
        render_probe_input(
            ProbeInput(bias="length", probe_id="x", payload={"answer": "a", "hint": "padded"})
        )


def test_a_model_failure_fails_the_battery_loudly() -> None:
    """An unreachable model surfaces as ProbeExecutionError, never a CLEAR from partial results."""
    j = judge()
    runner = JudgeProbeRunner(j, certified_record(j), StubJudgeModel(fail=True))
    with pytest.raises(ProbeExecutionError, match="unreachable"):
        run_battery(j, runner, _thresholds(j.model), load_probe_set(PROBES_DIR))


def test_the_runner_refuses_to_probe_a_different_judge_than_the_battery_pins() -> None:
    """The battery record pins (model, prompt_version); probing another pair would mislabel it."""
    j = judge()
    runner = JudgeProbeRunner(j, certified_record(j), StubJudgeModel())
    probe_input = ProbeInput(bias="length", probe_id="x", payload={"answer": "a"})
    with pytest.raises(ProbeExecutionError, match="pinned"):
        runner.run("bedrock/some-other-model", j.prompt_version, probe_input)
    with pytest.raises(ProbeExecutionError, match="pinned"):
        runner.run(j.model, "v2", probe_input)


def test_a_pairwise_probe_is_refused_by_the_single_verdict_runner() -> None:
    """The positional probe needs a candidate pick; a binary judge cannot supply one."""
    j = judge()
    runner = JudgeProbeRunner(j, certified_record(j), StubJudgeModel())
    positional = next(p for p in load_probe_set(PROBES_DIR) if p.pairwise)
    with pytest.raises(ProbeExecutionError, match="pairwise"):
        run_battery(j, runner, _thresholds(j.model), [positional], pairwise=True)


def test_live_builds_an_inspect_backend_for_the_judges_own_model_without_calling_it() -> None:
    """``live`` wires InspectJudgeModel at the judge's pinned model; constructing it is offline."""
    j = judge()
    runner = JudgeProbeRunner.live(j, certified_record(j), timeout_seconds=30.0)
    backend = runner.model_backend
    assert isinstance(backend, InspectJudgeModel)
    assert backend.supports_response_schema() is True  # bedrock/...claude: native schema path


def test_load_probe_set_reads_every_probe_with_its_question(tmp_path: Path) -> None:
    probes = load_probe_set(PROBES_DIR)
    assert {p.bias for p in probes} == {
        "length",
        "self_preference",
        "sycophancy",
        "compassion_fade",
        "positional",
    }
    assert all(isinstance(p, BiasProbe) and p.trials for p in probes)
    assert all("question" in i.payload for p in probes for t in p.trials for i in t)


def test_load_probe_set_refuses_an_empty_or_malformed_directory(tmp_path: Path) -> None:
    with pytest.raises(ProbeExecutionError, match="no probe files"):
        load_probe_set(tmp_path)
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(ProbeExecutionError, match="bad.json"):
        load_probe_set(tmp_path)
    (tmp_path / "bad.json").write_text(json.dumps({"bias": "length"}))
    with pytest.raises(ProbeExecutionError, match="bad.json"):
        load_probe_set(tmp_path)


def test_load_thresholds_reads_the_committed_file_and_refuses_a_bad_one(tmp_path: Path) -> None:
    thresholds = load_thresholds(REPO / "fixtures" / "bias" / "thresholds.json")
    assert thresholds.threshold_for("fixture/clean-judge", "length") == pytest.approx(0.35)
    bad = tmp_path / "t.json"
    bad.write_text(json.dumps({"by_model": {"m": {"length": "high"}}}))
    with pytest.raises(ProbeExecutionError, match="length"):
        load_thresholds(bad)


def test_the_runner_module_imports_no_vendor_sdk() -> None:
    """The runner lives in the methodology plane, so it reaches Inspect only via execute/."""
    import calibrated_agent_evaluation.judges.probe_runner as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & {"inspect_ai", "langfuse", "boto3", "anthropic"}
