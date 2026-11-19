# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the judge-facing commands share: the judge's model, its targets, and its human gold.

``calibrated-agent-evaluation calibrate``, ``calibrated-agent-evaluation bias`` and
``calibrated-agent-evaluation judge`` all run one judge file's judge. The model is
the judge's own (live, through the ``[inspect]`` extra) or a recorded cassette replayed with
``--cassette``. ``calibrated-agent-evaluation calibrate`` and ``calibrated-agent-evaluation judge``
also need the targets the judge reads, which
for a workflow judge is the whole trace and for a span judge each span of its agent with the path
that led to it, and ``calibrated-agent-evaluation calibrate`` needs the human gold each target is
graded against.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.schema.types import Label, Trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.cli.judgefile import JudgeFile
    from calibrated_agent_evaluation.judges import JudgeTarget

__all__ = ["GoldSet", "gold_for", "model_backend", "targets_for"]


def model_backend(args: argparse.Namespace, judge: Any) -> Any:
    """The judge's model: a ``--cassette`` replay or recording, else its live model via Inspect.

    ``--cassette-mode replay`` (the default) answers from the cassette alone and never calls a
    model, so a miss is a typed error. ``record`` calls the live model for each reply the
    cassette does not hold and writes it in, each write atomic and merged with what other
    recorders added, so a run that fails part way keeps the replies it paid for.
    """
    cassette = getattr(args, "cassette", None)
    mode = str(getattr(args, "cassette_mode", None) or "replay")
    if mode not in ("replay", "record"):
        raise UsageError(f"--cassette-mode must be replay or record, got {mode!r}")
    if not cassette:
        if mode == "record":
            raise CliError("--cassette-mode record needs --cassette PATH, the file to record into")
        _require_live(judge, "or replay a recorded run with --cassette")
        from calibrated_agent_evaluation.execute import InspectJudgeModel

        return InspectJudgeModel(judge.model, temperature=judge.temperature)

    from calibrated_agent_evaluation.execute import Cassette, CassetteJudgeModel, Mode

    path = Path(cassette)
    if mode == "replay":
        if not path.is_file():
            raise CliError(
                f"--cassette {path} is not a file; record one with --cassette-mode record"
            )
        return CassetteJudgeModel(
            Cassette(path, mode=Mode.REPLAY),
            supports_schema=True,
            temperature=judge.temperature,
            model_id=judge.model,
        )
    if path.exists() and not path.is_file():
        raise CliError(f"--cassette {path} is not a file to record into")
    _require_live(judge, "recording a cassette calls the judge's live model")
    return CassetteJudgeModel(
        Cassette(path, mode=Mode.RECORD),
        supports_schema=True,
        record_model=judge.model,
        temperature=judge.temperature,
        model_id=judge.model,
    )


def _require_live(judge: Any, why: str) -> None:
    from calibrated_agent_evaluation.cli.config import probe_extra

    probe = probe_extra("inspect_ai")
    if not probe.available:
        raise CliError(
            f"running {judge.model!r} needs the inspect extra: {probe.install_hint} ({why})"
        )


def targets_for(jf: JudgeFile, traces: list[Trace]) -> list[JudgeTarget]:
    """The targets the judge reads in ``traces``.

    A workflow judge reads each whole trace. A span judge reads every span of the judge file's
    agent, each with its ancestor path (:func:`calibrated_agent_evaluation.judges.span_target`); a
    trace with no span of that agent (a probed trace, or one where the agent did not run) gives
    none.
    """
    from calibrated_agent_evaluation.judges import span_target, workflow_target

    if jf.judge.level == "workflow":
        return [workflow_target(t) for t in traces]
    return [span_target(t, s) for t in traces for s in _agent_spans(jf, t)]


@dataclass(frozen=True)
class GoldSet:
    """The human-gold targets a judge is calibrated on, and what was left out and why.

    ``pairs`` holds each target with the human label it is graded against. ``left_out`` counts the
    human labels in scope that are not gold for this judge, by reason, so a calibration never
    shrinks its set without saying so.
    """

    pairs: list[tuple[JudgeTarget, Label]]
    left_out: dict[str, int] = field(default_factory=dict)

    @property
    def n_fail(self) -> int:
        """How many gold labels are FAIL."""
        return sum(1 for _, g in self.pairs if g.outcome == "FAIL")

    @property
    def n_pass(self) -> int:
        """How many gold labels are PASS."""
        return sum(1 for _, g in self.pairs if g.outcome == "PASS")


# Why a human label in scope is not gold for the judge being calibrated.
OTHER_MODE = "FAIL label(s) for another failure mode or agent"
DISAGREE = "target(s) whose human labels disagree for this mode (adjudicate them first)"
SPAN_FOR_WORKFLOW = "span-level label(s), which do not grade a workflow judge"
TRACE_FOR_SPAN = "trace-level label(s), which do not grade a span judge"
OTHER_AGENT = "span label(s) on another agent's span"


def gold_for(
    jf: JudgeFile, traces: list[Trace], labels_by_trace: dict[str, list[Label]]
) -> GoldSet:
    """The human gold for ``jf``'s judge on ``traces``, graded at the judge's level.

    A workflow judge for mode M is graded against each trace's trace-level human labels: a FAIL
    for M (and for the judge file's agent, when it names one, read off the label's culprit span)
    is a positive, a PASS a negative. A span judge is graded against the per-span human labels on
    each span of its agent the same way, including a span defect recorded on a trace whose outcome
    passed; trace-level labels never grade it. A FAIL for another mode is
    not evidence either way and is left out, as is a target whose labels say both (two raters who
    disagree), since its gold is unknown until adjudicated. Every left-out label is counted.
    """
    from calibrated_agent_evaluation.judges import span_target, workflow_target

    span_judge = jf.judge.level == "span"
    pairs: list[tuple[JudgeTarget, Label]] = []
    left_out: Counter[str] = Counter()
    for trace in traces:
        human = [lb for lb in labels_by_trace.get(trace.trace_id, []) if lb.kind == "human"]
        trace_level = [lb for lb in human if lb.span_id is None]
        span_level = [lb for lb in human if lb.span_id is not None]
        if not span_judge:
            left_out[SPAN_FOR_WORKFLOW] += len(span_level)
            gold = _gold_of(jf, trace, trace_level, left_out)
            if gold is not None:
                pairs.append((workflow_target(trace), gold))
            continue
        left_out[TRACE_FOR_SPAN] += len(trace_level)
        spans = _agent_spans(jf, trace)
        mine = {s.span_id for s in spans}
        left_out[OTHER_AGENT] += sum(1 for lb in span_level if lb.span_id not in mine)
        for span in spans:
            on_span = [lb for lb in span_level if lb.span_id == span.span_id]
            gold = _gold_of(jf, trace, on_span, left_out)
            if gold is not None:
                pairs.append((span_target(trace, span), gold))
    return GoldSet(pairs=pairs, left_out={k: v for k, v in left_out.items() if v})


def _gold_of(
    jf: JudgeFile, trace: Trace, labels: list[Label], left_out: Counter[str]
) -> Label | None:
    """The one gold label for a target, or ``None`` (no gold, or raters who disagree)."""
    positives, negatives, other = _classify(jf, trace, labels)
    left_out[OTHER_MODE] += len(other)
    if positives and negatives:
        left_out[DISAGREE] += 1
        return None
    return (positives or negatives or [None])[0]


def _classify(
    jf: JudgeFile, trace: Trace, labels: list[Label]
) -> tuple[list[Label], list[Label], list[Label]]:
    agents = {s.span_id: s.agent for s in trace.spans}
    positives: list[Label] = []
    negatives: list[Label] = []
    other: list[Label] = []
    for label in labels:
        # A span label is on the judge's agent by construction; a trace label names its agent
        # through the culprit span the labeler recorded in the walk-back.
        on_agent = (
            jf.agent is None
            or label.span_id is not None
            or agents.get(label.culprit_span or "") == jf.agent
        )
        if label.outcome == "PASS":
            negatives.append(label)
        elif label.failure_mode == jf.failure_mode and on_agent:
            positives.append(label)
        else:
            other.append(label)
    return positives, negatives, other


def _agent_spans(jf: JudgeFile, trace: Trace) -> list[Any]:
    return sorted((s for s in trace.spans if s.agent == jf.agent), key=lambda s: s.seq)
