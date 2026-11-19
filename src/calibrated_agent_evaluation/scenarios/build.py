# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build scenario slices from human FAIL labels and an answer key; pure, no I/O.

Only trace-level human labels (no ``span_id``) decide whether a trace becomes a scenario: a
span-level FAIL describes one step and can sit under a trace-level PASS. A trace becomes a
scenario when every rater's newest trace-level label on it is FAIL. Its expected answer is the
``expected`` each rater's newest FAIL label that records one gives, as
``calibrated-agent-evaluation score`` resolves it; traces whose raters disagree on the outcome or
record different expected outputs are left out and named.

The answer key supplies what labels don't hold: per failure mode, the expected tool order and
plain-language assertions. A scenario gets the ordered union of its modes' assertions and the one
tool order they share; a ``by_trace`` entry replaces each field it sets for that trace.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.labels import newest_per_rater, resolve_expected
from calibrated_agent_evaluation.scenarios.types import (
    AnswerEntry,
    AnswerKey,
    CalibratedAgentEvaluationScenarioMeta,
    DatasetSlice,
    PredefinedScenario,
    Scenario,
    Turn,
)
from calibrated_agent_evaluation.schema.types import Label, Trace

__all__ = [
    "CONFLICTING_EXPECTED",
    "NO_INPUT",
    "RATERS_DISAGREE",
    "BuildResult",
    "build_slice",
    "expected_text",
    "failure_modes_in_scope",
    "starter_key",
]

RATERS_DISAGREE = "their raters disagree on PASS or FAIL (adjudicate them first)"
CONFLICTING_EXPECTED = "their raters recorded different expected outputs (adjudicate them first)"
NO_INPUT = "they have no input to replay"


def expected_text(expected: dict[str, Any]) -> str:
    """A structured expected output as the compact, sorted-key JSON text a turn carries."""
    return json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class BuildResult:
    """The built slice (``None`` when no trace qualified) and what was left out, by reason."""

    slice: DatasetSlice | None
    left_out: dict[str, list[str]] = field(default_factory=dict)
    unused_modes: list[str] = field(default_factory=list)

    @property
    def n_scenarios(self) -> int:
        """How many scenarios the slice holds."""
        return 0 if self.slice is None else len(self.slice.scenarios)


@dataclass(frozen=True)
class _Failure:
    trace: Trace
    modes: list[str]
    all_fails: list[Label]


def failure_modes_in_scope(
    traces: Sequence[Trace], labels: Mapping[str, Sequence[Label]], failure_mode: str | None = None
) -> list[str]:
    """The failure modes, in first-seen order, on the traces that every rater labeled FAIL."""
    modes: dict[str, None] = {}
    for failure in _failures(traces, labels, failure_mode, left_out={}):
        modes.update(dict.fromkeys(failure.modes))
    return list(modes)


def starter_key(
    traces: Sequence[Trace], labels: Mapping[str, Sequence[Label]], failure_mode: str | None = None
) -> AnswerKey:
    """An answer key listing each failure mode in scope with empty lists, for a human to fill."""
    return AnswerKey(
        by_failure_mode={
            m: AnswerEntry() for m in failure_modes_in_scope(traces, labels, failure_mode)
        }
    )


def build_slice(
    traces: Sequence[Trace],
    labels: Mapping[str, Sequence[Label]],
    key: AnswerKey | None = None,
    failure_mode: str | None = None,
) -> BuildResult:
    """A predefined slice of one single-turn scenario per trace every rater labeled FAIL, by id."""
    if key is None:
        key = AnswerKey()
    in_scope = {t.trace_id for t in traces}
    unknown = sorted(set(key.by_trace) - in_scope)
    if unknown:
        raise ScenarioError(
            f"the answer key's by_trace names trace(s) not in scope: {unknown}; fix the ids or "
            "widen the scope"
        )
    left_out: dict[str, list[str]] = {}
    scenarios: list[Scenario] = []
    ambiguous: dict[str, list[str]] = {}
    for failure in _failures(traces, labels, failure_mode, left_out):
        trace = failure.trace
        # The same rule as `calibrated-agent-evaluation score`: each rater's newest label that
        # records an expected.
        resolved = resolve_expected(
            newest_per_rater(lb for lb in failure.all_fails if lb.expected is not None).values()
        )
        if resolved.conflicting:
            left_out.setdefault(CONFLICTING_EXPECTED, []).append(trace.trace_id)
            continue
        if not trace.input:
            left_out.setdefault(NO_INPUT, []).append(trace.trace_id)
            continue
        answers = _answers_for(trace.trace_id, failure.modes, key, ambiguous)
        scenarios.append(
            PredefinedScenario(
                scenario_id=trace.trace_id,
                turns=[
                    Turn(
                        input=trace.input,
                        expected_response=None
                        if resolved.value is None
                        else expected_text(resolved.value),
                    )
                ],
                expected_trajectory=answers.expected_trajectory,
                assertions=answers.assertions,
                calibrated_agent_evaluation=CalibratedAgentEvaluationScenarioMeta(
                    source_trace_id=trace.trace_id,
                    frame=trace.frame,
                    taxonomy_cell=dict(trace.taxonomy_cell),
                    failure_modes=failure.modes,
                    expected=resolved.value,
                ),
            )
        )
    if ambiguous:
        detail = "; ".join(f"{tid}: {modes}" for tid, modes in sorted(ambiguous.items()))
        raise ScenarioError(
            "the answer key gives different expected tool orders for failure modes that share a "
            f"trace ({detail}); set the trace's expected_trajectory under by_trace"
        )
    used = {m for s in scenarios for m in s.calibrated_agent_evaluation.failure_modes}
    unused = [m for m in key.by_failure_mode if m not in used]
    # Sorted, so the same store content builds the same slice whatever order a backend reads in.
    scenarios.sort(key=lambda s: s.scenario_id)
    built = DatasetSlice(form="predefined", scenarios=scenarios) if scenarios else None
    return BuildResult(slice=built, left_out=left_out, unused_modes=unused)


def _failures(
    traces: Sequence[Trace],
    labels: Mapping[str, Sequence[Label]],
    failure_mode: str | None,
    left_out: dict[str, list[str]],
) -> list[_Failure]:
    """The traces every rater's newest trace-level human label marks FAIL, with their modes."""
    found: list[_Failure] = []
    for trace in traces:
        newest = newest_per_rater(
            lb for lb in labels.get(trace.trace_id, ()) if lb.kind == "human" and lb.span_id is None
        )
        outcomes = {lb.outcome for lb in newest.values()}
        if outcomes != {"FAIL"}:
            if outcomes == {"PASS", "FAIL"}:
                left_out.setdefault(RATERS_DISAGREE, []).append(trace.trace_id)
            continue
        considered = [
            lb for lb in newest.values() if failure_mode is None or lb.failure_mode == failure_mode
        ]
        if not considered:
            continue
        modes = list(dict.fromkeys(lb.failure_mode for lb in considered if lb.failure_mode))
        wanted = {lb.failure_mode for lb in considered}
        all_fails = [
            lb
            for lb in labels.get(trace.trace_id, ())
            if lb.kind == "human"
            and lb.span_id is None
            and lb.outcome == "FAIL"
            and (failure_mode is None or lb.failure_mode in wanted)
        ]
        found.append(_Failure(trace=trace, modes=modes, all_fails=all_fails))
    return found


def _answers_for(
    trace_id: str, modes: list[str], key: AnswerKey, ambiguous: dict[str, list[str]]
) -> AnswerEntry:
    entries = [key.by_failure_mode[m] for m in modes if m in key.by_failure_mode]
    assertions = list(dict.fromkeys(a for e in entries for a in e.assertions))
    orders = list(
        dict.fromkeys(tuple(e.expected_trajectory) for e in entries if e.expected_trajectory)
    )
    override = key.by_trace.get(trace_id)
    sets_trajectory = override is not None and "expected_trajectory" in override.model_fields_set
    if len(orders) > 1 and not sets_trajectory:
        ambiguous[trace_id] = [m for m in modes if m in key.by_failure_mode]
    trajectory = list(orders[0]) if len(orders) == 1 else []
    if override is not None:
        if "assertions" in override.model_fields_set:
            assertions = list(override.assertions)
        if sets_trajectory:
            trajectory = list(override.expected_trajectory)
    return AnswerEntry(assertions=assertions, expected_trajectory=trajectory)
