# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The executable judge object the judge-execution runtime owns.

The two existing ``JudgeDefinition`` classes deliberately stop short of a runnable judge:
``enforce/judgedef.py`` is the *lint target* (axes, output-field order, checklist), and
``judges/types.py`` is the *level-carrying identity* (``judge_id``/``level``/``model``/
``prompt_version``) whose docstring notes that the full executable judge object lives elsewhere.
This module is that object.

``ExecutableJudge`` is a frozen, vendor-free data holder that (1) satisfies the calibration gate's
structural ``Judge`` protocol (``judge_id``/``model``/``prompt_version``), so it can be handed
straight to ``certifies``/``enforce_score_gate``; (2) carries the ``level`` it attributes at; and
(3) adds the prompt template + ``output_fields`` order the runtime needs to build the forced-output
schema. No vendor type appears here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from calibrated_agent_evaluation.judges.types import JudgeLevel


@dataclass(frozen=True)
class ExecutableJudge:
    """A fully runnable binary judge.

    ``model`` is the Inspect model string the runtime calls through the ``execute/`` seam;
    ``prompt_version`` labels ``prompt_template``. Calibration is pinned to ``(model,
    prompt_version)`` plus :attr:`fingerprint`, so a change to any of them invalidates the record
    (``certifies``). ``level`` is ``"workflow"`` (whole trace) or ``"span"`` (one agent's
    input+output); the deterministic ``"process"`` judge lives in the level layer and needs no
    runtime. ``output_fields`` is the ordered field list of the forced-structured output; a
    non-empty ``critique`` MUST come strictly before ``verdict``, enforced when the schema is built
    (:func:`calibrated_agent_evaluation.judges.verdict_schema.build_verdict_schema`).

    ``temperature`` is sent to the model only when set; ``None`` sends no sampling parameter and
    leaves the provider's default. Newer models reject the parameter (Claude 4.7 and later on
    Bedrock), so it is opt-in. It is part of :attr:`fingerprint`, so changing it voids the
    judge's records like any prompt edit; give the changed judge a new ``prompt_version``.

    ``failure_mode`` and ``agent`` name what the judge measures (a judge file always sets the
    mode, and ``agent`` for a span judge). They are part of :attr:`fingerprint` when set, so
    pointing a certified judge at another mode or agent voids its records: its error rates were
    measured on the old one.
    """

    judge_id: str
    model: str
    prompt_version: str
    level: JudgeLevel
    prompt_template: str
    output_fields: tuple[str, ...] = ("critique", "verdict")
    temperature: float | None = None
    failure_mode: str | None = None
    agent: str | None = None

    def __post_init__(self) -> None:
        t = self.temperature
        if t is not None and (isinstance(t, bool) or not isinstance(t, int | float) or t < 0):
            raise ValueError(f"temperature must be a non-negative number or None, got {t!r}")

    @property
    def fingerprint(self) -> str:
        """A hash of everything that decides this judge's verdicts.

        Covers the model, level, prompt template, output fields, sampling parameters and, when
        set, the failure mode and agent. Records stamp it, so editing the prompt in place without
        bumping ``prompt_version`` still voids the judge's calibration and bias records.
        """
        doc: dict[str, object] = {
            "model": self.model,
            "level": self.level,
            "prompt_template": self.prompt_template,
            "output_fields": list(self.output_fields),
            "sampling": self.sampling(),
        }
        # Added only when set, so a judge with no target keeps the fingerprint it always had.
        if self.failure_mode is not None:
            doc["failure_mode"] = self.failure_mode
        if self.agent is not None:
            doc["agent"] = self.agent
        blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def sampling(self) -> dict[str, float]:
        """The sampling parameters this judge sends: ``{"temperature": t}``, or none."""
        return {} if self.temperature is None else {"temperature": float(self.temperature)}
