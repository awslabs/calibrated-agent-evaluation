# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The judge file ``judge`` and ``run`` read: one binary judge, as JSON.

```json
{
  "judge_id": "orderbot-refund-resolved",
  "model": "bedrock/global.anthropic.claude-sonnet-4-6",
  "prompt_version": "v1",
  "level": "workflow",
  "prompt_template": "You are grading ...",
  "failure_mode": "trusted_broken_policy_tool",
  "agent": null
}
```

The first five fields are :class:`~calibrated_agent_evaluation.judges.ExecutableJudge`'s
(``output_fields`` is optional and defaults to critique-then-verdict). ``temperature`` is optional
and omitted by default, so no sampling parameter is sent; newer models reject one. ``failure_mode``
is the one mode the binary judge detects, and ``agent`` the crew agent that mode belongs to (omit it
for a single agent). Together
they are the :class:`~calibrated_agent_evaluation.run.ModeKey` the judge's verdicts are stamped with
and its trend is reported under. A ``"span"`` judge reads every span of ``agent``, so it must name
one; a ``"workflow"`` judge reads the whole trace. A judge file names a judge; whether that judge
may score is decided by its calibration and bias records, never by the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.errors import CliError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.enforce import Scale

__all__ = ["JudgeFile", "declares_agentcore_backend", "load_judge_file"]

#: The modules the prompt-judge layer imports from the ``[stats]`` extra.
JUDGE_FILE_MODULES = ("numpy", "scipy", "statsmodels")
_REQUIRED = ("judge_id", "model", "prompt_version", "level", "prompt_template", "failure_mode")
_OPTIONAL = ("agent", "output_fields", "temperature")
# Keys that describe a scoring shape the binary judge format refuses or cannot express.
_SHAPE_KEYS = ("axes", "scale", "checklist", "weights", "weighted_collapse")
_BINARY_SCALES = frozenset({"binary", "pass_fail", "pass-fail", "pass/fail"})


@dataclass(frozen=True)
class JudgeFile:
    """A parsed judge file: the runnable judge plus the failure mode it detects."""

    judge: Any  # calibrated_agent_evaluation.judges.ExecutableJudge, imported on use
    failure_mode: str
    agent: str | None
    path: Path


def load_judge_file(path: str | Path) -> JudgeFile:
    """Parse ``path``; a missing file, bad JSON, or a missing/unknown field is a ``CliError``.

    A prompt judge is built on the calibration engine, so a missing ``[stats]`` extra is refused
    with its install command before the judge layer is imported.
    """
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras(*JUDGE_FILE_MODULES)
    from calibrated_agent_evaluation.judges import ExecutableJudge

    src = Path(path)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read judge file {src}: {exc.strerror or exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"judge file {src} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CliError(f"judge file {src} must hold one JSON object")
    if doc.get("backend") == "agentcore":
        raise CliError(
            f"judge file {src} is an AgentCore judge (backend: agentcore), and this command takes "
            "a prompt judge file. `calibrated-agent-evaluation judge`, "
            "`calibrated-agent-evaluation calibrate`, `calibrated-agent-evaluation run` and "
            "`calibrated-agent-evaluation "
            "dump --judges-only` take AgentCore judge files; `calibrated-agent-evaluation judge "
            "--check` validates one"
        )
    _lint(doc, src)
    missing = [k for k in _REQUIRED if not isinstance(doc.get(k), str) or not doc[k].strip()]
    if missing:
        raise CliError(f"judge file {src} is missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(doc) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise CliError(f"judge file {src} has unknown field(s): {', '.join(unknown)}")
    if doc["level"] not in ("workflow", "span"):
        raise CliError(
            f"judge file {src}: level must be 'workflow' or 'span', got {doc['level']!r}"
        )
    agent = doc.get("agent")
    if agent is not None and (not isinstance(agent, str) or not agent.strip()):
        raise CliError(f"judge file {src}: agent must be a non-empty string or null")
    if doc["level"] == "span" and agent is None:
        raise CliError(
            f"judge file {src}: a span-level judge needs an agent, the crew agent whose spans it "
            "judges (a span judge is calibrated and rated per agent)"
        )
    kwargs: dict[str, Any] = {k: doc[k] for k in _REQUIRED if k != "failure_mode"}
    if "output_fields" in doc:
        fields = doc["output_fields"]
        if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
            raise CliError(f"judge file {src}: output_fields must be a list of strings")
        kwargs["output_fields"] = tuple(fields)
    if "temperature" in doc:
        temperature = doc["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise CliError(f"judge file {src}: temperature must be a number")
        if temperature < 0:
            raise CliError(f"judge file {src}: temperature must be non-negative")
        kwargs["temperature"] = float(temperature)
    kwargs["failure_mode"] = doc["failure_mode"]
    kwargs["agent"] = agent
    return JudgeFile(
        judge=ExecutableJudge(**kwargs),
        failure_mode=doc["failure_mode"],
        agent=agent,
        path=src,
    )


def _lint(doc: dict[str, Any], src: Path) -> None:
    """Lint the file's scoring shape; a refused anti-pattern raises its ``EnforcementError``.

    A file with no shape keys is linted as the one binary axis it declares, which catches a
    ``verdict`` ordered before ``critique``. A shape key the lint does not refuse (several binary
    axes, an unweighted checklist) is still not something one binary judge file can run, so it
    is refused in plain words, with the lint's nudges in the message.
    """
    from calibrated_agent_evaluation.enforce import (
        Axis,
        ChecklistItem,
        EnforcementContext,
        JudgeDefinition,
        lint_judge_definition,
    )

    judge_id = str(doc["judge_id"]) if isinstance(doc.get("judge_id"), str) else src.stem
    mode = str(doc["failure_mode"]) if isinstance(doc.get("failure_mode"), str) else "verdict"
    fields = doc.get("output_fields")
    output_fields = (
        tuple(fields)
        if isinstance(fields, list) and all(isinstance(f, str) for f in fields)
        else ("critique", "verdict")
    )
    axes = _axes(doc, src, mode)
    items = _checklist(doc, src)
    weighted = (
        doc.get("weighted_collapse") not in (None, False)
        or "weights" in doc
        or any("weight" in item for item in items)
    )
    definition = JudgeDefinition(
        judge_id=judge_id,
        axes=tuple(Axis(name=name, scale=scale) for name, scale in axes),
        output_fields=output_fields,
        checklist=tuple(
            ChecklistItem(
                name=str(item.get("name", f"item {i + 1}")),
                critical=bool(item.get("critical", False)),
            )
            for i, item in enumerate(items)
        ),
        weighted_collapse=weighted,
    )
    decisions = lint_judge_definition(definition, EnforcementContext())
    present = [k for k in _SHAPE_KEYS if k in doc]
    if not present:
        return
    nudges = "".join(f" {d.message} {d.nudge or ''}".rstrip() for d in decisions)
    raise CliError(
        f"judge file {src} declares {', '.join(present)}, which a judge file cannot express: a "
        "judge file holds one binary (PASS/FAIL) judge for one failure mode. Write one judge "
        "file per axis or checklist item." + nudges
    )


def _axes(doc: dict[str, Any], src: Path, mode: str) -> list[tuple[str, Scale]]:
    """The file's axes as ``(name, scale)``; ``scale`` is ``binary`` or ``likert``."""
    if "axes" not in doc:
        return [(mode, _scale(doc.get("scale", "binary")))]
    raw = doc["axes"]
    if not isinstance(raw, list):
        raise CliError(f"judge file {src}: axes must be a list")
    out: list[tuple[str, Scale]] = []
    for i, axis in enumerate(raw):
        if isinstance(axis, str):
            out.append((axis, _scale(doc.get("scale", "binary"))))
        elif isinstance(axis, dict):
            out.append(
                (str(axis.get("name", f"axis {i + 1}")), _scale(axis.get("scale", "binary")))
            )
        else:
            raise CliError(f"judge file {src}: each axis must be a name or an object")
    return out


def _scale(value: object) -> Scale:
    """``binary`` for a pass/fail scale; any other (``likert``, ``1-5``, ``0-10``) is Likert."""
    if isinstance(value, str) and value.strip().lower() in _BINARY_SCALES:
        return "binary"
    return "likert"


def _checklist(doc: dict[str, Any], src: Path) -> list[dict[str, Any]]:
    raw = doc.get("checklist", [])
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise CliError(f"judge file {src}: checklist must be a list of objects")
    return raw


def declares_agentcore_backend(path: str | Path) -> bool:
    """Whether ``path`` is a JSON judge file declaring ``backend: agentcore``.

    Reads only the parsed JSON, so no AgentCore code is imported; a file that cannot be read or
    parsed answers ``False`` and :func:`load_judge_file` reports the problem.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(doc, dict) and doc.get("backend") == "agentcore"
