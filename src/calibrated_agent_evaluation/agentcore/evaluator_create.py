# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a Calibrated Agent Evaluation prompt judge as an AgentCore LLM-as-judge evaluator.

:func:`build_evaluator_request` turns a prompt judge (its prompt, model and failure mode) into a
``CreateEvaluator`` request: level ``TRACE``, the judge's prompt followed by a fixed block that
places AgentCore's trace-level ``{context}`` and ``{assistant_turn}`` placeholders (and no
ground-truth placeholder, which online configs refuse), and a two-rung scale, ``0.0 FAIL`` and
``1.0 PASS``, that the AgentCore judge's outcome rule reads as FAIL and PASS.

:func:`create_evaluator` refuses a name already in use, shows the request through the shared
preview-and-confirm step, creates the evaluator, waits for it to become ``ACTIVE``, and writes an
``agentcore`` judge file naming it, so every later command reads it through the AgentCore judge
loader. An evaluator that fails or does not become active in time is not recorded.

This module takes the prompt judge's fields as a :class:`PromptJudge`, so it imports nothing from
the CLI, and imports the AWS SDK only inside its calls.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreResourceExists,
    AgentCoreResourceNotActive,
)
from calibrated_agent_evaluation.agentcore.online import WritePreview, call_aws, send_confirmed

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.guards import ReadOnlyGuard
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "CREATE_EVALUATOR_OPERATIONS",
    "TRACE_INSTRUCTIONS",
    "CreatedEvaluator",
    "PromptJudge",
    "bedrock_model_id",
    "build_evaluator_request",
    "create_evaluator",
    "create_evaluator_guard",
    "default_evaluator_name",
    "write_agentcore_judge_file",
]

_SERVICE = "bedrock-agentcore-control"
# AgentCore's evaluator name rule (CreateEvaluator.evaluatorName).
_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}")
# Statuses from which an evaluator being created never becomes ACTIVE.
_FAILED = frozenset({"CREATE_FAILED", "UPDATE_FAILED", "DELETING"})
_DESCRIPTION_LIMIT = 200
_BEDROCK_PREFIX = "bedrock/"

CREATE_EVALUATOR_OPERATIONS: frozenset[tuple[str, str]] = frozenset(
    (_SERVICE, name) for name in ("ListEvaluators", "CreateEvaluator", "GetEvaluator")
)
"""The calls creating an evaluator makes; the CLI's clients refuse any other."""

TRACE_INSTRUCTIONS = (
    "\n\nThe trace to grade follows. The conversation and the agent's tool calls:\n"
    "{context}\n\n"
    "The agent's final response:\n"
    "{assistant_turn}\n\n"
    "Give your verdict on the rating scale: FAIL if the trace shows the failure the instructions "
    "above describe, PASS if it does not."
)
"""The block placed after the judge's own prompt: AgentCore fills the two trace placeholders."""


@dataclass(frozen=True)
class PromptJudge:
    """The fields of a prompt judge file that the evaluator is built from.

    Attributes:
        judge_id: The judge's id; the ``agentcore`` judge file keeps it.
        model: The judge's model, as Calibrated Agent Evaluation routes it (``bedrock/<model id>``
            or a bare id).
        prompt_version: The judge's prompt version; it is part of the default evaluator name.
        prompt_template: The judge's prompt.
        failure_mode: The failure mode the judge detects.
        agent: The crew agent the failure mode belongs to, or ``None``.
        temperature: The judge's sampling temperature, or ``None`` for 0.
    """

    judge_id: str
    model: str
    prompt_version: str
    prompt_template: str
    failure_mode: str
    agent: str | None = None
    temperature: float | None = None


def create_evaluator_guard() -> ReadOnlyGuard:
    """A client guard that refuses every call creating an evaluator does not make."""
    from calibrated_agent_evaluation.agentcore.guards import ReadOnlyGuard

    return ReadOnlyGuard(CREATE_EVALUATOR_OPERATIONS)


@dataclass(frozen=True)
class CreatedEvaluator:
    """An evaluator created and active, and the ``agentcore`` judge file that names it.

    Attributes:
        evaluator_id: The evaluator id AgentCore generated.
        evaluator_arn: The evaluator's ARN.
        evaluator_name: The name it was created with.
        updated_at: Its ``updatedAt`` once active, as the AgentCore judge reports it for display.
        judge_file: The ``agentcore`` judge file written for it.
    """

    evaluator_id: str
    evaluator_arn: str
    evaluator_name: str
    updated_at: str
    judge_file: Path


def default_evaluator_name(judge: PromptJudge) -> str:
    """The evaluator name for ``judge``: its id and prompt version, in AgentCore's name rules.

    Two versions of one judge get different names, so each can be created as its own evaluator.
    """
    raw = f"{judge.judge_id}_{judge.prompt_version}"
    name = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    if not name[:1].isalpha():
        name = f"j_{name}"
    return name[:48]


def bedrock_model_id(model: str) -> str:
    """The Bedrock model id AgentCore runs for ``model``, with the library's routing prefix removed.

    Raises:
        AgentCoreError: ``model`` names another provider (an AgentCore LLM-as-judge evaluator
            runs a Bedrock model) or is empty.
    """
    model_id = model.removeprefix(_BEDROCK_PREFIX)
    if not model_id or "/" in model_id:
        raise AgentCoreError(
            f"the judge's model {model!r} is not an Amazon Bedrock model; an AgentCore evaluator "
            "runs a Bedrock model, so name one as bedrock/<model id>"
        )
    return model_id


def build_evaluator_request(
    judge_id: str,
    model: str,
    prompt_template: str,
    failure_mode: str,
    *,
    name: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """The ``CreateEvaluator`` request for a prompt judge, with no AWS call.

    Args:
        judge_id: The judge's id; conformed to AgentCore's name rules when ``name`` is not given.
        model: The judge's model (``bedrock/<model id>`` or a bare Bedrock model id).
        prompt_template: The judge's prompt; the trace block follows it.
        failure_mode: The failure mode, named in the scale's definitions.
        name: The evaluator name to create.
        temperature: The model's sampling temperature; 0 when not given.

    Returns:
        The request: level ``TRACE``, the instructions, the two-rung scale and the model.

    Raises:
        AgentCoreError: ``name`` breaks AgentCore's name rules, ``model`` is not a Bedrock model,
            or ``temperature`` is outside 0 to 1, the range AgentCore takes.
    """
    if temperature is not None and not 0.0 <= temperature <= 1.0:
        raise AgentCoreError(
            f"the judge's temperature {temperature} is outside 0 to 1, the range an AgentCore "
            "evaluator's model takes"
        )
    evaluator_name = name or re.sub(r"[^a-zA-Z0-9_]", "_", judge_id)
    if not _NAME.fullmatch(evaluator_name):
        raise AgentCoreError(
            f"the evaluator name {evaluator_name!r} is not one AgentCore accepts: it must start "
            "with a letter and hold only letters, digits and underscores, up to 48 characters"
        )
    return {
        "evaluatorName": evaluator_name,
        "level": "TRACE",
        "evaluatorConfig": {
            "llmAsAJudge": {
                "instructions": prompt_template + TRACE_INSTRUCTIONS,
                "ratingScale": {
                    "numerical": [
                        {
                            "value": 0.0,
                            "label": "FAIL",
                            "definition": f"The trace shows the failure mode {failure_mode}.",
                        },
                        {
                            "value": 1.0,
                            "label": "PASS",
                            "definition": (
                                f"The trace does not show the failure mode {failure_mode}."
                            ),
                        },
                    ]
                },
                "modelConfig": {
                    "bedrockEvaluatorModelConfig": {
                        "modelId": bedrock_model_id(model),
                        "inferenceConfig": {
                            "temperature": 0.0 if temperature is None else float(temperature)
                        },
                    }
                },
            }
        },
    }


def create_evaluator(
    judge: PromptJudge,
    *,
    out: Path,
    clients: AwsClients,
    confirm: Callable[[WritePreview], bool],
    name: str | None = None,
    wait_seconds: float = 120.0,
    poll_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> CreatedEvaluator:
    """Create ``judge`` as an AgentCore evaluator and write the ``agentcore`` judge file for it.

    Args:
        judge: The prompt judge to create.
        out: Where to write the ``agentcore`` judge file; it must not exist yet.
        clients: The AWS clients.
        confirm: Receives the ``CreateEvaluator`` preview; returns whether to send it.
        name: The evaluator name; :func:`default_evaluator_name` when not given.
        wait_seconds: How long to wait for the evaluator to become ``ACTIVE``.
        poll_seconds: How often to read its status while waiting.
        sleep: Waits between status reads (injectable for tests).
        clock: The monotonic clock the wait is measured on (injectable for tests).

    Returns:
        The created evaluator and the judge file written for it.

    Raises:
        AgentCoreError: ``out`` already exists, the request cannot be built, a call fails, or the
            judge file cannot be written once the evaluator exists (the message names it).
        AgentCoreResourceExists: An evaluator with the name already exists; none was created.
        AgentCoreRunDeclined: ``confirm`` declined; nothing was created.
        AgentCoreResourceNotActive: The evaluator failed or was not ``ACTIVE`` within the wait; it
            is not recorded, and the message names it so it can be deleted.
    """
    if out.exists():
        raise AgentCoreError(
            f"{out} already exists, so no evaluator was created; pass another output file or "
            "remove it first"
        )
    evaluator_name = name or default_evaluator_name(judge)
    request = build_evaluator_request(
        judge.judge_id,
        judge.model,
        judge.prompt_template,
        judge.failure_mode,
        name=evaluator_name,
        temperature=judge.temperature,
    )
    if evaluator_name in _existing_names(clients):
        raise AgentCoreResourceExists("evaluator", evaluator_name)
    request["description"] = (
        f"Calibrated Agent Evaluation judge {judge.judge_id} prompt {judge.prompt_version}, "
        "failure mode "
        f"{judge.failure_mode}"
    )[:_DESCRIPTION_LIMIT]
    request["clientToken"] = f"calibrated-agent-evaluation-{uuid.uuid4().hex}"
    try:
        created = send_confirmed(
            clients,
            _SERVICE,
            "CreateEvaluator",
            request,
            summary=(
                f"Creates the AgentCore evaluator {evaluator_name} from judge {judge.judge_id} "
                f"({judge.prompt_version}), level TRACE, scale 0.0 FAIL / 1.0 PASS."
            ),
            confirm=confirm,
        )
    except AgentCoreError as exc:
        if _error_code(exc) == "ConflictException":
            raise AgentCoreResourceExists("evaluator", evaluator_name) from exc
        raise
    evaluator_id = str(created["evaluatorId"])
    active = _wait_active(
        clients, evaluator_id, wait_seconds, poll_seconds=poll_seconds, sleep=sleep, clock=clock
    )
    try:
        write_agentcore_judge_file(
            out,
            judge_id=judge.judge_id,
            evaluator_id=evaluator_id,
            failure_mode=judge.failure_mode,
            agent=judge.agent,
        )
    except OSError as exc:
        raise AgentCoreError(
            f"the evaluator {evaluator_id} was created and is active, but its judge file {out} "
            f"could not be written ({type(exc).__name__}); write it by hand naming evaluator_id "
            f"{evaluator_id!r}, or delete the evaluator"
        ) from exc
    return CreatedEvaluator(
        evaluator_id=evaluator_id,
        evaluator_arn=str(active["evaluatorArn"]),
        evaluator_name=evaluator_name,
        updated_at=_iso(active.get("updatedAt")),
        judge_file=out,
    )


def write_agentcore_judge_file(
    path: Path, *, judge_id: str, evaluator_id: str, failure_mode: str, agent: str | None
) -> None:
    """Write an ``agentcore`` judge file the AgentCore judge loader reads back.

    Raises:
        OSError: The file cannot be created, or already exists.
    """
    doc: dict[str, Any] = {
        "backend": "agentcore",
        "judge_id": judge_id,
        "evaluator_id": evaluator_id,
        "failure_mode": failure_mode,
        "level": "workflow",
    }
    if agent is not None:
        doc["agent"] = agent
    with path.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=2) + "\n")


def _existing_names(clients: AwsClients) -> set[str]:
    """Every evaluator name in the account and region, read to exhaustion."""
    names: set[str] = set()
    params: dict[str, Any] = {}
    while True:
        page = call_aws(clients, _SERVICE, "ListEvaluators", params)
        names.update(str(e["evaluatorName"]) for e in page.get("evaluators", []))
        token = page.get("nextToken")
        if not token:
            return names
        params = {"nextToken": token}


def _wait_active(
    clients: AwsClients,
    evaluator_id: str,
    wait_seconds: float,
    *,
    poll_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Read the evaluator until it is ``ACTIVE``; raise on a failed status or when the wait ends."""
    deadline = clock() + wait_seconds
    created = (f"evaluator {evaluator_id}",)
    while True:
        got = call_aws(
            clients, _SERVICE, "GetEvaluator", {"evaluatorId": evaluator_id}, created=created
        )
        status = str(got.get("status", ""))
        if status == "ACTIVE":
            return got
        if status in _FAILED or clock() >= deadline:
            raise AgentCoreResourceNotActive("evaluator", evaluator_id, status)
        sleep(poll_seconds)


def _error_code(exc: BaseException) -> str | None:
    """The AWS error code behind a wrapped call failure, if it was a service error."""
    response = getattr(exc.__cause__, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        return code if isinstance(code, str) else None
    return None


def _iso(value: Any) -> str:
    """``updatedAt`` as the AgentCore judge reports it for display (ISO-8601 for a datetime)."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else ""
