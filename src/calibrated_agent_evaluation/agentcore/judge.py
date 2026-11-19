# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An AgentCore evaluator run as a judge here: identity, fingerprint and calibratability.

An AgentCore evaluator is driven through the same calibration and enforcement gates as a local
judge. This module builds the judge identity those gates pin to (it satisfies
:class:`calibrated_agent_evaluation.calibrate.engine.Judge` structurally) and decides whether an
evaluator can be certified at all.

AWS upgrades a built-in or third-party evaluator without notice, so there is no version that pins
its behaviour; those stay advisory and the existing gate refuses them. Only an evaluator whose
content a user actually controls can be certified: a ``Custom`` evaluator, a ``CustomDerived`` one
with a pinned model, or a ``CustomCode`` one pinned to a qualified Lambda version. The pin rides the
judge fingerprint, so an evaluator edit (a new ``updatedAt``, model or Lambda version) changes the
fingerprint and the existing ``pin_mismatch`` invalidates the stale calibration, with no new pin
mechanism here.

Vendor-free: the AWS read that produces the ``GetEvaluator`` document lives in the evaluate layer;
this module maps an already-decoded dict, so it holds no boto3.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreAdvisoryOnly,
    AgentCoreError,
    AgentCoreEvaluatorLevelUnsupported,
    AgentCoreEvaluatorUnreadable,
    AgentCoreNoPassDirection,
    AgentCoreRatingScaleMalformed,
    AgentCoreScoreOffScale,
)
from calibrated_agent_evaluation.schema.types import Outcome

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

JudgeLevel = Literal["workflow", "span", "session"]

# Two rung values are the same scale point within this tolerance, so a value that is an evaluator's
# own rung (e.g. 0.17 reported as 0.17000000001) is recognised despite float representation.
_SCALE_TOLERANCE = 1e-9

# AgentCore evaluators whose high score is the bad outcome. Nothing in the evaluator document says
# which way a score runs, so this is a maintained, live-verified fact: FailureAnalysis publishes
# 1.0 "Failures Found", and Toxicity and Bias scored a clean answer 0.0. An evaluator not listed
# keeps "a high score is a pass".
_INVERTED_EVALUATORS = frozenset(
    {"Builtin.FailureAnalysis", "ThirdParty.DeepEval.Toxicity", "ThirdParty.DeepEval.Bias"}
)
# Evaluators whose score has no pass direction at all: a refusal is right for a harmful request
# and wrong for a benign one, and a funny answer is neither a pass nor a fail.
_NO_PASS_DIRECTION = frozenset({"Builtin.Refusal", "ThirdParty.AutoEval.Humor"})


def scores_inverted(evaluator_ids: Iterable[str]) -> bool:
    """Whether the evaluator, named by any of ``evaluator_ids``, scores failures high.

    The ids are the names one evaluator goes by (its id, its name, the tail of its ARN). Raises
    :class:`AgentCoreNoPassDirection` when it is an evaluator whose score has no pass direction.
    """
    names = set(evaluator_ids)
    blocked = names & _NO_PASS_DIRECTION
    if blocked:
        raise AgentCoreNoPassDirection(
            f"evaluator {sorted(blocked)[0]} has no defined pass direction (whether a high score "
            "is good depends on the use), so no PASS/FAIL is derived from it; refused rather "
            "than guessed"
        )
    return bool(names & _INVERTED_EVALUATORS)


# The evaluator level each judge level runs: a workflow judge scores one trace (TRACE), a session
# judge one simulated-conversation trace as a session (SESSION). TOOL_CALL needs a tool-call span
# Calibrated Agent Evaluation does not serialize, so no judge level runs it.
_RUNNABLE_LEVELS: dict[str, str] = {"workflow": "TRACE", "session": "SESSION"}


def require_runnable_level(
    evaluator_level: str | None, *, evaluator_id: str, level: JudgeLevel = "workflow"
) -> None:
    """Refuse an evaluator whose ``GetEvaluator`` level the judge's ``level`` cannot send.

    Raises :class:`AgentCoreEvaluatorLevelUnsupported` unless ``evaluator_level`` is the one
    ``level`` runs (``TRACE`` for ``workflow``, ``SESSION`` for ``session``), naming both, or when
    ``level`` runs no evaluator at all (``span``).
    """
    expected = _RUNNABLE_LEVELS.get(level)
    if expected is not None and evaluator_level == expected:
        return
    found = f"a {evaluator_level} evaluator" if evaluator_level else "an evaluator with no level"
    if expected is None:
        raise AgentCoreEvaluatorLevelUnsupported(
            f"judge level {level!r} runs no AgentCore evaluator (workflow runs TRACE and session "
            f"runs SESSION evaluators); evaluator {evaluator_id!r} is {found}, so it is refused "
            "before any billed call"
        )
    raise AgentCoreEvaluatorLevelUnsupported(
        f"evaluator {evaluator_id!r} is {found}, and a {level} judge runs {expected} evaluators "
        "(workflow judges score one trace, session judges one simulated conversation; a TOOL_CALL "
        "evaluator needs tool-call spans Calibrated Agent Evaluation does not send), so it is "
        "refused before any "
        "billed call"
    )


def require_trace_level(evaluator_level: str | None, *, evaluator_id: str) -> None:
    """Refuse an evaluator that is not ``TRACE``-level (a ``workflow`` judge's level check).

    Raises :class:`AgentCoreEvaluatorLevelUnsupported` naming the level, or that there is none.
    """
    require_runnable_level(evaluator_level, evaluator_id=evaluator_id, level="workflow")


@dataclass(frozen=True)
class ScaleValue:
    """One rung of an evaluator's published numerical rating scale: a value and its label."""

    value: float
    label: str


class CalibratabilityClass(StrEnum):
    """Whether an AgentCore evaluator can be certified, or may only ever be advisory."""

    CALIBRATABLE = "calibratable"
    ADVISORY_ONLY = "advisory-only"


def _model_id(model_config: dict[str, Any]) -> str | None:
    """The pinned model id in an evaluator ``modelConfig``, across the two config shapes."""
    for key in ("bedrockEvaluatorModelConfig", "responsesEvaluatorModelConfig"):
        model_id = (model_config.get(key) or {}).get("modelId")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None


def _pinned_model(config: dict[str, Any]) -> str | None:
    """The pinned model of a ``Custom`` (llmAsAJudge) or ``CustomDerived`` evaluator, if any."""
    for section in ("llmAsAJudge", "derived"):
        model_config = (config.get(section) or {}).get("modelConfig")
        if isinstance(model_config, dict):
            model_id = _model_id(model_config)
            if model_id is not None:
                return model_id
    return None


def _qualified_lambda(config: dict[str, Any]) -> str | None:
    """A ``CustomCode`` evaluator's Lambda ARN when it is pinned to a numeric version, else None.

    Only a version-qualified ARN (``…:function:name:3``) pins the code that was certified; an
    unqualified ARN or a ``$LATEST``/alias qualifier can be repointed, so it cannot be certified.
    """
    arn = ((config.get("codeBased") or {}).get("lambdaConfig") or {}).get("lambdaArn")
    if not isinstance(arn, str) or ":" not in arn:
        return None
    qualifier = arn.rsplit(":", 1)[-1]
    return arn if qualifier.isdigit() else None


def _rating_scale(config: dict[str, Any]) -> tuple[ScaleValue, ...]:
    """The evaluator's published numerical rating scale, empty when it declares none.

    Only an llm-as-judge evaluator publishes a numerical scale; a code-based one reports a value
    whose meaning it owns, so there is no scale to enumerate and the tuple is empty.

    Raises :class:`AgentCoreRatingScaleMalformed` when a rung is not an object, has no numeric
    value, or repeats another rung's value: dropping or guessing such a rung would silently move
    the pass boundary, so the scale is refused instead of parsed partially. A NaN or infinite value
    has no place in the order, and a single rung has no pass and fail side; both are refused too.
    """
    numerical = ((config.get("llmAsAJudge") or {}).get("ratingScale") or {}).get("numerical") or []
    if not isinstance(numerical, list):
        raise AgentCoreRatingScaleMalformed(
            f"the evaluator's published rating scale is {numerical!r}, not a list of rungs; it "
            "cannot be read, so the judge is refused"
        )
    if len(numerical) == 1:
        raise AgentCoreRatingScaleMalformed(
            f"the evaluator's published rating scale has a single rung {numerical[0]!r}; one "
            "point has no pass and fail side, so the judge is refused"
        )
    rungs: list[ScaleValue] = []
    for rung in numerical:
        raw = rung.get("value") if isinstance(rung, dict) else None
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        if value is None or not math.isfinite(value):
            raise AgentCoreRatingScaleMalformed(
                f"the evaluator's published rating scale has a rung {rung!r} with no finite "
                "numeric value; the scale cannot be read without dropping it, so the judge is "
                "refused"
            )
        if any(abs(value - kept.value) <= _SCALE_TOLERANCE for kept in rungs):
            raise AgentCoreRatingScaleMalformed(
                f"the evaluator's published rating scale repeats the value {value}; two rungs on "
                "one point cannot be ordered, so the judge is refused"
            )
        rungs.append(ScaleValue(value=value, label=str(rung.get("label") or "")))
    return tuple(rungs)


def outcome_for_value(
    value: float, rating_scale: tuple[ScaleValue, ...], *, inverted: bool = False
) -> Outcome:
    """Collapse a numeric evaluator value to a binary outcome against its published scale.

    The verdict is read from the number, never the free-text label: the label string differs
    between the on-demand and batch APIs while the value is stable. The value is matched to the
    rung it denotes (within a float tolerance), and passes when that rung's position in the sorted
    scale is strictly above the middle position; the middle rung of an odd-length scale fails. The
    split is by rung position, not by the value midpoint, so a non-uniform scale splits at its
    middle rung. For an ``inverted`` evaluator (a high score is the bad outcome) the order is
    reversed, so the low rungs pass and the middle rung still fails. A value matching no rung,
    including any value against an empty scale, is refused rather than placed on a guess.
    """
    values = sorted(rung.value for rung in rating_scale)
    position = next(
        (i for i, rung in enumerate(values) if abs(value - rung) <= _SCALE_TOLERANCE), None
    )
    if position is None:
        raise AgentCoreScoreOffScale(
            f"value {value} is not on the evaluator's published scale {values}; the result cannot "
            "be placed on the scale, so it is refused rather than mapped to a binary outcome"
        )
    if inverted:
        position = len(values) - 1 - position
    return "PASS" if position > (len(values) - 1) / 2 else "FAIL"


def classify_evaluator(evaluator: dict[str, Any]) -> CalibratabilityClass:
    """Classify a ``GetEvaluator`` document as calibratable or advisory-only.

    ``Custom`` is calibratable; ``CustomDerived`` only with a pinned model; ``CustomCode`` only when
    pinned to a qualified Lambda version. Everything else — ``Builtin``, ``ThirdParty``, or an
    evaluator type this version does not recognise — is advisory-only, the conservative default.
    """
    config = evaluator.get("evaluatorConfig") or {}
    return _calibratability(
        evaluator.get("evaluatorType"), _pinned_model(config), _qualified_lambda(config)
    )


def _calibratability(
    evaluator_type: object, pinned_model: str | None, lambda_version_arn: str | None
) -> CalibratabilityClass:
    """The rule both :func:`classify_evaluator` and :class:`AgentCoreJudge` apply."""
    if evaluator_type == "Custom":
        return CalibratabilityClass.CALIBRATABLE
    if evaluator_type == "CustomDerived" and pinned_model:
        return CalibratabilityClass.CALIBRATABLE
    if evaluator_type == "CustomCode" and lambda_version_arn:
        return CalibratabilityClass.CALIBRATABLE
    return CalibratabilityClass.ADVISORY_ONLY


def require_calibratable(judge: AgentCoreJudge) -> None:
    """Refuse an advisory-only evaluator, naming why it can never be certified.

    An advisory-only evaluator can never obtain a certifying calibration record, so the existing
    score gate refuses it like any uncertified judge. This check makes the reason explicit (the
    evaluator type AWS can change) rather than a bare "no calibration". Nothing in Calibrated Agent
    Evaluation calls it yet: certifying and calibrating AgentCore judges comes in a later feature,
    which is expected to call it before calibrating.
    """
    if judge.calibratability is CalibratabilityClass.ADVISORY_ONLY:
        raise AgentCoreAdvisoryOnly(
            f"evaluator {judge.evaluator_id!r} is {judge.evaluator_type!r}, which AWS can change "
            "without notice; it stays advisory and cannot be certified or used as a gate. Use a "
            "Custom evaluator, a CustomDerived one with a pinned model, or a CustomCode one pinned "
            "to a qualified Lambda version to certify"
        )


def _iso(value: Any) -> str:
    """A stable ISO-8601 string for an evaluator ``updatedAt`` (a datetime or already a string)."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else ""


# Evaluators AWS owns and may change without an edit the caller makes, so their updatedAt is
# part of their version.
_AWS_OWNED_TYPES = frozenset({"Builtin", "ThirdParty"})


def config_digest(evaluator: dict[str, Any]) -> str:
    """A 16-hex digest of what decides an evaluator's verdicts, as ``GetEvaluator`` returns it.

    It covers the evaluator's type, level, description and ``evaluatorConfig`` (instructions,
    rating scale, model and its settings, or Lambda), and, for an evaluator AWS owns (``Builtin``,
    ``ThirdParty``), its ``updatedAt``. A customer evaluator's own ``updatedAt`` is left out:
    attaching it to an online config locks it and moves its ``updatedAt`` without changing it.
    The description is included because whether it reaches the judge model is not documented.
    """
    evaluator_type = evaluator.get("evaluatorType")
    doc: dict[str, Any] = {
        "evaluator_type": evaluator_type,
        "level": evaluator.get("level"),
        "description": evaluator.get("description"),
        "evaluator_config": evaluator.get("evaluatorConfig") or {},
    }
    if evaluator_type in _AWS_OWNED_TYPES:
        doc["updated_at"] = _iso(evaluator.get("updatedAt"))
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=_iso).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _require_readable(evaluator: dict[str, Any], name: str) -> None:
    """Refuse a custom LLM-as-judge evaluator read without its instructions or rating scale."""
    if evaluator.get("evaluatorType") != "Custom":
        return
    llm = (evaluator.get("evaluatorConfig") or {}).get("llmAsAJudge") or {}
    instructions = llm.get("instructions")
    missing = [
        field
        for field, present in (
            ("instructions", isinstance(instructions, str) and bool(instructions.strip())),
            ("ratingScale", bool(llm.get("ratingScale"))),
        )
        if not present
    ]
    if missing:
        raise AgentCoreEvaluatorUnreadable(
            f"GetEvaluator returned evaluator {name} without its {' or '.join(missing)}, so "
            "Calibrated Agent Evaluation "
            "could not tell if they change later and will not use it as a judge. Check that the "
            "evaluator has instructions and a rating scale, and, if it is encrypted with your own "
            "KMS key, that you have kms:Decrypt on that key"
        )


@dataclass(frozen=True)
class AgentCoreJudge:
    """An AgentCore evaluator as a judge, carrying the identity the calibration gates pin to.

    Satisfies :class:`calibrated_agent_evaluation.calibrate.engine.Judge`: ``model`` is the
    evaluator ARN and ``prompt_version`` is :func:`config_digest` of the evaluator, so a change to
    what decides its verdicts shows up as a changed pin, while a lock (which moves only
    ``updatedAt``) does not. The fingerprint folds in the ARN, that digest, type and the pinned
    model / Lambda version, so any change that alters the evaluator's behaviour voids its
    calibration through the existing gate. ``updated_at`` is kept for display.
    """

    judge_id: str
    evaluator_id: str
    evaluator_arn: str
    evaluator_type: str
    updated_at: str
    config_digest: str
    pinned_model: str | None = None
    lambda_version_arn: str | None = None
    rating_scale: tuple[ScaleValue, ...] = ()
    # The evaluator (or a derived evaluator's base) scores failures high, so its outcome is flipped.
    inverted: bool = False
    # A derived evaluator's base's updatedAt: the base owns the prompt and the scoring, so an edit
    # to it changes this judge's verdicts.
    base_updated_at: str | None = None
    # The evaluator's GetEvaluator level. Only TRACE runs; None (a judge built without it) is
    # refused, so leaving it out is never read as TRACE.
    evaluator_level: str | None = None
    level: JudgeLevel = "workflow"

    @property
    def model(self) -> str:
        """The evaluator ARN — the model the gate pins to."""
        return self.evaluator_arn

    @property
    def prompt_version(self) -> str:
        """The digest of the evaluator's configuration — an edit shows up as a changed pin."""
        return self.config_digest

    @property
    def calibratability(self) -> CalibratabilityClass:
        """Whether this evaluator can be certified, or may only ever be advisory."""
        return _calibratability(self.evaluator_type, self.pinned_model, self.lambda_version_arn)

    @property
    def fingerprint(self) -> str:
        """A 16-hex digest of everything that decides this evaluator's verdicts."""
        doc: dict[str, str] = {
            "evaluator_arn": self.evaluator_arn,
            "evaluator_type": self.evaluator_type,
            "config_digest": self.config_digest,
        }
        if self.pinned_model is not None:
            doc["pinned_model"] = self.pinned_model
        if self.lambda_version_arn is not None:
            doc["lambda_version_arn"] = self.lambda_version_arn
        # The scale and the direction decide the binary outcome, so a change to either (a new
        # scale, or an entry added to the direction table) must void a stored calibration.
        if self.rating_scale:
            doc["rating_scale"] = json.dumps([[r.value, r.label] for r in self.rating_scale])
        if self.inverted:
            doc["inverted"] = "true"
        if self.base_updated_at is not None:
            doc["base_updated_at"] = self.base_updated_at
        blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def judge_from_evaluator(
    judge_id: str,
    evaluator: dict[str, Any],
    *,
    base_evaluator: dict[str, Any] | None = None,
    level: JudgeLevel = "workflow",
) -> AgentCoreJudge:
    """Build an :class:`AgentCoreJudge` from a ``GetEvaluator`` document.

    A ``CustomDerived`` evaluator runs its base's scoring, and its own document carries no rating
    scale, so the caller passes the base's ``GetEvaluator`` document as ``base_evaluator``: the
    judge takes its scale and its score direction from the base. Raises ``ValueError`` when a
    derived evaluator comes without its base's document (or with another evaluator's),
    :class:`AgentCoreEvaluatorLevelUnsupported` for an evaluator whose own level is not ``TRACE``,
    :class:`AgentCoreEvaluatorUnreadable` for a ``Custom`` evaluator read without its instructions
    or rating scale, :class:`AgentCoreNoPassDirection` for an evaluator whose score has no pass
    direction, and
    :class:`AgentCoreRatingScaleMalformed` for a scale that cannot be read.
    """
    raw_level = evaluator.get("level")
    evaluator_level = raw_level if isinstance(raw_level, str) else None
    name = str(evaluator.get("evaluatorId") or evaluator.get("evaluatorArn") or "<unnamed>")
    require_runnable_level(evaluator_level, evaluator_id=name, level=level)
    _require_readable(evaluator, name)
    config = evaluator.get("evaluatorConfig") or {}
    scoring = config
    base_updated_at: str | None = None
    names = {str(evaluator.get("evaluatorId") or "")}
    base_id = (config.get("derived") or {}).get("baseEvaluatorId")
    if isinstance(base_id, str) and base_id:
        if base_evaluator is None or base_evaluator.get("evaluatorId") != base_id:
            raise ValueError(
                f"evaluator {evaluator.get('evaluatorId')!r} is derived from {base_id!r}; pass "
                "that base's GetEvaluator document as base_evaluator, since the base owns the "
                "rating scale its verdicts are read against"
            )
        scoring = base_evaluator.get("evaluatorConfig") or {}
        base_updated_at = _iso(base_evaluator.get("updatedAt"))
        names.add(base_id)
    inverted = scores_inverted(names)
    return AgentCoreJudge(
        judge_id=judge_id,
        evaluator_id=str(evaluator.get("evaluatorId") or ""),
        evaluator_arn=str(evaluator.get("evaluatorArn") or ""),
        evaluator_type=str(evaluator.get("evaluatorType") or ""),
        updated_at=_iso(evaluator.get("updatedAt")),
        config_digest=config_digest(evaluator),
        pinned_model=_pinned_model(config),
        lambda_version_arn=_qualified_lambda(config),
        rating_scale=_rating_scale(scoring),
        inverted=inverted,
        base_updated_at=base_updated_at,
        evaluator_level=evaluator_level,
        level=level,
    )


_JUDGE_FILE_REQUIRED = ("backend", "judge_id", "evaluator_id", "failure_mode")
_JUDGE_FILE_OPTIONAL = ("agent", "level")


@dataclass(frozen=True)
class AgentCoreJudgeFile:
    """A parsed ``agentcore`` judge file: which evaluator to run, and the mode it detects.

    ``agent`` is the crew agent the failure mode belongs to, as in a prompt judge file; ``level``
    is ``workflow`` (a TRACE evaluator scoring each trace) or ``session`` (a SESSION evaluator
    scoring each simulated conversation).
    """

    judge_id: str
    evaluator_id: str
    failure_mode: str
    agent: str | None
    level: JudgeLevel
    path: Path


def load_agentcore_judge_file(path: str | Path) -> AgentCoreJudgeFile:
    """Parse and validate an ``agentcore`` judge file, with no AWS call.

    The file is JSON: ``backend`` (``"agentcore"``), ``judge_id``, ``evaluator_id`` and
    ``failure_mode``, with an optional ``agent`` and an optional ``level``, ``"workflow"`` (the
    default) or ``"session"``. Raises :class:`AgentCoreError` for an unreadable file or a missing,
    unknown or invalid field. The evaluator itself is read when the judge is built
    (:func:`build_agentcore_judge`), and its own level must match the file's.
    """
    src = Path(path)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentCoreError(f"cannot read the AgentCore judge file {src}: {exc}") from exc
    if not isinstance(doc, dict):
        raise AgentCoreError(f"the AgentCore judge file {src} must hold one JSON object")
    missing = [
        k for k in _JUDGE_FILE_REQUIRED if not isinstance(doc.get(k), str) or not doc[k].strip()
    ]
    if missing:
        raise AgentCoreError(
            f"the AgentCore judge file {src} is missing required field(s): {', '.join(missing)}"
        )
    unknown = sorted(set(doc) - set(_JUDGE_FILE_REQUIRED) - set(_JUDGE_FILE_OPTIONAL))
    if unknown:
        raise AgentCoreError(
            f"the AgentCore judge file {src} has unknown field(s): {', '.join(unknown)}"
        )
    if doc["backend"] != "agentcore":
        raise AgentCoreError(f"{src} declares backend {doc['backend']!r}, not 'agentcore'")
    level = doc.get("level", "workflow")
    if not isinstance(level, str) or level not in _RUNNABLE_LEVELS:
        raise AgentCoreEvaluatorLevelUnsupported(
            f"the AgentCore judge file {src}: level must be 'workflow' (a TRACE evaluator on each "
            f"trace) or 'session' (a SESSION evaluator on each simulated conversation), got "
            f"{level!r}"
        )
    agent = doc.get("agent")
    if agent is not None and (not isinstance(agent, str) or not agent.strip()):
        raise AgentCoreError(
            f"the AgentCore judge file {src}: agent must be a non-empty string or null"
        )
    return AgentCoreJudgeFile(
        judge_id=doc["judge_id"],
        evaluator_id=doc["evaluator_id"],
        failure_mode=doc["failure_mode"],
        agent=agent,
        level=cast("JudgeLevel", level),
        path=src,
    )


def build_agentcore_judge(judge_file: AgentCoreJudgeFile, *, clients: AwsClients) -> AgentCoreJudge:
    """Build the judge ``judge_file`` names from its evaluator's ``GetEvaluator`` document.

    A ``CustomDerived`` evaluator's base is read too, since the base owns the rating scale and the
    score direction. Raises :class:`AgentCoreError` naming the evaluator (or base) the account does
    not have, or when the call fails, and the refusals of :func:`judge_from_evaluator`.
    """
    evaluator = _get_evaluator(judge_file.evaluator_id, clients=clients)
    base_id = ((evaluator.get("evaluatorConfig") or {}).get("derived") or {}).get("baseEvaluatorId")
    base = (
        _get_evaluator(base_id, clients=clients) if isinstance(base_id, str) and base_id else None
    )
    try:
        return judge_from_evaluator(
            judge_file.judge_id, evaluator, base_evaluator=base, level=judge_file.level
        )
    except ValueError as exc:
        # A base document that does not match the derived evaluator's base id is a service
        # response problem here, not a caller mistake, so it is reported as one.
        raise AgentCoreError(f"cannot build judge {judge_file.judge_id!r}: {exc}") from exc


def _get_evaluator(evaluator_id: str, *, clients: AwsClients) -> dict[str, Any]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        response = clients.client("bedrock-agentcore-control").get_evaluator(
            evaluatorId=evaluator_id
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "ResourceNotFoundException":
            raise AgentCoreError(
                f"the account has no AgentCore evaluator {evaluator_id!r} in this region"
            ) from exc
        raise AgentCoreError(f"GetEvaluator for {evaluator_id!r} failed: {exc}") from exc
    except BotoCoreError as exc:
        raise AgentCoreError(f"GetEvaluator for {evaluator_id!r} failed: {exc}") from exc
    return dict(response)
