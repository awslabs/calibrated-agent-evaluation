# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrated AgentCore judges in online evaluation: the writes and their safeguards.

:func:`create_online_config` attaches certified, bias-cleared judges to an AgentCore online
evaluation config on a deployed runtime.

Every write this adapter makes to AgentCore or CloudWatch goes through :func:`send_confirmed`: the
caller sees the exact request as a :class:`WritePreview` and confirms it, and a decline sends
nothing. Every AWS call goes through :func:`call_aws`, which turns an SDK failure into an
:class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError` naming the call and listing
any resource created earlier in the same request, so nothing is left behind unreported.

The AWS SDK is imported only inside these functions, so importing this module loads none of it.
A failed call's message carries the service's error code and its message, as the judge's
``Evaluate`` errors do: AgentCore's messages say what was wrong with the request (a role that
cannot be assumed, an evaluator that does not exist), and may repeat a name or ARN the caller
sent, never a credential.
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreEvaluatorUnreadable,
    AgentCoreNotCertified,
    AgentCoreResourceExists,
    AgentCoreResourceNotActive,
    AgentCoreRunDeclined,
    AgentCoreRuntimeNotFound,
    BilledCallBudgetExceeded,
)
from calibrated_agent_evaluation.agentcore.guards import BUDGET_ENV_VAR
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge, judge_from_evaluator
from calibrated_agent_evaluation.agentcore.online_record import (
    AgentCoreSessionLog,
    AttachedEvaluator,
    OnlineEvaluationRecord,
)
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.bias import BiasBatteryRecord

__all__ = [
    "FILTER_OPERATORS",
    "AttachedState",
    "AttachedStatus",
    "OnlineFilter",
    "OnlineJudge",
    "WritePreview",
    "aws_call_failed",
    "aws_error_code",
    "call_aws",
    "check_attached",
    "check_config_request",
    "create_online_config",
    "send_confirmed",
]

_CONTROL = "bedrock-agentcore-control"
# AgentCore's own bounds on a config (the service model); checked before any call, since the SDK
# checks only some of them and the rest would fail after a confirmed create.
_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}")
_FILTER_KEY = re.compile(r"[a-zA-Z0-9._-]{1,256}")
_MAX_FILTER_STRING = 1024
_MAX_FILTERS = 5
_MAX_EVALUATORS = 25
FILTER_OPERATORS: frozenset[str] = frozenset(
    {
        "Equals",
        "NotEquals",
        "GreaterThan",
        "LessThan",
        "GreaterThanOrEqual",
        "LessThanOrEqual",
        "Contains",
        "NotContains",
    }
)
"""The filter operators AgentCore accepts."""
_FAILED_STATUSES = frozenset({"CREATE_FAILED", "UPDATE_FAILED", "ERROR", "DELETING"})
_RUNTIME_ARN = re.compile(
    r"arn:aws[a-zA-Z-]*:bedrock-agentcore:([a-z0-9-]+):[0-9]{12}:"
    r"runtime/([a-zA-Z][a-zA-Z0-9_]{0,99}-[a-zA-Z0-9]{10})"
)


@dataclass(frozen=True)
class WritePreview:
    """One write as the caller confirms it: the call, its exact request and what it does.

    Attributes:
        service: The AWS service the call goes to, for example ``bedrock-agentcore-control``.
        operation: The API operation, for example ``CreateEvaluator``.
        request: The request parameters, exactly as they will be sent.
        summary: What the write does, in a line or two (with a cost, when it has one).
    """

    service: str
    operation: str
    request: Mapping[str, Any]
    summary: str

    def describe(self) -> str:
        """The preview as text: what it does, the call, and the request as JSON."""
        body = json.dumps(self.request, indent=2, sort_keys=True, default=str)
        return f"{self.summary}\n{self.service} {self.operation}:\n{body}"


def send_confirmed(
    clients: AwsClients,
    service: str,
    operation: str,
    request: Mapping[str, Any],
    *,
    summary: str,
    confirm: Callable[[WritePreview], bool],
    created: Sequence[str] = (),
) -> dict[str, Any]:
    """Show ``confirm`` the write, and send it only if ``confirm`` returns true.

    A non-interactive approval is a ``confirm`` that returns true.

    Args:
        clients: The AWS clients to send through.
        service: The AWS service, for example ``bedrock-agentcore-control``.
        operation: The API operation, for example ``CreateEvaluator``.
        request: The request parameters; exactly these are previewed and sent.
        summary: What the write does, shown with the request.
        confirm: Receives the :class:`WritePreview`; returns whether to send it.
        created: Resources created earlier in the same request, listed if this call fails.

    Returns:
        The service's response.

    Raises:
        AgentCoreRunDeclined: ``confirm`` returned false; nothing was sent.
        AgentCoreError: The call failed (see :func:`call_aws`).
    """
    preview = WritePreview(
        service=service, operation=operation, request=copy.deepcopy(dict(request)), summary=summary
    )
    if not confirm(preview):
        raise AgentCoreRunDeclined(
            f"{operation} was declined at its preview, so nothing was sent ({summary})"
        )
    return call_aws(clients, service, operation, preview.request, created=created)


def call_aws(
    clients: AwsClients,
    service: str,
    operation: str,
    request: Mapping[str, Any],
    *,
    created: Sequence[str] = (),
) -> dict[str, Any]:
    """Make one AWS call, turning an SDK failure into an :class:`AgentCoreError`.

    A refusal raised by a guard on the clients (a billed-call budget, a read-only allowlist) is
    raised as it is when nothing was created earlier; otherwise it is wrapped, naming the guard by
    its type and listing what was created.

    Args:
        clients: The AWS clients to call through.
        service: The AWS service.
        operation: The API operation, for example ``GetEvaluator``.
        request: The request parameters.
        created: Resources created earlier in the same request, listed in the error so the caller
            can delete them.

    Returns:
        The service's response.

    Raises:
        AgentCoreError: The service answered with an error; the call was not sent (credentials,
            a connection that could not be made, a bad parameter); or it failed once the request
            may have reached the service, so it may have taken effect. The message names the call
            and lists ``created``. A guard's refusal, as above.
    """
    from botocore import xform_name
    from botocore.exceptions import BotoCoreError, ClientError
    from urllib3.exceptions import HTTPError as URLLib3HTTPError

    client = clients.client(service)
    try:
        response: dict[str, Any] = getattr(client, xform_name(operation))(**request)
    except AgentCoreError as exc:
        # Only a guard on the clients raises a Calibrated Agent Evaluation error from inside an AWS
        # call.
        if not created:
            raise
        hint = (
            f"Raise {BUDGET_ENV_VAR}, or unset it to remove the limit"
            if isinstance(exc, BilledCallBudgetExceeded)
            else "Check the guards installed on the AWS clients"
        )
        raise AgentCoreError(
            f"a guard on the AWS clients ({type(exc).__name__}) refused the {service} call "
            f"{operation}, so it was not sent. {hint}{_left(created)}"
        ) from exc
    except ClientError as exc:
        error = exc.response.get("Error") or {}
        code = error.get("Code") or type(exc).__name__
        detail = f": {error['Message']}" if error.get("Message") else ""
        raise AgentCoreError(
            f"the {service} call {operation} failed with {code}{detail}{_left(created)}"
        ) from exc
    except _not_sent_errors() as exc:
        raise AgentCoreError(
            f"the {service} call {operation} was not sent: it failed with {type(exc).__name__} "
            f"before reaching the service. Fix the cause (credentials, network or a request "
            f"parameter) and run again{_left(created)}"
        ) from exc
    except (BotoCoreError, OSError, URLLib3HTTPError) as exc:
        # Raised once the request may have reached the service: a read timeout, a connection
        # closed while waiting, or an HTTP-layer error botocore let through as it is.
        raise AgentCoreError(
            f"the {service} call {operation} failed with {type(exc).__name__} after the request "
            f"may have reached the service, so it may have taken effect; check before running it "
            f"again{_left(created)}"
        ) from exc
    return response


def _not_sent_errors() -> tuple[type[Exception], ...]:
    """The SDK errors raised before a request reached the service.

    A request parameter the SDK refused; credentials that are missing or could not be loaded or
    refreshed; and a connection that could not be made. The same list as the runtime executor's.
    A connect timeout is an ``OSError`` like a read timeout, so these are caught first.
    """
    from botocore.exceptions import (
        ConnectTimeoutError,
        CredentialRetrievalError,
        EndpointConnectionError,
        NoCredentialsError,
        ParamValidationError,
        PartialCredentialsError,
        ProxyConnectionError,
        RefreshWithMFAUnsupportedError,
        SSLError,
        SSOError,
        TokenRetrievalError,
        UnauthorizedSSOTokenError,
    )

    return (
        ParamValidationError,
        NoCredentialsError,
        PartialCredentialsError,
        CredentialRetrievalError,
        RefreshWithMFAUnsupportedError,
        TokenRetrievalError,
        UnauthorizedSSOTokenError,
        SSOError,
        EndpointConnectionError,
        ConnectTimeoutError,
        SSLError,
        ProxyConnectionError,
    )


def _left(created: Sequence[str]) -> str:
    """The note listing what an earlier call in the same request created, or nothing."""
    if not created:
        return ""
    if len(created) == 1:
        return (
            f". Created earlier in this request, and still exists: {created[0]}; delete it if "
            "it is not wanted"
        )
    return (
        f". Created earlier in this request, and they still exist: {', '.join(created)}; delete "
        "them if they are not wanted"
    )


@dataclass(frozen=True)
class OnlineJudge:
    """A judge put forward for online evaluation, with the records that let it score.

    Attributes:
        judge: The AgentCore judge.
        calib: Its newest calibration record, or ``None`` when it has none.
        battery: Its newest bias battery record, or ``None`` when it has none.
    """

    judge: AgentCoreJudge
    calib: CalibrationRecord | None
    battery: BiasBatteryRecord | None


@dataclass(frozen=True)
class OnlineFilter:
    """One rule filter: AgentCore scores only sessions whose ``key`` compares true to ``value``.

    ``value`` is sent as a number, a boolean or a string by its Python type.
    """

    key: str
    operator: str
    value: str | float | bool


def create_online_config(
    judges: Sequence[OnlineJudge],
    *,
    name: str,
    runtime_arn: str,
    execution_role_arn: str,
    sampling_percentage: float,
    session_logs: Sequence[AgentCoreSessionLog],
    clients: AwsClients,
    confirm: Callable[[WritePreview], bool],
    endpoint: str = "DEFAULT",
    session_timeout_minutes: int = 15,
    filters: Sequence[OnlineFilter] = (),
    wait_seconds: float = 300.0,
    poll_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> OnlineEvaluationRecord:
    """Attach ``judges`` to a new AgentCore online evaluation config on a deployed runtime.

    Every judge must be calibratable, certified by its calibration record, cleared by a ``CLEAR``
    bias battery, and have the session logs of both runs, which the config's read-back skips; all
    of it is checked before any AWS call. The runtime is read with ``GetAgentRuntime``, and the
    config's data source derived from it: the service name ``<runtime name>.<endpoint>`` and the
    log group ``/aws/bedrock-agentcore/runtimes/<runtime id>-<endpoint>``. The request is shown to
    ``confirm`` with the cost per scored trace, created with a client token, and read until
    ``ACTIVE``.

    Args:
        judges: The judges to attach, with their records.
        name: The config's name.
        runtime_arn: The deployed runtime's ARN.
        execution_role_arn: The role AgentCore assumes to run the config.
        sampling_percentage: The share of sessions scored, 0.01 to 100.
        session_logs: The stored calibration and bias session logs.
        clients: The AWS clients.
        confirm: Receives the ``CreateOnlineEvaluationConfig`` preview; returns whether to send.
        endpoint: The runtime endpoint whose sessions are scored.
        session_timeout_minutes: How long a session is idle before it is scored, 1 to 1440.
        filters: Up to five rule filters.
        wait_seconds: How long to wait for the config to become ``ACTIVE``.
        poll_seconds: How often to read its status while waiting.
        sleep: Waits between status reads (injectable for tests).
        clock: The monotonic clock the wait is measured on (injectable for tests).

    Returns:
        The record of the config, its evaluators' pins and the sessions to skip; the caller
        stores it.

    Raises:
        AgentCoreError: A value AgentCore would refuse (before any call), or a call failed.
        AgentCoreAdvisoryOnly: A judge is ``Builtin`` or ``ThirdParty``; before any call.
        AgentCoreNotCertified: A judge lacks a record, a pass of the trust gates, or a session
            log; before any call.
        AgentCoreRuntimeNotFound: The runtime does not exist; before any write.
        AgentCoreResourceExists: A config with ``name`` exists; none was created.
        AgentCoreRunDeclined: ``confirm`` declined; nothing was created.
        AgentCoreResourceNotActive: The config failed or was not ``ACTIVE`` within the wait.
    """
    check_config_request(name, sampling_percentage, session_timeout_minutes, filters)
    if not judges:
        raise AgentCoreError("no judge to attach; give at least one certified AgentCore judge")
    ids = [o.judge.evaluator_id for o in judges]
    if len(set(ids)) != len(ids):
        raise AgentCoreError(
            "the same evaluator was given more than once; attach each judge once. Nothing was sent"
        )
    if len(ids) > _MAX_EVALUATORS:
        raise AgentCoreError(
            f"{len(ids)} judges were given; an online config takes at most {_MAX_EVALUATORS}. "
            "Nothing was sent"
        )
    attached: list[AttachedEvaluator] = []
    excluded: list[str] = []
    seen: set[str] = set()
    for online in judges:
        evaluator, sessions = _admitted(online, session_logs)
        attached.append(evaluator)
        for session in sessions:
            if session not in seen:
                seen.add(session)
                excluded.append(session)

    runtime_id = _runtime_id(runtime_arn, clients)
    try:
        runtime = call_aws(clients, _CONTROL, "GetAgentRuntime", {"agentRuntimeId": runtime_id})
    except AgentCoreError as exc:
        if aws_error_code(exc) == "ResourceNotFoundException":
            raise AgentCoreRuntimeNotFound(runtime_arn) from exc
        raise
    service_name = f"{runtime['agentRuntimeName']}.{endpoint}"
    log_group = f"/aws/bedrock-agentcore/runtimes/{runtime['agentRuntimeId']}-{endpoint}"
    if name in _config_names(clients):
        raise AgentCoreResourceExists("online evaluation config", name)

    rule: dict[str, Any] = {
        "samplingConfig": {"samplingPercentage": float(sampling_percentage)},
        "sessionConfig": {"sessionTimeoutMinutes": int(session_timeout_minutes)},
    }
    if filters:
        rule["filters"] = [_typed(f) for f in filters]
    request: dict[str, Any] = {
        "onlineEvaluationConfigName": name,
        "description": (
            "Calibrated Agent Evaluation online judges: "
            + ", ".join(o.judge.judge_id for o in judges)
        )[:200],
        "rule": rule,
        "dataSourceConfig": {
            "cloudWatchLogs": {"serviceNames": [service_name], "logGroupNames": [log_group]}
        },
        "evaluators": [{"evaluatorId": e.evaluator_id} for e in attached],
        "evaluationExecutionRoleArn": execution_role_arn,
        "enableOnCreate": True,
        "clientToken": f"calibrated-agent-evaluation-{uuid.uuid4().hex}",
    }
    summary = (
        f"Creates the AgentCore online evaluation config {name} on {service_name} (log group "
        f"{log_group}), scoring {sampling_percentage:g}% of its sessions once each has been idle "
        f"{session_timeout_minutes} minute(s), with {len(attached)} certified judge(s). "
        f"{_cost_note(clients, len(attached))}"
    )
    try:
        created = send_confirmed(
            clients,
            _CONTROL,
            "CreateOnlineEvaluationConfig",
            request,
            summary=summary,
            confirm=confirm,
        )
    except AgentCoreError as exc:
        # A conflict on create is the name taken since it was listed; AgentCore's own message is
        # kept on the chained error.
        if aws_error_code(exc) == "ConflictException":
            raise AgentCoreResourceExists("online evaluation config", name) from exc
        raise
    config_id = str(created["onlineEvaluationConfigId"])
    _wait_active(
        clients, config_id, wait_seconds, poll_seconds=poll_seconds, sleep=sleep, clock=clock
    )
    return OnlineEvaluationRecord(
        config_id=config_id,
        config_name=name,
        runtime_service_name=service_name,
        evaluators=tuple(attached),
        excluded_session_ids=tuple(excluded),
        sampling_percentage=float(sampling_percentage),
        session_timeout_minutes=int(session_timeout_minutes),
        created_at=_iso(created.get("createdAt")),
        filtered=bool(filters),
    )


AttachedState = Literal["certified", "changed", "missing"]


@dataclass(frozen=True)
class AttachedStatus:
    """Whether one evaluator attached to an online config is still the one that was certified.

    Attributes:
        evaluator_id: The AgentCore evaluator id.
        state: ``certified`` when its fingerprint matches the one recorded at attach time,
            ``changed`` when it does not, ``missing`` when the account no longer has it.
        reason: A plain sentence saying what was found.
    """

    evaluator_id: str
    state: AttachedState
    reason: str

    @property
    def certified(self) -> bool:
        """Whether its verdicts still come from the judge its calibration certified."""
        return self.state == "certified"


def check_attached(record: OnlineEvaluationRecord, *, clients: AwsClients) -> list[AttachedStatus]:
    """Read each evaluator ``record`` attached and say whether it is still the certified one.

    Each evaluator is read with ``GetEvaluator`` (and a derived evaluator's base too), rebuilt as a
    judge and its fingerprint compared with the one recorded. The fingerprint is made from what
    decides the verdicts, so the lock the online config puts on the evaluator, which moves only its
    ``updatedAt``, is not a change. An evaluator the account no longer has is ``missing``.

    Raises:
        AgentCoreError: A ``GetEvaluator`` call failed for any reason other than the evaluator not
            existing; a failed read says nothing about the evaluator, so it is not reported.
        AgentCoreEvaluatorUnreadable: An evaluator came back without its instructions or rating
            scale, so whether it changed cannot be told.
    """
    statuses: list[AttachedStatus] = []
    for attached in record.evaluators:
        evaluator_id = attached.evaluator_id
        document = _read_evaluator(clients, evaluator_id)
        if document is None:
            statuses.append(
                AttachedStatus(
                    evaluator_id, "missing", f"evaluator {evaluator_id} no longer exists"
                )
            )
            continue
        config = document.get("evaluatorConfig") or {}
        base_id = (config.get("derived") or {}).get("baseEvaluatorId")
        base = None
        if isinstance(base_id, str) and base_id:
            base = _read_evaluator(clients, base_id)
            if base is None:
                statuses.append(
                    AttachedStatus(
                        evaluator_id,
                        "missing",
                        f"evaluator {evaluator_id}'s base {base_id} no longer exists",
                    )
                )
                continue
        try:
            judge = judge_from_evaluator(evaluator_id, document, base_evaluator=base)
        except AgentCoreEvaluatorUnreadable:
            raise
        except (AgentCoreError, ValueError) as exc:
            # It can no longer be built as a judge (its level or scale changed, say): whatever it
            # is now, it is not the evaluator that was certified.
            statuses.append(
                AttachedStatus(
                    evaluator_id,
                    "changed",
                    f"evaluator {evaluator_id} changed since it was certified and can no longer "
                    f"be used as a judge: {exc}",
                )
            )
            continue
        if judge.fingerprint == attached.fingerprint:
            statuses.append(
                AttachedStatus(
                    evaluator_id,
                    "certified",
                    f"evaluator {evaluator_id} is unchanged since it was certified",
                )
            )
        else:
            statuses.append(
                AttachedStatus(
                    evaluator_id,
                    "changed",
                    f"evaluator {evaluator_id} changed since it was certified (fingerprint "
                    f"{attached.fingerprint}, now {judge.fingerprint}), so its calibration no "
                    "longer applies",
                )
            )
    return statuses


def _read_evaluator(clients: AwsClients, evaluator_id: str) -> dict[str, Any] | None:
    """``GetEvaluator`` for ``evaluator_id``, or ``None`` when the account does not have it."""
    try:
        return call_aws(clients, _CONTROL, "GetEvaluator", {"evaluatorId": evaluator_id})
    except AgentCoreError as exc:
        if aws_error_code(exc) == "ResourceNotFoundException":
            return None
        raise


def check_config_request(
    name: str, sampling: float, timeout: int, filters: Sequence[OnlineFilter]
) -> None:
    """Refuse a config value AgentCore would refuse, naming it, before any AWS call.

    Raises:
        AgentCoreError: The name, sampling percentage, session timeout or a filter is one
            AgentCore does not accept; the message names each.
    """
    problems: list[str] = []
    if not _NAME.fullmatch(name):
        problems.append(
            f"the name {name!r} must start with a letter and hold at most 48 letters, digits "
            "and underscores"
        )
    if not 0.01 <= sampling <= 100.0:
        problems.append(f"the sampling percentage {sampling!r} must be from 0.01 to 100")
    if not 1 <= timeout <= 1440:
        problems.append(f"the session timeout {timeout!r} must be from 1 to 1440 minutes")
    if len(filters) > _MAX_FILTERS:
        problems.append(f"{len(filters)} filters were given; AgentCore takes at most five")
    for f in filters:
        if not _FILTER_KEY.fullmatch(f.key):
            problems.append(
                f"the filter key {f.key!r} must be 1 to 256 letters, digits, dots, underscores "
                "or hyphens"
            )
        if isinstance(f.value, str) and not 1 <= len(f.value) <= _MAX_FILTER_STRING:
            problems.append(
                f"the filter value for {f.key!r} must be 1 to {_MAX_FILTER_STRING} characters"
            )
        if f.operator not in FILTER_OPERATORS:
            problems.append(
                f"the filter operator {f.operator!r} is not one AgentCore accepts "
                f"({', '.join(sorted(FILTER_OPERATORS))})"
            )
        value = f.value
        if isinstance(value, float) and not math.isfinite(value):
            problems.append(f"the filter value {value!r} for {f.key!r} is not a finite number")
    if problems:
        raise AgentCoreError(
            "the online config was not created, and nothing was sent: " + "; ".join(problems)
        )


def _admitted(
    online: OnlineJudge, session_logs: Sequence[AgentCoreSessionLog]
) -> tuple[AttachedEvaluator, list[str]]:
    """The judge as attached, and its runs' session ids; or the refusal saying what it lacks."""
    from calibrated_agent_evaluation.agentcore.judge import (
        require_calibratable,
        require_trace_level,
    )
    from calibrated_agent_evaluation.enforce.admission import admit_judge

    judge = online.judge
    require_calibratable(judge)
    require_trace_level(judge.evaluator_level, evaluator_id=judge.evaluator_id)
    missing = [
        what
        for what, record in (
            ("a calibration record", online.calib),
            ("a bias battery", online.battery),
        )
        if record is None
    ]
    if online.calib is None or online.battery is None:
        raise AgentCoreNotCertified(judge.judge_id, tuple(missing))
    if online.battery.verdict != "CLEAR":
        raise AgentCoreNotCertified(
            judge.judge_id,
            (
                f"a CLEAR bias battery (the latest, {online.battery.battery_id}, is "
                f"{online.battery.verdict})",
            ),
        )
    gate = admit_judge(judge, online.calib, online.battery)
    if not gate.admitted:
        raise AgentCoreNotCertified(
            judge.judge_id, (f"a pass of the trust gates ({gate.refusal_reason})",)
        )
    sessions: list[str] = []
    for kind, run_id, what in (
        ("calibration", online.calib.calib_id, "the session log of its calibration run"),
        ("bias", online.battery.battery_id, "the session log of its bias run"),
    ):
        log = next((s for s in session_logs if s.run_kind == kind and s.run_id == run_id), None)
        if log is None:
            missing.append(what)
        else:
            sessions.extend(log.session_ids)
    if missing:
        raise AgentCoreNotCertified(judge.judge_id, tuple(missing))
    evaluator = AttachedEvaluator(
        evaluator_id=judge.evaluator_id,
        evaluator_arn=judge.evaluator_arn,
        updated_at=judge.updated_at,
        fingerprint=judge.fingerprint,
        calib_id=online.calib.calib_id,
        battery_id=online.battery.battery_id,
    )
    return evaluator, sessions


def _runtime_id(runtime_arn: str, clients: AwsClients) -> str:
    """The runtime id in ``runtime_arn``; refused if it is not a runtime ARN in the clients' region.

    An endpoint ARN is refused too: the endpoint is given separately.
    """
    match = _RUNTIME_ARN.fullmatch(runtime_arn)
    if match is None:
        raise AgentCoreError(
            f"{runtime_arn!r} is not an AgentCore runtime ARN "
            "(arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>); nothing was sent"
        )
    region, runtime_id = match.groups()
    if clients.region and region != clients.region:
        raise AgentCoreError(
            f"the runtime {runtime_arn} is in {region}, but the AWS clients are in "
            f"{clients.region}; run with --region {region}. Nothing was sent"
        )
    return runtime_id


def _config_names(clients: AwsClients) -> set[str]:
    """Every online config name in the account and region, reading every page."""
    names: set[str] = set()
    request: dict[str, Any] = {}
    while True:
        page = call_aws(clients, _CONTROL, "ListOnlineEvaluationConfigs", request)
        names.update(
            str(c["onlineEvaluationConfigName"]) for c in page.get("onlineEvaluationConfigs", [])
        )
        token = page.get("nextToken")
        if not token:
            return names
        request = {"nextToken": token}


def _typed(f: OnlineFilter) -> dict[str, Any]:
    """A filter in AgentCore's shape, its value typed by its Python type."""
    value = f.value
    if isinstance(value, bool):
        typed: dict[str, Any] = {"booleanValue": value}
    elif isinstance(value, int | float):
        typed = {"doubleValue": float(value)}
    else:
        typed = {"stringValue": str(value)}
    return {"key": f.key, "operator": f.operator, "value": typed}


def _cost_note(clients: AwsClients, evaluators: int) -> str:
    """The cost per scored trace at the Price List rate, or why there is none."""
    from calibrated_agent_evaluation.agentcore import pricing

    try:
        rate = pricing.lookup_evaluation_rates(clients).custom_per_evaluation
    except AgentCoreError as exc:
        return f"No cost is shown: the AWS Price List lookup did not give a rate ({exc})."
    if rate is None:
        return "No cost is shown: the AWS Price List lists no custom-evaluator rate here."
    per_trace = rate * evaluators
    return (
        f"Cost: about ${per_trace:.4f} per trace scored ({evaluators} evaluator(s) at "
        f"${rate:.4f} each, the Price List rate); a session costs that once per trace it holds, "
        "and the evaluator model's own usage is charged on top."
    )


def _wait_active(
    clients: AwsClients,
    config_id: str,
    wait_seconds: float,
    *,
    poll_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    """Read the config until it is ``ACTIVE``; raise on a failed status or when the wait ends."""
    deadline = clock() + wait_seconds
    created = (f"online evaluation config {config_id}",)
    while True:
        got = call_aws(
            clients,
            _CONTROL,
            "GetOnlineEvaluationConfig",
            {"onlineEvaluationConfigId": config_id},
            created=created,
        )
        status = str(got.get("status", ""))
        if status == "ACTIVE":
            return
        reason = got.get("failureReason")
        if status in _FAILED_STATUSES or clock() >= deadline:
            raise AgentCoreResourceNotActive(
                "online evaluation config", config_id, status, str(reason) if reason else None
            )
        sleep(poll_seconds)


def aws_call_failed(exc: BaseException) -> bool:
    """Whether a wrapped failure was an AWS call failing (throttled, denied, timed out).

    Such a failure says nothing about the resource the call read, so a caller reports it rather
    than treat the resource as changed. It lives here so the CLI needs no SDK import.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    return isinstance(exc.__cause__, BotoCoreError | ClientError)


def aws_error_code(exc: BaseException) -> str | None:
    """The AWS error code behind a wrapped call failure, if it was a service error."""
    response = getattr(exc.__cause__, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        return code if isinstance(code, str) else None
    return None


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else ""
