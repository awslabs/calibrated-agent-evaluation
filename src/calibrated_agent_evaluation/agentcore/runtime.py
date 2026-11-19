# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore Runtime an executor invokes, the IDs and trace headers of each run, and the calls.

:class:`RuntimeTarget` names the runtime, endpoint qualifier, request payload format and optional
configuration bundle, and refuses any that can't be invoked as given. :func:`new_run_ids` makes a
run's session ID and W3C trace context once; :func:`traceparent` and :func:`baggage` turn them
into the headers every invocation of the run carries, so the agent's spans are found by ID rather
than by time. :class:`RunIds` holds an ID read back from a file to the form a new one has.
:func:`request_body` builds a prompt's payload and refuses one that is not standard JSON before
anything is sent. The ARN, bundle and endpoint-name patterns are those of the installed botocore
``bedrock-agentcore-control`` service model, matched whole because the model's are unanchored.

Three functions make the AWS calls of a run.

- :func:`resolve_runtime` is the read-only preflight. It refuses, before anything billed is sent,
  a runtime the executor can't call: one that only accepts OAuth, or doesn't serve HTTP.
- :func:`invoke` sends one run's invocation through clients built with
  :func:`runtime_client_config`, so the SDK sends each request exactly once: an SDK retry after
  the agent started would run it again in the same session. It retries only a busy session and
  throttling itself, with the same IDs and body, and refuses every other failure by name, saying
  whether the request was never sent or the agent may still have run.
- :func:`stop` stops a run's session with the SDK's standard retries, which are safe because the
  stop carries an idempotency token, and waits at most :data:`STOP_TIMEOUT_SECONDS`. It reports
  how the stop went rather than raising, because a run's result stands either way.

Every refusal raises
:class:`~calibrated_agent_evaluation.agentcore.runtime_errors.RuntimeInvocationError`, chained from
its cause, if any. Messages name operations, error codes, error types, session IDs, ARNs, protocols
and status codes, never the payload, the response or an error's text.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeAlias
from urllib.parse import quote

from botocore import xform_name
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
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
from pydantic import ConfigDict, field_validator, model_validator
from urllib3.exceptions import HTTPError as URLLib3HTTPError

from calibrated_agent_evaluation._finite import (
    MAX_JSON_DEPTH,
    NestedTooDeepError,
    NonFiniteNumberError,
    require_finite,
)
from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.agentcore.endpoint_label import DEFAULT_QUALIFIER
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, BilledCallBudgetExceeded
from calibrated_agent_evaluation.agentcore.guards import BUDGET_ENV_VAR
from calibrated_agent_evaluation.agentcore.run_manifest import VersionSource
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError
from calibrated_agent_evaluation.backends._bounded import call_bounded
from calibrated_agent_evaluation.probe.adapter import PAYLOAD_FORMATS

if TYPE_CHECKING:
    from botocore.client import BaseClient

    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "AGENT_RUNTIME_ARN",
    "CONFIGURATION_BUNDLE_ARN",
    "CONFIGURATION_BUNDLE_VERSION",
    "CONTROL_SERVICE",
    "CUSTOM_PAYLOAD_FORMAT",
    "DATA_SERVICE",
    "DEFAULT_QUALIFIER",
    "ENDPOINT_NAME",
    "NOT_SENT_ERRORS",
    "PARENT_SPAN_ID_PATTERN",
    "PAYLOAD_FORMAT_NAMES",
    "RETRIED_ERROR_CODES",
    "SESSION_ID_PATTERN",
    "STOP_TIMEOUT_SECONDS",
    "W3C_TRACE_ID_PATTERN",
    "InvocationProgress",
    "PayloadBuilder",
    "RawResponse",
    "RunIds",
    "RuntimeInfo",
    "RuntimeTarget",
    "StopResult",
    "baggage",
    "guard_hint",
    "guard_refusal",
    "invoke",
    "new_run_ids",
    "request_body",
    "resolve_runtime",
    "runtime_client_config",
    "stop",
    "traceparent",
    "turn_run_ids",
]

logger = logging.getLogger(__name__)

AGENT_RUNTIME_ARN = (
    "arn:aws(-[^:]+)?:bedrock-agentcore:[a-z0-9-]+:[0-9]{12}:"
    "runtime/[a-zA-Z][a-zA-Z0-9_]{0,47}-[a-zA-Z0-9]{10}"
)
"""The ``AgentRuntimeArn`` pattern: ``arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>``.

Matched whole, like the other patterns.
"""

CONFIGURATION_BUNDLE_ARN = (
    "arn:aws[a-zA-Z-]*:bedrock-agentcore:[a-z0-9-]+:[0-9]{12}:"
    "configuration-bundle/[a-zA-Z][a-zA-Z0-9-_]{0,99}-[a-zA-Z0-9]{10}"
)
"""The ``ConfigurationBundleArn`` pattern."""

CONFIGURATION_BUNDLE_VERSION = "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
"""The ``ConfigurationBundleVersion`` pattern: a lowercase UUID."""

ENDPOINT_NAME = "[a-zA-Z][a-zA-Z0-9_]{0,47}"
"""The ``EndpointName`` pattern a qualifier must match: a letter, then up to 47 letters, digits and
underscores.

The invocation's ``qualifier`` parameter has no pattern of its own and botocore doesn't check this
one on the client, so without it an empty or slashed qualifier would reach the service.
"""

SESSION_ID_PATTERN = "calibrated-agent-evaluation-[0-9a-f]{32}"
"""A run's runtime session ID: the name, a hyphen and 32 lowercase hex characters.

The prefix is ``calibrated-agent-evaluation-``, so an ID is 60 characters in all.

One whose hex is all zeros is invalid as well.
"""

W3C_TRACE_ID_PATTERN = "[0-9a-f]{32}"
"""A W3C trace ID: 32 lowercase hex characters. An all-zero one is invalid as well."""

PARENT_SPAN_ID_PATTERN = "[0-9a-f]{16}"
"""A W3C parent span ID: 16 lowercase hex characters. An all-zero one is invalid as well."""

CUSTOM_PAYLOAD_FORMAT = "custom"
"""The payload format for a request body the caller's own builder makes."""

PAYLOAD_FORMAT_NAMES: tuple[str, ...] = (
    *(payload_format.name for payload_format in PAYLOAD_FORMATS),
    CUSTOM_PAYLOAD_FORMAT,
)
"""Every accepted payload format: the probe's request shapes, then ``custom``."""

PayloadBuilder: TypeAlias = Callable[[str, str], Mapping[str, object]]
"""A custom request body: ``(prompt, session_id) -> JSON object``."""

_RUNTIME_ARN = re.compile(AGENT_RUNTIME_ARN)
_BUNDLE_ARN = re.compile(CONFIGURATION_BUNDLE_ARN)
_BUNDLE_VERSION = re.compile(CONFIGURATION_BUNDLE_VERSION)
_ENDPOINT_NAME = re.compile(ENDPOINT_NAME)
_SESSION_ID = re.compile(SESSION_ID_PATTERN)
_W3C_TRACE_ID = re.compile(W3C_TRACE_ID_PATTERN)
_PARENT_SPAN_ID = re.compile(PARENT_SPAN_ID_PATTERN)
_NAMED_FORMATS = {payload_format.name: payload_format for payload_format in PAYLOAD_FORMATS}

# The bundle keys are those the AgentCore SDK reads from a request's baggage to load a bundle. It
# keeps them as literals in its parser rather than exporting them, so they are repeated here.
_SESSION_KEY = "session.id"
_BUNDLE_ARN_KEY = "aws.agentcore.configbundle_arn"
_BUNDLE_VERSION_KEY = "aws.agentcore.configbundle_version"


class _RuntimeModel(CalibratedAgentEvaluationModel):
    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = RuntimeInvocationError


class RuntimeTarget(_RuntimeModel):
    """The AgentCore Runtime agent to invoke, and how to ask it.

    Attributes:
        runtime_arn: The runtime's ARN,
            ``arn:aws:bedrock-agentcore:<region>:<account>:runtime/<name>-<10-character suffix>``.
            Its region is where every call goes.
        qualifier: The endpoint every invocation goes to: a letter, then up to 47 letters,
            digits and underscores.
        payload_format: How a prompt becomes a request body: one of ``generic_message``,
            ``openai``, ``agentcore_sse`` and ``nested_agentcore``, or ``custom`` for a body the
            caller's own builder makes.
        config_bundle_arn: The configuration bundle the agent runs with. Name it together with
            its version, or name neither; an empty string counts as not named.
        config_bundle_version: That bundle's version, a lowercase UUID.
    """

    runtime_arn: str
    qualifier: str = DEFAULT_QUALIFIER
    payload_format: str
    config_bundle_arn: str | None = None
    config_bundle_version: str | None = None

    @field_validator("runtime_arn")
    @classmethod
    def _is_a_runtime_arn(cls, value: str) -> str:
        if _RUNTIME_ARN.fullmatch(value) is None:
            raise ValueError(
                f"{value!r} is not an AgentCore Runtime ARN; pass the runtime's own ARN, "
                "arn:aws:bedrock-agentcore:<region>:<account>:runtime/<name>-<10-character "
                "suffix>, not an endpoint or another resource's ARN"
            )
        return value

    @field_validator("qualifier")
    @classmethod
    def _is_an_endpoint_name(cls, value: str) -> str:
        if _ENDPOINT_NAME.fullmatch(value) is None:
            raise ValueError(
                f"{value!r} is not an endpoint name; name the runtime endpoint to invoke, a letter "
                "followed by up to 47 letters, digits and underscores, such as DEFAULT"
            )
        return value

    @field_validator("payload_format")
    @classmethod
    def _is_a_known_format(cls, value: str) -> str:
        if value not in PAYLOAD_FORMAT_NAMES:
            raise ValueError(
                f"{value!r} is not a payload format; use one of {', '.join(PAYLOAD_FORMAT_NAMES)}"
            )
        return value

    @field_validator("config_bundle_arn", "config_bundle_version", mode="before")
    @classmethod
    def _empty_is_not_named(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("config_bundle_arn")
    @classmethod
    def _is_a_bundle_arn(cls, value: str | None) -> str | None:
        if value is not None and _BUNDLE_ARN.fullmatch(value) is None:
            raise ValueError(
                f"{value!r} is not a configuration bundle ARN; expected "
                "arn:aws:bedrock-agentcore:<region>:<account>:configuration-bundle/<name>-"
                "<10-character suffix>"
            )
        return value

    @field_validator("config_bundle_version")
    @classmethod
    def _is_a_bundle_version(cls, value: str | None) -> str | None:
        if value is not None and _BUNDLE_VERSION.fullmatch(value) is None:
            raise ValueError(
                f"{value!r} is not a configuration bundle version; expected a lowercase UUID "
                "such as 0f8b6c1e-3a2d-4c5b-9e7f-1a2b3c4d5e6f"
            )
        return value

    @model_validator(mode="after")
    def _bundle_is_whole(self) -> Self:
        # Half a bundle would run the agent without the configuration its records name. Raised
        # typed: a model-level check runs outside the field errors' typed wrapping.
        arn_named = self.config_bundle_arn is not None
        if arn_named != (self.config_bundle_version is not None):
            named, missing = ("config_bundle_arn", "config_bundle_version")
            if not arn_named:
                named, missing = missing, named
            raise RuntimeInvocationError(
                f"{named} is set but {missing} is missing; name a configuration bundle by both "
                "its ARN and its version, or name neither"
            )
        return self

    @property
    def region(self) -> str:
        """The AWS region in the runtime ARN."""
        return self.runtime_arn.split(":")[3]

    @property
    def runtime_id(self) -> str:
        """The runtime's ID: the ARN's resource after ``runtime/``."""
        return self.runtime_arn.split(":", 5)[5].partition("/")[2]


class RunIds(_RuntimeModel):
    """The IDs one run is correlated by: its runtime session and its W3C trace context.

    Attributes:
        session_id: The runtime session: ``calibrated-agent-evaluation-`` and 32 lowercase hex
            characters, 60 in all, never all zeros.
        w3c_trace_id: The W3C trace ID: 32 lowercase hex characters, never all zeros.
        parent_span_id: The W3C parent span ID: 16 lowercase hex characters, never all zeros.
    """

    session_id: str
    w3c_trace_id: str
    parent_span_id: str

    # The messages name the field, never the value, which may have been read from a file.
    @field_validator("session_id")
    @classmethod
    def _is_a_session_id(cls, value: str) -> str:
        if _SESSION_ID.fullmatch(value) is None:
            raise ValueError(
                "not a run's session ID: expected calibrated-agent-evaluation- and 32 lowercase "
                "hex characters"
            )
        if not value.removeprefix("calibrated-agent-evaluation-").strip("0"):
            raise ValueError(
                "an all-zero session ID is invalid; expected calibrated-agent-evaluation- and 32 "
                "lowercase hex "
                "characters with a nonzero digit"
            )
        return value

    @field_validator("w3c_trace_id")
    @classmethod
    def _is_a_trace_id(cls, value: str) -> str:
        return _w3c_id(value, _W3C_TRACE_ID, "trace ID", 32)

    @field_validator("parent_span_id")
    @classmethod
    def _is_a_parent_span_id(cls, value: str) -> str:
        return _w3c_id(value, _PARENT_SPAN_ID, "parent span ID", 16)


def _w3c_id(value: str, pattern: re.Pattern[str], what: str, length: int) -> str:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"not a W3C {what}: expected {length} lowercase hex characters")
    if not value.strip("0"):
        raise ValueError(f"an all-zero W3C {what} is invalid; expected one with a nonzero digit")
    return value


def new_run_ids() -> RunIds:
    """New IDs for one run: a new runtime session and a new W3C trace and parent span.

    Every retry of the run reuses them; the next run gets new ones.

    Returns:
        The run's IDs.
    """
    return RunIds(
        session_id=f"calibrated-agent-evaluation-{uuid.uuid4().hex}",
        w3c_trace_id=_nonzero_hex(16),
        parent_span_id=_nonzero_hex(8),
    )


def turn_run_ids(session_id: str) -> RunIds:
    """IDs for the next turn of a conversation: its session kept, a new W3C trace and parent span.

    Every turn of one conversation is sent on the same runtime session, so the deployed agent
    keeps its memory of the earlier turns, and each turn is its own trace.

    Args:
        session_id: The conversation's session, from its first turn's :func:`new_run_ids`.

    Returns:
        The turn's IDs.

    Raises:
        RuntimeInvocationError: ``session_id`` is not a run's session ID.
    """
    return RunIds(
        session_id=session_id,
        w3c_trace_id=_nonzero_hex(16),
        parent_span_id=_nonzero_hex(8),
    )


def _nonzero_hex(nbytes: int) -> str:
    # W3C trace context treats an all-zero trace ID or parent ID as invalid, so redraw one.
    while True:
        value = secrets.token_hex(nbytes)
        if value.strip("0"):
            return value


def traceparent(ids: RunIds) -> str:
    """The W3C ``traceparent`` header of a run: version 00, its trace and parent span, sampled.

    Args:
        ids: The run's IDs.

    Returns:
        ``00-<trace ID>-<parent span ID>-01``.
    """
    return f"00-{ids.w3c_trace_id}-{ids.parent_span_id}-01"


def baggage(target: RuntimeTarget, ids: RunIds) -> str:
    """The W3C ``baggage`` header of a run: its session ID, and the target's bundle if it names one.

    The bundle entries use the keys the AgentCore SDK reads to load a configuration bundle in the
    agent. Values are percent-encoded.

    Args:
        target: The runtime being invoked.
        ids: The run's IDs.

    Returns:
        The comma-separated ``key=value`` entries.
    """
    entries = [(_SESSION_KEY, ids.session_id)]
    if target.config_bundle_arn is not None and target.config_bundle_version is not None:
        entries.append((_BUNDLE_ARN_KEY, target.config_bundle_arn))
        entries.append((_BUNDLE_VERSION_KEY, target.config_bundle_version))
    return ",".join(f"{key}={quote(value, safe='')}" for key, value in entries)


# --- request body ---------------------------------------------------------------------------


def request_body(
    target: RuntimeTarget,
    prompt: str,
    session_id: str,
    builder: PayloadBuilder | None = None,
    builder_name: str | None = None,
) -> bytes:
    """The request payload for one prompt: a JSON object, as UTF-8 bytes.

    A named payload format uses the probe's request shape of that name; ``custom`` calls
    ``builder(prompt, session_id)``. Call it before the run's first AWS call, so a body that can't
    be built costs nothing.

    Args:
        target: The runtime being invoked; its payload format decides how the body is made.
        prompt: The prompt to send.
        session_id: The run's session ID, which some formats carry in the body.
        builder: Builds the body for the ``custom`` format. Give none with a named format.
        builder_name: Names ``builder`` in messages. Give none with a named format; an empty
            name counts as none.

    Returns:
        The body, serialized without NaN or infinity, which are not JSON.

    Raises:
        RuntimeInvocationError: ``custom`` was given no builder, or a named format was given a
            builder or a builder name; the builder raised (named by the error's type) or returned
            something other than a mapping (named by its type); or the body holds a NaN or
            infinite number, nests more than 256 levels deep counted from the body, or does not
            serialize to JSON. Each message names the format or the builder and says nothing was
            invoked, never carrying the builder's error text or a value from the body.
    """
    payload_format = target.payload_format
    if payload_format == CUSTOM_PAYLOAD_FORMAT:
        if builder is None:
            raise RuntimeInvocationError(
                f"payload_format={CUSTOM_PAYLOAD_FORMAT!r} needs a payload builder to make the "
                "request body, and none was given, so nothing was invoked"
            )
        source = f"the payload builder {builder_name!r}" if builder_name else "the payload builder"
        try:
            built = builder(prompt, session_id)
            document = dict(built) if isinstance(built, Mapping) else None
        except Exception as exc:
            raise RuntimeInvocationError(
                f"{source} failed with {type(exc).__name__}, so nothing was invoked. Fix the "
                "builder, then run again"
            ) from exc
        if document is None:
            raise RuntimeInvocationError(
                f"{source} returned a {type(built).__name__}, not a JSON object, so nothing was "
                "invoked. Make it return a mapping of the request's fields"
            )
    else:
        # Refused here too, so a direct caller is held to the executor's pairing.
        stray = [
            given
            for given, present in (
                ("a payload builder", builder is not None),
                (f"the builder name {builder_name!r}", bool(builder_name)),
            )
            if present
        ]
        if stray:
            raise RuntimeInvocationError(
                f"{' and '.join(stray)} {'was' if len(stray) == 1 else 'were'} given with "
                f"payload_format={payload_format!r}, which builds its own body, so nothing was "
                f"invoked; use payload_format={CUSTOM_PAYLOAD_FORMAT!r} with a builder and its "
                f"name, or drop {'it' if len(stray) == 1 else 'them'}"
            )
        source = f"the payload format {payload_format!r}"
        document = dict(_NAMED_FORMATS[payload_format].build_body(prompt, session_id))
    try:
        # Checked first, so the answer doesn't depend on how deep this Python's encoder goes.
        require_finite(document, where="request body")
        # NaN and infinity are not JSON, though Python writes them unless told not to.
        encoded = json.dumps(document, allow_nan=False)
    except NonFiniteNumberError as exc:
        raise RuntimeInvocationError(
            f"{source} built a body holding a number JSON cannot represent (NaN or an "
            "infinity), so nothing was invoked. Make it return finite numbers only"
        ) from exc
    except NestedTooDeepError as exc:
        raise RuntimeInvocationError(
            f"{source} built a body that nests more than {MAX_JSON_DEPTH} levels deep, deeper "
            "than Calibrated Agent Evaluation sends as JSON, so nothing was invoked. Make it "
            "return a flatter object"
        ) from exc
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuntimeInvocationError(
            f"{source} built a body that does not serialize to JSON ({type(exc).__name__}), so "
            "nothing was invoked. Make it return a JSON object: strings, finite numbers, booleans, "
            "null, lists and objects"
        ) from exc
    return encoded.encode("utf-8")


# --- preflight ------------------------------------------------------------------------------

CONTROL_SERVICE = "bedrock-agentcore-control"
"""The AWS service the read-only preflight calls: the runtime's endpoint and its version."""

DATA_SERVICE = "bedrock-agentcore"
"""The AWS service a run invokes and whose sessions it stops."""

_INVOKE = f"{DATA_SERVICE}:InvokeAgentRuntime"
_GET_RUNTIME = "GetAgentRuntime"
_HTTP = "HTTP"


class RuntimeInfo(_RuntimeModel):
    """What a successful preflight learned about the runtime version that will serve the runs.

    Attributes:
        runtime_version: The runtime version.
        version_source: ``endpoint`` when it is the endpoint's live version, ``latest`` when the
            endpoint reported none and it is the runtime's latest version.
    """

    runtime_version: str
    version_source: VersionSource


def resolve_runtime(base: AwsClients, target: RuntimeTarget) -> RuntimeInfo:
    """Read the runtime version the target's endpoint serves; refuse one the library can't call.

    Reads the endpoint with ``GetAgentRuntimeEndpoint``, then the runtime at the endpoint's live
    version with ``GetAgentRuntime``, or with no version (the latest) when the endpoint reports
    none. Both calls are read-only and go through ``base``, keeping its retries and guards.

    Args:
        base: The clients for the target's region.
        target: The runtime and the qualifier to resolve.

    Returns:
        The runtime version and where it came from.

    Raises:
        RuntimeInvocationError: A preflight call failed or a guard refused it, naming the
            operation and the error code or type; the runtime read answered without its version;
            the runtime has a custom JWT authorizer, so it only accepts OAuth; or it serves a
            protocol other than HTTP, named. A runtime with no protocol configuration counts as
            HTTP.
    """
    control = base.client(CONTROL_SERVICE)
    endpoint = _preflight(
        control,
        "GetAgentRuntimeEndpoint",
        target,
        agentRuntimeId=target.runtime_id,
        endpointName=target.qualifier,
    )
    live_version = endpoint.get("liveVersion")
    source: VersionSource
    if live_version is not None:
        source = "endpoint"
        runtime = _preflight(
            control,
            _GET_RUNTIME,
            target,
            agentRuntimeId=target.runtime_id,
            agentRuntimeVersion=live_version,
        )
    else:
        source = "latest"
        runtime = _preflight(control, _GET_RUNTIME, target, agentRuntimeId=target.runtime_id)
    authorizer = runtime.get("authorizerConfiguration") or {}
    if authorizer.get("customJWTAuthorizer") is not None:
        raise RuntimeInvocationError(
            f"runtime {target.runtime_arn} has a custom JWT authorizer, so it only accepts OAuth "
            "calls. OAuth-only runtimes are not supported yet, so nothing was invoked; use a "
            "runtime that accepts AWS IAM (SigV4) calls"
        )
    protocol = (runtime.get("protocolConfiguration") or {}).get("serverProtocol", _HTTP)
    if protocol != _HTTP:
        raise RuntimeInvocationError(
            f"runtime {target.runtime_arn} serves the {protocol!r} protocol, and Calibrated Agent "
            "Evaluation only "
            "invokes HTTP runtimes, so nothing was invoked; use a runtime deployed with the HTTP "
            "protocol"
        )
    version = runtime.get("agentRuntimeVersion")
    if not isinstance(version, str) or not version:
        # The service always names it, but botocore doesn't check what it parses, and every run
        # record names its version.
        raise RuntimeInvocationError(
            f"the preflight call {CONTROL_SERVICE}:{_GET_RUNTIME} for runtime "
            f"{target.runtime_arn} answered without the runtime's version, so nothing was "
            "invoked. Check that the runtime and its endpoint have a deployed version, then run "
            "again"
        )
    return RuntimeInfo(runtime_version=version, version_source=source)


def _preflight(
    control: BaseClient, operation: str, target: RuntimeTarget, **params: Any
) -> dict[str, Any]:
    qualified = f"{CONTROL_SERVICE}:{operation}"
    try:
        return getattr(control, xform_name(operation))(**params)
    except AgentCoreError as exc:
        # Only a guard on the clients raises a Calibrated Agent Evaluation error from inside an AWS
        # call.
        raise RuntimeInvocationError(
            f"{guard_refusal(exc, qualified)}, so nothing was invoked. {guard_hint(exc)}"
        ) from exc
    except Exception as exc:
        # A service error, an HTTP-layer error botocore let through, or any other, such as one a
        # caller's own handler raised.
        failure = _error_code(exc) if isinstance(exc, ClientError) else ""
        raise RuntimeInvocationError(
            f"the preflight call {qualified} for runtime {target.runtime_arn} failed with "
            f"{failure or type(exc).__name__}, so nothing was invoked. Check that the runtime and "
            f"its endpoint exist in {target.region}, that AWS is reachable, and that your AWS "
            "identity may read them"
        ) from exc


# --- invocation -----------------------------------------------------------------------------

_INVOKE_CONNECT_TIMEOUT_SECONDS = 10
_BACKOFF_CAP_SECONDS = 8

RETRIED_ERROR_CODES: frozenset[str] = frozenset(
    {"RetryableConflictException", "ThrottlingException"}
)
"""The invocation errors retried with backoff: a session busy with an earlier request (409), and
throttling (429). The service documents both as temporary, and both are taken to mean the request
was turned away before the agent handled it, so retrying it doesn't run the agent twice."""

NOT_SENT_ERRORS: tuple[type[Exception], ...] = (
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
"""The SDK errors raised before an invocation reached the service, so the agent did not run.

A request parameter the SDK refused; credentials that are missing, or that could not be loaded or
refreshed (an expired SSO token, a failed credential process, a refresh that needs an MFA code);
and a connection that could not be made. Any other SDK error from an invocation may come after
the request was sent. A connect timeout is an ``OSError``, just like a read timeout, so these are
told apart before the broader errors.
"""

# How an AWS call fails short of a service error. botocore wraps most of urllib3's errors, but lets
# some through as they are: an SSL or content-decoding error while a body is read, and any urllib3
# error while it reads the whole answer of a call that does not stream, which it also never retries.
_CALL_FAILURES: tuple[type[Exception], ...] = (BotoCoreError, OSError, URLLib3HTTPError)

# Raised once the request may have reached the service, while waiting for or reading the response.
_AFTER_SENDING = _CALL_FAILURES


def runtime_client_config(read_timeout: float) -> Config:
    """The client settings for invocations: a 10-second connect timeout and one SDK attempt.

    Merge it over a set of clients with
    :meth:`~calibrated_agent_evaluation.agentcore.session.AwsClients.with_config`, so the SDK sends
    each request exactly once: an SDK retry after the agent started would run it again in the same
    session. :func:`invoke` retries only a busy session and throttling itself.

    Args:
        read_timeout: Seconds to wait for each read of the response, the whole stream included.

    Returns:
        The settings to merge.
    """
    return Config(
        connect_timeout=_INVOKE_CONNECT_TIMEOUT_SECONDS,
        read_timeout=read_timeout,
        # The mode is named: left out, it would come from the environment or the AWS profile.
        retries={"mode": "standard", "total_max_attempts": 1},
    )


class RawResponse(_RuntimeModel):
    """An invocation's response, read whole and closed.

    Attributes:
        status_code: The HTTP status, always 200 here.
        content_type: The response's content type, as sent.
        lines: The body's lines, as bytes without their line breaks.
        response_traceparent: The ``traceparent`` the runtime sent back, or ``None``.
        attempts: How many attempts were sent, the one that succeeded included.
    """

    status_code: int
    content_type: str | None
    lines: tuple[bytes, ...]
    response_traceparent: str | None
    attempts: int


@dataclass
class InvocationProgress:
    """What one run's invocation reached so far, kept up to date by :func:`invoke` on each attempt.

    An attempt counts when the service answered it, or when it failed after it may have been sent;
    one a guard refused, or that failed before the request left, does not. So after a failed
    :func:`invoke`, ``sent`` says whether the run may have opened a session.

    Attributes:
        attempts: The attempts that may have reached the service.
        status_code: The HTTP status of the last answer, or ``None`` before any answer.
        content_type: The last answer's content type, or ``None``.
        response_traceparent: The ``traceparent`` the last answer carried, or ``None``.
    """

    attempts: int = 0
    status_code: int | None = None
    content_type: str | None = None
    response_traceparent: str | None = None

    @property
    def sent(self) -> bool:
        """Whether any attempt may have reached the service."""
        return self.attempts > 0


def _note_answer(
    progress: InvocationProgress, status: object, content_type: object, response_traceparent: object
) -> None:
    progress.attempts += 1
    progress.status_code = status if isinstance(status, int) else None
    progress.content_type = content_type if isinstance(content_type, str) else None
    progress.response_traceparent = (
        response_traceparent if isinstance(response_traceparent, str) else None
    )


def invoke(
    invocation_client: BaseClient,
    target: RuntimeTarget,
    ids: RunIds,
    body: bytes,
    *,
    attempt_limit: int,
    sleep: Callable[[float], object],
    progress: InvocationProgress | None = None,
) -> RawResponse:
    """Send one run's invocation, retrying only a busy session or throttling, and read the response.

    Every attempt carries the same session ID, trace headers and body. A
    ``RetryableConflictException`` or ``ThrottlingException`` is retried after 1, 2, 4, then 8
    seconds, up to ``attempt_limit`` attempts; any other failure ends the run at once. The
    response body is read whole and closed here.

    Args:
        invocation_client: A ``bedrock-agentcore`` client built with
            :func:`runtime_client_config` merged over its settings.
        target: The runtime, qualifier and bundle to invoke.
        ids: The run's IDs.
        body: The request payload, JSON.
        attempt_limit: The most attempts to send, the first one included.
        sleep: Waits the given number of seconds between attempts.
        progress: Updated on each attempt, so the caller can read what was reached when this
            raises. Pass a new one for each run.

    Returns:
        The response.

    Raises:
        RuntimeInvocationError: ``attempt_limit`` is below 1; the last allowed attempt still found
            the session busy or was throttled; the service refused the request with any other
            error, named by its code; a guard on the clients refused the attempt before it was
            sent; the request was never sent; the request failed after it was sent, or with an
            error not known to come before sending, so the agent may still have run; or the status
            was a success other than 200. Every message names the session.
    """
    if attempt_limit < 1:
        raise RuntimeInvocationError(
            f"attempt_limit={attempt_limit} allows no invocation; pass 1 or more"
        )
    params = {
        "agentRuntimeArn": target.runtime_arn,
        "qualifier": target.qualifier,
        "runtimeSessionId": ids.session_id,
        "payload": body,
        "contentType": "application/json",
        "accept": "text/event-stream, application/json",
        "traceParent": traceparent(ids),
        "baggage": baggage(target, ids),
    }
    session = ids.session_id
    reached = InvocationProgress() if progress is None else progress
    attempt = 0
    while True:
        attempt += 1
        try:
            response = _send(invocation_client, params, session, reached)
        except ClientError as exc:
            code = _error_code(exc) or type(exc).__name__
            if code not in RETRIED_ERROR_CODES:
                raise RuntimeInvocationError(
                    f"{_INVOKE} for session {session} failed with {code}, which is not retried. "
                    "Look the session up in the agent's logs, and check the runtime, the payload "
                    "format and your access to them"
                ) from exc
            if attempt >= attempt_limit:
                tried = f"{attempt} attempt" if attempt == 1 else f"all {attempt} attempts"
                raise RuntimeInvocationError(
                    f"{_INVOKE} for session {session} failed with {code} on {tried}: the session "
                    "stayed busy or the runtime kept throttling. Wait and run again, or allow "
                    "more attempts"
                ) from exc
            sleep(min(2 ** (attempt - 1), _BACKOFF_CAP_SECONDS))
            continue
        return _read(response, session, attempt)


def _send(
    client: BaseClient, params: dict[str, Any], session: str, progress: InvocationProgress
) -> dict[str, Any]:
    # One SDK attempt. A service error is left to the caller, which decides whether to retry it.
    try:
        response = client.invoke_agent_runtime(**params)  # pyright: ignore[reportAttributeAccessIssue]
    except AgentCoreError as exc:
        # Only a guard on the clients raises a Calibrated Agent Evaluation error from inside an AWS
        # call.
        raise RuntimeInvocationError(
            f"{guard_refusal(exc, _INVOKE)} for session {session}, so it was not sent and no "
            f"further invocation is sent in this run. {guard_hint(exc)}"
        ) from exc
    except NOT_SENT_ERRORS as exc:
        raise RuntimeInvocationError(
            f"{_INVOKE} for session {session} was not sent: it failed with {type(exc).__name__} "
            "before the request reached the runtime, so the agent did not run. Fix the cause "
            "(credentials, network, proxy or a request parameter) and run again"
        ) from exc
    except ClientError as exc:
        metadata = exc.response.get("ResponseMetadata") or {}
        headers = metadata.get("HTTPHeaders") or {}
        _note_answer(
            progress,
            metadata.get("HTTPStatusCode"),
            headers.get("content-type"),
            headers.get("traceparent"),
        )
        raise
    except _AFTER_SENDING as exc:
        progress.attempts += 1
        raise _may_have_run(exc, session) from exc
    except Exception as exc:
        # Not known to come before sending (a caller's own handler, say), so it may have gone out.
        progress.attempts += 1
        raise _may_have_run(exc, session, known_sent=False) from exc
    _note_answer(
        progress,
        response.get("statusCode"),
        response.get("contentType"),
        response.get("traceParent"),
    )
    return response


def _read(response: dict[str, Any], session: str, attempts: int) -> RawResponse:
    stream = response.get("response")
    try:
        status = response.get("statusCode")
        if status != 200:
            # botocore raises ClientError from 300 up: this is a 202 or 204, say, or no status.
            answered = "no HTTP status" if status is None else f"HTTP status {status}"
            raise RuntimeInvocationError(
                f"the runtime answered session {session} with {answered}, not 200, so the run has "
                "no answer to keep. Check that the agent answers within the request"
            )
        try:
            lines = () if stream is None else tuple(stream.iter_lines())
        except Exception as exc:
            # Any failure here, not only _AFTER_SENDING: the request was sent, so the agent may
            # have run.
            raise _may_have_run(exc, session) from exc
    finally:
        if stream is not None:
            stream.close()
    return RawResponse(
        status_code=status,
        content_type=response.get("contentType"),
        lines=lines,
        response_traceparent=response.get("traceParent"),
        attempts=attempts,
    )


def _may_have_run(
    exc: BaseException, session: str, *, known_sent: bool = True
) -> RuntimeInvocationError:
    if known_sent:
        when = "after the request was sent"
    else:
        # Not known to come before or after sending, so the message doesn't say it was sent.
        when = "when the request may already have been sent"
    return RuntimeInvocationError(
        f"{_INVOKE} for session {session} failed with {type(exc).__name__} {when}, so the agent "
        "may still have run. Look the session up in the agent's logs before running it again; if "
        "it timed out, allow a longer read timeout"
    )


# --- stopping a session ---------------------------------------------------------------------

StopResult: TypeAlias = Literal["stopped", "already_gone", "failed"]
"""How a stop went: stopped; already gone, which counts as stopped; or failed after retries."""

STOP_TIMEOUT_SECONDS = 30.0
"""How long :func:`stop` waits for a ``StopRuntimeSession`` call, retries included, before giving
up on it."""

_STOP_THREAD = "calibrated-agent-evaluation-agentcore-stop"


def stop(
    stop_client: BaseClient,
    target: RuntimeTarget,
    session_id: str,
    *,
    timeout_seconds: float = STOP_TIMEOUT_SECONDS,
) -> StopResult:
    """Stop a run's runtime session, so its microVM stops billing.

    Uses the SDK's standard retries, which are safe: the SDK fills in an idempotency token and
    reuses it across retries. The call runs on a worker thread and is waited for at most
    ``timeout_seconds``; one still running then is left to finish on its own and counts as failed.

    Args:
        stop_client: A ``bedrock-agentcore`` client with the standard retries.
        target: The runtime and qualifier the session was opened on.
        session_id: The session to stop.
        timeout_seconds: How long to wait for the call.

    Returns:
        ``stopped``; ``already_gone`` when the service no longer has the session; or ``failed``
        when the stop failed with any error (after the SDK's retries, for those it retries) or
        did not return in time, which is logged as a warning naming the session.

    Raises:
        BaseException: An interrupt, or any other error that is not an ``Exception``, raised
            during the call, raised again here in the caller's thread.
    """

    def stop_session() -> None:
        stop_client.stop_runtime_session(  # pyright: ignore[reportAttributeAccessIssue]
            agentRuntimeArn=target.runtime_arn,
            qualifier=target.qualifier,
            runtimeSessionId=session_id,
        )

    outcome = call_bounded(stop_session, timeout_seconds=timeout_seconds, name=_STOP_THREAD)
    if not outcome.finished:
        logger.warning(
            "could not stop runtime session %s: the stop did not return within %g seconds; it may "
            "stay billed until its idle timeout ends",
            session_id,
            timeout_seconds,
        )
        return "failed"
    error = outcome.error
    if error is None:
        return "stopped"
    if not isinstance(error, Exception):
        # An interrupt: the caller must see it, and the session still reads as not stopped.
        raise error
    if isinstance(error, ClientError):
        code = _error_code(error)
        if code == "ResourceNotFoundException":
            return "already_gone"
        failure = code or type(error).__name__
    else:
        # An HTTP-layer error botocore let through, say: the run's result stands, so it is
        # reported rather than raised.
        failure = type(error).__name__
    logger.warning(
        "could not stop runtime session %s (%s); it may stay billed until its idle timeout ends",
        session_id,
        failure,
    )
    return "failed"


# --- shared ---------------------------------------------------------------------------------


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code") or "")


def guard_refusal(refusal: AgentCoreError, operation: str) -> str:
    """How a refusal by a guard on the AWS clients is named in a message: the guard and the call.

    The guard is named by its type, never its text, so nothing a caller's guard wrote can reach
    the message.

    Args:
        refusal: The error the guard raised.
        operation: The call it refused, such as ``bedrock-agentcore:InvokeAgentRuntime``.

    Returns:
        ``a guard on the AWS clients (<type>) refused <operation>``.
    """
    return f"a guard on the AWS clients ({type(refusal).__name__}) refused {operation}"


def guard_hint(refusal: AgentCoreError) -> str:
    """What to do about a guard's refusal: raise or unset the billed-call budget, or check guards.

    Args:
        refusal: The error the guard raised.

    Returns:
        One sentence naming the next step.
    """
    if isinstance(refusal, BilledCallBudgetExceeded):
        return f"Raise {BUDGET_ENV_VAR}, or unset it to remove the limit"
    return "Check the guards installed on the AWS clients"
