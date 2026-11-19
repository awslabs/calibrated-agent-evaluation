# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentCore Code Interpreter sandbox: an opt-in ``SandboxBackend`` over Bedrock AgentCore.

Lives in ``calibrated_agent_evaluation.agentcore`` (every AgentCore adapter does), behind the
``[agentcore]`` lazy import so a bare install never loads ``boto3``. It runs code in a managed Code
Interpreter session and maps the service result onto the vendor-free
:class:`~calibrated_agent_evaluation.execute.sandbox.SandboxResult` the local sandbox also returns,
so the execution scorers are byte-identical across both backends.

Verified surface (``bedrock-agentcore==1.24.2``, introspected 2026-10-21):
``code_interpreter_client.code_session(region, session=..., identifier=...) -> CodeInterpreter``;
``CodeInterpreter.execute_code(code, language="python", clear_context=False) -> dict`` (wraps the
``invoke_code_interpreter`` data-plane call). Confirmed live (2026-10-22, us-west-2): the
response is ``{"sessionId", "stream": <EventStream>}`` whose
result arrives as a streamed ``{"result": {"structuredContent": {stdout, stderr, exitCode,
executionTime}, "isError": ...}}`` event — so the stream is drained inside the session context
(:func:`_result_from_stream`) and mapped by :func:`_normalize`, which raises :class:`AgentCoreError`
on a shape it does not recognise (an unexpected response is a typed refusal, never a wrong verdict).

Identity and billed-call limits are enforced by this adapter (``require_role`` + the budget):
``code_session`` builds its own ``bedrock-agentcore`` client from the session rather than taking a
guarded one of ours.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, BilledCallBudgetExceeded
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget
from calibrated_agent_evaluation.agentcore.session import AwsClients, require_role
from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteError,
    CassetteMiss,
    model_stamp,
)
from calibrated_agent_evaluation.execute.errors import SandboxError
from calibrated_agent_evaluation.execute.sandbox import SandboxResult

_CI_OPERATION = ("bedrock-agentcore", "InvokeCodeInterpreter")
_DEFAULT_IDENTIFIER = "aws.codeinterpreter.v1"

#: Test seam: ``(code, language, clear_context) -> raw service response``. Lets unit tests exercise
#: the mapping, budget, identity and cassette paths without a live account; the default builds
#: the real ``code_session`` and is exercised by the opt-in live parity test.
ExecuteFn = Callable[[str, str, bool], dict[str, Any]]


class CodeInterpreterSandbox:
    """A ``SandboxBackend`` backed by Bedrock AgentCore Code Interpreter.

    Policy refusals (billed-call budget, wrong identity) surface as their own
    :class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError` subclasses; an
    execution/response failure surfaces as
    :class:`~calibrated_agent_evaluation.execute.errors.SandboxError` (the seam's contract, which
    the execution scorer re-wraps into ``ScorerError``).
    """

    def __init__(
        self,
        *,
        region: str,
        clients: AwsClients | None = None,
        cassette: Cassette | None = None,
        budget: BilledCallBudget | None = None,
        role_arn: str | None = None,
        identifier: str = _DEFAULT_IDENTIFIER,
        execute: ExecuteFn | None = None,
    ) -> None:
        self._region = region
        self._clients = clients
        self._cassette = cassette
        self._budget = budget
        self._role_arn = role_arn
        self._identifier = identifier
        self._execute = execute
        self._model = model_stamp(f"agentcore-code-interpreter:{identifier}", {})

    def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        # The caller (the execution scorer's harness) enforces the real timeout in-process, so it
        # behaves the same here as on the local sandbox; the managed session's own timeout is only a
        # backstop.
        timeout_s: float,  # noqa: ARG002
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run ``code`` in a managed Code Interpreter session and return a normalized result.

        Raises:
            WrongAwsIdentity: a role is required and the caller is not it (pre-flight).
            BilledCallBudgetExceeded: the billed-call budget is used up (before the call is sent).
            SandboxError: the language is unsupported, seeded files were asked for, the call failed,
                the response was unparseable, or (REPLAY) nothing was recorded for it.
        """
        if language != "python":
            # Validate up front, before a session is started or the budget is charged, so an
            # unsupported language is a typed refusal here rather than a bare ValueError escaping
            # from deep in the SDK after we have already paid for the call.
            raise SandboxError(
                f"the Code Interpreter sandbox runs Python only in v1, got language={language!r}"
            )
        if files:
            # The seam declares SandboxError for run-failures; a capability gap is one of those.
            raise SandboxError(
                "the Code Interpreter sandbox does not support seeded files in v1; inline the "
                "inputs into the code"
            )
        if self._role_arn is not None:
            if self._clients is None:
                raise AgentCoreError(
                    "a role_arn was set but no AwsClients was provided, so the identity cannot be "
                    "checked; pass clients= or drop role_arn"
                )
            require_role(self._clients, self._role_arn)
        request = {
            "code": code,
            "language": language,
            "clearContext": True,
            "identifier": self._identifier,
        }
        normalized = self._dispatch(request, code, language)
        return _to_result(normalized)

    def _dispatch(self, request: dict[str, Any], code: str, language: str) -> dict[str, Any]:
        def live_call() -> dict[str, Any]:
            if self._budget is not None:
                self._budget.charge(_CI_OPERATION)
            return _normalize(self._invoke(code, language))

        try:
            if self._cassette is None:
                return live_call()
            return self._cassette.play(request, live_call, model=self._model)
        except BilledCallBudgetExceeded:
            raise  # a policy refusal surfaces as itself, not as a sandbox failure
        except CassetteMiss as exc:
            raise SandboxError(
                f"no recorded Code Interpreter response for this request (cassette REPLAY mode): "
                f"{exc}"
            ) from exc
        except AgentCoreError as exc:
            raise SandboxError(f"Code Interpreter response could not be parsed: {exc}") from exc
        except CassetteError as exc:
            raise SandboxError(f"Code Interpreter cassette error: {exc}") from exc
        except (BotoCoreError, ClientError) as exc:
            raise SandboxError(
                f"Code Interpreter call failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _invoke(self, code: str, language: str) -> dict[str, Any]:
        if self._execute is not None:
            return _result_from_stream(self._execute(code, language, True))
        from bedrock_agentcore.tools.code_interpreter_client import code_session

        session = self._clients.boto_session if self._clients is not None else None
        with code_session(self._region, session=session, identifier=self._identifier) as ci:
            raw = ci.execute_code(code, language=language, clear_context=True)
            # The result MUST be drained from the event stream inside the session context.
            return _result_from_stream(raw)


def _result_from_stream(raw: dict[str, Any]) -> dict[str, Any]:
    """Materialize the Code Interpreter result from the response event stream.

    Confirmed against the live service (2026-10-22): the ``invoke_code_interpreter`` response
    is ``{"sessionId": ..., "stream": <EventStream>}`` and the result (``structuredContent`` +
    ``isError``) arrives as a streamed ``{"result": {...}}`` event, which must be consumed before
    the session context exits. A test seam / cassette returns the already-materialized ``result``
    dict (no ``stream``), which passes through unchanged.
    """
    stream = raw.get("stream")
    if stream is None:
        return raw
    for event in stream:
        if isinstance(event, dict) and isinstance(event.get("result"), dict):
            return event["result"]
    raise AgentCoreError("Code Interpreter response stream carried no 'result' event")


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a Code Interpreter response to ``{stdout, stderr, exit_code, is_error}``.

    Accepts ``structuredContent`` at the top level or nested under ``result`` (both forms appear in
    AWS examples). Raises :class:`AgentCoreError` if no ``structuredContent`` is present, or if it
    carries no ``exitCode``, so an unrecognised shape is refused rather than silently scored as a
    pass. When the service flags ``isError`` but reports exit code 0, the exit code is forced
    non-zero so the scorer does not read a failed run as a pass.
    """
    content = raw
    if "structuredContent" not in content and isinstance(content.get("result"), dict):
        content = content["result"]
    structured = content.get("structuredContent")
    if not isinstance(structured, dict):
        raise AgentCoreError(
            f"unexpected Code Interpreter response: no 'structuredContent' (top-level keys "
            f"{sorted(raw)})"
        )
    if "exitCode" not in structured:
        # Missing exitCode must not default to 0: that would score a response we cannot read as a
        # PASS. Refuse it as a typed error (mapped to SandboxError) so the scorer gives no verdict.
        raise AgentCoreError(
            "unexpected Code Interpreter response: structuredContent has no "
            f"'exitCode' (keys {sorted(structured)})"
        )
    is_error = bool(content.get("isError", False))
    # A non-numeric exitCode/executionTime is a shape we don't recognise: refuse it as a typed
    # AgentCoreError (which _dispatch maps to SandboxError) rather than let int()/float() raise a
    # bare ValueError/TypeError that would escape the sandbox boundary un-wrapped.
    try:
        exit_code = int(structured["exitCode"])
        duration_s = float(structured.get("executionTime", 0.0) or 0.0)
    except (TypeError, ValueError) as exc:
        raise AgentCoreError(
            f"unexpected Code Interpreter response: non-numeric exitCode/executionTime ({exc})"
        ) from exc
    if is_error and exit_code == 0:
        exit_code = 1
    return {
        "stdout": str(structured.get("stdout", "")),
        "stderr": _clean_stderr(str(structured.get("stderr", ""))),
        "exit_code": exit_code,
        "is_error": is_error,
        "duration_s": duration_s,
    }


_IPYTHON_EXIT_NOTE = "To exit: use 'exit', 'quit', or Ctrl-D."


def _clean_stderr(stderr: str) -> str:
    """Drop the Code Interpreter shell's housekeeping exit note from stderr.

    The managed runtime runs code in an IPython-style shell that appends a ``To exit: use 'exit',
    'quit', or Ctrl-D.`` warning (both a ``UserWarning`` summary and the ``warn(...)`` source echo)
    to stderr. Left in, it becomes the last stderr line, which the execution scorer would show as a
    FAIL's critique instead of the real error. Removing it keeps the critique human-readable and
    makes the AgentCore stderr match the local sandbox's. Does not change any verdict.
    """
    kept = [line for line in stderr.splitlines() if _IPYTHON_EXIT_NOTE not in line]
    return "\n".join(kept).rstrip()


def _to_result(normalized: dict[str, Any]) -> SandboxResult:
    return SandboxResult(
        stdout=normalized["stdout"],
        stderr=normalized["stderr"],
        exit_code=normalized["exit_code"],
        timed_out=False,
        limits_applied=True,  # the managed sandbox enforces its own CPU/memory/network isolation
        duration_s=normalized.get("duration_s", 0.0),  # structuredContent.executionTime (live)
    )
