# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed errors for the AgentCore plug-in. Vendor-free, so they import on a bare install."""

from __future__ import annotations

from calibrated_agent_evaluation._errors import (
    CalibratedAgentEvaluationDataError,
    MissingExtraError,
)

__all__ = [
    "AgentCoreAdvisoryOnly",
    "AgentCoreAmbiguousSessionResult",
    "AgentCoreAppliedBundleMismatch",
    "AgentCoreArmHalfStamped",
    "AgentCoreArmSessionAmbiguous",
    "AgentCoreArmSessionSplit",
    "AgentCoreArmSourceIdRepeated",
    "AgentCoreArmsMissing",
    "AgentCoreArmsReadChanged",
    "AgentCoreBlankAssertion",
    "AgentCoreContentMissing",
    "AgentCoreContentUnserializable",
    "AgentCoreDevelopmentInputMissing",
    "AgentCoreDevelopmentSourceChanged",
    "AgentCoreDevelopmentSourceUnread",
    "AgentCoreError",
    "AgentCoreEvaluateLimitExceeded",
    "AgentCoreEvaluatorLevelUnsupported",
    "AgentCoreEvaluatorResultError",
    "AgentCoreExtraMissing",
    "AgentCoreIndexingTooLow",
    "AgentCoreMetricNotPublished",
    "AgentCoreMixedArm",
    "AgentCoreMixedSession",
    "AgentCoreNoPassDirection",
    "AgentCoreNotAConversation",
    "AgentCoreNotCertified",
    "AgentCoreNotReplayable",
    "AgentCorePartialResult",
    "AgentCoreRatingScaleMalformed",
    "AgentCoreReferenceDropped",
    "AgentCoreReplayProvenanceError",
    "AgentCoreResourceExists",
    "AgentCoreResourceNotActive",
    "AgentCoreResultUnaddressable",
    "AgentCoreRunDeclined",
    "AgentCoreRuntimeNotFound",
    "AgentCoreScoreIncomplete",
    "AgentCoreScoreOffScale",
    "AgentCoreSettlingLag",
    "AgentCoreSliceReused",
    "AgentCoreTruncatedResult",
    "AgentCoreUncertifiedJudge",
    "AwsCredentialsMissing",
    "BilledCallBudgetExceeded",
    "DatasetDecodeError",
    "DatasetInputError",
    "DatasetNameConflict",
    "DatasetPublishError",
    "DatasetSchemaUnsupported",
    "DatasetVersionNotFound",
    "DetectorCallError",
    "DetectorCallLimitExceeded",
    "DetectorConfigError",
    "DetectorError",
    "DetectorOutputError",
    "DetectorSessionTooLarge",
    "DraftNamingError",
    "PolicySpanMalformed",
    "ReadOnlyViolation",
    "SessionConversionError",
    "VocabularyParseError",
    "WrongAwsIdentity",
]


class AgentCoreError(CalibratedAgentEvaluationDataError):
    """An AgentCore integration call or configuration failed."""


class AgentCoreAdvisoryOnly(AgentCoreError):
    """An evaluator AWS can change without notice was asked to certify or gate.

    Built-in and third-party evaluators have no version that pins their behaviour, so they stay
    advisory and are refused at certification; only a Custom / pinned CustomDerived / pinned
    CustomCode evaluator can be certified.
    """


class AgentCoreExtraMissing(MissingExtraError, AgentCoreError):
    """An AgentCore module was asked for but the ``[agentcore]`` extra is not installed."""

    def __init__(self, purpose: str) -> None:
        super().__init__(("agentcore",), purpose)


class ReadOnlyViolation(AgentCoreError):
    """A read-only command tried an AWS operation outside its allowlist; refused before sending.

    ``operation`` is the refused call, ``(service, operation)``, or ``None`` when not named.
    """

    def __init__(self, message: str, *, operation: tuple[str, str] | None = None) -> None:
        super().__init__(message)
        self.operation = operation


class BilledCallBudgetExceeded(AgentCoreError):
    """A billed AWS operation would exceed the billed-call budget; refused before sending.

    ``operation`` is the refused call, ``(service, operation)``, or ``None`` when not named.
    """

    def __init__(self, message: str, *, operation: tuple[str, str] | None = None) -> None:
        super().__init__(message)
        self.operation = operation


class AwsCredentialsMissing(AgentCoreError):
    """No AWS credentials could be resolved for the requested profile."""


class WrongAwsIdentity(AgentCoreError):
    """The credentials belong to a different principal than the one required."""


class DatasetDecodeError(AgentCoreError):
    """A dataset example has a field the codec does not map; refused rather than dropped."""


class DatasetSchemaUnsupported(AgentCoreError):
    """A dataset uses a schema type the codec does not support."""


class DatasetVersionNotFound(AgentCoreError):
    """The requested dataset or dataset version does not exist (or was deleted)."""


class DatasetNameConflict(AgentCoreError):
    """A dataset with the requested name exists but is not the one this slice was published to."""


class DatasetPublishError(AgentCoreError):
    """Publishing a slice could not complete; the message says what state the dataset is left in."""


class DatasetInputError(AgentCoreError):
    """A dataset name or version given to publish or pull is one AgentCore can't accept."""


class AgentCoreContentMissing(AgentCoreError):
    """A span's message content was not found in the joined ``otel-rt-logs`` records.

    Raised rather than yielding a span with empty input/output/tool content: an under-joined
    trace reads as a complete one, which is the silent-loss failure this ingest exists to prevent.
    """


class AgentCorePartialResult(AgentCoreError):
    """A CloudWatch Omni query returned ``partialResultsDetected``.

    The rows are an unknown subset of the match, so returning them would under-count silently.
    Refused rather than returned.
    """


class AgentCoreTruncatedResult(AgentCoreError):
    """A CloudWatch Logs Insights query matched more records than it returned.

    ``recordsMatched`` exceeded the rows retrieved, so the trace set is incomplete. Refused rather
    than returned truncated.
    """


class AgentCoreIndexingTooLow(AgentCoreError):
    """X-Ray indexes below 100% while a read goes through X-Ray or the trace list.

    A read that depends on the index would see only the indexed fraction, so the result is an
    unknown subset. Refused, naming the observed percentage.
    """


class AgentCoreSettlingLag(AgentCoreError):
    """The requested window ends inside the read-after-write settling lag.

    A trace written moments ago is not yet fully readable, so the window would return a partial
    trace. Refused rather than returned mid-write.
    """


class AgentCoreMixedSession(AgentCoreError):
    """A single trace carries more than one ``session.id``.

    The AgentCore ``Evaluate`` API rejects such input, and fusing the sessions would mislabel the
    trace. Refused by default; an explicit re-key opts into collapsing them to the runtime session.
    """


class AgentCoreScoreIncomplete(AgentCoreError):
    """An evaluator result lacks a field a ``Label`` requires (trace/session id, or a value).

    Also raised when an ``Evaluate`` call returns no result at all for the trace it targeted.
    Refused rather than stored as a label with an empty or inferred field, or a run returning fewer
    verdicts than traces, which would enter an unfounded or incomplete set into the store.
    """


# The failure detector reads any exception a model call raises as a context overflow when its text
# holds one of a few substrings, and then retries in chunks instead of re-raising. The detector
# errors can reach it from inside the model stack, so none of their fixed messages may hold any of
# those substrings, in any letter case.


class DetectorError(AgentCoreError):
    """The failure detector or the draft namer refused an input, a configuration or a response."""


class SessionConversionError(DetectorError):
    """A trace cannot be converted into the session the failure detector reads."""


class DetectorSessionTooLarge(DetectorError):
    """A trace is too large for the detector's model to read in one call."""


class DetectorOutputError(DetectorError):
    """The detector's model returned a response that cannot be read faithfully."""


class DetectorCallError(DetectorError):
    """A model call made for the detector or the namer failed; the cause is chained."""


class DetectorConfigError(DetectorError):
    """The detector or the namer was built with a configuration it cannot run under."""


class DetectorCallLimitExceeded(DetectorError):
    """A live model call would exceed ``max_live_calls``; refused before it was sent.

    ``limit`` is the ``max_live_calls`` value that refused it.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            "a live model call was refused before it was sent, because it would exceed "
            f"max_live_calls={limit}; raise max_live_calls to allow more live calls, or replay "
            "the calls from a cassette"
        )


class VocabularyParseError(DetectorError):
    """The detector's category table has a row the library cannot read as a vocabulary entry.

    ``row`` is the number of the table row that failed and ``check`` names the check it failed.
    """

    def __init__(self, row: int, check: str) -> None:
        self.row = row
        self.check = check
        super().__init__(
            f"the failure detector's category table cannot be read: row {row} fails the check "
            f"'{check}'; the installed strands-agents-evals prompt template may have changed"
        )


class DraftNamingError(DetectorError):
    """The namer's response holds no usable vocabulary name for every cluster."""


class AgentCoreScoreOffScale(AgentCoreError):
    """An evaluator result's numeric value is not on the evaluator's published rating scale.

    The binary outcome is read from the value against the scale the evaluator declares; a value
    the scale does not define cannot be placed, so it is refused rather than mapped on a guess.
    """


class AgentCoreEvaluatorLevelUnsupported(AgentCoreError):
    """An evaluator's level is one the library cannot send it, so no judge is built or run from it.

    A TRACE-level evaluator scores one trace and a SESSION-level evaluator scores one conversation
    trace; a judge file's ``level`` must match its evaluator's. A TOOL_CALL evaluator needs a
    tool-call span Calibrated Agent Evaluation does not serialize, so it is refused, as is a level
    mismatch, before any billed call rather than sent a request the service refuses or mis-scores.
    """


class AgentCoreNotAConversation(AgentCoreError):
    """A session-level judge was given a trace that is not a simulated conversation.

    A SESSION evaluator scores a whole conversation; a single-turn trace carries no transcript, so
    it is refused before any billed call rather than scored as a one-turn session.
    """


class AgentCoreEvaluateLimitExceeded(AgentCoreError):
    """A trace would exceed an ``Evaluate`` input limit (span documents or assertions per call).

    Refused before any billed call, naming the trace and the limit, rather than sent a request the
    service rejects or silently shortened.
    """


class AgentCoreBlankAssertion(AgentCoreError):
    """A conversation's assertion is empty or only whitespace, so it asserts nothing.

    Sent as is, it would be scored like a real assertion and the verdict would read as a check
    that never took place. Refused before any billed call, naming the trace and the assertion.
    """


class AgentCoreAmbiguousSessionResult(AgentCoreError):
    """A session evaluator answered one conversation's call with more than one result.

    A session call scores one session, so one result is expected; storing several would count the
    conversation more than once, and keeping one would be a guess. The run is refused and no label
    from it is stored.
    """


class AgentCoreResultUnaddressable(AgentCoreError):
    """An evaluator result names a trace or span the request did not send it.

    A result is addressed by the ids its own call targeted: a trace that call did not target, or a
    span that is not a span of the trace the result names, has no Calibrated Agent Evaluation id to
    file the verdict under. The run is refused rather than storing a verdict under an unknown id or
    in another trace's cluster.
    """


class AgentCoreNoPassDirection(AgentCoreError):
    """An evaluator's score has no defined pass direction, so it cannot become a binary outcome.

    Whether a high score is good depends on the use (a refusal is right for a harmful request and
    wrong for a benign one), so no PASS/FAIL is guessed: the judge is refused when built and the
    evaluator's result records are refused when mapped.
    """


class AgentCoreRatingScaleMalformed(AgentCoreError):
    """An evaluator published a numerical rating scale that cannot be read as ordered rungs.

    A rung that is not an object, carries no numeric value, or repeats another rung's value would
    have to be dropped or guessed to build the scale, which silently moves the pass boundary. The
    judge is refused instead.
    """


class AgentCoreEvaluatorUnreadable(AgentCoreError):
    """``GetEvaluator`` returned a custom evaluator without the content its version is made from.

    An LLM-as-judge evaluator's version is a digest of its configuration, so one read without its
    instructions or rating scale (``includedData=METADATA_ONLY``) would keep its version when
    either changes. The judge is refused instead.
    """


class AgentCoreEvaluatorResultError(AgentCoreError):
    """An evaluator result carried a per-result error code on an otherwise successful response.

    AgentCore returns an HTTP 200 whose ``evaluationResults`` entry carries an ``errorCode`` (for
    example a missing span log event); that result is not a verdict, so the run is refused and no
    label from it is stored, rather than recording a verdict the evaluator did not produce or
    silently storing a smaller set than the run covered.
    """


class AgentCoreReferenceDropped(AgentCoreError):
    """The service reported that it ignored supplied reference (ground-truth) inputs.

    A non-empty ``ignoredReferenceInputFields`` means the evaluator scored without the ground truth
    the caller provided, so the verdict answers a different question than intended. The result
    still carries a value, so the run is refused, and no label from it is stored, rather than the
    verdict being stored as if the reference had been used.
    """


class AgentCoreRunDeclined(AgentCoreError):
    """The caller declined a billed run at its estimate or a write at its preview; none was sent."""


class AgentCoreContentUnserializable(AgentCoreError):
    """A trace cannot be serialized into an Evaluate payload, so it is refused before any call.

    Raised when the trace carries no message content (an outcome-only or ``probed`` trace has no
    input/output/tool content, so the evaluator would score an empty payload); when a span's input
    or output, or a tool call's arguments or result, holds a value that cannot be written as JSON;
    when a span time does not parse or has no UTC offset; when its span timing is partial (some
    spans timed, others not), since real and synthetic times are never mixed; or, in an untimed
    trace, when a span names a parent outside the trace or cannot be reached from a root span, so
    synthetic timing has nowhere to place it. Refused rather than issuing a billed call.
    """


class PolicySpanMalformed(AgentCoreError):
    """A Gateway span carrying a Policy decision could not be read whole.

    Raised for a per-call decision span whose tool, decision or mode is missing or unknown, whose
    list attribute is in an unknown encoding, and for a span carrying the decision attribute that
    is neither a per-call nor a tools/list authorization. Guessing would add or drop a decision.
    """


class AgentCoreArmsMissing(AgentCoreError):
    """An A/B arms read found no trace it can assign to the audited test.

    No span in the window carries both arm attributes, or no stamped trace names the audited
    experiment. Refused rather than stored as a read with every trace unassigned.
    """


class AgentCoreArmHalfStamped(AgentCoreError):
    """A span carries one of the two arm attributes without the other.

    A request whose baggage named one key stamps its spans by halves, so such a trace's arm cannot
    be read safely; the trace is refused.
    """


class AgentCoreMixedArm(AgentCoreError):
    """One trace's spans name two experiments or two variants."""


class AgentCoreArmSessionSplit(AgentCoreError):
    """One runtime session's traces sit in two variants, so the session's arm is ambiguous."""


class AgentCoreArmSessionAmbiguous(AgentCoreError):
    """A trace's ``/invocations`` server spans name two runtime sessions."""


class AgentCoreArmSourceIdRepeated(AgentCoreError):
    """Two or more traces arrived under one source id, so none of them can be assigned."""


class AgentCoreArmsReadChanged(AgentCoreError):
    """A re-read under a stored read id delivered other traces or assigned one differently.

    A stored read never changes; the re-read is refused and nothing from it is written.
    """


class AgentCoreAppliedBundleMismatch(AgentCoreError):
    """A trace lacks the bundle attribute or value the operator required, or carries another."""


class AgentCoreReplayProvenanceError(AgentCoreError):
    """A replayed trace can't be tied to its arm's bundle or compared across the replays."""


class AgentCoreDevelopmentInputMissing(AgentCoreError):
    """A development set names a trace with no stored input, or holds no trace at all."""


class AgentCoreDevelopmentSourceUnread(AgentCoreError):
    """A development read did not cover a log group the recommendation's source names."""


class AgentCoreDevelopmentSourceChanged(AgentCoreError):
    """An online evaluation's configuration changed after the development window began."""


class AgentCoreSliceReused(AgentCoreError):
    """A held-out slice already decided another change, so it can't decide this one."""


class AgentCoreUncertifiedJudge(AgentCoreError):
    """A judge's verdicts feed the re-test without the records ``judge`` needs."""


class AgentCoreNotReplayable(AgentCoreError):
    """A test or recommendation has no pair of bundle versions to replay."""


class AgentCoreResourceExists(AgentCoreError):
    """An evaluator or online config with the requested name already exists; none was created.

    ``kind`` names the resource type (``"evaluator"`` or ``"online config"``) and ``name`` the
    existing resource, so the caller can pick another name or delete the old one.
    """

    def __init__(self, kind: str, name: str) -> None:
        super().__init__(
            f"{kind} {name} already exists, so a second one with that name was not created; "
            f"choose another name or delete the existing {kind} first"
        )
        self.kind = kind
        self.name = name


class AgentCoreResourceNotActive(AgentCoreError):
    """A created evaluator or online config failed, or did not become active within its wait.

    ``status`` is the status AgentCore last reported and ``failure_reason`` its reason, when it
    gave one. The resource is not recorded as created.
    """

    def __init__(
        self, kind: str, name: str, status: str, failure_reason: str | None = None
    ) -> None:
        failed = status.endswith("FAILED") or status in {"ERROR", "DELETING"}
        if failed:
            detail = f"reported {status}: {failure_reason or 'no reason given'}"
        else:
            detail = f"was still {status} when the wait ran out"
            if failure_reason:
                detail += f" ({failure_reason})"
        super().__init__(
            f"{kind} {name} {detail}, so it is not recorded as created; look it up in the "
            "AgentCore console, delete it if it is not wanted, and create it again"
        )
        self.kind = kind
        self.name = name
        self.status = status
        self.failure_reason = failure_reason


class AgentCoreMetricNotPublished(AgentCoreError):
    """CloudWatch did not confirm a metric ``PutMetricData`` sent; it is not reported as published.

    ``error_code`` is the AWS error code alone (``ClientError.response["Error"]["Code"]``), never
    CloudWatch's message, which can repeat the value that was rejected.
    """

    def __init__(self, metric: str, error_code: str) -> None:
        super().__init__(
            f"CloudWatch did not confirm the metric {metric} ({error_code}), so it was not "
            "published; "
            "check the caller's cloudwatch:PutMetricData permission and the namespace, then "
            "publish again"
        )
        self.metric = metric
        self.error_code = error_code


class AgentCoreRuntimeNotFound(AgentCoreError):
    """The runtime named for an online config does not exist in the account and region."""

    def __init__(self, runtime: str) -> None:
        super().__init__(
            f"the AgentCore runtime {runtime} was not found in this account and region, so no "
            "online config was created; check the runtime ARN and the region"
        )
        self.runtime = runtime


class AgentCoreNotCertified(AgentCoreError):
    """A judge was put forward for online evaluation without the evidence it needs.

    ``missing`` lists what is missing, each as a noun phrase, for example ``"a certifying
    calibration record"``, ``"a CLEAR bias battery (the latest is BIASED)"`` or ``"the session
    log of its bias run"``.
    """

    def __init__(self, judge_id: str, missing: tuple[str, ...]) -> None:
        if not missing:
            raise ValueError("AgentCoreNotCertified needs at least one missing item to name")
        super().__init__(
            f"judge {judge_id} is not ready for online evaluation: it is missing "
            f"{'; '.join(missing)}. Nothing was sent to AgentCore"
        )
        self.judge_id = judge_id
        self.missing = missing
