# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amazon Bedrock AgentCore plug-in: opt-in adapters behind the library's existing seams.

Nothing here is a default, and importing this package loads no AWS SDK. Adapter names load on
first access; without the ``[agentcore]`` extra that access raises :class:`AgentCoreExtraMissing`
naming the command that installs it.

The plug-in has six parts:

- the read-only account check behind ``calibrated-agent-evaluation doctor --agentcore``
  (:func:`run_checks`);
- the window ingest behind ``calibrated-agent-evaluation ingest --source agentcore``, which reads a
  runtime's traces for a time window from AgentCore Observability (:class:`AgentCoreTraceSource`,
  :func:`ingest_agentcore`);
- the Runtime executor behind ``calibrated-agent-evaluation generate --run --executor agentcore``,
  which sends each generated input to an agent deployed to AgentCore Runtime and stores one trace
  per input (:class:`AgentCoreRuntimeExecutor`), reading each run's span tree with the per-run span
  reader (:class:`AgentCoreSpanReader`);
- dataset publish and pull behind ``calibrated-agent-evaluation dataset publish`` and ``pull``,
  which publish the test scenarios built from the failures reviewers caught as a version of an
  AgentCore evaluation dataset, and read a version back (:func:`publish`, :func:`pull`);
- the Code Interpreter sandbox behind
  ``calibrated-agent-evaluation score --scorer code --sandbox agentcore``, which runs an agent's
  generated code in a managed AgentCore Code Interpreter session (:class:`CodeInterpreterSandbox`);
- the ``strands-agents-evals`` failure detector: :func:`failure_vocabulary` reads the detector's
  failure categories as typed entries, :class:`FailureDetector` runs it on a normalized trace and
  returns a :class:`Detection` or a typed refusal, and :class:`DetectorDraftNamer` names draft
  failure-mode clusters from that vocabulary for
  ``calibrated_agent_evaluation.cluster.build_draft``. Both call only the strands ``Model`` the
  caller passes, refuse a live call past ``max_live_calls`` (0 by default), and can record and
  replay their calls from a cassette. The result types and the detector errors are vendor-free, so
  they import without the extra.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreAppliedBundleMismatch,
    AgentCoreArmHalfStamped,
    AgentCoreArmSessionAmbiguous,
    AgentCoreArmSessionSplit,
    AgentCoreArmsMissing,
    AgentCoreArmSourceIdRepeated,
    AgentCoreArmsReadChanged,
    AgentCoreContentMissing,
    AgentCoreError,
    AgentCoreExtraMissing,
    AgentCoreIndexingTooLow,
    AgentCoreMixedArm,
    AgentCoreMixedSession,
    AgentCorePartialResult,
    AgentCoreScoreIncomplete,
    AgentCoreSettlingLag,
    AgentCoreTruncatedResult,
    AwsCredentialsMissing,
    BilledCallBudgetExceeded,
    DatasetDecodeError,
    DatasetInputError,
    DatasetNameConflict,
    DatasetPublishError,
    DatasetSchemaUnsupported,
    DatasetVersionNotFound,
    DetectorCallError,
    DetectorCallLimitExceeded,
    DetectorConfigError,
    DetectorError,
    DetectorOutputError,
    DetectorSessionTooLarge,
    DraftNamingError,
    PolicySpanMalformed,
    ReadOnlyViolation,
    SessionConversionError,
    VocabularyParseError,
    WrongAwsIdentity,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.adot import adot_to_otlp, xray_trace_id_to_otel
    from calibrated_agent_evaluation.agentcore.arms import (
        ARM_ARN_ATTRIBUTE,
        ARM_VARIANT_ATTRIBUTE,
        ArmRead,
        KeptTrace,
        RawTraceLog,
        merge_arms_read,
        read_arms,
        runtime_session_id,
        span_stamp,
    )
    from calibrated_agent_evaluation.agentcore.audit_types import (
        AbTestClaimRecord,
        ArmAssignment,
        BundleVersionRef,
    )
    from calibrated_agent_evaluation.agentcore.code_interpreter import CodeInterpreterSandbox
    from calibrated_agent_evaluation.agentcore.datasets import (
        DatasetPlan,
        dataset_guard,
        plan_publish,
        publish,
        pull,
    )
    from calibrated_agent_evaluation.agentcore.datasets_codec import (
        content_digest,
        decode_example,
        encode_example,
    )
    from calibrated_agent_evaluation.agentcore.detectors.detect import FailureDetector
    from calibrated_agent_evaluation.agentcore.detectors.models import detector_bedrock_model
    from calibrated_agent_evaluation.agentcore.detectors.namer import DetectorDraftNamer
    from calibrated_agent_evaluation.agentcore.detectors.session import trace_to_session
    from calibrated_agent_evaluation.agentcore.detectors.types import (
        DetectedFailure,
        Detection,
        DetectorProvenance,
        FailureVocabulary,
        NamingResult,
        VocabularyEntry,
    )
    from calibrated_agent_evaluation.agentcore.detectors.vocabulary import failure_vocabulary
    from calibrated_agent_evaluation.agentcore.doctor import (
        DEFAULT_CHECKS,
        DoctorTarget,
        read_only_guard,
        run_checks,
    )
    from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget, ReadOnlyGuard
    from calibrated_agent_evaluation.agentcore.ingest import ingest_agentcore
    from calibrated_agent_evaluation.agentcore.optimization import (
        AUDIT_READ_OPERATIONS,
        INGEST_READ_OPERATIONS,
        audit_clients,
        read_ab_test,
        read_bundle_version,
    )
    from calibrated_agent_evaluation.agentcore.policy_spans import (
        AGENTCORE_POLICY_VOCABULARY,
        AgentCorePolicyDecisionSource,
        agent_reported_error_trial,
        extract_decisions,
    )
    from calibrated_agent_evaluation.agentcore.readers import OmniReader, agentcore_source
    from calibrated_agent_evaluation.agentcore.run_manifest import RunRecord
    from calibrated_agent_evaluation.agentcore.run_spans import RunSpanQuery, RunSpanReader
    from calibrated_agent_evaluation.agentcore.runtime import RuntimeTarget
    from calibrated_agent_evaluation.agentcore.runtime_errors import (
        AgentReportedError,
        RuntimeInvocationError,
    )
    from calibrated_agent_evaluation.agentcore.runtime_executor import (
        AgentCoreRuntimeExecutor,
        AgentCoreRuntimeTask,
        AgentCoreTaskBuilder,
    )
    from calibrated_agent_evaluation.agentcore.runtime_probe import RuntimeProbeAdapter
    from calibrated_agent_evaluation.agentcore.scores import read_scores
    from calibrated_agent_evaluation.agentcore.session import (
        AwsClients,
        caller_is_role,
        require_role,
    )
    from calibrated_agent_evaluation.agentcore.settle import SettlePolicy
    from calibrated_agent_evaluation.agentcore.sources import (
        AgentCoreSourceConfig,
        AgentCoreTraceSource,
        TimeWindow,
    )
    from calibrated_agent_evaluation.agentcore.span_reader import AgentCoreSpanReader

__all__ = [
    "AGENTCORE_POLICY_VOCABULARY",
    "ARM_ARN_ATTRIBUTE",
    "ARM_VARIANT_ATTRIBUTE",
    "AUDIT_READ_OPERATIONS",
    "DEFAULT_CHECKS",
    "INGEST_READ_OPERATIONS",
    "AbTestClaimRecord",
    "AgentCoreAppliedBundleMismatch",
    "AgentCoreArmHalfStamped",
    "AgentCoreArmSessionAmbiguous",
    "AgentCoreArmSessionSplit",
    "AgentCoreArmSourceIdRepeated",
    "AgentCoreArmsMissing",
    "AgentCoreArmsReadChanged",
    "AgentCoreContentMissing",
    "AgentCoreError",
    "AgentCoreExtraMissing",
    "AgentCoreIndexingTooLow",
    "AgentCoreMixedArm",
    "AgentCoreMixedSession",
    "AgentCorePartialResult",
    "AgentCorePolicyDecisionSource",
    "AgentCoreRuntimeExecutor",
    "AgentCoreRuntimeTask",
    "AgentCoreScoreIncomplete",
    "AgentCoreSettlingLag",
    "AgentCoreSourceConfig",
    "AgentCoreSpanReader",
    "AgentCoreTaskBuilder",
    "AgentCoreTraceSource",
    "AgentCoreTruncatedResult",
    "AgentReportedError",
    "ArmAssignment",
    "ArmRead",
    "AwsClients",
    "AwsCredentialsMissing",
    "BilledCallBudget",
    "BilledCallBudgetExceeded",
    "BundleVersionRef",
    "CodeInterpreterSandbox",
    "DatasetDecodeError",
    "DatasetInputError",
    "DatasetNameConflict",
    "DatasetPlan",
    "DatasetPublishError",
    "DatasetSchemaUnsupported",
    "DatasetVersionNotFound",
    "DetectedFailure",
    "Detection",
    "DetectorCallError",
    "DetectorCallLimitExceeded",
    "DetectorConfigError",
    "DetectorDraftNamer",
    "DetectorError",
    "DetectorOutputError",
    "DetectorProvenance",
    "DetectorSessionTooLarge",
    "DoctorTarget",
    "DraftNamingError",
    "FailureDetector",
    "FailureVocabulary",
    "KeptTrace",
    "NamingResult",
    "OmniReader",
    "PolicySpanMalformed",
    "RawTraceLog",
    "ReadOnlyGuard",
    "ReadOnlyViolation",
    "RunRecord",
    "RunSpanQuery",
    "RunSpanReader",
    "RuntimeInvocationError",
    "RuntimeProbeAdapter",
    "RuntimeTarget",
    "SessionConversionError",
    "SettlePolicy",
    "TimeWindow",
    "VocabularyEntry",
    "VocabularyParseError",
    "WrongAwsIdentity",
    "adot_to_otlp",
    "agent_reported_error_trial",
    "agentcore_source",
    "audit_clients",
    "caller_is_role",
    "content_digest",
    "dataset_guard",
    "decode_example",
    "detector_bedrock_model",
    "encode_example",
    "extract_decisions",
    "failure_vocabulary",
    "ingest_agentcore",
    "merge_arms_read",
    "plan_publish",
    "publish",
    "pull",
    "read_ab_test",
    "read_arms",
    "read_bundle_version",
    "read_only_guard",
    "read_scores",
    "require_role",
    "run_checks",
    "runtime_session_id",
    "span_stamp",
    "trace_to_session",
    "xray_trace_id_to_otel",
]

_LAZY: dict[str, str] = {
    "AGENTCORE_POLICY_VOCABULARY": "calibrated_agent_evaluation.agentcore.policy_spans",
    "AgentCorePolicyDecisionSource": "calibrated_agent_evaluation.agentcore.policy_spans",
    "agent_reported_error_trial": "calibrated_agent_evaluation.agentcore.policy_spans",
    "extract_decisions": "calibrated_agent_evaluation.agentcore.policy_spans",
    "AbTestClaimRecord": "calibrated_agent_evaluation.agentcore.audit_types",
    "ArmAssignment": "calibrated_agent_evaluation.agentcore.audit_types",
    "BundleVersionRef": "calibrated_agent_evaluation.agentcore.audit_types",
    "ARM_ARN_ATTRIBUTE": "calibrated_agent_evaluation.agentcore.arms",
    "ARM_VARIANT_ATTRIBUTE": "calibrated_agent_evaluation.agentcore.arms",
    "ArmRead": "calibrated_agent_evaluation.agentcore.arms",
    "KeptTrace": "calibrated_agent_evaluation.agentcore.arms",
    "RawTraceLog": "calibrated_agent_evaluation.agentcore.arms",
    "merge_arms_read": "calibrated_agent_evaluation.agentcore.arms",
    "read_arms": "calibrated_agent_evaluation.agentcore.arms",
    "runtime_session_id": "calibrated_agent_evaluation.agentcore.arms",
    "span_stamp": "calibrated_agent_evaluation.agentcore.arms",
    "AUDIT_READ_OPERATIONS": "calibrated_agent_evaluation.agentcore.optimization",
    "INGEST_READ_OPERATIONS": "calibrated_agent_evaluation.agentcore.optimization",
    "audit_clients": "calibrated_agent_evaluation.agentcore.optimization",
    "read_ab_test": "calibrated_agent_evaluation.agentcore.optimization",
    "read_bundle_version": "calibrated_agent_evaluation.agentcore.optimization",
    "AwsClients": "calibrated_agent_evaluation.agentcore.session",
    "caller_is_role": "calibrated_agent_evaluation.agentcore.session",
    "require_role": "calibrated_agent_evaluation.agentcore.session",
    "BilledCallBudget": "calibrated_agent_evaluation.agentcore.guards",
    "CodeInterpreterSandbox": "calibrated_agent_evaluation.agentcore.code_interpreter",
    "DEFAULT_CHECKS": "calibrated_agent_evaluation.agentcore.doctor",
    "DoctorTarget": "calibrated_agent_evaluation.agentcore.doctor",
    "read_only_guard": "calibrated_agent_evaluation.agentcore.doctor",
    "run_checks": "calibrated_agent_evaluation.agentcore.doctor",
    "ReadOnlyGuard": "calibrated_agent_evaluation.agentcore.guards",
    "content_digest": "calibrated_agent_evaluation.agentcore.datasets_codec",
    "decode_example": "calibrated_agent_evaluation.agentcore.datasets_codec",
    "encode_example": "calibrated_agent_evaluation.agentcore.datasets_codec",
    "DatasetPlan": "calibrated_agent_evaluation.agentcore.datasets",
    "dataset_guard": "calibrated_agent_evaluation.agentcore.datasets",
    "plan_publish": "calibrated_agent_evaluation.agentcore.datasets",
    "publish": "calibrated_agent_evaluation.agentcore.datasets",
    "pull": "calibrated_agent_evaluation.agentcore.datasets",
    "AgentCoreTraceSource": "calibrated_agent_evaluation.agentcore.sources",
    "AgentCoreSourceConfig": "calibrated_agent_evaluation.agentcore.sources",
    "TimeWindow": "calibrated_agent_evaluation.agentcore.sources",
    "adot_to_otlp": "calibrated_agent_evaluation.agentcore.adot",
    "xray_trace_id_to_otel": "calibrated_agent_evaluation.agentcore.adot",
    "read_scores": "calibrated_agent_evaluation.agentcore.scores",
    "SettlePolicy": "calibrated_agent_evaluation.agentcore.settle",
    "AgentCoreSpanReader": "calibrated_agent_evaluation.agentcore.span_reader",
    "ingest_agentcore": "calibrated_agent_evaluation.agentcore.ingest",
    "OmniReader": "calibrated_agent_evaluation.agentcore.readers",
    "agentcore_source": "calibrated_agent_evaluation.agentcore.readers",
    # Its own module: it is also an execution-plane error, and importing
    # calibrated_agent_evaluation.execute eagerly would make a bare import of this package load the
    # execution plane.
    "RuntimeInvocationError": "calibrated_agent_evaluation.agentcore.runtime_errors",
    "AgentReportedError": "calibrated_agent_evaluation.agentcore.runtime_errors",
    "RuntimeTarget": "calibrated_agent_evaluation.agentcore.runtime",
    "RunSpanQuery": "calibrated_agent_evaluation.agentcore.run_spans",
    "RunSpanReader": "calibrated_agent_evaluation.agentcore.run_spans",
    "RunRecord": "calibrated_agent_evaluation.agentcore.run_manifest",
    "AgentCoreRuntimeExecutor": "calibrated_agent_evaluation.agentcore.runtime_executor",
    "AgentCoreRuntimeTask": "calibrated_agent_evaluation.agentcore.runtime_executor",
    "AgentCoreTaskBuilder": "calibrated_agent_evaluation.agentcore.runtime_executor",
    "VocabularyEntry": "calibrated_agent_evaluation.agentcore.detectors.types",
    "FailureVocabulary": "calibrated_agent_evaluation.agentcore.detectors.types",
    "DetectorProvenance": "calibrated_agent_evaluation.agentcore.detectors.types",
    "DetectedFailure": "calibrated_agent_evaluation.agentcore.detectors.types",
    "Detection": "calibrated_agent_evaluation.agentcore.detectors.types",
    "NamingResult": "calibrated_agent_evaluation.agentcore.detectors.types",
    "detector_bedrock_model": "calibrated_agent_evaluation.agentcore.detectors.models",
    "failure_vocabulary": "calibrated_agent_evaluation.agentcore.detectors.vocabulary",
    "trace_to_session": "calibrated_agent_evaluation.agentcore.detectors.session",
    "FailureDetector": "calibrated_agent_evaluation.agentcore.detectors.detect",
    "DetectorDraftNamer": "calibrated_agent_evaluation.agentcore.detectors.namer",
    "RuntimeProbeAdapter": "calibrated_agent_evaluation.agentcore.runtime_probe",
}

_VENDOR_ROOTS = frozenset({"boto3", "botocore", "bedrock_agentcore", "strands", "strands_evals"})


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only a missing vendor SDK means the extra is absent; any other missing module is a bug.
        if (exc.name or "").partition(".")[0] not in _VENDOR_ROOTS:
            raise
        raise AgentCoreExtraMissing(f"calibrated_agent_evaluation.agentcore.{name}") from exc
    value = getattr(module, name)
    globals()[name] = value
    return value
