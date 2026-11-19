# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The black-box probing toolkit: evaluate an agent you don't own, honestly.

Backend logic (no CLI, no UI) for probing an unowned HTTP agent: capability discovery,
repeated-trial robustness, mandatory limits-of-inference framing, judge-biased upper-bound rates,
refuse-by-default authorization, and the three-layer agent-security battery. ``probe_with`` runs
the same capability discovery over any :class:`ProbeAdapter` the caller builds, for a target no
URL names.

``probe/`` imports **no** vendor SDK: the transport is stdlib (`adapter.py`), and the sole
`inspect_evals` import lives in the sanctioned
`calibrated_agent_evaluation.execute.benchmark_tasks`. Probe results are normalized ``probed``
Traces with `ownership="probed"`, `spans == []`.
"""

from __future__ import annotations

from calibrated_agent_evaluation.probe.adapter import EndpointAdapter, ProbeAdapter, ProbeResponse
from calibrated_agent_evaluation.probe.authorize import confirm_authorization, require_authorized
from calibrated_agent_evaluation.probe.battery import run_battery
from calibrated_agent_evaluation.probe.benchmarks import (
    DEFAULT_BATTERY,
    TEXT_ONLY_BATTERY,
    BenchmarkSpec,
    Resolution,
    broke_from_score,
    installable_battery,
    layer_of,
    requires_tool_environment,
    resolve,
)
from calibrated_agent_evaluation.probe.capability import CapabilityMap, FailureHypothesis
from calibrated_agent_evaluation.probe.connection import (
    ApiKeyAuth,
    Auth,
    BearerAuth,
    ConnectionConfig,
    HeaderAuth,
    SubprotocolAuth,
)
from calibrated_agent_evaluation.probe.errors import (
    AuthorizationError,
    EndpointUnreachableError,
    PolicyCaseError,
    PolicyDecisionsMissing,
    PolicyModeMixed,
    ProbeError,
    ProbeMissingExtraError,
    ProbeUncorrectableError,
)
from calibrated_agent_evaluation.probe.fingerprint import endpoint_fingerprint
from calibrated_agent_evaluation.probe.framing import LimitsFraming
from calibrated_agent_evaluation.probe.harness import DEFAULT_SEEDS, ProbeReport, probe, probe_with
from calibrated_agent_evaluation.probe.policy import (
    POLICY_FINDING_KIND,
    AgentErrorTrial,
    PolicyCase,
    PolicyCaseFile,
    PolicyDecision,
    TrialDecisions,
    TrialRef,
    case_finding,
    classify,
    decisions_from_finding,
    load_policy_cases,
)
from calibrated_agent_evaluation.probe.policy_battery import (
    DecisionSource,
    PolicyBatteryResult,
    run_policy_battery,
)
from calibrated_agent_evaluation.probe.rate import UpperBoundRate, upper_bound_rate
from calibrated_agent_evaluation.probe.record import ProbeBatteryRecord, ProbeFinding
from calibrated_agent_evaluation.probe.redact import scrub
from calibrated_agent_evaluation.probe.robustness import (
    RobustnessResult,
    all_trials_held,
    capability_robustness,
    clopper_pearson,
    pass_hat_k,
    security_result,
)
from calibrated_agent_evaluation.probe.store import dump_probes, load_probes
from calibrated_agent_evaluation.probe.version import PROBE_SCHEMA_VERSION
from calibrated_agent_evaluation.probe.ws_adapter import WebSocketEndpointAdapter

__all__ = [
    "DEFAULT_BATTERY",
    "DEFAULT_SEEDS",
    "POLICY_FINDING_KIND",
    "PROBE_SCHEMA_VERSION",
    "TEXT_ONLY_BATTERY",
    "AgentErrorTrial",
    "ApiKeyAuth",
    "Auth",
    "AuthorizationError",
    "BearerAuth",
    "BenchmarkSpec",
    "CapabilityMap",
    "ConnectionConfig",
    "DecisionSource",
    "EndpointAdapter",
    "EndpointUnreachableError",
    "FailureHypothesis",
    "HeaderAuth",
    "LimitsFraming",
    "PolicyBatteryResult",
    "PolicyCase",
    "PolicyCaseError",
    "PolicyCaseFile",
    "PolicyDecision",
    "PolicyDecisionsMissing",
    "PolicyModeMixed",
    "ProbeAdapter",
    "ProbeBatteryRecord",
    "ProbeError",
    "ProbeFinding",
    "ProbeMissingExtraError",
    "ProbeReport",
    "ProbeResponse",
    "ProbeUncorrectableError",
    "Resolution",
    "RobustnessResult",
    "SubprotocolAuth",
    "TrialDecisions",
    "TrialRef",
    "UpperBoundRate",
    "WebSocketEndpointAdapter",
    "all_trials_held",
    "broke_from_score",
    "capability_robustness",
    "case_finding",
    "classify",
    "clopper_pearson",
    "confirm_authorization",
    "decisions_from_finding",
    "dump_probes",
    "endpoint_fingerprint",
    "installable_battery",
    "layer_of",
    "load_policy_cases",
    "load_probes",
    "pass_hat_k",
    "probe",
    "probe_with",
    "require_authorized",
    "requires_tool_environment",
    "resolve",
    "run_battery",
    "run_policy_battery",
    "scrub",
    "security_result",
    "upper_bound_rate",
]
