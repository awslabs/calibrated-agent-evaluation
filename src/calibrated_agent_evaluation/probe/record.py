# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auditable probe-finding envelope, and the stored record a battery run writes.

Every probing finding is wrapped in a :class:`ProbeFinding` carrying the inputs/context that
produced it -- a capture timestamp + endpoint fingerprint, the mandatory :class:`LimitsFraming`,
and any benchmark substitutions applied -- so a human can later reconstruct what was probed and
under what caveats. ``endpoint_fingerprint`` is required and non-empty by construction (an
unfingerprinted finding is not auditable).

:class:`ProbeBatteryRecord` is what ``calibrated-agent-evaluation probe --battery`` persists: the
findings of one run plus the provenance of the run that produced them, so the bounds in it can be
read back and reproduced.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

from pydantic import ConfigDict, Field, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation.probe.errors import ProbeError
from calibrated_agent_evaluation.probe.framing import LimitsFraming
from calibrated_agent_evaluation.probe.version import PROBE_SCHEMA_VERSION

__all__ = ["ProbeBatteryRecord", "ProbeFinding"]


class ProbeFinding(CalibratedAgentEvaluationModel):
    """One auditable probing finding.

    ``captured_at`` (ISO-8601) + ``endpoint_fingerprint`` stamp the non-stationarity context;
    ``framing`` is the mandatory limits-of-inference; ``substitutions`` records any benchmark
    substitutions surfaced; ``payload`` holds the finding-specific data.

    A ``capability_map`` finding's ``payload["detected_format"]`` is the request shape every
    prompt was sent in: the one the HTTP probe detected, or, from ``probe_with``, the one the
    caller named, since that probe detects nothing. It is empty when no shape applies, as for a
    WebSocket probe.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ProbeError

    kind: str
    captured_at: str
    endpoint_fingerprint: str
    framing: LimitsFraming
    payload: dict[str, Any] = Field(default_factory=dict)
    substitutions: tuple[str, ...] = ()
    schema_version: str = PROBE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_auditable(self) -> Self:
        # A finding with no endpoint fingerprint cannot be attributed to a probe run and is
        # not auditable -- refuse it rather than persisting an un-attributable record.
        if not self.endpoint_fingerprint:
            raise ProbeError(
                f"ProbeFinding({self.kind!r}) has an empty endpoint_fingerprint; every Stage-0 "
                "output must be stamped with an endpoint fingerprint"
            )
        if not self.captured_at:
            raise ProbeError(f"ProbeFinding({self.kind!r}) has an empty captured_at timestamp")
        return self


class ProbeBatteryRecord(CalibratedAgentEvaluationModel):
    """The findings of one benchmark-battery run against one endpoint.

    ``findings`` holds one :class:`ProbeFinding` per benchmark that ran, each carrying the trial
    count, the break count and the one-sided 95% upper bound on the break rate. There is no point
    estimate: a black-box target has no calibration set, so the bound is the honest signal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ProbeError

    probe_id: str
    endpoint_fingerprint: str
    findings: tuple[ProbeFinding, ...]
    schema_version: str = PROBE_SCHEMA_VERSION
    provenance: ProvenanceStamp
