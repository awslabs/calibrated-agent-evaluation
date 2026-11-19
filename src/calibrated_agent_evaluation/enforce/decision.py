# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The auditable enforcement decision.

Every path in the layer -- a hard refusal, a warning+nudge, a display-default -- emits an
:class:`EnforcementDecision` so a human can later reconstruct *why* a number was refused, warned,
or shaped, from stored artifacts alone. A hard refusal both raises a typed error (the
control flow) and records a decision (the audit trail).

``inputs`` is validated non-empty at construction: a decision with no recorded inputs is
refused, not silently accepted -- an audit record with nothing to audit is a silent-failure in
disguise.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.enforce.errors import EnforcementError
from calibrated_agent_evaluation.enforce.version import ENFORCE_SCHEMA_VERSION

# Every decision is exactly one of these three tiers.
Tier = Literal["hard_refusal", "warning", "display_default"]
_TIERS: frozenset[str] = frozenset({"hard_refusal", "warning", "display_default"})


class EnforcementDecision(CalibratedAgentEvaluationModel):
    """One recorded enforcement decision.

    ``rule`` is the governing id (e.g. ``""``, ``""``); ``tier`` is exactly one of
    :data:`Tier`; ``inputs`` are the evaluated inputs (non-empty); ``outcome`` is
    ``"refused"``/``"warned"``/``"shaped"``; ``message`` is the human-facing explanation; ``nudge``
    is the structural steer where applicable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = EnforcementError

    rule: str
    tier: Tier
    inputs: dict[str, str]
    outcome: str
    message: str
    nudge: str | None = None
    schema_version: str = ENFORCE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        # Reject an out-of-set tier loudly rather than persisting an un-auditable value.
        if self.tier not in _TIERS:
            raise EnforcementError(
                f"EnforcementDecision.tier {self.tier!r} is not one of {sorted(_TIERS)}"
            )
        # An audit record with no recorded inputs cannot answer "why" -- refuse it.
        if not self.inputs:
            raise EnforcementError(
                f"EnforcementDecision for rule {self.rule!r} has empty inputs; a decision must "
                "record the inputs that produced it"
            )
