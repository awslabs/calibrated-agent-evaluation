# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error hierarchy for the data plane.

Every data-plane failure is a typed, explained refusal; never swallowed and never a
silent fallback (conventions: "no silent failure"). All errors subclass
:class:`CalibratedAgentEvaluationDataError` so a caller can catch the whole family at the plane
boundary.
"""

from __future__ import annotations

from calibrated_agent_evaluation._errors import (
    CalibratedAgentEvaluationDataError as CalibratedAgentEvaluationDataError,
)
from calibrated_agent_evaluation._errors import MissingExtraError as MissingExtraError


class SchemaError(CalibratedAgentEvaluationDataError):
    """An artifact violates the normalized-schema contract.

    Raised on a missing/empty ``trace_id`` or a ``Label`` whose ``expected`` is
    set while ``critique`` is empty/missing; ``expected`` may never substitute
    for the human critique.
    """


class UnregisteredArtifactError(CalibratedAgentEvaluationDataError):
    """``dump``/``load`` hit an artifact type with no registered codec.

    Raised rather than silently skipping the artifact; a silent skip would make the
    no-lock-in round-trip lie.
    """


class SchemaVersionError(CalibratedAgentEvaluationDataError):
    """``load`` read an artifact whose ``schema_version`` ≠ the running ``SCHEMA_VERSION``.

    Raised rather than coercing the data; refuse-loudly is the honest v1 behavior;
    migrators are later work.
    """


class BackendError(CalibratedAgentEvaluationDataError):
    """A configured backend is unreachable or rejected a request.

    Raised without any silent fallback to the local backend; failure surfaces honestly,
    never silently degrades worlds.
    """


class TraceNotFoundError(BackendError):
    """The backend answered, and it holds no trace with the requested id.

    Distinct from its parent so a caller can tell "this id is free" from "the read failed": only
    the first may be acted on as an absence.
    """
