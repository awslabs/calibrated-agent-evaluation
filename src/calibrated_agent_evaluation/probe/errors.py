# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the black-box probing toolkit.

Every refusal/failure that crosses a public surface is a typed, explained error in the project
family -- never a raw ``urllib``/``http.client``/``OSError`` escaping, and never a silent
empty-success on a dead endpoint (conventions: "no silent failure", "wrap external errors at the
boundary"). All errors subclass :class:`ProbeError`, which subclasses the
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` base so a
caller can catch the whole methodology-plane family at the plane boundary.

  * :class:`AuthorizationError`       -- a probe attempted without confirmed authorization.
  * :class:`EndpointUnreachableError` -- endpoint unreachable / circuit opened.
  * :class:`ProbeMissingExtraError`   -- a WebSocket (``ws``/``wss`` scheme) probe requested without
    the ``[probe-ws]`` extra installed; refuse (fail-closed) rather than silently fall back to HTTP.
  * :class:`PolicyCaseError`, :class:`PolicyDecisionsMissing`, :class:`PolicyModeMixed` -- a
    policy-decision battery refused its case file, its decisions or a case mixing gateway modes.
  * :class:`ProbeUncorrectableError` -- a prevalence-corrected point rate was requested for
    a black-box target (no calibration set exists -- only an upper bound is honest).
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class ProbeError(CalibratedAgentEvaluationDataError):
    """Base class for every probing-toolkit error."""


class AuthorizationError(ProbeError):
    """A probe was attempted before the user confirmed authorization.

    Refuse-by-default: no HTTP request is sent until authorization is confirmed and recorded.
    """


class EndpointUnreachableError(ProbeError):
    """The target endpoint is unreachable or returned a transport/HTTP error.

    Raised rather than recording an empty/degraded result as a success -- a silent empty-success on
    a dead endpoint is the worst kind of silent failure. Also raised when the consecutive-failure
    circuit breaker opens.
    """


class ProbeMissingExtraError(ProbeError):
    """WebSocket (``ws``/``wss``) probe requested without the ``[probe-ws]`` extra installed.

    Fail-closed: the harness refuses rather than silently falling back to HTTP or no-opping. The
    message names the extra and its install command so the caller can fix it in one step.
    """


class ProbeUncorrectableError(ProbeError):
    """A prevalence-corrected point rate was requested for a black-box target.

    A black-box target has no calibration set, so the true rate is not identifiable -- only a
    judge-biased upper bound is honest. The upper-bound interval is the signal.
    """


class PolicyCaseError(ProbeError):
    """A policy case file can't be read or is invalid, or its tool names match no decision.

    Raised before any AWS call for a file that is missing, unreadable, not JSON or invalid, and
    after the trials when decisions were read but none names a tool any case lists, which is the
    shape of a case file written in the wrong tool-name form.
    """


class PolicyDecisionsMissing(ProbeError):
    """A policy battery read no decision it could score, so no finding would mean anything.

    Raised when no trial of the battery got a per-call decision, or when a control case got none
    in any of its trials.
    """


class PolicyModeMixed(ProbeError):
    """One policy case's decisions came under two gateway modes, which can't be pooled."""
