# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error for the trajectory normalizer.

``NormalizationError`` is a member of the ``CalibratedAgentEvaluationDataError`` family, so a
caller can catch the whole data-plane family at one boundary. Every failure the normalizer
surfaces; an unrecognized/unsupported source, a missing opt-in extra, a
vendor SDK / ``json`` / filesystem error while reading a source, or a mapped
``Trace`` that fails ``validate()``; is raised as this typed, explained
error, chained with ``raise ... from exc`` so the original cause is never lost (conventions:
"wrap external errors at the boundary"; "no silent failure").
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class NormalizationError(CalibratedAgentEvaluationDataError):
    """The normalizer could not produce a valid ``Trace`` from a source.

    Raised (never swallowed and never partially returned) on an unrecognized/unsupported
    source, framework ``kind``, or OTel semantic convention; a missing opt-in extra
    (the message names the install command for it); a vendor/parser/
    filesystem read failure (chained via ``from``); or a mapped ``Trace`` that fails
    the schema ``validate()``.
    """
