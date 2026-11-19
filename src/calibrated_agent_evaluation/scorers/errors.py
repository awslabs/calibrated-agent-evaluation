# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the scorers module.

Every failure that crosses the ``scorers/`` boundary surfaces as a ``ScorerError`` or
``ScorerLaneError``; never a raw ``autoevals`` exception, ``TypeError``, or ``ValueError``.
"""

from __future__ import annotations


class ScorerError(Exception):
    """A typed error from the scorers module.

    Wraps vendor/input-validation exceptions via chaining (``raise ScorerError(...) from exc``).
    """


class ScorerLaneError(Exception):
    """Raised when an EXPLORATION scorer is used in a CI-gate context.

    The two-lane rule: generic generation-quality metrics are banned from CI; only structured-target
    scorers with ``lane=CI_GATE`` may gate deployments.
    """
