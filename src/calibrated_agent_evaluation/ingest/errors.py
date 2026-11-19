# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error for the ingest surface.

``IngestError`` subclasses
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError`, so the CLI's
``DOMAIN_ERRORS`` boundary already catches it (one diagnostic line, exit 1) with no change to
``cli/errors.py``. It is the loud, actionable refusal behind the no-silent-loss guarantee: a source
that would produce an empty or lossy Trace, a duplicate-span-id trace, or a broken capture is
refused with a message that names the fix; never silently dropped.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class IngestError(CalibratedAgentEvaluationDataError):
    """An ingest source would lose data, or cannot be ingested faithfully.

    Raised (never swallowed) when a normalized Trace would drop tool calls the source carried,
    when a source has duplicate span ids, or when a captured agent raises
    or emits no spans. The message names the offending source and the fix.
    """


class IngestAbortedError(IngestError):
    """The store failed part-way through a batch, so the rest of the batch was not ingested.

    The traces written before the failure stay written. ``written`` names them, so the caller can
    tell what is in the store without re-reading it; the message names the source being ingested
    when the store failed.
    """

    def __init__(self, message: str, *, written: tuple[str, ...]) -> None:
        super().__init__(message)
        self.written = written
