# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure modes a labeler proposes from the review form, for the codebook owner to settle.

The picker offers the codebook's modes. When a trace shows a failure none of them names, the
labeler proposes a new one (a name and a note): the FAIL label is saved with that name as its
failure mode, and a :class:`~calibrated_agent_evaluation.review.record.ProposedModeRecord` flags it
for the codebook owner, who adopts or merges it with
``calibrated-agent-evaluation cluster --ratify``. Until then the proposal is offered in every
session on the same codebook version, so the next labeler picks it instead of proposing it again.

The review app takes a :class:`ProposalStore` by injection; the CLI's stores the proposals as
``proposed-mode`` records beside the local store.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from calibrated_agent_evaluation.review.record import ProposedModeRecord

__all__ = ["ProposalStore"]


@runtime_checkable
class ProposalStore(Protocol):
    """Where proposed failure modes are kept, shared by every session on the store."""

    def pending(self, version: str) -> tuple[ProposedModeRecord, ...]:
        """The proposals under ``version`` not yet adopted or merged, oldest first."""
        ...

    def propose(self, proposal: ProposedModeRecord) -> ProposedModeRecord:
        """Store ``proposal``, or return the pending one with the same name if there is one.

        Names are compared with :func:`~calibrated_agent_evaluation.review.record.mode_key` (case
        and spacing ignored), so two sessions proposing one mode at once store it once.

        Raises:
            ReviewError: The proposal could not be stored.
        """
        ...
