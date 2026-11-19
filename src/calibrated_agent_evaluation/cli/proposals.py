# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review app's proposal store: ``proposed-mode`` records under ``<root>/records/``.

Every session on the same local root reads the same records, so a mode proposed in one session
is offered in the next render of every other. Proposing holds the kind's lock across the
read-check-append, so two sessions proposing one name at once store it once.
"""

from __future__ import annotations

from pathlib import Path

from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.record import (
    ProposedModeRecord,
    mode_key,
    pending_proposals,
)

__all__ = ["KIND", "RecordsProposalStore"]

KIND = "proposed-mode"


class RecordsProposalStore:
    """Proposed failure modes stored as records under one local root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def pending(self, version: str) -> tuple[ProposedModeRecord, ...]:
        """The proposals under ``version`` no codebook record has adopted or merged."""
        from calibrated_agent_evaluation.cli.records import load_records_of
        from calibrated_agent_evaluation.cli.stages.codebook import stored_codebooks

        try:
            return pending_proposals(
                load_records_of(self._root, KIND), stored_codebooks(self._root), version
            )
        except CliError as exc:
            raise ReviewError(f"the proposed failure modes could not be read: {exc}") from exc

    def propose(self, proposal: ProposedModeRecord) -> ProposedModeRecord:
        """Store ``proposal`` unless a pending one has the same name; return the stored one."""
        from calibrated_agent_evaluation.cli.records import append_records, records_lock

        try:
            with records_lock(self._root, KIND):
                for existing in self.pending(proposal.codebook_version):
                    if mode_key(existing.name) == mode_key(proposal.name):
                        return existing
                append_records(self._root, KIND, [proposal], locked=True)
        except CliError as exc:
            raise ReviewError(f"the proposed failure mode was not stored: {exc}") from exc
        return proposal
