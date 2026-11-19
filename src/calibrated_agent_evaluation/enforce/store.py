# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trips for EnforcementDecision and JudgeDefinition.

Each kind is written to its own ``<kind>.{csv,json}`` pair under a directory and read back field
for field. A stored ``schema_version`` other than the running ``ENFORCE_SCHEMA_VERSION``, and
every read or parse failure, raises :class:`EnforcementStoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.enforce.decision import EnforcementDecision
from calibrated_agent_evaluation.enforce.errors import EnforcementStoreError
from calibrated_agent_evaluation.enforce.judgedef import JudgeDefinition
from calibrated_agent_evaluation.enforce.version import ENFORCE_SCHEMA_VERSION

__all__ = ["dump_decisions", "dump_definitions", "load_decisions", "load_definitions"]

_DECISIONS = RecordStore(
    EnforcementDecision,
    stem="enforcement_decision",
    schema_version=ENFORCE_SCHEMA_VERSION,
    error=EnforcementStoreError,
    noun="enforcement decision",
)
_DEFINITIONS = RecordStore(
    JudgeDefinition,
    stem="judge_definition",
    schema_version=ENFORCE_SCHEMA_VERSION,
    error=EnforcementStoreError,
    noun="judge definition",
)


def dump_decisions(
    decisions: Sequence[EnforcementDecision], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``decisions`` to ``enforcement_decision.{csv,json}`` under ``path``.

    Args:
        decisions: The decisions to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_decisions` reads back (both are always written).

    Raises:
        EnforcementStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _DECISIONS.dump(decisions, path, fmt)


def load_decisions(path: str | Path, fmt: str = "csv") -> list[EnforcementDecision]:
    """Read enforcement decisions back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_decisions` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored decisions.

    Raises:
        EnforcementStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _DECISIONS.load(path, fmt)


def dump_definitions(
    definitions: Sequence[JudgeDefinition], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``definitions`` to ``judge_definition.{csv,json}`` under ``path``.

    Args:
        definitions: The judge definitions to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_definitions` reads back (both are always written).

    Raises:
        EnforcementStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _DEFINITIONS.dump(definitions, path, fmt)


def load_definitions(path: str | Path, fmt: str = "csv") -> list[JudgeDefinition]:
    """Read judge definitions back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_definitions` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored definitions.

    Raises:
        EnforcementStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _DEFINITIONS.load(path, fmt)
