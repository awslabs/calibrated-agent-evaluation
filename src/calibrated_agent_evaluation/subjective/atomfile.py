# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an :class:`~calibrated_agent_evaluation.subjective.AtomSet` from a JSON file.

The file declares one criterion's split into objective atoms (gated) and subjective dimensions
(tracked)::

    {
      "criterion": "good claim decision letter",
      "objective": [{"name": "cites_exclusion", "description": "names the exclusion applied"}],
      "subjective": [{"name": "tone", "description": "reads as courteous and plain"}]
    }

``description`` is optional. Unknown keys are refused, and the split is checked by
:func:`~calibrated_agent_evaluation.subjective.define_atoms` (no duplicate names, no name in both
lanes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.subjective.classify import define_atoms
from calibrated_agent_evaluation.subjective.errors import SubjectiveError
from calibrated_agent_evaluation.subjective.types import AtomSet

__all__ = ["atom_set_from_dict", "load_atom_set"]


class _Entry(CalibratedAgentEvaluationModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SubjectiveError

    name: str
    description: str = ""


class _AtomFile(CalibratedAgentEvaluationModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SubjectiveError

    criterion: str
    objective: tuple[_Entry, ...] = ()
    subjective: tuple[_Entry, ...] = ()


def atom_set_from_dict(doc: object) -> AtomSet:
    """Build an :class:`AtomSet` from a decoded atom-set file.

    Raises:
        SubjectiveError: The document is not an object of the shape above, or the split is
            invalid.
    """
    parsed = _AtomFile.model_validate(doc)
    if not parsed.criterion.strip():
        raise SubjectiveError("an atom set needs a non-empty criterion")
    return define_atoms(
        parsed.criterion,
        [(e.name, e.description) for e in parsed.objective],
        [(e.name, e.description) for e in parsed.subjective],
    )


def load_atom_set(path: str | Path) -> AtomSet:
    """Read and check the atom-set file at ``path``.

    Raises:
        SubjectiveError: The file cannot be read, is not JSON, or does not declare a valid split.
    """
    file = Path(path)
    try:
        doc = json.loads(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SubjectiveError(f"cannot read the atom set {file}: {exc.strerror or exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubjectiveError(f"the atom set {file} is not valid JSON: {exc}") from exc
    try:
        return atom_set_from_dict(doc)
    except SubjectiveError as exc:
        raise SubjectiveError(f"the atom set {file} is not valid: {exc}") from exc
