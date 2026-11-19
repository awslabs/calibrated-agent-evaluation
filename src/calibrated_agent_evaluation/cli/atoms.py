# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an AtomSet JSON file: which dimensions of a criterion are gated and which are tracked.

The file mirrors :class:`calibrated_agent_evaluation.subjective.AtomSet`::

    {
      "criterion": "refund handling",
      "objective_atoms": [{"name": "over_refund", "description": "refunded above policy"}],
      "subjective_dimensions": [{"name": "tone", "description": "warmth of the reply"}]
    }

``description`` is optional. The lanes are checked by
:func:`calibrated_agent_evaluation.subjective.define_atoms`, so an empty set, a blank or duplicate
name, or a name in both lanes is refused with its
:class:`~calibrated_agent_evaluation.subjective.SubjectiveError`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.subjective import AtomSet, define_atoms


class _Dimension(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""


class _AtomSetFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    criterion: str
    objective_atoms: tuple[_Dimension, ...] = ()
    subjective_dimensions: tuple[_Dimension, ...] = ()


def load_atom_set(path: str | Path) -> AtomSet:
    """The :class:`AtomSet` in the JSON file at ``path``.

    Raises :class:`CliError` for a file that cannot be read, is not JSON or does not have the
    shape above, and :class:`~calibrated_agent_evaluation.subjective.SubjectiveError` for lanes
    ``define_atoms`` refuses.
    """
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read the atoms file {file}: {exc.strerror or exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CliError(f"the atoms file {file} is not valid JSON: {exc}") from exc
    try:
        doc = _AtomSetFile.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'file'}: {err['msg']}"
            for err in exc.errors()
        )
        raise CliError(
            f"the atoms file {file} is not an AtomSet ({problems}); it needs a criterion, "
            "objective_atoms and subjective_dimensions, each a list of {name, description}"
        ) from exc
    return define_atoms(
        doc.criterion,
        [(d.name, d.description) for d in doc.objective_atoms],
        [(d.name, d.description) for d in doc.subjective_dimensions],
    )
