# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read pairwise preferences from a JSON Lines file.

One object per line with exactly the five fields of a
:class:`~calibrated_agent_evaluation.subjective.PreferencePair`::

    {"trace_id_a": "CLM-4471", "trace_id_b": "CLM-4473", "dimension": "tone",
     "winner": "b", "judge_kind": "human"}

``winner`` is ``"a"`` or ``"b"``; ``judge_kind`` is ``human``, ``llm-judge`` or ``code``. Blank
lines are skipped. Every line is checked against the atom set with
:func:`~calibrated_agent_evaluation.subjective.record_preference`, and one bad line refuses the
whole file, naming every bad line, so a partial import never reaches the store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.subjective.errors import SubjectiveError
from calibrated_agent_evaluation.subjective.preference import record_preference
from calibrated_agent_evaluation.subjective.types import AtomSet, PairJudgeKind, PreferencePair

__all__ = ["read_preference_file"]

# Bad lines named in a refusal; the count is always given in full.
_SHOW = 5


class _Line(CalibratedAgentEvaluationModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = SubjectiveError

    trace_id_a: str
    trace_id_b: str
    dimension: str
    winner: Literal["a", "b"]
    judge_kind: PairJudgeKind


def read_preference_file(path: str | Path, atom_set: AtomSet) -> list[tuple[int, PreferencePair]]:
    """Every preference in the file at ``path`` with its 1-based line number, checked.

    Raises:
        SubjectiveError: The file cannot be read, holds no preference, or any line is not valid
            JSON, lacks a field, carries an unknown one, or fails ``record_preference`` (an
            objective or undeclared dimension, a trace compared with itself).
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise SubjectiveError(f"cannot read the pairs file {file}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise SubjectiveError(f"the pairs file {file} is not UTF-8 text: {exc}") from exc
    pairs: list[tuple[int, PreferencePair]] = []
    bad: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            line = _Line.model_validate(json.loads(raw))
            pair = PreferencePair(
                trace_id_a=line.trace_id_a,
                trace_id_b=line.trace_id_b,
                dimension=line.dimension,
                winner=line.winner,
                judge_kind=line.judge_kind,
            )
            pairs.append((number, record_preference(pair, atom_set)))
        except json.JSONDecodeError as exc:
            bad.append(f"line {number}: not valid JSON ({exc.msg})")
        except SubjectiveError as exc:
            bad.append(f"line {number}: {exc}")
    if bad:
        more = f"; and {len(bad) - _SHOW} more" if len(bad) > _SHOW else ""
        raise SubjectiveError(
            f"{len(bad)} line(s) of the pairs file {file} were refused, so nothing was imported: "
            + "; ".join(bad[:_SHOW])
            + more
        )
    if not pairs:
        raise SubjectiveError(f"the pairs file {file} holds no preference")
    return pairs
