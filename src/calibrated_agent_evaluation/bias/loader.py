# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a bias-probe set and per-model thresholds from JSON files.

The probe and threshold files follow the committed ``fixtures/bias/`` format, so a caller outside
this repo (a judge for an external agent) can run the battery from its own probe directory without
going through a demo script. Each trial's ``question`` is copied into every input's payload:
the judge has to see what it is grading, and
:func:`calibrated_agent_evaluation.judges.render_probe_input` renders it.

Pure file reading, no vendor import. Any read or shape failure raises
:class:`~calibrated_agent_evaluation.bias.errors.ProbeExecutionError` naming the file, before a
single judge call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from calibrated_agent_evaluation.bias.errors import ProbeExecutionError
from calibrated_agent_evaluation.bias.probe import BiasProbe, ProbeInput
from calibrated_agent_evaluation.bias.record import ADVISORY, MUST_PASS, BiasName
from calibrated_agent_evaluation.bias.thresholds import BiasThresholds

_KNOWN_BIASES: frozenset[str] = frozenset((*MUST_PASS, *ADVISORY, "positional"))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeExecutionError(f"cannot read bias file {path}: {exc}") from exc


def _malformed(path: Path, why: str) -> ProbeExecutionError:
    return ProbeExecutionError(f"malformed bias file {path}: {why}")


def _trial_inputs(trial: Any, bias: BiasName, path: Path) -> tuple[ProbeInput, ...]:
    question = trial.get("question")
    inputs = []
    for position, inp in enumerate(trial["inputs"]):
        payload = dict(inp["payload"])
        if question is not None:
            payload.setdefault("question", question)
        toward = inp.get("toward")
        if toward is not None and (position == 0 or not isinstance(toward, str) or not toward):
            raise _malformed(
                path,
                f"trial {trial.get('trial_id')!r}: 'toward' must be a verdict label on a perturbed "
                "input; the first input is the unperturbed reference and cannot declare one",
            )
        inputs.append(
            ProbeInput(bias=bias, probe_id=str(inp["probe_id"]), payload=payload, toward=toward)
        )
    if not inputs:
        raise _malformed(path, f"trial {trial.get('trial_id')!r} has no inputs")
    return tuple(inputs)


def _probe_from_doc(doc: Any, path: Path) -> BiasProbe:
    try:
        bias = doc["bias"]
        trials_doc = doc["trials"]
        must_pass = bool(doc["must_pass"])
    except (KeyError, TypeError) as exc:
        raise _malformed(path, f"missing field {exc!r}") from exc
    if bias not in _KNOWN_BIASES:
        raise _malformed(path, f"unknown bias {bias!r}; expected one of {sorted(_KNOWN_BIASES)}")
    try:
        trials = tuple(_trial_inputs(t, cast(BiasName, bias), path) for t in trials_doc)
    except (KeyError, TypeError, AttributeError) as exc:
        raise _malformed(path, f"bad trial entry {exc!r}") from exc
    return BiasProbe(
        bias=cast(BiasName, bias),
        is_must_pass=must_pass,
        trials=trials,
        pairwise=bool(doc.get("pairwise", False)),
    )


def load_probe_set(directory: str | Path) -> list[BiasProbe]:
    """Load every ``*.json`` probe file under ``directory``, sorted by file name.

    Raises :class:`ProbeExecutionError` when the directory holds no probe files, or any file is
    unreadable or missing ``bias`` / ``must_pass`` / ``trials`` (an empty battery would certify
    nothing, so it is refused here rather than reaching the gate).
    """
    root = Path(directory)
    files = sorted(root.glob("*.json")) if root.is_dir() else []
    if not files:
        raise ProbeExecutionError(f"no probe files (*.json) found under {root}")
    return [_probe_from_doc(_read_json(f), f) for f in files]


def load_thresholds(path: str | Path) -> BiasThresholds:
    """Load per-model, per-bias thresholds from a ``{"by_model": {model: {bias: float}}}`` file.

    Raises :class:`ProbeExecutionError` on an unreadable file or a non-numeric threshold.
    """
    file = Path(path)
    doc = _read_json(file)
    try:
        items = [(str(m), dict(b)) for m, b in doc["by_model"].items()]
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise _malformed(
            file, f"expected {{'by_model': {{model: {{bias: float}}}}}}: {exc!r}"
        ) from exc
    by_model: dict[str, dict[BiasName, float]] = {}
    for model, biases in items:
        row: dict[BiasName, float] = {}
        for bias, value in biases.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise _malformed(file, f"threshold for {model}/{bias} is {value!r}, not a number")
            row[cast(BiasName, bias)] = float(value)
        by_model[model] = row
    return BiasThresholds(by_model=by_model)
