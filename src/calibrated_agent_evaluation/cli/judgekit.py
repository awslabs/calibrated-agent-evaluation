# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A judges-only dump: what a fresh store (a CI runner's, say) needs to admit and score with judges.

``calibrated-agent-evaluation dump DIR --judges-only --judge FILE`` exports, for each judge file,
the newest calibration record pinned to it, the newest bias battery pinned to it and the judge file
itself, and refuses a judge the trust chain would not admit (so the dump never carries a judge that
cannot score). No trace, label or other record is exported: the gold behind the calibration stays
where it was labeled. ``calibrated-agent-evaluation load DIR`` recognizes the dump by its manifest,
merges the records into the local root and copies each judge file to ``<root>/judges/``.

The layout is ``DIR/judges-only.json`` (the manifest), ``DIR/judges/<file>`` and
``DIR/records/{calibration,bias}/``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, ValidationError

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.cli.judgefile import JudgeFile

__all__ = ["MANIFEST", "JudgeKitManifest", "dump_judges", "is_judge_kit", "load_judges"]

MANIFEST = "judges-only.json"
_KIT_VERSION = "1"


class JudgeKitEntry(CalibratedAgentEvaluationModel):
    """One exported judge: its file and the records that admit it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_id: str
    file: str
    fingerprint: str | None
    calib_id: str
    battery_id: str


class JudgeKitManifest(CalibratedAgentEvaluationModel):
    """The manifest of a judges-only dump."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kit_version: str = _KIT_VERSION
    judges: tuple[JudgeKitEntry, ...]


def is_judge_kit(directory: Path) -> bool:
    """Whether ``directory`` holds a judges-only dump."""
    return (directory / MANIFEST).is_file()


def dump_judges(
    root: Path,
    dest: Path,
    judge_paths: list[str],
    *,
    open_judge: Callable[[str], JudgeFile] | None = None,
) -> JudgeKitManifest:
    """Export each judge file with its admitting calibration and bias records into ``dest``.

    ``open_judge`` loads each file; the default reads a prompt judge file. The CLI passes one that
    also builds an ``agentcore`` judge file's judge from its evaluator.

    Raises:
        CliError: No judge file given, ``dest`` already holds files, two files share a judge id
            or a file name, or a judge would not be admitted.
    """
    from calibrated_agent_evaluation.cli.config import require_extras
    from calibrated_agent_evaluation.cli.judgefile import JUDGE_FILE_MODULES, load_judge_file
    from calibrated_agent_evaluation.cli.records import (
        latest_for_judge,
        load_records_of,
        records_dir,
    )

    # Admission reads calibration and bias records through the [stats] engine.
    require_extras(*JUDGE_FILE_MODULES)
    from calibrated_agent_evaluation.enforce.admission import admit_judge

    opener = open_judge or load_judge_file

    if not judge_paths:
        raise UsageError(
            "calibrated-agent-evaluation dump --judges-only needs --judge FILE for each judge to "
            "export (the judge "
            "file is part of the export, and the store does not hold it)"
        )
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise CliError(f"{dest} already exists and is not an empty directory; dump into a new one")
    calibrations = load_records_of(root, "calibration")
    batteries = load_records_of(root, "bias")
    entries: list[JudgeKitEntry] = []
    calibs: list[Any] = []
    bias: list[Any] = []
    sources: list[Path] = []
    for path in judge_paths:
        jf = opener(path)
        judge = jf.judge
        if any(e.judge_id == judge.judge_id for e in entries):
            raise UsageError(f"two --judge files name judge {judge.judge_id!r}; export it once")
        if any(e.file == jf.path.name for e in entries):
            raise UsageError(f"two --judge files are named {jf.path.name}; rename one")
        calib = latest_for_judge(calibrations, judge)
        battery = latest_for_judge(batteries, judge)
        if calib is None or battery is None:
            missing = "calibration record" if calib is None else "bias battery"
            where = records_dir(root, "calibration" if calib is None else "bias")
            raise CliError(
                f"judge {judge.judge_id!r} has no {missing} under {where}, so a store loaded from "
                "this dump could not admit it; certify it and run its bias battery first"
            )
        gate = admit_judge(judge, calib, battery)
        if not gate.admitted:
            raise CliError(
                f"judge {judge.judge_id!r} would not be admitted, so it is not exported: "
                f"{gate.refusal_reason}"
            )
        entries.append(
            JudgeKitEntry(
                judge_id=judge.judge_id,
                file=jf.path.name,
                fingerprint=getattr(judge, "fingerprint", None),
                calib_id=calib.calib_id,
                battery_id=battery.battery_id,
            )
        )
        calibs.append(calib)
        bias.append(battery)
        sources.append(jf.path)

    from calibrated_agent_evaluation.bias.store import dump_battery
    from calibrated_agent_evaluation.calibrate import dump_records

    manifest = JudgeKitManifest(judges=tuple(entries))
    try:
        (dest / "judges").mkdir(parents=True, exist_ok=True)
        for src in sources:
            shutil.copyfile(src, dest / "judges" / src.name)
        (dest / MANIFEST).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise CliError(f"cannot write the judges-only dump to {dest}: {exc}") from exc
    dump_records(calibs, records_dir(dest, "calibration"), fmt="json")
    dump_battery(bias, records_dir(dest, "bias"), fmt="json")
    return manifest


def load_judges(src: Path, root: Path) -> tuple[JudgeKitManifest, dict[str, int], list[Path]]:
    """Merge a judges-only dump's records into ``root`` and copy its judge files.

    Everything is read and checked before anything is written: every judge file must load and
    match the fingerprint in the manifest (an ``agentcore`` file, whose fingerprint needs its
    evaluator, must instead name the manifest's judge id and the evaluator its calibration record
    was made for, so loading makes no AWS call), and a judge file already in ``<root>/judges/``
    with different content is refused.

    Returns:
        The manifest, the count of new records per kind, and the restored judge files.

    Raises:
        CliError: A malformed manifest, a missing or changed judge file, or a clash in ``root``.
    """
    from calibrated_agent_evaluation.cli.judgefile import (
        declares_agentcore_backend,
        load_judge_file,
    )
    from calibrated_agent_evaluation.cli.records import load_records_of, prepare_import

    try:
        manifest = JudgeKitManifest.model_validate_json((src / MANIFEST).read_text("utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise CliError(f"{src / MANIFEST} is not a judges-only manifest: {exc}") from exc
    if manifest.kit_version != _KIT_VERSION:
        raise CliError(
            f"{src / MANIFEST} is kit version {manifest.kit_version!r}; this version reads "
            f"{_KIT_VERSION!r}. Dump it again with this version"
        )
    target = root / "judges"
    planned: list[tuple[Path, Path]] = []
    calibrations = load_records_of(src, "calibration")
    for entry in manifest.judges:
        file = src / "judges" / entry.file
        if not file.is_file() or Path(entry.file).name != entry.file:
            raise CliError(f"the judges-only dump in {src} is missing judges/{entry.file}")
        if declares_agentcore_backend(file):
            _check_agentcore_entry(file, entry, src, calibrations)
        else:
            jf = load_judge_file(file)
            if jf.judge.judge_id != entry.judge_id or getattr(jf.judge, "fingerprint", None) != (
                entry.fingerprint
            ):
                raise CliError(
                    f"judges/{entry.file} in {src} is not the judge the dump recorded "
                    f"({entry.judge_id!r}); it was edited after the dump. Dump it again"
                )
        dest = target / entry.file
        if dest.exists() and dest.read_bytes() != file.read_bytes():
            raise CliError(
                f"{dest} already exists with different content; load into a fresh --root or "
                "move it aside"
            )
        planned.append((file, dest))
    write_records = prepare_import(src, root)
    counts = write_records()
    try:
        target.mkdir(parents=True, exist_ok=True)
        for file, dest in planned:
            shutil.copyfile(file, dest)
    except OSError as exc:
        raise CliError(f"cannot copy the judge files into {target}: {exc}") from exc
    return manifest, counts, [dest for _, dest in planned]


def _check_agentcore_entry(
    file: Path, entry: JudgeKitEntry, src: Path, calibrations: list[Any]
) -> None:
    """Check an ``agentcore`` judge file in a dump against its manifest entry, with no AWS call.

    Its fingerprint can't be recomputed without reading the evaluator, so the file must name the
    judge the manifest recorded and the evaluator its calibration record was made for. An
    evaluator edited since the dump is caught when the judge is next built.

    Raises:
        CliError: The file does not load, or names another judge or another evaluator.
    """
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import load_agentcore_judge_file

    try:
        loaded = load_agentcore_judge_file(file)
    except AgentCoreError as exc:
        raise CliError(f"judges/{entry.file} in {src}: {exc}") from exc
    if loaded.judge_id != entry.judge_id:
        raise CliError(
            f"judges/{entry.file} in {src} names judge {loaded.judge_id!r}, but the dump recorded "
            f"{entry.judge_id!r}; it was edited after the dump. Dump it again"
        )
    record = next((c for c in calibrations if c.calib_id == entry.calib_id), None)
    if record is None or not record.model.endswith(f"evaluator/{loaded.evaluator_id}"):
        made_for = record.model if record is not None else "nothing in the dump"
        raise CliError(
            f"judges/{entry.file} in {src} names evaluator {loaded.evaluator_id!r}, but its "
            f"calibration record {entry.calib_id!r} was made for {made_for}; it was edited after "
            "the dump. Dump it again"
        )
