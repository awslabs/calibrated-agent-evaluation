#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the failure detector's and the draft namer's committed cassettes against live Bedrock.

Makes four billed Bedrock ``ConverseStream`` calls, and only with ``--yes``; without it the script
prints the plan and sends nothing:

1. to 3. one ``FailureDetector`` detects ORD-1001, ORD-1002 and CLM-4471 and records the three
   calls into ``fixtures/raw/cassettes/detector_fixture.cassette.json``. One detector and one
   cassette, because a caller that detects several traces replays them all from one file.
4. ``build_draft`` names the clusters the seven gold FAIL critiques form, through a
   ``DetectorDraftNamer``, and the call is recorded into ``namer_gold_fails.cassette.json``. The
   request is built by ``name_gold_fail_clusters`` below, so a replay that builds it the same way
   finds it.

It then replays both cassettes offline, the detections on traces read back from a local store,
and writes each result's oracle view under ``fixtures/expected/detectors/``. It stops and reports,
writing no oracle, when a call fails, when a replay raises, or when none of the three detections
lists a failure. The report quotes each Calibrated Agent Evaluation error in the chain of causes and
names any other error by its type alone, because its message can echo a credential. It never records
again on its own: a cassette that already exists is refused, because a second recording is another
set of billed calls. ``--oracles-only`` sends nothing: it replays the committed cassettes and writes
the oracles again, for when the oracle view changes or a recording stopped before its oracles were
written.

The recording's model id comes from ``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID`` (default
``global.anthropic.claude-sonnet-4-6``) and its credentials from ``AWS_PROFILE``. A replay reads
neither: it runs under the model id stamped on the cassette entries, so an id exported for some
other run cannot break it. Every entry of both cassettes must carry the same id; a cassette with an
entry stamped with no model id, or entries naming more than one, is refused before anything is
replayed.

Usage:
    uv run python scripts/record_detector_cassettes.py              # print the plan, send nothing
    AWS_PROFILE=<profile> uv run python scripts/record_detector_cassettes.py --yes
    uv run python scripts/record_detector_cassettes.py --oracles-only   # offline, sends nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from strands.models.model import Model

    from calibrated_agent_evaluation.agentcore import Detection, DetectorDraftNamer, NamingResult
    from calibrated_agent_evaluation.cluster import DraftCodebook
    from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError
    from calibrated_agent_evaluation.schema.types import Trace

FIXTURES = _REPO_ROOT / "fixtures"
CASSETTES = FIXTURES / "raw" / "cassettes"
DETECTOR_CASSETTE = CASSETTES / "detector_fixture.cassette.json"
NAMER_CASSETTE = CASSETTES / "namer_gold_fails.cassette.json"
ORACLES = FIXTURES / "expected" / "detectors"
NAMER_ORACLE = ORACLES / "namer_gold_fails.json"

#: The traces the committed detector cassette holds one detection each for, in recording order.
DETECTED_TRACE_IDS: tuple[str, ...] = ("ORD-1001", "ORD-1002", "CLM-4471")

#: The cut the gold FAIL critiques are clustered at.
CLUSTER_THRESHOLD = 0.5


def detection_oracle(trace_id: str) -> Path:
    """The committed oracle view of the detection of ``trace_id``."""
    return ORACLES / f"{trace_id}.detection.json"


def fixture_trace(trace_id: str) -> Trace:
    """The normalized fixture trace ``trace_id``, as ``just seed-fixture`` stores it."""
    from calibrated_agent_evaluation.schema.serde import trace_from_dict

    path = FIXTURES / "normalized" / f"{trace_id}.json"
    return trace_from_dict(json.loads(path.read_text(encoding="utf-8")))


def gold_fail_critiques() -> list[str]:
    """The critiques of the FAIL labels in ``fixtures/labels/gold.json``, in file order."""
    from calibrated_agent_evaluation.schema.serde import label_from_dict

    rows = json.loads((FIXTURES / "labels" / "gold.json").read_text(encoding="utf-8"))
    return [label_from_dict(row).critique for row in rows if row.get("outcome") == "FAIL"]


def _sha256_embeddings(texts: list[str]) -> list[list[float]]:
    # Deterministic stand-in vectors: the same vector for the same text in every process.
    return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


def name_gold_fail_clusters(namer: DetectorDraftNamer) -> DraftCodebook:
    """Cluster the gold FAIL critiques and name the clusters through ``build_draft``.

    ``build_draft`` hands the namer the clusters in label order, which is the request the
    committed namer cassette recorded; calling ``name_clusters`` over the critiques in another
    order would send a request no recording holds.
    """
    from calibrated_agent_evaluation.cluster import build_draft, hierarchical_cluster

    critiques = gold_fail_critiques()
    embeddings = _sha256_embeddings(critiques)
    assignments = hierarchical_cluster(embeddings, threshold=CLUSTER_THRESHOLD)
    return build_draft(critiques, embeddings, assignments, namer=namer)


def oracle_text(view: dict[str, Any]) -> str:
    """An oracle view as the committed oracle files hold it."""
    return json.dumps(view, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


#: Three detections and one naming call.
BILLED_CALLS = len(DETECTED_TRACE_IDS) + 1

ModelFactory = Callable[[], "Model"]
#: Builds the replay's model for the model id the cassettes are stamped with.
ReplayFactory = Callable[[str], "Model"]

# A failure report follows the chain of causes this many links down.
_CAUSES_SHOWN = 5


class RecordingStopped(Exception):
    """The recording or its replay stopped; nothing more is sent and no oracle is written."""


class UnusableModelStamps(RecordingStopped):
    """The cassette entries do not name one model to replay them under, so nothing is replayed."""


@dataclass(frozen=True)
class Outputs:
    """Where the recording writes: the two cassettes and the oracle files."""

    detector_cassette: Path = DETECTOR_CASSETTE
    namer_cassette: Path = NAMER_CASSETTE
    namer_oracle: Path = NAMER_ORACLE
    oracle_dir: Path | None = None

    def detection_oracle(self, trace_id: str) -> Path:
        """The oracle file of the detection of ``trace_id``."""
        if self.oracle_dir is None:
            return detection_oracle(trace_id)
        return self.oracle_dir / f"{trace_id}.detection.json"


def plan_text(*, model_id: str, profile: str | None, region: str | None, out: Outputs) -> str:
    """The plan printed before anything is sent."""
    clusters = f"the {len(gold_fail_critiques())} gold FAIL critiques"
    lines = [
        f"Recording plan: {BILLED_CALLS} billed Bedrock ConverseStream calls",
        f"  model id: {model_id}  (CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID)",
        f"  profile:  {profile or '(AWS_PROFILE is not set)'}  (AWS_PROFILE)",
        f"  region:   {region or '(none configured: pass --region)'}",
        "",
    ]
    for number, trace_id in enumerate(DETECTED_TRACE_IDS, start=1):
        lines.append(f"  call {number}: detect {trace_id}  -> {_shown(out.detector_cassette)}")
    lines.append(
        f"  call {BILLED_CALLS}: name the clusters build_draft forms from {clusters}  -> "
        f"{_shown(out.namer_cassette)}"
    )
    lines += [
        "",
        "  Then: replay both cassettes offline, under the model id the recording stamps on",
        "  their entries, and write the oracle views:",
        *(f"    {_shown(out.detection_oracle(t))}" for t in DETECTED_TRACE_IDS),
        f"    {_shown(out.namer_oracle)}",
    ]
    existing = [p for p in (out.detector_cassette, out.namer_cassette) if p.exists()]
    if existing:
        lines += [
            "",
            "  Already recorded, so --yes refuses to run: "
            + ", ".join(_shown(p) for p in existing),
        ]
    return "\n".join(lines)


def record(model_factory: ModelFactory, out: Outputs) -> None:
    """Make the four live calls, recording each into its cassette.

    Raises:
        RecordingStopped: A call or its reading failed; what was recorded before it stays.
    """
    from calibrated_agent_evaluation.agentcore import (
        DetectorDraftNamer,
        DetectorError,
        FailureDetector,
    )
    from calibrated_agent_evaluation.agentcore import failure_vocabulary as vocabulary
    from calibrated_agent_evaluation.cluster import ClusteringError
    from calibrated_agent_evaluation.execute import Mode

    try:
        detector = FailureDetector(
            model_factory(),
            cassette_path=out.detector_cassette,
            cassette_mode=Mode.RECORD,
            max_live_calls=len(DETECTED_TRACE_IDS),
        )
    except DetectorError as exc:
        raise RecordingStopped(_failure("building the detector", exc)) from exc
    for number, trace_id in enumerate(DETECTED_TRACE_IDS, start=1):
        try:
            detection = detector.detect(fixture_trace(trace_id))
        except DetectorError as exc:
            raise RecordingStopped(_failure(f"call {number} (detect {trace_id})", exc)) from exc
        print(f"  call {number}: {trace_id}: {_failures(detection)}")
    try:
        namer = DetectorDraftNamer(
            model_factory(),
            vocabulary(),
            cassette_path=out.namer_cassette,
            cassette_mode=Mode.RECORD,
            max_live_calls=1,
        )
        draft = name_gold_fail_clusters(namer)
    except (DetectorError, ClusteringError) as exc:
        raise RecordingStopped(_failure(f"call {BILLED_CALLS} (name the clusters)", exc)) from exc
    print(f"  call {BILLED_CALLS}: named {len(draft.clusters)} clusters")


def replay(model_factory: ReplayFactory, out: Outputs) -> tuple[list[Detection], NamingResult]:
    """Replay both cassettes with no live call allowed, the traces read back from a local store.

    The replay runs under :func:`recorded_model_id`, the id the cassette entries are stamped with,
    which ``model_factory`` is called with.

    Raises:
        UnusableModelStamps: The entries do not name one model; nothing is replayed.
        RecordingStopped: A replay raised; nothing is recorded again.
    """
    from calibrated_agent_evaluation.agentcore import (
        DetectorDraftNamer,
        DetectorError,
        FailureDetector,
    )
    from calibrated_agent_evaluation.agentcore import failure_vocabulary as vocabulary
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cluster import ClusteringError
    from calibrated_agent_evaluation.execute import Mode

    model_id = recorded_model_id(out)
    detections: list[Detection] = []
    try:
        detector = FailureDetector(
            model_factory(model_id),
            cassette_path=out.detector_cassette,
            cassette_mode=Mode.REPLAY,
            max_live_calls=0,
        )
        with tempfile.TemporaryDirectory() as root, LocalBackend(root) as store:
            for trace_id in DETECTED_TRACE_IDS:
                store.put_trace(fixture_trace(trace_id))
            for trace_id in DETECTED_TRACE_IDS:
                try:
                    detections.append(detector.detect(store.get_trace(trace_id)))
                except DetectorError as exc:
                    raise RecordingStopped(_failure(f"the replay of {trace_id}", exc)) from exc
        namer = DetectorDraftNamer(
            model_factory(model_id),
            vocabulary(),
            cassette_path=out.namer_cassette,
            cassette_mode=Mode.REPLAY,
            max_live_calls=0,
        )
        name_gold_fail_clusters(namer)
    except (DetectorError, ClusteringError) as exc:
        raise RecordingStopped(_failure("the replay", exc)) from exc
    return detections, namer.results[-1]


def recorded_model_id(out: Outputs) -> str:
    """The one model id every entry of both cassettes is stamped with.

    A cassette refuses to replay an entry under any other id, so the replay takes its id from the
    recording, not from ``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID``, which may name another
    model in this shell.

    Raises:
        UnusableModelStamps: A cassette cannot be read, holds no entry, or holds an entry stamped
            with no model id, or the entries name more than one id. The message names the
            cassettes and the ids found in each, and quotes no request or response.
    """
    found = {path: _stamped_ids(path) for path in (out.detector_cassette, out.namer_cassette)}
    ids = sorted({model_id for found_ids in found.values() for model_id in found_ids})
    if len(ids) > 1:
        held = "; ".join(
            f"{_shown(path)} holds {', '.join(repr(i) for i in sorted(found_ids))}"
            for path, found_ids in found.items()
        )
        raise UnusableModelStamps(
            f"the cassettes were recorded under more than one model id ({held}), and a replay "
            "under any one of them would refuse the entries recorded under another. Nothing was "
            "replayed and no oracle was written: record both cassettes under one model, once the "
            "owner approves the billed calls"
        )
    return ids[0]


def _stamped_ids(path: Path) -> set[str]:
    """The model ids the entries of the cassette at ``path`` are stamped with; at least one.

    Raises:
        UnusableModelStamps: The cassette cannot be read, holds no entry, or holds an entry with
            no model id.
    """
    nothing_done = "Nothing was replayed and no oracle was written"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # These messages say where the read failed, never what the file holds.
        raise UnusableModelStamps(
            f"the cassette {_shown(path)} cannot be read for its model stamps "
            f"({type(exc).__name__}: {exc}). {nothing_done}: fix or restore the file"
        ) from exc
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise UnusableModelStamps(
            f"the cassette {_shown(path)} has no list of entries, so it names no model to replay "
            f"under. {nothing_done}: fix or restore the file"
        )
    if not entries:
        raise UnusableModelStamps(
            f"the cassette {_shown(path)} holds no entry, so it names no model to replay under. "
            f"{nothing_done}: record it again, once the owner approves the billed calls"
        )
    ids: set[str] = set()
    for index, entry in enumerate(entries):
        stamp = entry.get("model") if isinstance(entry, dict) else None
        model_id = stamp.get("id") if isinstance(stamp, dict) else None
        if not (isinstance(model_id, str) and model_id):
            raise UnusableModelStamps(
                f"the cassette {_shown(path)} has an entry (entry {index}) stamped with no model "
                "id, so which model answered it is not known and a replay could credit its answer "
                f"to the wrong model. {nothing_done}: record the cassette again, once the owner "
                "approves the billed calls"
            )
        ids.add(model_id)
    return ids


def write_oracles(detections: Sequence[Detection], naming: NamingResult, out: Outputs) -> None:
    """Write each result's oracle view, once at least one detection lists a failure.

    Raises:
        RecordingStopped: No detection lists a failure, so no oracle could show one.
    """
    if not any(detection.failures for detection in detections):
        raise RecordingStopped(
            "none of the three replayed detections lists a failure, so no oracle was written: "
            "the acceptance rows need at least one detection with a failure. Ask the owner "
            "before recording again; a second recording is another set of billed calls"
        )
    for detection in detections:
        _write(out.detection_oracle(detection.trace_id), oracle_text(detection.oracle_view()))
    _write(out.namer_oracle, oracle_text(naming.oracle_view()))


def run(
    model_factory: ModelFactory, replay_factory: ReplayFactory, out: Outputs
) -> tuple[list[Detection], NamingResult]:
    """Record, replay under the model id the recording stamped, and write the oracles.

    Raises:
        RecordingStopped: A cassette already exists, a call or a replay failed, the recorded
            entries do not name one model, or no detection lists a failure.
    """
    existing = [p for p in (out.detector_cassette, out.namer_cassette) if p.exists()]
    if existing:
        raise RecordingStopped(
            f"{', '.join(_shown(p) for p in existing)} already recorded, so nothing was sent: a "
            "second recording is another set of billed calls. Ask the owner, and delete the "
            "file yourself only once they approve a new recording"
        )
    record(model_factory, out)
    detections, naming = replay(replay_factory, out)
    write_oracles(detections, naming, out)
    return detections, naming


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=f"make the {BILLED_CALLS} billed calls (without it the plan is printed, nothing sent)",
    )
    parser.add_argument(
        "--region", default=None, help="the Bedrock region (default: the profile's region)"
    )
    parser.add_argument(
        "--oracles-only",
        action="store_true",
        help="send nothing: replay the committed cassettes and write the oracles again",
    )
    args = parser.parse_args(argv)

    from calibrated_agent_evaluation.agentcore import detector_bedrock_model

    out = Outputs()
    if args.oracles_only:
        # CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID is not read: the replay runs under the id the
        # cassettes carry.
        return _oracles_only(detector_bedrock_model, out)

    from calibrated_agent_evaluation.flagship.errors import FlagshipError
    from calibrated_agent_evaluation.flagship.runner import live_bedrock_model_id

    try:
        model_id = live_bedrock_model_id(os.environ)
    except FlagshipError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    profile = os.environ.get("AWS_PROFILE") or None
    region = args.region or _configured_region(profile)
    print(plan_text(model_id=model_id, profile=profile, region=region, out=out))
    if not args.yes:
        recorded = out.detector_cassette.exists() or out.namer_cassette.exists()
        then = (
            "--oracles-only writes the oracles from the recording without sending anything"
            if recorded
            else f"run again with --yes to make the {BILLED_CALLS} calls"
        )
        print(f"\nNothing was sent; {then}.")
        return 0
    if profile is None:
        print("\nREFUSED: set AWS_PROFILE to the profile the calls are billed to", file=sys.stderr)
        return 2
    if region is None:
        print("\nREFUSED: no region is configured; pass --region", file=sys.stderr)
        return 2

    print(f"\nRecording {BILLED_CALLS} calls...")
    try:
        detections, naming = run(
            lambda: detector_bedrock_model(model_id, region=region), detector_bedrock_model, out
        )
    except RecordingStopped as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 1
    _report_oracles(detections, naming, out)
    return 0


def _oracles_only(replay_factory: ReplayFactory, out: Outputs) -> int:
    """Replay the committed cassettes, with no live call allowed, and write the oracles again."""
    missing = [p for p in (out.detector_cassette, out.namer_cassette) if not p.exists()]
    if missing:
        print(
            f"REFUSED: {', '.join(_shown(p) for p in missing)} not recorded; the oracles are "
            "written from a recording",
            file=sys.stderr,
        )
        return 2
    print(
        "Replaying the committed cassettes offline, under the model id stamped on their entries; "
        "nothing is sent."
    )
    try:
        detections, naming = replay(replay_factory, out)
        write_oracles(detections, naming, out)
    except UnusableModelStamps as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except RecordingStopped as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 1
    _report_oracles(detections, naming, out)
    return 0


def _report_oracles(detections: Sequence[Detection], naming: NamingResult, out: Outputs) -> None:
    model_id = naming.provenance.model_id
    print(f"\nReplayed offline under {model_id}, the model id the cassettes are stamped with,")
    print("and wrote the oracles:")
    for detection in detections:
        print(f"  {_shown(out.detection_oracle(detection.trace_id))}: {_failures(detection)}")
    print(f"  {_shown(out.namer_oracle)}: {', '.join(naming.names)}")


def _configured_region(profile: str | None) -> str | None:
    """The region the environment or the profile configures, read from the config files only.

    ``None`` when none is configured, or when the profile cannot be read; a live run then
    refuses to start until ``--region`` is passed.
    """
    from botocore.exceptions import BotoCoreError
    from botocore.session import Session

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        return region
    try:
        session = Session(profile=profile) if profile else Session()
        configured = session.get_config_variable("region")
    except BotoCoreError:
        return None
    return configured if isinstance(configured, str) and configured else None


def _failures(detection: Detection) -> str:
    count = len(detection.failures)
    if count == 0:
        return "no failure"
    categories = ", ".join(f"{f.category} ({f.confidence})" for f in detection.failures)
    return f"{count} failure{'s' if count > 1 else ''}: {categories}"


def _failure(step: str, exc: CalibratedAgentEvaluationDataError) -> str:
    message = f"{step} failed with {type(exc).__name__}: {exc}"
    cause, depth = exc.__cause__, 0
    while cause is not None and depth < _CAUSES_SHOWN:
        message += f"\n  caused by {_cause_line(cause)}"
        cause, depth = cause.__cause__, depth + 1
    return (
        f"{message}\nNothing more was sent and no oracle was written. Whatever was recorded "
        "before this step stays in its cassette. Ask the owner before recording again"
    )


def _cause_line(cause: BaseException) -> str:
    """A Calibrated Agent Evaluation error by its type and message; any other error by its type
    alone.

    Calibrated Agent Evaluation writes its own errors' messages to be shown, and the CLI prints
    them. Another library's message can echo a credential, so it is left out, as the detector's own
    errors leave it out of theirs.
    """
    from calibrated_agent_evaluation.cli.errors import DOMAIN_ERRORS

    if isinstance(cause, DOMAIN_ERRORS):
        return f"{type(cause).__name__}: {_cut(str(cause))}"
    return f"{type(cause).__name__} (its message is not shown: it can hold a credential)"


def _cut(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else f"{text[:limit]} (cut to its first {limit} characters)"


def _shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise RecordingStopped(
            f"the oracle {_shown(path)} could not be written ({exc}); the cassettes are kept, so "
            "run with --oracles-only to write the oracles from them without sending anything"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
