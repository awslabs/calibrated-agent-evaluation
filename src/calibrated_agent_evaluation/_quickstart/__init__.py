# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled quickstart corpus: a tiny, zero-dependency sample that ships inside the wheel.

The default install packages only ``src/calibrated_agent_evaluation`` (no ``fixtures/``), so an
installed user has no data to run ``calibrated-agent-evaluation quickstart`` against. This package
bundles a ~56 KB slice of the reference corpus
(six normalized traces + their gold labels) as importable package data, read via
:mod:`importlib.resources` so it resolves no matter where the wheel is installed. It lets a fresh
install produce a real result offline, with no credentials and no files.

The data copies the committed reference corpus (``fixtures/normalized`` + ``fixtures/labels``);
it is sample input for the quickstart, not the acceptance corpus the test suite runs against.
"""

from __future__ import annotations

import gzip
import json
from importlib.resources import files  # nosemgrep: python37-compatibility-importlib2
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Label, Trace

_DATA = files(__name__).joinpath("data")


def load_sample_corpus() -> tuple[list[Trace], list[Label]]:
    """Return the bundled sample traces and gold labels, parsed into schema objects.

    Reads the packaged JSON via ``importlib.resources`` (not a filesystem path), so it works from an
    installed wheel, a zip, or the source tree alike.
    """
    normalized = _DATA.joinpath("normalized")
    traces = [
        trace_from_dict(json.loads(entry.read_text(encoding="utf-8")))
        for entry in sorted(normalized.iterdir(), key=lambda p: p.name)
        if entry.name.endswith(".json")
    ]
    gold_text = _DATA.joinpath("labels", "gold.json").read_text(encoding="utf-8")
    labels = [label_from_dict(item) for item in json.loads(gold_text)]
    return traces, labels


# The certify set: real OrderBot runs with SYNTHETIC rule-derived gold, built by
# corpus/generate/quickstart_set.py, big enough for a judge to certify and be rate-safe.
_CERTIFY = _DATA.joinpath("certify")
KIT_DIR = "quickstart"
CASSETTE_FILE = "judge.cassette.json"
THRESHOLDS_FILE = "bias-thresholds.json"
# Two versions of one judge: v1 certifies on round 1 and is refused for bias; v2 is the one prompt
# revision that result prompted, scored on round 2 only, since a locked test is single-use.
JUDGE_FILES = {"v1": "judge-v1.json", "v2": "judge-v2.json"}
JUDGE_ROUNDS = {"v1": "1", "v2": "2"}
# The taxonomy-cell key that scopes each round's calibration
# (`calibrated-agent-evaluation calibrate --cell`).
ROUND_KEY = "label_round"


def probes_dir(rnd: str) -> str:
    """The bundle and kit directory holding hunted round ``rnd``'s bias probes."""
    return f"bias-probes-{rnd}"


def documented_commands(root: Path) -> list[list[str]]:
    """The commands ``quickstart --certify`` prints for a store at ``root``, in order.

    Each is an argv without the ``calibrated-agent-evaluation --root ROOT`` prefix. Both judge
    versions are calibrated on their own round and run through the bias battery; only v2 goes on to
    score the production sample and the CI gate, since a judge the battery refuses cannot score.
    """
    kit = root / KIT_DIR
    cassette = ["--cassette", str(kit / CASSETTE_FILE)]
    thresholds = ["--thresholds", str(kit / THRESHOLDS_FILE)]
    argvs: list[list[str]] = []
    for version, rnd in JUDGE_ROUNDS.items():
        judge = ["--judge", str(kit / JUDGE_FILES[version])]
        argvs.append(
            ["calibrate", *judge, *cassette, "--rate-safe", "--cell", f"{ROUND_KEY}={rnd}"]
        )
        argvs.append(
            ["bias", *judge, "--probes", str(kit / probes_dir(rnd)), *thresholds, *cassette]
        )
    final = ["--judge", str(kit / JUDGE_FILES["v2"])]
    argvs.append(["judge", *final, "--frame", "production-representative", *cassette])
    argvs.append(["run", *final, "--ci"])
    return argvs


def _jsonl_gz(name: str) -> list[dict[str, Any]]:
    text = gzip.decompress(_CERTIFY.joinpath(name).read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def load_certify_set() -> tuple[list[Trace], list[Label]]:
    """Return the certify set's traces (hunted calibration set + production sample) and gold."""
    traces = [trace_from_dict(doc) for doc in _jsonl_gz("traces.jsonl.gz")]
    labels = [label_from_dict(doc) for doc in _jsonl_gz("gold.jsonl.gz")]
    return traces, labels


def certify_manifest() -> dict[str, Any]:
    """The certify set's manifest: sizes, the written policy, what the hunt left out."""
    manifest: dict[str, Any] = json.loads(
        _CERTIFY.joinpath("manifest.json").read_text(encoding="utf-8")
    )
    return manifest


def write_certify_kit(dest: Path) -> Path:
    """Write both judge files, their recorded cassette and each round's bias probes under ``dest``.

    Returns the kit directory. The cassette holds the judge model's recorded replies for every
    question :func:`documented_commands` asks, so the commands run with ``--cassette`` and no
    credentials.
    """
    kit = dest / KIT_DIR
    kit.mkdir(parents=True, exist_ok=True)
    for name in (*JUDGE_FILES.values(), THRESHOLDS_FILE):
        (kit / name).write_bytes(_CERTIFY.joinpath(name).read_bytes())
    cassette = gzip.decompress(_CERTIFY.joinpath(CASSETTE_FILE + ".gz").read_bytes())
    (kit / CASSETTE_FILE).write_bytes(cassette)
    for rnd in JUDGE_ROUNDS.values():
        probes = kit / probes_dir(rnd)
        probes.mkdir(exist_ok=True)
        for entry in _CERTIFY.joinpath(probes_dir(rnd)).iterdir():
            if entry.name.endswith(".json"):
                (probes / entry.name).write_bytes(entry.read_bytes())
    return kit
