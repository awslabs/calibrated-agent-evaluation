# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation define` builds the discovery frame; `generate --frame` reads it."""

from __future__ import annotations

import json
from itertools import combinations, product

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK

GRID = {
    "feature": ["refund", "exchange", "cancel"],
    "persona": ["new", "returning"],
    "tier": ["a", "b"],
}


def _taxonomy(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(GRID))
    return path


def test_define_writes_a_frame_that_covers_every_pair(tmp_path, capsys):
    out = tmp_path / "frame.json"
    rc = main(
        [
            "--root",
            str(tmp_path / "s"),
            "define",
            "--taxonomy",
            str(_taxonomy(tmp_path)),
            "--out",
            str(out),
        ]
    )
    assert rc == EXIT_OK
    doc = json.loads(out.read_text())
    assert doc["frame"] == "discovery"
    assert doc["coverage_strength"] == 2
    cells = [row["cell"] for row in doc["rows"]]
    for a, b in combinations(GRID, 2):
        for va, vb in product(GRID[a], GRID[b]):
            assert any(c[a] == va and c[b] == vb for c in cells), (a, va, b, vb)
    summary = capsys.readouterr().out
    assert f"{len(cells)} row(s) covering every 2-way combination of 3 dimension(s)" in summary


def test_generate_reads_the_frame_define_wrote(tmp_path, capsys):
    frame, inputs = tmp_path / "frame.json", tmp_path / "inputs.jsonl"
    root = str(tmp_path / "s")
    main(["--root", root, "define", "--taxonomy", str(_taxonomy(tmp_path)), "--out", str(frame)])
    assert (
        main(["--root", root, "generate", "--frame", str(frame), "--out", str(inputs)]) == EXIT_OK
    )
    rows = json.loads(frame.read_text())["rows"]
    written = [json.loads(line) for line in inputs.read_text().splitlines()]
    assert [w["cell"] for w in written] == [r["cell"] for r in rows]


def test_define_over_the_store_says_which_dimensions_it_padded(seeded_local_root, capsys):
    assert main(["define", "--trace", "ORD-1001"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "taxonomy of the stored traces" in out
    assert "placeholder second value" in out
    assert "frame not kept" in out


def test_define_with_too_high_a_strength_is_refused(tmp_path, capsys):
    rc = main(
        [
            "--root",
            str(tmp_path / "s"),
            "define",
            "--taxonomy",
            str(_taxonomy(tmp_path)),
            "--strength",
            "4",
        ]
    )
    assert rc == EXIT_DOMAIN_ERROR
    assert "SamplingConfigError" in capsys.readouterr().err


def test_define_on_an_unknown_trace_is_refused(seeded_local_root, capsys):
    assert main(["define", "--trace", "NOPE"]) == EXIT_DOMAIN_ERROR
    assert "NOPE" in capsys.readouterr().err


def test_a_frame_file_that_is_not_a_frame_is_refused(tmp_path, capsys):
    bad = tmp_path / "frame.json"
    bad.write_text(json.dumps({"rows": []}))
    rc = main(["--root", str(tmp_path / "s"), "generate", "--frame", str(bad)])
    assert rc == EXIT_DOMAIN_ERROR
    assert "not a calibrated-agent-evaluation define frame" in capsys.readouterr().err
