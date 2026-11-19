# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation label --serve` with a codebook: the version it stamps and the modes
its form offers.

uvicorn is stubbed (the app is built for real and handed back); the source order is file, then
stored ratified codebook for --codebook-version, then open coding.
"""

from __future__ import annotations

import json
from typing import Any

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE


def _serve(monkeypatch, root, argv) -> tuple[int, Any]:
    import uvicorn

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **_kw: served.setdefault("app", app))
    monkeypatch.chdir(root)
    rc = main(["label", "--serve", "--rater", "alice", *argv])
    return rc, served.get("app")


def _codebook_file(path, version="7", names=("wrong_tool", "late_refund")):
    path.write_text(json.dumps({"version": version, "modes": [{"name": n} for n in names]}))
    return path


def _ratify(root, tmp_path, version="1"):
    doc = {
        "codebook_version": version,
        "ratified_by": "carol",
        "assign": [{"trace_id": "ORD-1001", "mode": "trusted_broken_policy_tool"}],
    }
    path = tmp_path / f"ratify-{version}.json"
    path.write_text(json.dumps(doc))
    assert main(["cluster", "--ratify", str(path)]) == EXIT_OK


def test_a_codebook_file_supplies_the_modes_and_the_version(
    seeded_local_root, tmp_path, monkeypatch, capsys
):
    book = _codebook_file(tmp_path / "codebook.json")
    rc, app = _serve(monkeypatch, seeded_local_root, ["--codebook", str(book)])
    assert rc == EXIT_OK
    assert app.state.codebook_version == "7"
    assert [m.name for m in app.state.failure_modes] == ["wrong_tool", "late_refund"]
    assert "failure modes: 2 from" in capsys.readouterr().err


def test_a_codebook_version_that_contradicts_the_file_is_refused(
    seeded_local_root, tmp_path, monkeypatch, capsys
):
    book = _codebook_file(tmp_path / "codebook.json")
    rc, app = _serve(
        monkeypatch, seeded_local_root, ["--codebook", str(book), "--codebook-version", "8"]
    )
    assert rc == EXIT_DOMAIN_ERROR
    assert app is None
    assert "does not match the version" in capsys.readouterr().err


def test_ratified_modes_are_offered_for_their_version(
    seeded_local_root, tmp_path, monkeypatch, capsys
):
    _ratify(seeded_local_root, tmp_path)
    capsys.readouterr()
    rc, app = _serve(monkeypatch, seeded_local_root, ["--codebook-version", "1"])
    assert rc == EXIT_OK
    assert app.state.codebook_version == "1"
    assert [m.name for m in app.state.failure_modes] == ["trusted_broken_policy_tool"]


def test_without_a_version_the_session_is_unversioned_and_names_the_stored_one(
    seeded_local_root, tmp_path, monkeypatch, capsys
):
    _ratify(seeded_local_root, tmp_path, version="3")
    capsys.readouterr()
    rc, app = _serve(monkeypatch, seeded_local_root, [])
    assert rc == EXIT_OK
    assert app.state.codebook_version is None
    assert app.state.failure_modes == ()
    err = capsys.readouterr().err
    assert "open coding, unversioned" in err
    assert "--codebook-version 3" in err


def test_a_version_with_no_list_is_open_coding_under_that_version(
    seeded_local_root, monkeypatch, capsys
):
    rc, app = _serve(monkeypatch, seeded_local_root, ["--codebook-version", "9"])
    assert rc == EXIT_OK
    assert app.state.codebook_version == "9"
    assert app.state.failure_modes == ()
    assert "open coding under codebook version '9'" in capsys.readouterr().err


def test_a_malformed_codebook_file_is_a_typed_refusal(
    seeded_local_root, tmp_path, monkeypatch, capsys
):
    for content, message in (
        ("{", "not valid JSON"),
        ('{"version": "1", "modes": []}', "modes"),
        ('{"version": "1", "modes": [{"name": "a"}, {"name": "a"}]}', "unique"),
    ):
        path = tmp_path / "bad.json"
        path.write_text(content)
        rc, app = _serve(monkeypatch, seeded_local_root, ["--codebook", str(path)])
        assert rc == EXIT_DOMAIN_ERROR
        assert app is None
        assert message in capsys.readouterr().err


def test_an_empty_codebook_version_is_refused(seeded_local_root, monkeypatch, capsys):
    rc, _app = _serve(monkeypatch, seeded_local_root, ["--codebook-version", " "])
    assert rc == EXIT_USAGE
    assert "--codebook-version" in capsys.readouterr().err


def test_the_printed_launch_command_carries_the_codebook_flags(seeded_local_root, capsys):
    assert main(["label", "--codebook-version", "2", "--codebook", "book.json"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "--codebook book.json --codebook-version 2" in out
