# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building the review apps writes nothing into the working directory.

FastHTML creates a ``.sesskey`` file in the current directory for its session middleware unless it
is given a key. The review apps use no sessions, so they must not leave that file behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK
from calibrated_agent_evaluation.subjective import atom_set_from_dict

pytest.importorskip("fasthtml")

_ATOMS = {
    "criterion": "claim letter",
    "objective": [{"name": "cites_exclusion"}],
    "subjective": [{"name": "tone", "description": "courteous and plain"}],
}


@pytest.fixture
def store(tmp_path, monkeypatch, capsys) -> Path:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    assert main(["--root", str(root), "quickstart"]) == EXIT_OK
    capsys.readouterr()
    return root


@pytest.fixture
def empty_cwd(tmp_path, monkeypatch) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def test_building_the_review_app_writes_no_file_into_the_cwd(store, empty_cwd) -> None:
    from calibrated_agent_evaluation.review.app import build_app

    build_app(LocalBackend(root=store), rater_id="rater-1")
    assert list(empty_cwd.iterdir()) == []


def test_building_the_pairwise_app_writes_no_file_into_the_cwd(store, empty_cwd) -> None:
    from calibrated_agent_evaluation.review.pairwise import build_pairwise_app

    backend = LocalBackend(root=store)
    build_pairwise_app(
        backend,
        atom_set=atom_set_from_dict(_ATOMS),
        rater_id="rater-1",
        traces=[backend.get_trace(t) for t in ("CLM-4471", "CLM-4473")],
        store=lambda records: None,
    )
    assert list(empty_cwd.iterdir()) == []
