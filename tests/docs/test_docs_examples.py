# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The docs site's captured example output matches what the code prints today.

Each fragment's committed golden under ``fixtures/expected/docs/`` is what a page shows. A change
that moves a number or rewords a message fails here, before the site can drift. Refresh with
``just docs-goldens`` after reviewing the diff.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_docs_examples() -> ModuleType:
    """Import ``docs/site/docs_examples.py``, which lives outside any package on the path."""
    path = Path(__file__).resolve().parents[2] / "docs" / "site" / "docs_examples.py"
    spec = importlib.util.spec_from_file_location("docs_examples", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


docs_examples = _load_docs_examples()
_BY_NAME = {f.name: f for f in docs_examples.FRAGMENTS}


@pytest.mark.parametrize("name", sorted(_BY_NAME))
def test_fragment_matches_its_golden(name: str) -> None:
    fragment = _BY_NAME[name]
    assert fragment.golden.exists(), f"no golden for {name}; run `just docs-goldens`"
    actual = docs_examples.produce(fragment)
    assert actual == fragment.golden.read_text(encoding="utf-8"), (
        f"{name} changed; review the output, then run `just docs-goldens`"
    )


def test_every_golden_has_a_fragment() -> None:
    known = {f.golden.name for f in docs_examples.FRAGMENTS} | {docs_examples.GOLDEN_MANIFEST.name}
    stray = sorted(p.name for p in docs_examples.GOLDEN_DIR.iterdir() if p.name not in known)
    assert not stray, f"goldens with no fragment: {stray}"


def test_the_manifest_lists_every_golden() -> None:
    listed = docs_examples.GOLDEN_MANIFEST.read_text(encoding="utf-8").splitlines()
    assert listed == sorted(f.golden.name for f in docs_examples.FRAGMENTS)


def test_refreshing_goldens_removes_only_files_it_wrote_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keep = docs_examples.Fragment("keep", ("calibrated-agent-evaluation", "--help"))
    monkeypatch.setattr(docs_examples, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(docs_examples, "GOLDEN_MANIFEST", tmp_path / "MANIFEST")
    monkeypatch.setattr(docs_examples, "FRAGMENTS", (keep,))
    (tmp_path / "retired.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("someone else's file\n", encoding="utf-8")
    (tmp_path / "MANIFEST").write_text("keep.txt\nretired.txt\n", encoding="utf-8")

    docs_examples.write_goldens({"keep": "new\n"})

    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "new\n"
    assert not (tmp_path / "retired.txt").exists()
    assert (tmp_path / "notes.txt").exists()
    assert (tmp_path / "MANIFEST").read_text(encoding="utf-8") == "keep.txt\n"


def test_scrub_replaces_store_paths_run_ids_and_timestamps(tmp_path: Path) -> None:
    stamp = "2026-10-16T06:10:26Z"
    text = f"\x1b[1mrecorded run-20261016T061026Z-0ff5b9 in {tmp_path}/records at {stamp}  "
    assert docs_examples.scrub(text, tmp_path) == (
        "recorded run-<id> in $CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT/records at <time>\n"
    )


@pytest.mark.parametrize(
    "error",
    [
        "invalid choice: 'bogus' (choose from 'judge', 'scorer', 'human')",
        "invalid choice: 'bogus' (choose from judge, scorer, human)",
    ],
    ids=["quoted", "unquoted"],
)
def test_scrub_reads_argparse_choices_the_same_whichever_way_python_quotes_them(
    tmp_path: Path, error: str
) -> None:
    # CPython patch releases differ on whether an invalid-choice error quotes the choices.
    assert docs_examples.scrub(error, tmp_path) == (
        "invalid choice: 'bogus' (choose from judge, scorer, human)\n"
    )


def test_unexpected_exit_code_is_an_error() -> None:
    wrong = docs_examples.Fragment(
        "probe", ("calibrated-agent-evaluation", "run", "--verdicts", "human", "--ci")
    )
    with pytest.raises(docs_examples.FragmentError, match="exited 1, expected 0"):
        docs_examples.produce(docs_examples.Fragment(wrong.name, wrong.argv, seed_quickstart=True))


def test_fragments_run_without_aws_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    env = docs_examples._offline_env(tmp_path)
    assert "AWS_ACCESS_KEY_ID" not in env
    assert env["CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"] == "replay"
    assert env["AWS_PROFILE"] == "calibrated-agent-evaluation-docs-no-such-profile"
