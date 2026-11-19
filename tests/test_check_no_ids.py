# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The id guard catches every spelling of a planning id and fails loudly on a bad invocation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_ids.py"
# Ids are assembled from parts so this file passes the guard it tests.
_FR = "FR"
_DASH = "-"


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


@pytest.mark.parametrize(
    "line",
    [
        "# see " + "fr" + _DASH + "001\n",
        "# covers " + "F" + _DASH + "12\n",
        "# covers " + _FR + "001\n",
        "def test_x_" + "p006" + "():\n",
        "class Test" + _FR + "010" + "LazyImport:\n",
        "# carried over from " + "ticket" + " 16\n",
        'MESSAGE = "' + _FR + _DASH + '012"\n',
        'raise ValueError("refused (' + _FR + _DASH + '012)")\n',
    ],
)
def test_each_id_spelling_is_flagged(tmp_path: Path, line: str) -> None:
    source = tmp_path / "module.py"
    source.write_text(line, encoding="utf-8")
    result = _run(str(source))
    assert result.returncode == 1, result.stdout
    assert "module.py:1" in result.stdout


@pytest.mark.parametrize(
    "line",
    [
        'make_span(trace_id="T' + _DASH + '1", span_id="T' + _DASH + '1/root")\n',
        'rule = "ENF' + _DASH + '2"\n',
        "region = 'us-east-1'\n",
    ],
)
def test_data_that_looks_like_an_id_is_not_flagged(tmp_path: Path, line: str) -> None:
    source = tmp_path / "data.py"
    source.write_text(line, encoding="utf-8")
    result = _run(str(source))
    assert result.returncode == 0, result.stdout


def test_a_missing_path_is_an_error(tmp_path: Path) -> None:
    result = _run(str(tmp_path / "nowhere"))
    assert result.returncode == 2
    assert "no such path" in result.stderr


def test_the_default_roots_resolve_from_any_directory(tmp_path: Path) -> None:
    result = _run(cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no such path" not in result.stderr


def test_help_prints_usage_and_scans_nothing() -> None:
    result = _run("--help")
    assert result.returncode == 0
    assert "--all-roots" in result.stdout
    assert ".py:" not in result.stdout


def test_all_roots_takes_no_paths(tmp_path: Path) -> None:
    result = _run("--all-roots", str(tmp_path))
    assert result.returncode == 2
    assert "takes no paths" in result.stderr


def test_all_roots_adds_the_code_trees_the_docs_site_and_the_root_docs() -> None:
    spec = importlib.util.spec_from_file_location("check_no_ids", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    roots = (*module.DEFAULT_ROOTS, *module.EXTRA_ROOTS)
    assert {"scripts", "corpus", "fixtures", "skills", "pyproject.toml"} <= set(roots)
    assert {"docs/site", "README.md", "CONTRIBUTING.md", "CHANGELOG.md"} <= set(roots)
    assert not [r for r in roots if r == "docs" or (r.startswith("docs/") and r != "docs/site")]


def test_a_fixture_note_is_scanned_and_a_recording_is_not() -> None:
    spec = importlib.util.spec_from_file_location("check_no_ids", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    root = module.REPO_ROOT
    assert not module._skipped(root / "fixtures" / "bias" / "thresholds.json")
    assert not module._skipped(root / "fixtures" / "raw" / "otel" / "PROVENANCE.md")
    assert module._skipped(root / "fixtures" / "raw" / "otel" / "ORD-1001.attrs.otlp.json")
    assert module._skipped(root / "fixtures" / "raw" / "cassettes" / "orderbot_otel.cassette.json")
    assert module._skipped(root / "fixtures" / "expected" / "docs" / "ts-usage-error.txt")
