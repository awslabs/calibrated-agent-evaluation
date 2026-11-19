# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``latest_verdicts``: a judge re-run supersedes its earlier verdicts instead of adding to them.

A live model words its critique differently each run, so the byte-identity dedup keeps both rows of
a re-run. These tests read real stores (LocalBackend and the Langfuse adapter over its verified
stub) and check that the newest verdict per target, mode and judge version is the one kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from _helpers import make_label
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema import latest_verdicts
from test_backends_langfuse import StubLangfuseClient


def _verdict(critique: str, outcome: str = "PASS", **over: object):
    fields: dict[str, object] = {
        "trace_id": "T-1",
        "kind": "llm-judge",
        "outcome": outcome,
        "critique": critique,
        "failure_mode": "m",
        "rater_id": "judge@abc",
        **over,
    }
    return make_label(**fields)


def test_a_re_run_keeps_only_the_newest_verdict_on_a_local_store(tmp_path: Path) -> None:
    backend = LocalBackend(root=tmp_path / "store")
    backend.put_score(_verdict("first wording", "FAIL"))
    backend.put_score(_verdict("second wording", "PASS"))
    stored = backend.get_scores("T-1")
    assert len(stored) == 2
    assert latest_verdicts(stored) == [_verdict("second wording", "PASS")]


def test_verdicts_on_other_modes_spans_or_judge_versions_are_all_kept() -> None:
    labels = [
        _verdict("a"),
        _verdict("b", failure_mode="other"),
        _verdict("c", span_id="T-1/orderbot"),
        _verdict("d", rater_id="judge@def"),
    ]
    assert latest_verdicts(labels) == labels


def test_human_labels_and_unattributed_verdicts_are_never_collapsed() -> None:
    labels = [
        make_label(critique="one", rater_id="alice"),
        make_label(critique="two", rater_id="alice"),
        _verdict("x", rater_id=None),
        _verdict("y", rater_id=None),
    ]
    assert latest_verdicts(labels) == labels


def test_langfuse_reads_back_oldest_first_so_the_newest_verdict_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Langfuse returns scores in no write order; the adapter orders them by their write stamp."""
    import calibrated_agent_evaluation.backends.langfuse as lf_mod

    lf = LangfuseBackend(cast("LangfuseLike", StubLangfuseClient()))
    stamps = iter(["2026-01-02T00:00:00.000000+00:00", "2026-01-01T00:00:00.000000+00:00"])
    monkeypatch.setattr(lf_mod, "_written_at", lambda: next(stamps))
    newer = _verdict("put first, stamped later", "FAIL")
    older = _verdict("put second, stamped earlier", "PASS")
    lf.put_score(newer)
    lf.put_score(older)

    assert lf.get_scores("T-1") == [older, newer]
    assert latest_verdicts(lf.get_scores("T-1")) == [newer]
