# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level CLI error-boundary tests.

The boundary catches the (deliberately non-uniform) typed-error family —
``CalibratedAgentEvaluationDataError``, ``ExecutionError``, ``ScorerError``,
``CassetteError``/``CassetteMiss``, ``CliError`` — turning each into one diagnostic line + exit 1,
with NO traceback. An exception outside that family is a genuine bug and must NOT be swallowed (it
propagates).
"""

from __future__ import annotations

import argparse

import pytest

from calibrated_agent_evaluation.cli import dispatch, handlers
from calibrated_agent_evaluation.cli.errors import DOMAIN_ERRORS, CliError
from calibrated_agent_evaluation.execute.cassette import CassetteError, CassetteMiss
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.schema.errors import (
    BackendError,
    CalibratedAgentEvaluationDataError,
)
from calibrated_agent_evaluation.scorers.errors import ScorerError


@pytest.fixture
def raising_handler(monkeypatch):
    """Install a handler for the `probe` stage that raises whatever exception it's given."""

    def _install(exc: Exception):
        def _handler(_args: argparse.Namespace) -> dispatch.StageOutcome:
            raise exc

        monkeypatch.setattr(handlers, "for_stage", lambda _stage: _handler)

    return _install


@pytest.mark.parametrize(
    "exc",
    [
        CalibratedAgentEvaluationDataError("data plane failed"),
        BackendError("backend unreachable"),
        ExecutionError("eval failed"),
        ScorerError("scorer failed"),
        CassetteError("cassette corrupt"),
        CassetteMiss("no cassette entry"),
        CliError("missing extra"),
    ],
)
def test_domain_error_becomes_exit_1_no_traceback(exc, raising_handler, capsys):
    """Every DOMAIN_ERRORS member → exit 1, one diagnostic line, no traceback."""
    raising_handler(exc)

    rc = dispatch.main(["probe"])

    assert rc == dispatch.EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    # One diagnostic line naming the error type; not a multi-line Python traceback.
    assert type(exc).__name__ in err
    assert "Traceback (most recent call last)" not in err
    assert err.strip().count("\n") == 0


def test_unexpected_exception_is_not_swallowed(raising_handler):
    """Edge: an exception OUTSIDE the family propagates (no silent swallow)."""
    raising_handler(RuntimeError("genuine bug"))

    with pytest.raises(RuntimeError, match="genuine bug"):
        dispatch.main(["probe"])


def test_every_family_member_is_in_the_boundary_tuple():
    """The boundary tuple is the explicit non-uniform family (contract)."""
    names = {c.__name__ for c in DOMAIN_ERRORS}
    assert {
        "CalibratedAgentEvaluationDataError",
        "ExecutionError",
        "ScorerError",
        "CassetteError",
        "CassetteMiss",
        "CliError",
    } <= names
