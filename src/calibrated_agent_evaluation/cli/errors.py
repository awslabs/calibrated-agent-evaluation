# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-layer error type + the top-level error-boundary family.

The methodology plane's typed errors are deliberately **not** a single hierarchy: `execute/`'s
``ExecutionError`` and `scorers/`'s ``ScorerError``/``ScorerLaneError`` subclass plain ``Exception``
(not ``CalibratedAgentEvaluationDataError``), and the cassette layer's
``CassetteError``/``CassetteMiss`` subclass ``RuntimeError``. So the CLI's top-level boundary cannot
catch the whole surface with one ``except CalibratedAgentEvaluationDataError``; it catches an
explicit **tuple** of the real base classes (:data:`DOMAIN_ERRORS`). Anything outside that tuple is
a genuine bug and is NOT swallowed; it propagates so it is visible ("no silent failure").
"""

from __future__ import annotations

from calibrated_agent_evaluation.execute.cassette import CassetteError, CassetteMiss
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation.scorers.errors import ScorerError

__all__ = [
    "DOMAIN_ERRORS",
    "CliError",
    "GoldSetReusedError",
    "JudgeNotCertifiedError",
    "StaleReadError",
    "UsageError",
]


class CliError(Exception):
    """A CLI-layer failure that is not a specific feature's domain error.

    Raised for bad invocation config the argparse layer can't catch and (critically) for a stage
    that needs an optional extra which is not installed: the CLI wraps the would-be ``ImportError``
    in a ``CliError`` whose message names the exact install command to run,
    so a missing extra is a typed, actionable refusal rather than a raw traceback.
    """


class UsageError(CliError):
    """The command line is wrong: a flag is missing, out of range, or does not go with another.

    Exits 2, as argparse's own usage errors do, where a refusal of valid input exits 1.
    """


class StaleReadError(CliError):
    """A read that would miss writes an earlier command made to an eventually consistent store.

    Raised after a bounded wait, naming the writes the store does not show yet
    (mod:`calibrated_agent_evaluation.cli.consistency`), so a stage never computes over a
    partial store with a clean exit.
    """


class JudgeNotCertifiedError(CliError):
    """A command that needs a certified judge found none for the judge file it was given.

    ``reason`` says why in plain words (no record, a REFUSED record, or a record certifying another
    version of the judge); ``judge_path`` is the file to calibrate and ``root`` the ``--root`` the
    command was given, which the printed next step repeats.
    """

    def __init__(
        self, message: str, *, reason: str, judge_path: str, root: str | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.judge_path = judge_path
        self.root = root


class GoldSetReusedError(CliError):
    """``calibrate`` was given gold that already tested a judge for the same mode.

    ``prior_result`` is the earlier test's outcome (``CERTIFIED`` or ``REFUSED``); evaluating a
    locked test at all uses it up, whatever the result. ``n_overlap`` is how many of the new
    gold's traces were in the used set, or ``None`` when the two sets are the same.
    """

    def __init__(
        self,
        message: str,
        *,
        judge_id: str,
        prior_result: str | None = None,
        n_overlap: int | None = None,
    ) -> None:
        super().__init__(message)
        self.judge_id = judge_id
        self.prior_result = prior_result
        self.n_overlap = n_overlap


# The top-level boundary family. Explicit tuple because the methodology plane's typed errors do NOT
# share one base (see module docstring). Every member is a *typed, explained* refusal the CLI turns
# into a one-line diagnostic + exit 1; an exception outside this tuple is an unexpected bug
# and is left to propagate (never silently swallowed).
DOMAIN_ERRORS: tuple[type[Exception], ...] = (
    # Includes the enforcement layer's EnforcementError (judge-file lint refusals). It is not
    # listed by name: importing calibrated_agent_evaluation.enforce loads the stats stack, and this
    # module must stay importable on a bare install.
    CalibratedAgentEvaluationDataError,
    ExecutionError,
    ScorerError,
    CassetteError,
    CassetteMiss,
    CliError,
)
