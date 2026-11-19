# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a stage's result reaches the terminal: plain language first, statistics on request.

A stage returns a :class:`~calibrated_agent_evaluation.cli.dispatch.StageOutcome` whose ``summary``
says what happened and what to do next in engineer terms, and whose ``detail`` holds the statistical
account (bounds, methods, thresholds, record paths). :class:`Presenter` prints the summary, and the
detail only under ``--explain``. A typed error is printed the same way: an :class:`ErrorExplainer`
turns the exception into a plain headline and next step, and ``--explain`` adds the exception's own
message. An error with no explainer is printed as its type and message, as before.

The presenter writes to the streams it is given, so it is tested without running the CLI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from calibrated_agent_evaluation.cli.dispatch import StageOutcome

EXPLAIN_HINT = "Run with --explain for the numbers behind this."


@dataclass(frozen=True)
class PlainError:
    """A typed error in plain words: what happened, and the next action (if there is one)."""

    headline: str
    next_step: str | None = None


ErrorExplainer = Callable[[BaseException], PlainError]


@dataclass(frozen=True)
class Presenter:
    """Print stage outcomes and typed errors, with the statistical detail behind ``--explain``.

    ``explainers`` maps an exception class name to the function that phrases it; the lookup walks
    the exception's class hierarchy, so a subclass without its own entry uses its parent's.
    """

    out: TextIO
    err: TextIO
    explain: bool = False
    explainers: Mapping[str, ErrorExplainer] = field(default_factory=dict)

    def outcome(self, command: str, outcome: StageOutcome, *, ok: bool) -> None:
        """Print a stage's summary, plus its detail under ``--explain``.

        Args:
            command: The subcommand that produced the outcome.
            outcome: What the stage reported.
            ok: Whether the stage succeeded; a failure goes to stderr.
        """
        stream = self.out if ok else self.err
        prefix = f"calibrated-agent-evaluation {command}: "
        # Ingest and capture reports already open with the command name.
        prefixed = ok or outcome.summary.startswith(prefix)
        summary = outcome.summary if prefixed else f"{prefix}{outcome.summary}"
        print(summary, file=stream)
        if outcome.detail:
            if self.explain:
                print("", file=stream)
                print(outcome.detail, file=stream)
            else:
                print(EXPLAIN_HINT, file=stream)

    def error(self, command: str, exc: BaseException) -> None:
        """Print a typed error as a plain headline with the next step.

        The error's class name is an implementation detail, so it is shown only under
        ``--explain``, where it tells a bug report which error was raised.

        Args:
            command: The subcommand that raised.
            exc: The error to describe.
        """
        typed = f"{type(exc).__name__}: {exc}"
        explainer = self._explainer_for(exc)
        if explainer is None:
            shown = typed if self.explain else str(exc)
            print(f"calibrated-agent-evaluation {command}: {shown}", file=self.err)
            return
        plain = explainer(exc)
        print(f"calibrated-agent-evaluation {command}: {plain.headline}", file=self.err)
        if plain.next_step:
            print(f"Next: {plain.next_step}", file=self.err)
        if self.explain:
            print("", file=self.err)
            print(typed, file=self.err)
        else:
            print("(Run with --explain for the details.)", file=self.err)

    def _explainer_for(self, exc: BaseException) -> ErrorExplainer | None:
        for cls in type(exc).__mro__:
            found = self.explainers.get(cls.__name__)
            if found is not None:
                return found
        return None
