# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI dispatcher: subcommand → feature call, exit codes, error boundary.

``main`` parses argv, resolves the stage handler, runs it, and translates the result into a process
exit code:

  * **0**: the stage succeeded.
  * **1**: a typed domain error (any member of
    :data:`~calibrated_agent_evaluation.cli.errors.DOMAIN_ERRORS`): the boundary prints a plain
    diagnostic through :class:`~calibrated_agent_evaluation.cli.present.Presenter` and exits; no
    traceback reaches the user. A ``RunGateDecision`` that fails the CI gate also exits non-zero
    here.
  * **2**: a usage error (a missing, malformed or out-of-range argument, or flags that do not
    go together), raised by the parser or as a
    :class:`~calibrated_agent_evaluation.cli.errors.UsageError` by a stage.
  * **3**: ``calibrated-agent-evaluation run --fail-on-drift`` found input drift or a stale judge
    while the gate itself passed (or was not asked to fail the process). A failing gate under
    ``--ci`` stays **1**.

Handlers are looked up in :data:`_HANDLERS`. Each reuses its feature module; a
handler that a stage hasn't wired yet raises a typed ``CliError`` (still exit 1, still no
traceback), never a bare ``NotImplementedError`` traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.cli.errors import DOMAIN_ERRORS, UsageError
from calibrated_agent_evaluation.cli.parser import (
    META_COMMANDS,
    STAGES,
    TOOL_COMMANDS,
    TRUST_COMMANDS,
    parse_args,
)
from calibrated_agent_evaluation.cli.plain import default_error_explainers
from calibrated_agent_evaluation.cli.present import Presenter

_COMMANDS: tuple[str, ...] = STAGES + TRUST_COMMANDS + META_COMMANDS + TOOL_COMMANDS

# A stage handler takes the parsed args and returns a StageOutcome. Handlers live in cli.stages.*
# and are imported lazily inside main() so importing calibrated_agent_evaluation.cli never pulls a
# stage's optional deps at module load.
Handler = Callable[[argparse.Namespace], "StageOutcome"]

EXIT_OK = 0
EXIT_DOMAIN_ERROR = 1
EXIT_USAGE = 2
EXIT_DRIFT = 3


@dataclass(frozen=True)
class StageOutcome:
    """Result of one ``calibrated-agent-evaluation <stage>`` dispatch.

    ``exit_code`` is the process code the CLI returns. ``summary`` is the plain-language message
    printed on success (to stdout) or on a handled failure (to stderr). ``detail`` is the
    statistical account behind it, printed only under ``--explain``. A non-zero ``exit_code``
    from a handler (e.g. a failing CI gate) is honored as-is.
    """

    stage: str
    exit_code: int
    summary: str
    detail: str = ""


def _load_handler(stage: str) -> Handler:
    """Lazily import the handler for ``stage`` (keeps optional deps off the import path)."""
    from calibrated_agent_evaluation.cli import handlers

    return handlers.for_stage(stage)


def _explain_from_env() -> bool:
    """Whether the environment turns ``--explain`` on for every command, e.g. for full CI logs.

    ``CALIBRATED_AGENT_EVALUATION_EXPLAIN=1`` does.
    """
    return os.environ.get("CALIBRATED_AGENT_EVALUATION_EXPLAIN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tolerate_unencodable_output() -> None:
    """Escape what the terminal's encoding cannot show instead of crashing on it.

    Under a non-UTF-8 locale (``LC_ALL=C``, Latin-1) printing an arrow or a non-ASCII trace id
    raised ``UnicodeEncodeError`` after the stage had already done its work.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``calibrated-agent-evaluation``. Returns a process exit code."""
    _tolerate_unencodable_output()
    args = parse_args(argv)

    command: str | None = getattr(args, "command", None)
    if command is None:
        # No subcommand: print help and exit 0 (a bare `calibrated-agent-evaluation` is not an
        # error).
        from calibrated_agent_evaluation.cli.parser import build_parser

        build_parser().print_help()
        return EXIT_OK

    if command not in _COMMANDS:  # pragma: no cover - argparse restricts this, defensive only
        print(f"calibrated-agent-evaluation: unknown command {command!r}", file=sys.stderr)
        return EXIT_USAGE

    presenter = Presenter(
        out=sys.stdout,
        err=sys.stderr,
        explain=bool(getattr(args, "explain", False)) or _explain_from_env(),
        explainers=default_error_explainers(),
    )
    try:
        handler = _load_handler(command)
        outcome = handler(args)
    except UsageError as exc:
        presenter.error(command, exc)
        return EXIT_USAGE
    except DOMAIN_ERRORS as exc:
        # The one place the CLI catches the (non-uniform) typed-error family: a plain diagnostic,
        # no traceback. Anything outside DOMAIN_ERRORS is a genuine bug; it propagates.
        presenter.error(command, exc)
        return EXIT_DOMAIN_ERROR
    except ModuleNotFoundError as exc:
        # The backstop for a stage that reached an optional engine before checking for it: a
        # module an extra provides becomes that extra's install command. Any other missing
        # module is a bug and propagates.
        missing = _missing_extra(command, exc)
        if missing is None:
            raise
        presenter.error(command, missing)
        return EXIT_DOMAIN_ERROR

    presenter.outcome(command, outcome, ok=outcome.exit_code == EXIT_OK)
    return outcome.exit_code


def _missing_extra(command: str, exc: ModuleNotFoundError) -> BaseException | None:
    from calibrated_agent_evaluation._errors import MissingExtraError
    from calibrated_agent_evaluation.cli.config import extra_for_missing_module

    extra = extra_for_missing_module(exc.name)
    if extra is None:
        return None
    return MissingExtraError((extra,), f"`calibrated-agent-evaluation {command}`")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
