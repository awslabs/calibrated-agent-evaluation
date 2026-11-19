# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for Calibrated Agent Evaluation (``calibrated-agent-evaluation``).

The ``calibrated-agent-evaluation`` CLI runs the whole
``probe → define → generate → label → cluster → judge → run → improve`` pipeline (plus
``dump``/``load``) over the code each feature already ships; it reuses the methodology plane, it
never re-implements it. The package layout:

  * ``cli.parser``  ; the argparse subcommand surface (all ten stages).
  * ``cli.config``  ; backend/model/extra resolution + graceful degradation.
  * ``cli.dispatch``: maps a subcommand to its feature call; owns exit codes + the
    top-level typed-error boundary.
  * ``cli.errors``  ; ``CliError`` + the (non-uniform) error-boundary family tuple.
  * ``cli.stages.*``: one thin module per stage that needs more than a direct dispatch line.

The default install carries one runtime dependency, pydantic; every stage
that needs an optional extra degrades with a typed install-hint rather than an ``ImportError``.
The console entry point is ``calibrated-agent-evaluation = calibrated_agent_evaluation.cli:main``.
"""

from __future__ import annotations

from calibrated_agent_evaluation.cli.dispatch import main

__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
