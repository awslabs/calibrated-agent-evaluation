# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nox sessions for Calibrated Agent Evaluation.

- ``nox -s tests``     runs the suite across supported Python versions.
- ``nox -s lint``      runs Ruff (lint + format check).
- ``nox -s typecheck`` runs Pyright.

Environments are created and populated with uv. Running the full ``tests`` matrix
downloads any missing CPython interpreters via uv on first use.
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "tests"]

PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]


def _sync_dev(session: nox.Session) -> None:
    """Install what ``just install`` does, so a session runs the whole suite."""
    session.run_install(
        "uv",
        "sync",
        "--all-extras",
        "--group",
        "dev",
        "--group",
        "corpus",
        "--active",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite with coverage."""
    _sync_dev(session)
    session.run("pytest", "--cov", "--cov-report=term-missing")


@nox.session(python="3.13")
def lint(session: nox.Session) -> None:
    """Check lint rules and formatting with Ruff."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.13")
def typecheck(session: nox.Session) -> None:
    """Static type-check with Pyright."""
    _sync_dev(session)
    session.run("pyright")
