# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The root of the library's typed error hierarchy.

It lives outside :mod:`calibrated_agent_evaluation.schema` so that the model base in
:mod:`calibrated_agent_evaluation._model`, which the schema types themselves build on, can name it
without importing the schema package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibrated_agent_evaluation._install import InstallSource

__all__ = ["CalibratedAgentEvaluationDataError", "MissingExtraError"]


class CalibratedAgentEvaluationDataError(Exception):
    """Base class for every data-plane error."""


class MissingExtraError(CalibratedAgentEvaluationDataError):
    """A capability was asked for whose optional extra is not installed.

    ``hint`` is the one command that installs ``extras`` into this installation, matched to how
    Calibrated Agent Evaluation was installed (checkout, editable, uv tool, wheel or index).
    """

    def __init__(
        self, extras: tuple[str, ...], purpose: str, *, source: InstallSource | None = None
    ) -> None:
        # Imported here: _install builds on pydantic, and this module must stay importable first.
        from calibrated_agent_evaluation._install import _running_source, hint_for

        self.extras = extras
        self.hint = hint_for(source or _running_source(), extras)
        names = ", ".join(f"[{e}]" for e in extras)
        super().__init__(f"{purpose} needs the {names} extra; install it with: {self.hint}")


# Public home is calibrated_agent_evaluation.schema.errors, which re-exports them; tracebacks and
# the reference say so.
CalibratedAgentEvaluationDataError.__module__ = "calibrated_agent_evaluation.schema.errors"
MissingExtraError.__module__ = "calibrated_agent_evaluation.schema.errors"
