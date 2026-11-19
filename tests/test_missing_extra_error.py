# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A missing optional extra is a typed error that carries the command to install it."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation._install import InstallSource, install_hint
from calibrated_agent_evaluation.schema.errors import (
    CalibratedAgentEvaluationDataError,
    MissingExtraError,
)


def test_names_the_extras_and_the_install_command_for_this_installation() -> None:
    err = MissingExtraError(("agentcore",), "AgentCore integration")

    assert isinstance(err, CalibratedAgentEvaluationDataError)
    assert err.extras == ("agentcore",)
    assert err.hint == install_hint("agentcore")
    assert "[agentcore]" in str(err)
    assert err.hint in str(err)


def test_the_hint_follows_the_install_kind() -> None:
    tool = InstallSource(
        kind="tool", path="/opt/calibrated_agent_evaluation", tool_extras=("stats",)
    )

    err = MissingExtraError(("agentcore",), "AgentCore integration", source=tool)

    assert err.hint.startswith("uv tool install ")
    assert "stats,agentcore" in err.hint
    assert not err.hint.startswith("uv sync")


def test_catchable_as_the_package_error_family() -> None:
    with pytest.raises(
        CalibratedAgentEvaluationDataError, match=r"needs the \[synth\], \[agentcore\] extra"
    ):
        raise MissingExtraError(("synth", "agentcore"), "Simulated AgentCore sessions")
