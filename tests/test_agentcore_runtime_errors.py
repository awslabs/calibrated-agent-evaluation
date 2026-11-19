# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime executor's error belongs to both typed families, and the package stays light."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from calibrated_agent_evaluation.agentcore import AgentCoreError, RuntimeInvocationError
from calibrated_agent_evaluation.execute import ExecutionError


@pytest.mark.parametrize("family", [ExecutionError, AgentCoreError])
def test_a_runtime_invocation_error_is_caught_by_the_execution_and_agentcore_families(
    family: type[Exception],
) -> None:
    with pytest.raises(family) as raised:
        raise RuntimeInvocationError("the runtime refused the session")

    assert type(raised.value) is RuntimeInvocationError
    assert str(raised.value) == "the runtime refused the session"


def test_the_execution_plane_loads_only_when_the_runtime_error_is_first_asked_for() -> None:
    code = """
        import sys

        def execution_plane_loaded():
            prefix = ["calibrated_agent_evaluation", "execute"]
            return any(m.split(".")[:2] == prefix for m in sys.modules)

        import calibrated_agent_evaluation.agentcore
        print("before:", execution_plane_loaded())
        calibrated_agent_evaluation.agentcore.RuntimeInvocationError
        print("after:", execution_plane_loaded())
        """

    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=True
    ).stdout

    assert "before: False" in out
    assert "after: True" in out
