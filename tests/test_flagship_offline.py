# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flagship cassette-replay path completes fully offline.

The hermetic CI proof (cassette-replay + local backend) runs both agents end-to-end with socket
creation hard-disabled -- no model-provider call, no backend network. This is the state-driven
negative: while in cassette mode with the network disabled, the proof still completes.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.flagship import FlagshipConfig, World, prove

_REPO = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO / "fixtures"


def _backend_factory(tmp_path: Path):
    def factory(_world: World) -> TraceStoreBackend:
        return get_backend(None, local_root=tmp_path / "backend")

    return factory


def test_cassette_proof_completes_with_network_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cassette + local proof runs with socket creation disabled."""
    from _flagship_helpers import orderbot_plan

    def _no_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the cassette-replay proof must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_network)

    config = FlagshipConfig(
        agents=("orderbot",),
        worlds=("local",),
        model_plane="cassette",
        fixtures_root=str(_FIXTURES),
    )
    report = prove(config, [orderbot_plan()], backend_factory=_backend_factory(tmp_path))
    assert report.verdict == "proven", report.reasons
