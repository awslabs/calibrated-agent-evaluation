# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""BedrockRenderer tests.

The renderer satisfies the ``Renderer`` protocol by calling the ``generate_text`` seam.
Tests run against a stub that mirrors the real ``InspectJudgeModel.generate_text(*, system, user)``
signature and assert on the round-tripped behavior (a non-empty NL string; typed refusals) — never
on calls into the stub. The live check against real Bedrock is the opt-in integration test.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.execute.bedrock_renderer import BedrockRenderer
from calibrated_agent_evaluation.synth.errors import SynthError
from calibrated_agent_evaluation.synth.types import Renderer


class _StubModel:
    """Mirrors InspectJudgeModel.generate_text — returns a scripted completion or raises."""

    def __init__(self, *, returns: str | None = None, raises: Exception | None = None) -> None:
        self._returns = returns
        self._raises = raises

    def generate_text(self, *, system: str, user: str) -> str:
        if self._raises is not None:
            raise self._raises
        assert system  # the renderer must pass a real system prompt
        assert user  # ...and a real user prompt
        return self._returns if self._returns is not None else ""


def _renderer_with(model: _StubModel) -> BedrockRenderer:
    r = BedrockRenderer.__new__(BedrockRenderer)
    r._model = model  # type: ignore[attr-defined]
    return r


def test_render_returns_nonempty_nl_for_valid_cell():
    """Render(cell) returns a non-empty NL string for a valid taxonomy cell."""
    renderer = _renderer_with(_StubModel(returns="Please refund my out-of-window return."))

    text = renderer.render({"feature": "refund", "persona": "consumer"})

    assert text == "Please refund my out-of-window return."
    assert text.strip()


def test_satisfies_the_renderer_protocol():
    """BedrockRenderer structurally satisfies the Renderer protocol (contract)."""
    renderer = _renderer_with(_StubModel(returns="x"))
    assert isinstance(renderer, Renderer)


def test_whitespace_output_is_refused():
    """Negative: whitespace-only model output → SynthError, no fabricated input."""
    renderer = _renderer_with(_StubModel(returns="   \n  "))

    with pytest.raises(SynthError, match="empty text.*refusing to fabricate"):
        renderer.render({"feature": "refund"})


def test_unreachable_model_raises_syntherror_wrapping_vendor_exc():
    """Negative: an unreachable/erroring model → SynthError wrapping the vendor exception."""
    boom = RuntimeError("bedrock unreachable")
    renderer = _renderer_with(_StubModel(raises=boom))

    with pytest.raises(SynthError) as ei:
        renderer.render({"feature": "refund"})

    assert ei.value.__cause__ is boom  # chained from the vendor exception, no raw escape
