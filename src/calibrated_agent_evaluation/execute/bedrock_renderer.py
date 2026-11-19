# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BedrockRenderer``: the concrete LLM tuple→NL renderer deferred to the caller.

Satisfies the ``Renderer`` protocol (``render(cell) -> str``). Renders a taxonomy cell into a
realistic natural-language input via the shipped Inspect model seam
(``InspectJudgeModel.generate_text``; the plain-text path, NOT the verdict/structured path). It is
vendor-isolated in ``execute/`` (the vendor-adapter exemption package), opt-in behind the
``[bedrock]`` extra;
the default ``calibrated-agent-evaluation generate`` uses the offline template renderer instead.

An unreachable/empty/error model response raises
:class:`~calibrated_agent_evaluation.synth.errors.SynthError` (chained from the vendor exception);
the renderer NEVER fabricates text or returns an empty input (an empty render is refused by
``synth.generate`` too).
"""

from __future__ import annotations

from calibrated_agent_evaluation.execute.judge_model import InspectJudgeModel
from calibrated_agent_evaluation.synth.errors import SynthError

_SYSTEM = (
    "You write a single realistic end-user input for an AI agent under evaluation. "
    "Return ONLY the user's message text — no preamble, no quotes, no explanation."
)


class BedrockRenderer:
    """Concrete ``Renderer`` on Bedrock via the Inspect model seam.

    ``model`` is an Inspect model string (e.g. ``"bedrock/global.anthropic.claude-sonnet-4-6"``).
    Construction is cheap (no network); the model call happens in :meth:`render`.
    """

    def __init__(self, model: str, *, timeout_seconds: float = 60.0) -> None:
        self._model = InspectJudgeModel(model, timeout_seconds=timeout_seconds)

    def render(self, cell: dict[str, str]) -> str:
        """Render one taxonomy cell into a non-empty NL input.

        Raises :class:`SynthError` if the model is unreachable/errors (wrapping the vendor
        exception) or returns empty/whitespace text; never fabricates or returns an empty input.
        """
        prompt = self._prompt_for(cell)
        try:
            text = self._model.generate_text(system=_SYSTEM, user=prompt)
        except Exception as exc:
            # Wrap the vendor/execution exception into the synth error family (no raw escape).
            raise SynthError(
                f"LLM renderer failed for cell {cell}: {type(exc).__name__}: {exc}"
            ) from exc
        if not text or not text.strip():
            raise SynthError(
                f"LLM renderer returned empty text for cell {cell}; refusing to fabricate an input"
            )
        return text.strip()

    @staticmethod
    def _prompt_for(cell: dict[str, str]) -> str:
        """Build the user prompt describing the taxonomy cell to realize."""
        dims = "; ".join(f"{k}: {v}" for k, v in sorted(cell.items()))
        return f"Write one input that exercises these dimensions — {dims}."
