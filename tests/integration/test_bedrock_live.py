# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live checks of the concrete Bedrock renderer + Titan embedder.

Marked ``integration`` (not run in the default hermetic suite): these bind the REAL
``BedrockRenderer`` and ``BedrockEmbedder`` to live Bedrock to keep the stub-based unit tests honest
against SDK/service drift — the both-worlds discipline (a stub alone can't prove the real surface).
They skip cleanly when the ``[bedrock]`` extra is absent or no AWS credentials are configured. Run
with ``AWS_PROFILE=<your-profile> just test-integration -k bedrock_live``.
"""

from __future__ import annotations

import os

import pytest
from _bedrock import inspect_bedrock_model

pytestmark = pytest.mark.integration


def _require_bedrock() -> None:
    """Skip unless the extra is installed and AWS credentials are present."""
    pytest.importorskip("boto3")
    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        pytest.skip(
            "no AWS credentials configured (set AWS_PROFILE to a profile with Bedrock access)"
        )


def test_real_bedrock_renderer_renders_nonempty_nl() -> None:
    """The real BedrockRenderer renders a non-empty NL input for a corpus cell."""
    _require_bedrock()
    from calibrated_agent_evaluation.execute.bedrock_renderer import BedrockRenderer

    renderer = BedrockRenderer(model=inspect_bedrock_model())
    text = renderer.render(
        {"feature": "refund", "scenario": "out_of_window", "persona": "consumer"}
    )

    assert isinstance(text, str)
    assert text.strip()
    # The output is the model's NL, not the cell echoed back verbatim.
    assert text != "refund"


def test_real_titan_embedder_returns_ordered_1to1_vectors() -> None:
    """The real Titan embedder returns exactly one vector per input, in order."""
    _require_bedrock()
    from calibrated_agent_evaluation.execute.bedrock_embedder import BedrockEmbedder

    texts = [
        "The agent over-refunded an out-of-window return.",
        "The agent correctly denied an out-of-window return.",
    ]
    vectors = BedrockEmbedder().embed(texts)

    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) and v for v in vectors)
    # Distinct inputs yield distinct embeddings (a sanity check the mapping is real, not constant).
    assert vectors[0] != vectors[1]
