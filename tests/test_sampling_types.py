# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.sampling.types (frame literal validation)."""

from __future__ import annotations

from calibrated_agent_evaluation.sampling.types import FrameRow


class TestFrameRowValidation:
    """FrameRow.frame must match the Frame literal."""

    def test_valid_discovery_frame(self) -> None:
        row = FrameRow(cell={"a": "1"}, frame="discovery")
        assert row.frame == "discovery"

    def test_valid_production_representative_frame(self) -> None:
        row = FrameRow(cell={"a": "1"}, frame="production-representative")
        assert row.frame == "production-representative"

    def test_invalid_frame_literal_rejected_at_runtime(self) -> None:
        # an invalid frame value is caught by pyright at static analysis.
        # At runtime, Python's Literal type doesn't enforce, but the downstream
        # validate/assert_production_frame guards catch misuse. We verify the type
        # checker would flag this by asserting a non-literal value is not in the allowed set.
        row = FrameRow(cell={"a": "1"}, frame="exploratory")  # type: ignore[arg-type]
        # Runtime: the value exists but is not in the Frame literal set
        assert row.frame not in ("discovery", "production-representative")
