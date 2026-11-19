# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real AgentCore runtime trace, as the span reader decoded it, for tests that need one.

The trace is the first of the recorded online-evaluation window under
``fixtures/raw/agentcore/online/``: one OrderBot run on a deployed runtime, with its ``aws/spans``
records, its ``otel-rt-logs`` content records and its session ids.
"""

from __future__ import annotations

import copy
import functools
import gzip
import json
from pathlib import Path
from typing import Any

ONLINE_TRACES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "agentcore" / "online"
) / "traces.json.gz"

#: The user's request in the captured run, as its content records carry it.
CAPTURED_QUESTION = "Requesting a refund on Q-30187 (Over-ear headphones). It stopped working."


@functools.cache
def _recorded() -> list[dict[str, Any]]:
    return json.loads(gzip.decompress(ONLINE_TRACES.read_bytes()))


def captured_run() -> dict[str, Any]:
    """A fresh copy of the captured run: its trace id, span and content records, and sessions."""
    return copy.deepcopy(_recorded()[0])
