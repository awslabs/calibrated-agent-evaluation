# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""ExecutorBackend protocol: the execution plane's public contract.

Defined exclusively against normalized schema types (``Trace``) and Python stdlib types. No
Inspect, boto3, or other vendor type appears in the protocol's method signatures. Concrete
implementations (``InspectExecutor``) structurally satisfy this protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from calibrated_agent_evaluation.schema.types import Trace


@runtime_checkable
class ExecutorBackend(Protocol):
    """Run agents and return normalized, validated Traces.

    The methodology plane programs against this protocol; it never sees Inspect or boto3 types.
    """

    def run(
        self,
        task: Any,
        *,
        log_dir: Path | None = None,
    ) -> list[Trace]:
        """Execute an agent task and return normalized Traces.

        Raises ``ExecutionError`` on any failure (eval error, cassette miss, unreachable model,
        invalid trace from normalization). Never returns an unvalidated Trace.
        """
        ...
