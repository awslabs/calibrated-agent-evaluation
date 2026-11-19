# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup reachability probe: an unreachable backend opens no session."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from calibrated_agent_evaluation.schema import BackendError


def test_unreachable_backend_raises_and_opens_no_session(
    make_client: Callable[..., Any], make_backend: Callable[..., Any]
) -> None:
    backend = make_backend(traces=[], query_error=BackendError("backend unreachable"))
    with pytest.raises(BackendError):
        make_client(backend)  # build_app probes via query() before opening a session
