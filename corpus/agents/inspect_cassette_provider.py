# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cassette-backed Inspect model provider — re-exports from the canonical shipped location.

The provider was promoted to ``src/calibrated_agent_evaluation/execute/cassette_provider.py`` (a
review fix to the executor) so that ``model="cassette/..."`` replay works for pip-installed users
who have no ``corpus/`` tree. This module re-exports the full public surface so existing corpus
scripts (``corpus/generate/record.py``, ``corpus/agents/orderbot_inspect.py``) continue to work
unchanged.

Importing this module still triggers the ``@modelapi(name="cassette")`` registration side-effect
(it runs at import time in the canonical module), so the ``import ... # noqa: F401`` idiom the
corpus scripts rely on keeps registering the provider.
"""

from calibrated_agent_evaluation.execute.cassette_provider import (
    DEFAULT_INNER_MODEL,
    CassetteModelAPI,
    cassette_provider,
    configure,
)

__all__ = [
    "DEFAULT_INNER_MODEL",
    "CassetteModelAPI",
    "cassette_provider",
    "configure",
]
