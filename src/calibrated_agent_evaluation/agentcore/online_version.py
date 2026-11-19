# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema version for the AgentCore online-evaluation records.

Isolated in its own module (like ``bias/version.py``) so the records can pin the version without
an import cycle.
"""

from __future__ import annotations

__all__ = ["ONLINE_SCHEMA_VERSION"]

ONLINE_SCHEMA_VERSION = "1"
"""Stamped on every online-evaluation record and session log; a load of another one is refused."""
