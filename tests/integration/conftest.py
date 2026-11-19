# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the repo-root ``corpus`` package importable for integration tests.

Same pattern as ``tests/corpus/conftest.py`` — ``corpus/`` lives at the repo root, not under
``src/``, so it is not on ``sys.path`` by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
