# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the repo-root ``corpus`` package importable for the corpus tests.

``corpus/`` is the corpus-generation tree (companion spec); it lives at the repo root, not under
``src/``, so it is not on ``sys.path`` by default when pytest collects from ``tests/``. The corpus
integrity tests import it (to replay the capture harness offline), so we prepend the repo root here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
