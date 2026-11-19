# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a JSON fixture stored as ``<name>.json``, or gzipped as ``<name>.json.gz`` when it is large.

The large fixtures are kept gzipped so a code review diff stays small; callers name the plain
``.json`` path either way.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """The parsed JSON at ``path``, or at ``path`` with ``.gz`` added when only that copy exists."""
    gzipped = path.with_name(path.name + ".gz")
    if not path.exists() and gzipped.exists():
        return json.loads(gzip.decompress(gzipped.read_bytes()))
    return json.loads(path.read_text(encoding="utf-8"))
