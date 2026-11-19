# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail the docs build when a captured example no longer matches what the code prints.

Run by the mkdocs ``gen-files`` plugin. Regenerates every fragment in ``docs_examples.py`` and
compares it with the golden that pages include; any difference stops the build with the diff and
the command that refreshes the goldens. Set ``CALIBRATED_AGENT_EVALUATION_DOCS_SKIP_EXAMPLES=1`` to
skip the check (``just docs`` does, so live reload stays fast).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from docs_examples import FragmentError, diff_against_goldens, produce_all

if os.environ.get("CALIBRATED_AGENT_EVALUATION_DOCS_SKIP_EXAMPLES") != "1":
    try:
        problems = diff_against_goldens(produce_all())
    except FragmentError as exc:
        raise SystemExit(f"gen_examples: {exc}") from exc
    if problems:
        raise SystemExit(
            "gen_examples: captured example output changed; review the diff, then run "
            "`just docs-goldens`:\n\n" + "\n".join(problems)
        )
