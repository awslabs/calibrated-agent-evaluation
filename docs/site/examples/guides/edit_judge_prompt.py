# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy a judge file with one sentence added to its prompt and ``prompt_version`` left unchanged.

The edit a reviewer might make in place, without bumping the version: the certified record no
longer matches the judge's fingerprint, so the scoring stages refuse it.

    python docs/site/examples/guides/edit_judge_prompt.py JUDGE.json OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

source, target = Path(sys.argv[1]), Path(sys.argv[2])
judge = json.loads(source.read_text(encoding="utf-8"))
judge["prompt_template"] += " Be strict about the return window."
target.write_text(json.dumps(judge, indent=2) + "\n", encoding="utf-8")
