# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Show what a ``calibrated-agent-evaluation`` command prints when an optional extra it needs is
not installed.

Hides the extra's packages from the import system (as if they were never installed), then runs
the command in-process. The docs build has every extra installed, so this is how the missing-extra
messages are captured. Run from the repository root::

    uv run --all-extras python docs/site/examples/guides/missing_extra.py stats run
"""

from __future__ import annotations

import sys

HIDDEN = {
    "stats": ("scipy", "statsmodels"),
    "ui": ("fasthtml", "uvicorn"),
    "langfuse": ("langfuse",),
    "agentcore": ("boto3", "botocore", "bedrock_agentcore", "strands_evals"),
}


def main(extra: str, argv: list[str]) -> int:
    for module in HIDDEN[extra]:
        sys.modules[module] = None  # type: ignore[assignment]  # a None entry makes imports fail
    from calibrated_agent_evaluation.cli import main

    print(f"$ calibrated-agent-evaluation {' '.join(argv)}    # without the {extra} extra")
    sys.stdout.flush()
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
