# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Show which endpoint a judge file's ``model`` string resolves to, without calling the model.

Loads the judge file the way ``calibrated-agent-evaluation judge`` does, then asks Inspect which
provider and base URL that model string selects. Nothing is sent to the endpoint. Run from the
repository root with the ``inspect`` extra (and ``openai``, which Inspect's OpenAI-compatible
providers use)::

    uv run --extra inspect python docs/site/examples/guides/local_judge_endpoint.py \\
        docs/site/examples/guides/local-judge.json
"""

from __future__ import annotations

import sys

from inspect_ai.model import get_model

from calibrated_agent_evaluation.cli.judgefile import load_judge_file

judge_file = load_judge_file(sys.argv[1])
model = judge_file.judge.model
print(f"judge {judge_file.judge.judge_id!r} uses model {model!r}")
api = get_model(model).api
print(f"  provider {type(api).__name__}, endpoint {api.base_url}")
