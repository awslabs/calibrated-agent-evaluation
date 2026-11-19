# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live test: a judge reads one prompt for a trace in either world.

Langfuse returns a whole-number float such as ``149.0`` as ``149``. The traces compare equal, but
the judge prompt used to differ by world, so the committed OrderBot cassette missed on a trace
read back from Langfuse and a live judge saw different text for the same trace. Here a fresh copy
of ORD-1002 (whose amounts are ``149.0``) is written to the dev stack, read back, and rendered:
the prompt, the cassette fingerprint and the replayed verdict match the local read.

Skipped unless the stack is up: ``just langfuse-up`` then ``just test-integration``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest

from _helpers import langfuse_reachable
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.execute import CassetteJudgeModel
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode, fingerprint
from calibrated_agent_evaluation.judges import (
    ExecutableJudge,
    build_verdict_schema,
    render_span,
    render_trace,
    workflow_target,
)
from calibrated_agent_evaluation.schema.serde import trace_from_dict
from calibrated_agent_evaluation.schema.types import Trace

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
_CASSETTE = _FIXTURES / "raw" / "cassettes" / "judge_orderbot_resolved.cassette.json"
_PROMPT = (
    "You are grading whether an OrderBot agent resolved a customer's refund request correctly, "
    "per the store's return policy. A refund is correct only if the return is within the stated "
    "return window. Write your critique first (cite the policy and the facts), then give a binary "
    "verdict: 'pass' if the agent handled the refund correctly, 'fail' if it did not."
)


@pytest.fixture
def langfuse() -> LangfuseBackend:
    pytest.importorskip("langfuse")
    try:
        backend = LangfuseBackend.from_credentials(
            public_key=os.environ.get(
                "LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"
            ),
            secret_key=os.environ.get(
                "LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"
            ),
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
    except Exception as exc:
        pytest.skip(f"no Langfuse backend ({exc})")
    if not langfuse_reachable(backend._client):  # type: ignore[attr-defined]
        pytest.skip("no reachable Langfuse (run `just langfuse-up`)")
    return backend


def _fresh(trace_id: str) -> Trace:
    source = trace_from_dict(
        json.loads((_FIXTURES / "normalized" / f"{trace_id}.json").read_text(encoding="utf-8"))
    )
    new_id = f"RND-{uuid.uuid4().hex[:10]}"
    spans = [
        replace(s, trace_id=new_id, span_id=s.span_id.replace(trace_id, new_id, 1))
        for s in source.spans
    ]
    return replace(source, trace_id=new_id, spans=spans)


def _read_back(backend: LangfuseBackend, trace: Trace) -> Trace:
    deadline = time.monotonic() + 60
    while True:
        try:
            got = backend.get_trace(trace.trace_id)
            if len(got.spans) == len(trace.spans):
                return got
        except Exception:
            if time.monotonic() > deadline:
                raise
        if time.monotonic() > deadline:
            pytest.fail(f"{trace.trace_id} did not come back whole within 60s")
        time.sleep(1)


def _judge() -> ExecutableJudge:
    return ExecutableJudge(
        judge_id="orderbot-refund-resolved",
        model="bedrock/global.anthropic.claude-sonnet-4-6",
        prompt_version="demo-v1",
        level="workflow",
        prompt_template=_PROMPT,
    )


def _request(content: str) -> dict[str, object]:
    return {
        "system": _PROMPT,
        "user": content,
        "schema": build_verdict_schema(_judge()),
        "via": "response_schema",
    }


def test_a_trace_renders_and_replays_the_same_from_either_world(langfuse, tmp_path) -> None:  # type: ignore[no-untyped-def]
    trace = _fresh("ORD-1002")
    with LocalBackend(root=tmp_path / "local") as local:
        local.put_trace(trace)
        from_local = local.get_trace(trace.trace_id)
    with langfuse as lf:
        lf.put_trace(trace)
        from_langfuse = _read_back(lf, trace)

    # The defect's precondition, so this test cannot pass vacuously: the worlds disagree on type.
    lf_lookup = from_langfuse.spans[0].tool_calls[0].result
    local_lookup = from_local.spans[0].tool_calls[0].result
    assert isinstance(local_lookup, dict)
    assert isinstance(lf_lookup, dict)
    assert repr(local_lookup["amount_usd"]) == "149.0"
    assert repr(lf_lookup["amount_usd"]) == "149"

    assert render_trace(from_langfuse) == render_trace(from_local)
    span = from_langfuse.spans[0]
    assert render_span(from_langfuse, span) == render_span(from_local, from_local.spans[0])
    assert "'amount_usd': 149," in render_trace(from_langfuse)

    request_lf = _request(render_trace(from_langfuse))
    assert fingerprint(request_lf) == fingerprint(_request(render_trace(from_local)))

    # The committed cassette holds the verdict for this trace, so a Langfuse read replays it.
    model = CassetteJudgeModel(Cassette(_CASSETTE, mode=Mode.REPLAY), supports_schema=True)
    target = workflow_target(from_langfuse)
    reply = model.generate_structured(
        system=_PROMPT, user=target.content, schema=build_verdict_schema(_judge())
    )
    assert reply.fields["verdict"] == "pass"
