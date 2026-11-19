#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the AgentCore Runtime executor's offline fixtures, or a live run to a local directory.

Without ``--live`` the script makes no AWS call. It runs ``AgentCoreRuntimeExecutor`` in record
mode against a stubbed runtime and a fake span reader, and writes the three committed fixtures:

- ``fixtures/raw/cassettes/agentcore_runtime_CLM-4471.cassette.json``: two ``agentcore_sse``
  entries, one for CLM-4471's ``input.narrative`` and one for the prompt
  ``calibrated-agent-evaluation generate``'s offline template renderer writes for CLM-4471's
  taxonomy cell;
- ``fixtures/raw/frames/agentcore_CLM-4471.frame.json``: a one-row discovery frame for that cell,
  in the form ``calibrated-agent-evaluation define --out`` writes;
- ``fixtures/expected/agentcore_runtime/CLM-4471.trace.json``: the narrative run's trace, as the
  recording run returned it, with CLM-4471's taxonomy cell and the executor's default frame
  (``production-representative``). The cassette stores neither the frame nor the cell, so a
  replay compared with it must use the same ones.

Provenance: these are synthetic recordings, made offline.

- The runtime is a botocore ``Stubber`` on real ``bedrock-agentcore`` and
  ``bedrock-agentcore-control`` clients built from the installed service models with fake
  credentials (``tests/_agentcore_runtime_fakes.py``). It is named by the fixed synthetic ARN
  ``arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/calibrated_agent_evaluation_fixture-AbCdEf1234``
  (111122223333 is AWS's documentation account), with the qualifier ``DEFAULT`` and no
  configuration bundle. Its preflight reports endpoint ``DEFAULT`` serving runtime version 1, with
  the HTTP protocol and no custom JWT authorizer.
- Each invocation answers 200 with an event stream in the form the AgentCore SDK writes, one
  ``data: "<text>"`` event per line of CLM-4471's recorded final answer
  (``fixtures/normalized/CLM-4471.json``, ``output.answer``), and no ``traceparent``. Every stop
  succeeds.
- The span tree is CLM-4471's five-agent ClaimsCrew tree from the same file, re-keyed to each
  run's trace ID by the fake span reader.
- The executor makes each run's session ID and W3C trace context itself, so every re-recording
  carries new ones. Its clock is a synthetic one that starts at 2026-10-26T12:00:00Z and moves on
  one second each time it is read.

With ``--live`` the script records real runs of a deployed runtime into a directory the caller
names, which must be outside this repository: a live recording carries the real runtime ARN and
account ID, so it is never committed. It writes two things there and nothing else: the cassette
``agentcore_runtime_live.cassette.json``, and the run manifest pair ``agentcore_run_record.json``
and ``agentcore_run_record.csv``, which hold one record per run. It needs the AgentCore span
reader (``calibrated_agent_evaluation.agentcore.AgentCoreSpanReader``). Invocations are billed, so
it prints what it will call and refuses to call it without ``--yes``.
``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`` caps the billed calls, the reader's
included.

Usage:
    uv run python scripts/record_agentcore_cassette.py
    uv run python scripts/record_agentcore_cassette.py --live --runtime-arn ARN \\
        --payload-format NAME [--qualifier Q] [--profile P] --prompt TEXT [--prompt TEXT ...] \\
        --out DIR --yes
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.agentcore import (
    AgentCoreRuntimeExecutor,
    AgentCoreRuntimeTask,
    AwsClients,
    BilledCallBudget,
    RuntimeInvocationError,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.run_manifest import manifest_path
from calibrated_agent_evaluation.agentcore.runtime import PAYLOAD_FORMAT_NAMES
from calibrated_agent_evaluation.execute import Mode
from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Trace

REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS = REPO_ROOT / "tests"
_FIXTURES = REPO_ROOT / "fixtures"

FIXTURE_RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234"
)
CASSETTE = _FIXTURES / "raw" / "cassettes" / "agentcore_runtime_CLM-4471.cassette.json"
FRAME = _FIXTURES / "raw" / "frames" / "agentcore_CLM-4471.frame.json"
EXPECTED_TRACE = _FIXTURES / "expected" / "agentcore_runtime" / "CLM-4471.trace.json"

LIVE_CASSETTE_NAME = "agentcore_runtime_live.cassette.json"

_CLAIM = _FIXTURES / "normalized" / "CLM-4471.json"
_RUNTIME_VERSION = "1"
_ROLE_ARN = "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-fixture-runtime"
_CLOCK_START = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_FORMATS = tuple(name for name in PAYLOAD_FORMAT_NAMES if name != "custom")


class _SyntheticClock:
    """A UTC clock that starts at a fixed time and moves on one second each time it is read."""

    def __init__(self) -> None:
        self._now = _CLOCK_START

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


def _no_wait(_seconds: float) -> None:
    return None


def _json(document: Any) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


# --- the synthetic fixtures -----------------------------------------------------------------


def _event_stream(answer: str) -> bytes:
    # The AgentCore SDK writes each streamed value as one ``data:`` event of its JSON text.
    chunks = answer.splitlines(keepends=True)
    return "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks).encode(
        "utf-8"
    )


def _queue_preflight(clients: Any, target: RuntimeTarget) -> None:
    control = clients.stub("bedrock-agentcore-control")
    control.add_response(
        "get_agent_runtime_endpoint",
        {
            "agentRuntimeEndpointArn": f"{target.runtime_arn}/runtime-endpoint/{target.qualifier}",
            "agentRuntimeArn": target.runtime_arn,
            "status": "READY",
            "createdAt": _CLOCK_START,
            "lastUpdatedAt": _CLOCK_START,
            "name": target.qualifier,
            "id": target.qualifier,
            "liveVersion": _RUNTIME_VERSION,
        },
        {"agentRuntimeId": target.runtime_id, "endpointName": target.qualifier},
    )
    control.add_response(
        "get_agent_runtime",
        {
            "agentRuntimeArn": target.runtime_arn,
            "agentRuntimeName": target.runtime_id.rpartition("-")[0],
            "agentRuntimeId": target.runtime_id,
            "agentRuntimeVersion": _RUNTIME_VERSION,
            "createdAt": _CLOCK_START,
            "lastUpdatedAt": _CLOCK_START,
            "roleArn": _ROLE_ARN,
            "status": "READY",
            "lifecycleConfiguration": {},
            "protocolConfiguration": {"serverProtocol": "HTTP"},
        },
        {"agentRuntimeId": target.runtime_id, "agentRuntimeVersion": _RUNTIME_VERSION},
    )


def _queue_run(clients: Any, target: RuntimeTarget, answer: str, fakes: Any) -> None:
    clients.stub("bedrock-agentcore", view=True).add_response(
        "invoke_agent_runtime",
        {
            "response": fakes.streaming_body(_event_stream(answer)),
            "contentType": "text/event-stream",
            "statusCode": 200,
        },
    )
    clients.stub("bedrock-agentcore").add_response(
        "stop_runtime_session",
        {"statusCode": 200},
        fakes.stop_params(runtime_arn=target.runtime_arn, qualifier=target.qualifier),
    )


def record_fixtures() -> None:
    """Record the two synthetic runs and write the cassette, the frame and the expected trace."""
    # The stubbed runtime and the fake span reader are the unit tests' own doubles, reused so
    # the fixtures come from exactly the surfaces the tests check the executor against.
    if str(_TESTS) not in sys.path:
        sys.path.insert(0, str(_TESTS))
    fakes = importlib.import_module("_agentcore_runtime_fakes")
    from calibrated_agent_evaluation.cli.stages.generate import TemplateRenderer
    from calibrated_agent_evaluation.cli.taxonomy import frame_to_doc
    from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow

    claim = trace_from_dict(json.loads(_CLAIM.read_text(encoding="utf-8")))
    cell = dict(claim.taxonomy_cell)
    answer = str(claim.output["answer"])
    prompts = (str(claim.input["narrative"]), TemplateRenderer().render(cell))
    target = RuntimeTarget(runtime_arn=FIXTURE_RUNTIME_ARN, payload_format="agentcore_sse")

    with (
        fakes.StubbedAwsClients(region=target.region) as clients,
        tempfile.TemporaryDirectory(
            prefix="calibrated_agent_evaluation_agentcore_fixture_"
        ) as scratch,
    ):
        _queue_preflight(clients, target)
        for _ in prompts:
            _queue_run(clients, target, answer, fakes)
        recorded = Path(scratch) / CASSETTE.name
        with AgentCoreRuntimeExecutor(
            target,
            span_reader=fakes.FakeRunSpanReader(clients),
            clients=clients,
            cassette_path=recorded,
            cassette_mode=Mode.RECORD,
            log_dir=Path(scratch) / "logs",
            sleep=_no_wait,
            clock=_SyntheticClock(),
        ) as executor:
            traces = [
                executor.run(AgentCoreRuntimeTask(prompt=prompt, taxonomy_cell=dict(cell)))[0]
                for prompt in prompts
            ]
        clients.assert_no_pending_responses()
        _write(CASSETTE, recorded.read_text(encoding="utf-8"))
    frame = FrameResult(rows=[FrameRow(cell=dict(cell), frame="discovery")], frame="discovery")
    _write(FRAME, _json(frame_to_doc(frame)))
    _write(EXPECTED_TRACE, _json(to_dict(traces[0])))


# --- a live recording -----------------------------------------------------------------------


def _outside_the_repository(out: str) -> Path:
    directory = Path(out).expanduser().resolve()
    if directory.is_relative_to(REPO_ROOT.resolve()):
        raise SystemExit(
            f"--out {out} is inside this repository ({REPO_ROOT}). A live recording carries the "
            "real runtime ARN and account ID and must never be committed, so pass a directory "
            "outside the repository"
        )
    if directory.exists() and not directory.is_dir():
        raise SystemExit(f"--out {out} is not a directory; pass a directory to record into")
    return directory


def _plan(target: RuntimeTarget, prompts: Sequence[str], directory: Path) -> str:
    count = f"{len(prompts)} input" + ("" if len(prompts) == 1 else "s")
    return (
        f"Will invoke AgentCore Runtime {target.runtime_arn} (qualifier {target.qualifier}, "
        f"payload format {target.payload_format}) for {count}, one billed invocation each plus "
        f"retries and the span reader's billed reads, and record them into {directory}"
    )


def record_live(args: argparse.Namespace) -> None:
    """Record real runs of a deployed runtime into ``args.out``, outside the repository."""
    directory = _outside_the_repository(args.out)
    try:
        target = RuntimeTarget(
            runtime_arn=args.runtime_arn,
            qualifier=args.qualifier,
            payload_format=args.payload_format,
        )
    except RuntimeInvocationError as exc:
        raise SystemExit(str(exc)) from exc
    print(_plan(target, args.prompt, directory))
    if not args.yes:
        raise SystemExit("nothing was called: pass --yes to make these billed calls")
    try:
        budget = BilledCallBudget.from_env(os.environ)
    except AgentCoreError as exc:
        raise SystemExit(f"nothing was called: {exc}") from exc
    import calibrated_agent_evaluation.agentcore

    reader_class: Callable[[AwsClients], Any] | None = getattr(
        calibrated_agent_evaluation.agentcore, "AgentCoreSpanReader", None
    )
    if reader_class is None:
        raise SystemExit(
            "live runs need the AgentCore span reader "
            "(calibrated_agent_evaluation.agentcore.AgentCoreSpanReader) to "
            "read each run's spans, and this installation has none, so nothing was called"
        )
    try:
        clients = AwsClients.from_profile(
            region=target.region, profile=args.profile, handlers=[budget]
        )
    except AgentCoreError as exc:
        raise SystemExit(f"nothing was called: {exc}") from exc
    traces: list[Trace] = []
    # One set of clients for the reader and the executor, so one budget covers both.
    with (
        clients,
        AgentCoreRuntimeExecutor(
            target,
            span_reader=reader_class(clients),
            clients=clients,
            cassette_path=directory / LIVE_CASSETTE_NAME,
            cassette_mode=Mode.RECORD,
            log_dir=directory,
        ) as executor,
    ):
        for prompt in args.prompt:
            try:
                traces.extend(executor.run(AgentCoreRuntimeTask(prompt=prompt)))
            except RuntimeInvocationError as exc:
                raise SystemExit(f"recording stopped: {exc}") from exc
    print(f"recorded {len(traces)} run(s) into {directory / LIVE_CASSETTE_NAME}")
    print(f"run manifest: {manifest_path(directory)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record the AgentCore Runtime executor's offline fixtures (the default), or record "
            "live runs into a directory outside the repository (--live)."
        )
    )
    parser.add_argument(
        "--live", action="store_true", help="record real, billed runs instead of the fixtures"
    )
    live = parser.add_argument_group("--live options")
    live.add_argument("--runtime-arn", help="the runtime to invoke")
    live.add_argument("--qualifier", default="DEFAULT", help="the endpoint (default: DEFAULT)")
    live.add_argument("--payload-format", choices=_FORMATS, help="how a prompt becomes a body")
    live.add_argument("--profile", help="the AWS profile (default: the default credential chain)")
    live.add_argument("--prompt", action="append", default=[], help="an input; repeat for more")
    live.add_argument("--out", help="the directory to record into, outside the repository")
    live.add_argument("--yes", action="store_true", help="make the billed calls")
    return parser


_LIVE_OPTIONS = ("runtime_arn", "payload_format", "profile", "out", "yes")


def main(argv: Sequence[str] | None = None) -> int:
    """Record the fixtures, or a live run with ``--live``."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.live:
        given = [name for name in _LIVE_OPTIONS if getattr(args, name)]
        if given or args.prompt or args.qualifier != "DEFAULT":
            parser.error(
                "--runtime-arn, --payload-format, --qualifier, --profile, --prompt, "
                "--out and --yes only apply with --live"
            )
        record_fixtures()
        return 0
    missing = [
        flag
        for flag, value in (
            ("--runtime-arn", args.runtime_arn),
            ("--payload-format", args.payload_format),
            ("--prompt", args.prompt),
            ("--out", args.out),
        )
        if not value
    ]
    if missing:
        parser.error(f"--live needs {', '.join(missing)}")
    record_live(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
