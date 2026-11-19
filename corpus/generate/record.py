# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the reference agents against Bedrock and freeze the ``fixtures/raw/`` corpus.

This is the one-time capture harness (``corpus-companion.md`` §2). It runs each reference agent,
routing the live model calls through committed cassettes, and writes the frozen wire-format traces
the trajectory normalizer is coded against:

    fixtures/raw/
      otel/
        ORD-1001.events.otlp.json     # OrderBot #1 (hand-coded OTel), content in span EVENTS
        ORD-1001.attrs.otlp.json      # same run, content in span ATTRIBUTES (both content places)
        ORD-1001.build2.otlp.json     # OrderBot #2 (Anthropic+MCP) — the target-agnosticism twin
        ORD-1002.events.otlp.json     # in-window clean PASS control (OTel build)
        ORD-DEGRADED.otlp.json        # under-instrumented run -> normalizer must degrade to probed
      strands/
        CLM-4471.otlp.json            # ClaimsCrew on Strands (real strands-native span names)
      inspect/
        ORD-1001.eval  ORD-1002.eval  CLM-4471.eval   # real Inspect .eval logs
      cassettes/                      # committed model-call recordings (offline replay)

Modes (env ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE``):
  * ``replay`` (default here) — no Bedrock needed; replays the committed cassettes. CI / a fresh
    clone re-freezes byte-identical output offline.
  * ``record`` / ``auto`` — calls live Bedrock (set ``AWS_PROFILE=<your-profile>``), (re)writes
    cassettes.

Usage:
    AWS_PROFILE=<your-profile> CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=record uv run python -m
    corpus.generate.record uv run python -m corpus.generate.record            # replay the committed
    cassettes (offline)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from corpus.agents import (
    claimscrew_strands,
    claimscrew_swarm,
    orderbot_agent_as_tool,
    orderbot_mcp,
    orderbot_otel,
)
from corpus.agents.otel_emit import ContentLocation, emit_degraded_trace
from corpus.generate.bedrock_client import make_client
from corpus.generate.cassette import Cassette, Mode, mode_from_env
from corpus.generate.otel_capture import SpanCapture, write_otlp_json
from corpus.generate.scenarios import (
    CLAIMSCREW_ATTRIBUTION_TRACES,
    CLAIMSCREW_FAIL,
    ORDERBOT_CLEAN,
    ORDERBOT_FAIL,
    load_scenario,
)

logger = logging.getLogger("calibrated_agent_evaluation.corpus.record")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = _REPO_ROOT / "fixtures" / "raw"
_CASSETTES = _RAW / "cassettes"
_REGION = "us-west-2"


def _cassette(name: str, mode: Mode) -> Cassette:
    return Cassette(_CASSETTES / f"{name}.cassette.json", mode=mode)


def record_orderbot_otel(mode: Mode) -> None:
    """OrderBot #1 (hand-coded OTel): freeze events + attributes variants and the clean control."""
    client = make_client(_CASSETTES / "orderbot_otel.cassette.json", mode=mode, region=_REGION)

    # ORD-1001 in both content locations (same logical run, both OTel content placements).
    fail = load_scenario(ORDERBOT_FAIL)
    for location, suffix in (
        (ContentLocation.EVENTS, "events"),
        (ContentLocation.ATTRIBUTES, "attrs"),
    ):
        with SpanCapture("orderbot") as cap:
            orderbot_otel.run(client, fail, cap, content_location=location)
        write_otlp_json(cap.finished_spans, _RAW / "otel" / f"ORD-1001.{suffix}.otlp.json")
        logger.info("froze otel/ORD-1001.%s.otlp.json", suffix)

    # ORD-1002 clean control (events placement).
    clean = load_scenario(ORDERBOT_CLEAN)
    with SpanCapture("orderbot") as cap:
        orderbot_otel.run(client, clean, cap, content_location=ContentLocation.EVENTS)
    write_otlp_json(cap.finished_spans, _RAW / "otel" / "ORD-1002.events.otlp.json")
    logger.info("froze otel/ORD-1002.events.otlp.json")

    # ORD-DEGRADED: same run, but emitted with thin instrumentation (normalizes to probed).
    with SpanCapture("orderbot") as cap:
        record = orderbot_otel.run_record_only(client, fail)
        emit_degraded_trace(
            cap.tracer("calibrated_agent_evaluation.orderbot.degraded"),
            record,
            trace_input=fail.trace_input,
        )
    write_otlp_json(cap.finished_spans, _RAW / "otel" / "ORD-DEGRADED.otlp.json")
    logger.info("froze otel/ORD-DEGRADED.otlp.json")


def record_orderbot_mcp(mode: Mode) -> None:
    """OrderBot #2 (Anthropic + stdio-MCP): freeze the target-agnosticism twin of ORD-1001."""
    cas = _cassette("orderbot_mcp", mode)
    fail = load_scenario(ORDERBOT_FAIL)
    with SpanCapture("orderbot-mcp") as cap:
        asyncio.run(orderbot_mcp.run(cas, fail, cap, region=_REGION))
    write_otlp_json(cap.finished_spans, _RAW / "otel" / "ORD-1001.build2.otlp.json")
    logger.info("froze otel/ORD-1001.build2.otlp.json")


def record_claimscrew_strands(mode: Mode) -> None:
    """ClaimsCrew on Strands: freeze the real strands-native OTel export (legacy semconv).

    Freezes the marquee CLM-4471 (manifestation != origination) plus the crew-attribution
    contrast/trap crew traces (CLM-4473 different-culprit, CLM-4475 outcome-pass/span-fail,
    CLM-4477 disputed) — each a genuine Strands ``Graph`` run whose defect is injected via its
    normalized oracle's tool results, never hand-shaped (fixtures/README.md).
    """
    for trace_id in (CLAIMSCREW_FAIL, *CLAIMSCREW_ATTRIBUTION_TRACES):
        # CLM-4471 keeps its original cassette name (a committed file); the attribution traces get
        # their own per-trace cassettes so each Graph run replays offline independently.
        cassette_name = (
            "claimscrew_strands"
            if trace_id == CLAIMSCREW_FAIL
            else f"claimscrew_strands_{trace_id}"
        )
        cas = _cassette(cassette_name, mode)
        scenario = load_scenario(trace_id)
        with SpanCapture("claimscrew") as cap:
            claimscrew_strands.run(cas, scenario, cap, region=_REGION)
        write_otlp_json(cap.finished_spans, _RAW / "strands" / f"{trace_id}.otlp.json")
        logger.info("froze strands/%s.otlp.json", trace_id)


def record_claimscrew_swarm(mode: Mode) -> None:
    """Claims-triage Swarm: freeze the real strands-native OTel export (the handoff_to_agent tool).

    Proves the normalizer's strands_adapter Swarm path against real output: the capture carries an
    ``invoke_swarm`` root and a real ``execute_tool handoff_to_agent`` span.
    """
    cas = _cassette("claimscrew_swarm", mode)
    with SpanCapture("claims-swarm") as cap:
        claimscrew_swarm.run(cas, cap, region=_REGION)
    write_otlp_json(cap.finished_spans, _RAW / "strands" / "CLM-7720-swarm.otlp.json")
    logger.info("froze strands/CLM-7720-swarm.otlp.json")


def record_orderbot_agent_as_tool(mode: Mode) -> None:
    """Agent-as-Tools OrderBot: freeze the real strands-native OTel export (sub-agent under a tool).

    Proves the normalizer's strands_adapter Agent-as-Tools path and sub-agent discriminator against
    real output: the capture nests an ``invoke_agent policy_advisor`` span under the parent's
    ``execute_tool policy_advisor`` span, alongside a leaf ``execute_tool issue_refund``.
    """
    cas = _cassette("orderbot_agent_as_tool", mode)
    with SpanCapture("orderbot-as-tool") as cap:
        orderbot_agent_as_tool.run(cas, cap, region=_REGION)
    write_otlp_json(cap.finished_spans, _RAW / "strands" / "ORD-3310-as-tool.otlp.json")
    logger.info("froze strands/ORD-3310-as-tool.otlp.json")


def record_inspect(mode: Mode) -> None:
    """Mint the real Inspect ``.eval`` logs for OrderBot (fail + clean) and ClaimsCrew."""
    from inspect_ai import eval as inspect_eval

    from corpus.agents import inspect_cassette_provider as prov
    from corpus.agents.orderbot_inspect import claimscrew_fail, orderbot_clean, orderbot_fail

    inspect_out = _RAW / "inspect"
    inspect_out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("ORD-1001", orderbot_fail),
        ("ORD-1002", orderbot_clean),
        ("CLM-4471", claimscrew_fail),
    ]
    for trace_id, task_fn in jobs:
        cas = _cassette(f"inspect_{trace_id}", mode)
        prov.configure(cas, inner_model="bedrock/global.anthropic.claude-sonnet-4-6")
        with _temp_log_dir() as log_dir:
            logs = inspect_eval(
                task_fn(),
                model="cassette/inspect",
                log_dir=str(log_dir),
                log_format="eval",
                display="none",
            )
            if logs[0].status != "success":
                msg = logs[0].error.message if logs[0].error else "unknown"
                raise RuntimeError(f"inspect eval for {trace_id} failed: {msg}")
            minted = next(log_dir.glob("*.eval"))
            shutil.copy(minted, inspect_out / f"{trace_id}.eval")
            # print (not logger): Inspect's eval() reconfigures the root logger / display, which
            # swallows later logging output — print survives it so the milestone is visible.
            print(f"[record] froze inspect/{trace_id}.eval", flush=True)


@contextmanager
def _temp_log_dir() -> Generator[Path]:
    """A scratch dir for Inspect to write the .eval into before we copy the frozen one out."""
    d = tempfile.mkdtemp(prefix="calibrated-agent-evaluation-inspect-")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the reference agents -> fixtures/raw/.")
    parser.add_argument(
        "--only",
        choices=["otel", "mcp", "strands", "swarm", "as-tool", "inspect"],
        action="append",
        help="capture only the named target(s); default captures all",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    mode = mode_from_env(default=Mode.REPLAY)
    logger.info(
        "capture mode: %s (set CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=record + "
        "AWS_PROFILE=<your-profile> to relive)",
        mode,
    )
    _CASSETTES.mkdir(parents=True, exist_ok=True)

    targets = args.only or ["otel", "mcp", "strands", "swarm", "as-tool", "inspect"]
    runners = {
        "otel": record_orderbot_otel,
        "mcp": record_orderbot_mcp,
        "strands": record_claimscrew_strands,
        "swarm": record_claimscrew_swarm,
        "as-tool": record_orderbot_agent_as_tool,
        "inspect": record_inspect,
    }
    for target in targets:
        print(f"[record] === capturing {target} ===", flush=True)
        runners[target](mode)

    print(f"[record] done. froze fixtures/raw/ ({_RAW})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
