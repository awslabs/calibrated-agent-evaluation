# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``capture``: run an instrumented in-process agent, capture its trace, and ingest it.

The agent is given as ``--agent module:callable`` (a no-argument, OpenTelemetry-instrumented
entrypoint), NOT an arbitrary subprocess: OTel capture installs an in-process tracer provider, and a
child process cannot emit into it. The captured spans are encoded to OTLP-JSON
and run through the same ingest path as ``calibrated-agent-evaluation ingest``, so the
no-silent-loss gate applies; a run
that emits no span is refused, not persisted as an empty trace.

Reuses the ``calibrated_agent_evaluation.ingest`` feature module (lazy import); needs the
``[strands]`` extra for the OTel SDK, whose absence is a typed error naming the extra.
"""

from __future__ import annotations

import argparse

from calibrated_agent_evaluation.cli.dispatch import StageOutcome


def capture_handler(args: argparse.Namespace) -> StageOutcome:
    """Load ``--agent module:callable``, run it under capture, and ingest the captured trace."""
    from calibrated_agent_evaluation.cli.agentspec import load_agent
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.cli.stages.ingest import report_outcome
    from calibrated_agent_evaluation.ingest import capture_and_ingest

    spec = args.agent
    agent = load_agent(spec)

    service = getattr(args, "service", "agent")
    # No `or service` fallback. `--service` itself defaults to the constant "agent", so an
    # unspecified --trace-id made every capture on the machine resolve to the same id -- and the
    # store persists with INSERT OR REPLACE keyed on trace_id, so run 2 silently replaced run 1
    # while the report said "1 ingested" and exit 0. A None trace id makes capture_and_ingest
    # derive one from the captured OTel trace, which is unique per run by construction.
    trace_id = getattr(args, "trace_id", None)
    with open_backend(args) as backend:
        report = capture_and_ingest(
            agent,
            service_name=service,
            trace_id=trace_id,
            backend=backend,
            dry_run=getattr(args, "dry_run", False),
            require_owned=getattr(args, "require_owned", False),
            overwrite=getattr(args, "overwrite", False),
            agent_label=spec,
        )
    return report_outcome("capture", report)
