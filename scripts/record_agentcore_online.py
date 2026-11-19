#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record an online evaluation config's traces and results for an offline replay.

A replay serves these through a fake span reader and botocore ``Stubber``, so it runs the shipped
read-back and publish code with no AWS account. It reads, read-only, one window of a live config's
runtime traces (the span reader ``calibrated-agent-evaluation online read`` uses) and the config's
own results, and writes to ``fixtures/raw/agentcore/online/``:

- ``traces.json.gz``: each trace as the span reader decoded it (span records, content records,
  session ids);
- ``results.json.gz``: the config's result records for those traces;
- ``evaluator.json``: the attached evaluator's ``GetEvaluator`` document;
- ``record.json``: the online record ``calibrated-agent-evaluation online attach`` stored, with the
  calibration and bias records it was attached with.

The account-id field of every ARN (the 12 digits between the fourth and fifth colon) is replaced
with ``111122223333``, and every AWS access key id the spans carry (``aws.auth.account.access_key``)
with a placeholder, before anything is written. So is the host of the AWS-internal cell that served
each invocation (a ``*.aws.dev`` name the runtime's server span carries in ``http.server_name`` and
``http.url``), which names AWS infrastructure rather than the account. Nothing else is touched:
session ids, span durations and request ids can hold 12-digit runs too, and are kept as recorded.
A file that would still hold the account's own id anywhere, or an access key id, is refused. The
placeholder account changes the evaluator's ARN, and so its fingerprint, so the recorded online,
calibration and bias records are re-pinned to the fingerprint the scrubbed evaluator gives; their
statistics are left as they were measured.

Usage (credentials for the account, read-only calls only):
    uv run --all-extras python scripts/record_agentcore_online.py STORE CONFIG_ID SINCE UNTIL
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = _REPO / "fixtures" / "raw" / "agentcore" / "online"
TRACES = FIXTURE_DIR / "traces.json.gz"
RESULTS = FIXTURE_DIR / "results.json.gz"
EVALUATOR = FIXTURE_DIR / "evaluator.json"
RECORD = FIXTURE_DIR / "record.json"
# The account-id field of an ARN, between two colons (``arn:aws:svc:region:ACCOUNT:resource``).
_ARN_ACCOUNT_ID = re.compile(r"(?<=:)\d{12}(?=:)")
_PLACEHOLDER = "111122223333"
_ACCESS_KEY_ID = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA|AROA|AIDA)[A-Z0-9]{16}(?![A-Z0-9])")
_KEY_PLACEHOLDER = "REDACTED-ACCESS-KEY-ID"
_SERVICE_HOST = re.compile(r"(?<![a-z0-9.-])(?:[a-z0-9-]+\.)+aws\.dev(?![a-z0-9-])")
_HOST_PLACEHOLDER = "agentcore-runtime.invalid"


def scrub(value: Any) -> Any:
    """``value`` with each ARN's account id, access key id and AWS-internal host replaced."""
    text = json.dumps(value, default=str)
    text = _ARN_ACCOUNT_ID.sub(_PLACEHOLDER, text)
    text = _ACCESS_KEY_ID.sub(_KEY_PLACEHOLDER, text)
    text = _SERVICE_HOST.sub(_HOST_PLACEHOLDER, text)
    return json.loads(text)


def repin(evaluator: dict[str, Any], stored: dict[str, Any], judge_id: str) -> dict[str, Any]:
    """``stored`` with the recorded fingerprint replaced by the scrubbed evaluator's."""
    from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator

    old = stored["online"]["evaluators"][0]["fingerprint"]
    new = judge_from_evaluator(judge_id, evaluator).fingerprint
    return json.loads(json.dumps(stored).replace(old, new))


def write(path: Path, value: Any, *, account_id: str) -> None:
    """Write ``value`` scrubbed, refusing it if ``account_id`` or an access key id is still in it.

    The account id is looked for anywhere in the text, so one left outside an ARN, or inside a
    URL-encoded ARN (``%3A<id>%3A``), is caught rather than written.
    """
    text = json.dumps(scrub(value), indent=None, sort_keys=True, default=str)
    if account_id in text or _ACCESS_KEY_ID.search(text):
        raise SystemExit(f"refusing to write {path}: it would hold an AWS account or key id")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    else:
        path.write_text(text + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    from calibrated_agent_evaluation.agentcore import AwsClients, ReadOnlyGuard
    from calibrated_agent_evaluation.agentcore.online import call_aws
    from calibrated_agent_evaluation.agentcore.readback import results_group
    from calibrated_agent_evaluation.agentcore.readers import LogsInsightsReader, run_logs_insights
    from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow
    from calibrated_agent_evaluation.cli.records import load_records_of

    store, config_id, since, until = Path(argv[0]), argv[1], argv[2], argv[3]
    (record,) = [r for r in load_records_of(store, "agentcore-online") if r.config_id == config_id]
    attached = record.evaluators[0]
    calib = next(
        c for c in load_records_of(store, "calibration") if c.calib_id == attached.calib_id
    )
    battery = next(b for b in load_records_of(store, "bias") if b.battery_id == attached.battery_id)
    config = AgentCoreSourceConfig(
        runtime=record.runtime_service_name, window=TimeWindow(start=since, end=until)
    )
    allowed = {
        ("bedrock-agentcore-control", "GetEvaluator"),
        ("logs", "StartQuery"),
        ("logs", "GetQueryResults"),
        ("logs", "FilterLogEvents"),
    }
    with AwsClients.from_profile(region="us-east-1", handlers=[ReadOnlyGuard(allowed)]) as clients:
        traces = [
            dataclasses.asdict(raw) for raw in LogsInsightsReader(clients, config).read_traces()
        ]
        results = run_logs_insights(clients.client("logs"), results_group(config_id), config)
        evaluator = call_aws(
            clients,
            "bedrock-agentcore-control",
            "GetEvaluator",
            {"evaluatorId": attached.evaluator_id},
        )
    evaluator.pop("ResponseMetadata", None)
    parts = attached.evaluator_arn.split(":")
    account_id = parts[4] if len(parts) > 4 else ""
    if not re.fullmatch(r"\d{12}", account_id):
        raise SystemExit(
            f"evaluator ARN {attached.evaluator_arn} names no account, so the fixtures cannot be "
            "checked for it; record a config whose evaluator is the account's own"
        )
    evaluator = scrub(evaluator)
    stored = scrub(
        {
            "online": record.model_dump(mode="json"),
            "calibration": calib.model_dump(mode="json"),
            "bias": battery.model_dump(mode="json"),
        }
    )
    write(TRACES, traces, account_id=account_id)
    write(RESULTS, results, account_id=account_id)
    write(EVALUATOR, evaluator, account_id=account_id)
    write(RECORD, repin(evaluator, stored, calib.judge_id), account_id=account_id)
    print(f"wrote {len(traces)} trace(s) and {len(results)} result(s) to {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
