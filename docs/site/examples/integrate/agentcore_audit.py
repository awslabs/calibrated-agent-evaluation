#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B audit example: record what an AgentCore A/B test claimed and which arm each trace ran in.

Offline. The A/B test and its bundle versions are served through botocore's ``Stubber`` over the
installed service models, and the arm read takes a fixture window through a trace-source double,
so no AWS call is made. Each step prints what ``calibrated-agent-evaluation audit`` prints and
checks it against the oracle the fixtures carry.

Steps:
    abtest          record a fixture A/B test (``--fixture``: stopped, a-a, not-started,
                    running, target, missing, variant-both, variant-neither,
                    stopped-while-running)
    arms            read the arms of a fixture window (``--fixture``: main, half-stamped,
                    repeated-source-id, unstamped, other-experiment-only, second-window,
                    changed-stamp, grown, shrunk), after recording the stopped test when the
                    root holds no claim; ``--claim-record`` and the applied-bundle flags pass
                    through
    recommendation  register the recommendation's target and record a fixture recommendation
                    created an hour after that registration (``--fixture``: completed,
                    late-prereg, failed, inline, batch, spans-missing)
    replay          build a replay set's local stores from their JSON (``--set``: h1, h1-judged,
                    h2) into ``--to DIR``
    retest          record the stopped test, read its arms, register the target and re-test it
                    on the H1 roots with ``calibrated-agent-evaluation audit retest``
    report          the re-test, then ``calibrated-agent-evaluation audit report``, checked against
    its golden langfuse-seed   write one replay root's traces and verdicts (``--from DIR``) into the
                    Langfuse project ``--project`` (credentials from ``LANGFUSE_PUBLIC_KEY``,
                    ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_HOST``)
    guard           show each write and list call the audit never makes refused by the
                    read-only guard before it is sent

The record root is ``--root``, else ``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT``, else a temporary
directory.

Usage, from the repository root:
    uv run --all-extras python docs/site/examples/integrate/agentcore_audit.py
        [--root DIR] STEP [--fixture NAME] [options]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

_RAW = _REPO_ROOT / "fixtures" / "raw" / "agentcore" / "abtest"
_GOLDEN = _REPO_ROOT / "fixtures" / "expected" / "audit"
_BUNDLE_ID = "fixture-bundle-AbCdEf1234"
_TIMES = {
    "startedAt",
    "stoppedAt",
    "maxDurationExpiresAt",
    "createdAt",
    "updatedAt",
    "analysisTimestamp",
    "versionCreatedAt",
}
_TESTS = {
    "stopped": ("abtest_stopped.json", "bundle_version_treatment.json"),
    "stopped-while-running": ("abtest_stopped_while_running.json", "bundle_version_treatment.json"),
    "aa": ("abtest_aa.json", "bundle_version_aa.json"),
    "a-a": ("abtest_aa.json", "bundle_version_aa.json"),
    "not-started": ("abtest_not_started.json", "bundle_version_treatment.json"),
    "running": ("abtest_running.json", "bundle_version_treatment.json"),
    "target": ("abtest_target.json", None),
    "variant-both": ("abtest_variant_both.json", None),
    "variant-neither": ("abtest_variant_neither.json", None),
}
# The per-arm counts of each window with no applied-bundle flag, from the fixture's table.
_ARM_COUNTS = {
    "main": {"C.traces": 5, "C.runtime_sessions": 2, "T1.traces": 4, "refused": 5},
    "half-stamped": {"C.traces": 0, "T1.traces": 1, "refused": 2, "refused.half_stamped": 2},
    "repeated-source-id": {"C.traces": 0, "T1.traces": 1, "refused.shared_id": 1},
}


def _load(name: str) -> dict[str, Any]:
    return json.loads((_RAW / name).read_text(encoding="utf-8"))


def _response(doc: Any) -> Any:
    if isinstance(doc, dict):
        return {
            k: datetime.fromisoformat(v) if k in _TIMES and isinstance(v, str) else _response(v)
            for k, v in doc.items()
        }
    if isinstance(doc, list):
        return [_response(v) for v in doc]
    return doc


def _offline_credentials() -> None:
    os.environ.update({"AWS_ACCESS_KEY_ID": "demo", "AWS_SECRET_ACCESS_KEY": "demo"})
    os.environ.pop("AWS_PROFILE", None)
    os.environ.pop("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", None)


def record_test(root: Path, fixture: str = "stopped") -> Any:
    """Record the fixture A/B test into ``root`` and print its vendor-claim block."""
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.audit_report import claim_lines
    from calibrated_agent_evaluation.agentcore.optimization import audit_clients, read_ab_test
    from calibrated_agent_evaluation.cli.records import append_records

    _offline_credentials()
    if fixture == "missing":
        return _record_missing()
    test_file, treatment = _TESTS[fixture]
    test = _load(test_file)
    with audit_clients(region="us-west-2") as clients:
        data = Stubber(clients.client("bedrock-agentcore"))
        control = Stubber(clients.client("bedrock-agentcore-control"))
        data.add_response("get_ab_test", _response(test), {"abTestId": test["abTestId"]})
        variant = test["variants"][1].get("variantConfiguration") or {}
        if len(variant) == 1:
            versions = ["bundle_version_control.json"] + ([treatment] if treatment else [])
            for name in versions:
                doc = _load(name)
                control.add_response(
                    "get_configuration_bundle_version",
                    _response(doc),
                    {"bundleId": _BUNDLE_ID, "versionId": doc["versionId"]},
                )
        with data, control:
            record = read_ab_test(clients, test["abTestId"])
    append_records(root, "agentcore-abtest", [record])
    print("\n".join(claim_lines(record)))
    return record


class _Window:
    """A trace source that delivers one fixture window."""

    def __init__(self, name: str) -> None:
        from calibrated_agent_evaluation.backends.base import SourceTrace

        docs = _read_json(_RAW / "stamped" / f"{name}.json")
        self._items = [SourceTrace.model_validate(d) for d in docs]

    def pull(self) -> Iterator[Any]:
        yield from self._items


_WINDOWS = {"second-window": ("2026-10-28T19:30:00+00:00", "2026-10-28T20:30:00+00:00")}


def read_window(
    root: Path,
    fixture: str = "main",
    *,
    claim_record: str | None = None,
    attribute: str | None = None,
    require: tuple[tuple[str, str], ...] = (),
) -> Any:
    """Read the arms of a fixture window against the stored claim, store them, print the counts."""
    from calibrated_agent_evaluation.agentcore.arms import merge_arms_read, read_arms
    from calibrated_agent_evaluation.agentcore.audit_report import arms_lines
    from calibrated_agent_evaluation.cli.records import append_records, load_records_of
    from calibrated_agent_evaluation.cli.stages.audit import resolve_claim_record

    claim = resolve_claim_record(
        load_records_of(root, "agentcore-abtest"), "fixture-abtest-stopped", claim_record
    )
    bounds = _WINDOWS.get(fixture, ("2026-10-28T19:00:00+00:00", "2026-10-28T20:00:00+00:00"))

    class _Bounds:
        start, end = bounds

    read = read_arms(
        _Window(fixture),
        ab_test_arn=claim.ab_test_arn,
        runtime="fixture-runtime",
        window=_Bounds(),
        applied_bundle_attribute=attribute,
        applied_bundle_require=require,
        bundle_versions={v.name: v.bundle.version_id for v in claim.variants if v.bundle},
    )
    new = merge_arms_read(load_records_of(root, "agentcore-arms"), read)
    stored = append_records(root, "agentcore-arms", new) if new else 0
    print(
        "\n".join(
            arms_lines(
                read_id=read.read_id,
                counts=read.counts,
                stored=stored,
                omni=False,
                rekey_sessions=False,
                arns_seen=[a for a in read.arns_seen if a != claim.ab_test_arn],
            )
        )
    )
    return read


# The writes and the list calls the audit never makes, each tried through its guarded clients.
_GUARDED = (
    ("bedrock-agentcore", "StartRecommendation"),
    ("bedrock-agentcore", "CreateABTest"),
    ("bedrock-agentcore", "UpdateABTest"),
    ("bedrock-agentcore-control", "UpdateConfigurationBundle"),
    ("bedrock-agentcore", "ListABTests"),
    ("bedrock-agentcore", "DeleteABTest"),
)


def _minimal_input(shape: Any) -> Any:
    """The smallest value the installed model's validation accepts for ``shape``."""
    kind = shape.type_name
    if kind == "structure":
        if getattr(shape, "is_tagged_union", False):
            name = next(iter(shape.members))
            return {name: _minimal_input(shape.members[name])}
        return {name: _minimal_input(shape.members[name]) for name in shape.required_members}
    if kind == "list":
        return [_minimal_input(shape.member) for _ in range(int(shape.metadata.get("min", 0)))]
    if kind == "map":
        return {}
    if kind == "string":
        return "x" * max(1, int(shape.metadata.get("min", 1)))
    if kind in ("integer", "long"):
        return max(1, int(shape.metadata.get("min", 1)))
    if kind in ("float", "double"):
        return 1.0
    if kind == "boolean":
        return True
    if kind == "timestamp":
        return datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    if kind == "blob":
        return b"x"
    return {}


def _sent_counter(sent: list[str], operation: str) -> Any:
    """A ``before-send`` handler that records ``operation`` reaching the wire."""

    def handler(**_: Any) -> None:
        sent.append(operation)

    return handler


def show_guard() -> str:
    """Try each call the audit never makes; the guard refuses all of them before they are sent.

    A ``before-send`` counter on each client checks that no request reached the wire. (A botocore
    ``Stubber`` can't witness this: it checks a call at parameter build, before the guard runs.)
    """
    from calibrated_agent_evaluation.agentcore.errors import ReadOnlyViolation
    from calibrated_agent_evaluation.agentcore.optimization import audit_clients

    _offline_credentials()
    sent: list[str] = []
    lines: list[str] = []
    with audit_clients(region="us-west-2") as clients:
        for service, operation in _GUARDED:
            client = clients.client(service)
            client.meta.events.register("before-send", _sent_counter(sent, operation))
            model = client.meta.service_model.operation_model(operation)
            params = _minimal_input(model.input_shape) if model.input_shape is not None else {}
            method = next(
                m for m, op in client.meta.method_to_api_mapping.items() if op == operation
            )
            try:
                getattr(client, method)(**params)
            except ReadOnlyViolation as exc:
                lines.append(f"ReadOnlyViolation ({operation}): {exc}")
            else:
                raise SystemExit(f"guard step: {operation} was not refused")
    if sent:
        raise SystemExit(f"guard step: {sent} reached the wire")
    lines.append(f"0 requests sent; {len(_GUARDED)} calls refused before sending")
    print("\n".join(lines))
    return "\n".join(lines)


def check_claim(record: Any) -> None:
    """The stopped test's record equals the oracle written without Calibrated Agent Evaluation
    code."""
    expected = json.loads((_GOLDEN / "abtest_verdict.json").read_text(encoding="utf-8"))
    got = record.model_dump(mode="json")
    for key in ("record_id", "read_at", "provenance"):
        got.pop(key)
    if got != expected:
        raise SystemExit("abtest step: the claim record differs from abtest_verdict.json")


def check_counts(read: Any, fixture: str) -> None:
    """The window's per-arm counts equal the fixture table's."""
    for key, value in _ARM_COUNTS[fixture].items():
        if read.counts.get(key) != value:
            raise SystemExit(f"arms step: {key} is {read.counts.get(key)}, the table says {value}")


def _record_missing() -> None:
    """Read a test AgentCore says does not exist: a typed refusal, and nothing stored."""
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.optimization import audit_clients, read_ab_test

    with audit_clients(region="us-west-2") as clients:
        data = Stubber(clients.client("bedrock-agentcore"))
        data.add_client_error("get_ab_test", service_error_code="ResourceNotFoundException")
        with data:
            read_ab_test(clients, "fixture-abtest-missing")


# --- replay roots, the re-test and the report ---------------------------------------------------

_AGENTCORE_RAW = _REPO_ROOT / "fixtures" / "raw" / "agentcore"


def _read_json(path: Path) -> Any:
    """A JSON fixture, from its gzipped ``.json.gz`` copy when the plain file isn't there."""
    gzipped = path.with_name(path.name + ".gz")
    if not path.exists() and gzipped.exists():
        return json.loads(gzip.decompress(gzipped.read_bytes()))
    return json.loads(path.read_text(encoding="utf-8"))


_TARGET = "missing_date_range"
_SETS = {
    "h1": (
        "abtest/replay",
        ("control", "treatment", "development", "development-overlap", "development-messages"),
    ),
    "h1-judged": ("abtest/replay-judged", ("control", "treatment")),
    "h2": ("recommendations/replay", ("input", "result", "development")),
}


def build_root(name: str, dest: Path) -> Path:
    """Write the local store the JSON description ``name`` holds into ``dest``."""
    from calibrated_agent_evaluation.agentcore.audit_types import RunProvenance
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.schema.types import Label, Trace

    doc = _read_json(_AGENTCORE_RAW / f"{name}.json")
    with LocalBackend(root=dest) as backend:
        for t in doc["traces"]:
            backend.put_trace(
                Trace(
                    trace_id=t["trace_id"],
                    frame="production-representative",
                    ownership="probed",
                    input=t["input"],
                    output=t["output"],
                    taxonomy_cell={"feature": "analytics"},
                )
            )
        for lb in doc["labels"]:
            backend.put_score(Label(span_id=None, **lb))
    if doc["run_provenance"]:
        records = [RunProvenance.model_validate(r) for r in doc["run_provenance"]]
        append_records(dest, "agentcore-run-provenance", records)
    return dest


def _judge_records(name: str, root: Path) -> None:
    """The calibration and bias records a judged re-test's record root holds, from ``name``."""
    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.bias import BiasBatteryRecord, ProbeResult
    from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord, Interval
    from calibrated_agent_evaluation.cli.records import append_records

    doc = _read_json(_AGENTCORE_RAW / f"{name}.json")
    calibration = [
        CalibrationRecord(
            provenance=capture_provenance(now=datetime.fromisoformat(c["run_at"])),
            calib_id=c["calib_id"],
            judge_id=c["judge_id"],
            judge_fingerprint=c["judge_fingerprint"],
            model="synthetic",
            prompt_version="v1",
            hunted_set_id=f"hunted-{c['calib_id']}",
            n_pos=80,
            n_neg=80,
            tpr=Bound(point=0.9, lb95=0.83, n=80),
            tnr=Bound(point=0.9, lb95=0.81, n=80),
            youden_j=0.8,
            result=c["result"],
            rate_set_id="synthetic",
            corrected_rate=Interval(
                point=0.06,
                lo=0.03,
                hi=0.1,
                method="synthetic",
                level=0.95,
                sided="two-sided",
                n=160,
            ),
            locked_test_consumed=True,
            trace_ids=(f"{c['calib_id']}-gold",),
        )
        for c in doc["calibration"]
    ]
    bias = [
        BiasBatteryRecord(
            provenance=capture_provenance(now=datetime.fromisoformat(b["run_at"])),
            battery_id=b["battery_id"],
            judge_id=b["judge_id"],
            judge_fingerprint=b["judge_fingerprint"],
            model="synthetic",
            prompt_version="v1",
            is_pairwise=False,
            results=(
                ProbeResult(
                    bias="length",
                    status="run",
                    n=4,
                    biased_count=0,
                    magnitude=0.0,
                    magnitude_ub95=0.05,
                    threshold=0.2,
                    passed=True,
                    is_must_pass=True,
                ),
            ),
            verdict=b["verdict"],
        )
        for b in doc["bias"]
    ]
    append_records(root, "calibration", calibration)
    append_records(root, "bias", bias)


def build_set(which: str, to: Path) -> dict[str, Path]:
    """Build every store of the replay set ``which`` under ``to``; returns them by name."""
    base, names = _SETS[which]
    built = {n: build_root(f"{base}/{n}", to / n) for n in names}
    if which == "h1-judged":
        built["development"] = build_root("abtest/replay/development", to / "development")
        for records in ("records-certified", "records-uncertified-guard"):
            _judge_records(f"{base}/{records}", to / records)
            built[records] = to / records
    return built


def register(root: Path, *, frozen: datetime | None = None) -> str:
    """Register the replay sets' target with a minimum reduction of 0.10; its ``frozen_at``."""
    from calibrated_agent_evaluation.cli.records import load_ledger, save_ledger
    from calibrated_agent_evaluation.improve import PreRegEntry, register_prereg

    entry = PreRegEntry(prereg_id=f"p-{_TARGET}", target_metric=_TARGET, success_threshold=0.10)
    ledger = register_prereg(load_ledger(root), entry, now=frozen)
    save_ledger(root, ledger)
    return str(ledger.entries[-1].frozen_at)


_RECOMMENDATIONS = {
    "completed": "rec_system_prompt.json",
    "late-prereg": "rec_system_prompt.json",
    "failed": "rec_failed.json",
    "inline": "rec_inline_text.json",
    "batch": "rec_batch.json",
    "spans-missing": "rec_spans_missing.json",
}


def record_recommendation(root: Path, fixture: str = "completed") -> Any:
    """Register the target, then record a fixture recommendation created an hour after it."""
    from datetime import timedelta

    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.audit_report import recommendation_lines
    from calibrated_agent_evaluation.agentcore.optimization import (
        audit_clients,
        read_recommendation,
    )
    from calibrated_agent_evaluation.cli.records import append_records

    _offline_credentials()
    frozen = datetime.fromisoformat(register(root))
    created = frozen + timedelta(hours=-1 if fixture == "late-prereg" else 1)
    doc = json.loads(
        (_AGENTCORE_RAW / "recommendations" / _RECOMMENDATIONS[fixture]).read_text(encoding="utf-8")
    )
    doc["createdAt"] = created.isoformat()
    doc["updatedAt"] = (created + timedelta(minutes=20)).isoformat()
    with audit_clients(region="us-west-2") as clients:
        data = Stubber(clients.client("bedrock-agentcore"))
        control = Stubber(clients.client("bedrock-agentcore-control"))
        data.add_response("get_recommendation", doc, {"recommendationId": doc["recommendationId"]})
        refs = [
            source["configurationBundle"]["versionId"]
            for config in doc["recommendationConfig"].values()
            for source in config.values()
            if isinstance(source, dict) and "configurationBundle" in source
        ] + [
            r["configurationBundle"]["versionId"]
            for r in (doc.get("recommendationResult") or {}).values()
            if "configurationBundle" in r
        ]
        by_version = {
            _load(n)["versionId"]: _load(n)
            for n in ("bundle_version_control.json", "bundle_version_treatment.json")
        }
        for version in refs:
            control.add_response(
                "get_configuration_bundle_version",
                _response(by_version[version]),
                {"bundleId": _BUNDLE_ID, "versionId": version},
            )
        with data, control:
            record = read_recommendation(clients, doc["recommendationId"])
    append_records(root, "agentcore-recommendation", [record])
    print("\n".join(recommendation_lines(record)))
    return record


def retest(root: Path) -> int:
    """Record the stopped test, read its arms, register the target, re-test on the H1 roots."""
    import contextlib
    import io

    from calibrated_agent_evaluation.cli import main as calibrated_agent_evaluation_main
    from calibrated_agent_evaluation.cli.records import load_records_of

    with contextlib.redirect_stdout(io.StringIO()):
        if not load_records_of(root, "agentcore-abtest"):
            record_test(root, "stopped")
        if not load_records_of(root, "agentcore-arms"):
            read_window(root, "main")
    register(root, frozen=datetime.fromisoformat("2026-10-01T09:00:00+00:00"))
    roots = build_set("h1", root / "replay")
    return calibrated_agent_evaluation_main(
        [
            "--root",
            str(root),
            "audit",
            "retest",
            _TARGET,
            "--ab-test-id",
            "fixture-abtest-stopped",
            "--before-root",
            str(roots["control"]),
            "--after-root",
            str(roots["treatment"]),
            "--development-root",
            str(roots["development"]),
            "--verdicts",
            "human",
        ]
    )


def report(root: Path) -> str:
    """The re-test, then the report, which must equal ``report.txt`` with its values filled."""
    import contextlib
    import io

    from calibrated_agent_evaluation.cli import main as calibrated_agent_evaluation_main
    from calibrated_agent_evaluation.cli.records import load_records_of

    with contextlib.redirect_stdout(io.StringIO()):
        if retest(root) != 0:
            raise SystemExit("report step: the re-test did not accept the H1 change")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = calibrated_agent_evaluation_main(
            ["--root", str(root), "audit", "report", "--ab-test-id", "fixture-abtest-stopped"]
        )
    if code != 0:
        raise SystemExit(f"report step: calibrated-agent-evaluation audit report exited {code}")
    [audit] = load_records_of(root, "agentcore-audit")
    [claim] = load_records_of(root, "agentcore-abtest")
    golden = (_GOLDEN / "report.txt").read_text(encoding="utf-8")
    for placeholder, value in {
        "<improve-id>": audit.improve_id,
        "<audit-id>": audit.audit_id,
        "<frozen-at>": audit.prereg_frozen_at,
        "<read-at>": claim.read_at,
        "<claim-record>": claim.record_id,
        "<arms-read>": str(audit.arms_read_id),
    }.items():
        golden = golden.replace(placeholder, value)
    if out.getvalue() != golden:
        raise SystemExit("report step: the report differs from fixtures/expected/audit/report.txt")
    print(out.getvalue(), end="")
    return out.getvalue()


def langfuse_seed(project: str, source: Path) -> int:
    """Write one replay root's traces and verdicts into a Langfuse project; returns the count."""
    from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend
    from calibrated_agent_evaluation.backends.local import LocalBackend

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        raise SystemExit(f"set the {project} project's LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    with LocalBackend(root=source) as local:
        traces = list(local.query())
        labels = [lb for t in traces for lb in local.get_scores(t.trace_id)]
    # Leaving the block flushes the queued writes, so a write that failed surfaces here.
    with LangfuseBackend.from_credentials(pk, sk, host, timeout_seconds=10) as target:
        for trace in traces:
            target.put_trace(trace)
        for label in labels:
            target.put_score(label)
    print(f"wrote {len(traces)} trace(s) and {len(labels)} verdict(s) to {project} at {host}")
    return len(traces)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--root", default=None, help="the record root")
    parser.add_argument(
        "step",
        choices=(
            "abtest",
            "arms",
            "recommendation",
            "replay",
            "retest",
            "report",
            "langfuse-seed",
            "guard",
        ),
    )
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--claim-record", default=None)
    parser.add_argument("--applied-bundle-attribute", default=None)
    parser.add_argument("--applied-bundle-require", action="append", default=[])
    parser.add_argument("--set", dest="replay_set", choices=sorted(_SETS), default="h1")
    parser.add_argument("--to", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--from", dest="source", default=None)
    args = parser.parse_args()
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.cli.errors import CliError

    with tempfile.TemporaryDirectory(prefix="calibrated-agent-evaluation-audit-") as tmp:
        given = args.root or os.environ.get("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT")
        root = Path(given) if given else Path(tmp)
        try:
            return _run(args, root)
        except (AgentCoreError, CliError) as exc:
            print(f"refused: {type(exc).__name__}: {exc}")
            return 1


def _run(args: argparse.Namespace, root: Path) -> int:
    from calibrated_agent_evaluation.cli.records import load_records_of

    if args.step == "abtest":
        fixture = args.fixture or "stopped"
        record = record_test(root, fixture)
        if fixture == "stopped":
            check_claim(record)
    elif args.step == "arms":
        fixture = args.fixture or "main"
        if not load_records_of(root, "agentcore-abtest"):
            check_claim(record_test(root, "stopped"))
        require = tuple(tuple(r.split("=", 1)) for r in args.applied_bundle_require)
        read = read_window(
            root,
            fixture,
            claim_record=args.claim_record,
            attribute=args.applied_bundle_attribute,
            require=require,  # type: ignore[arg-type]
        )
        if fixture in _ARM_COUNTS and args.applied_bundle_attribute is None:
            check_counts(read, fixture)
    elif args.step == "recommendation":
        record_recommendation(root, args.fixture or "completed")
    elif args.step == "replay":
        if not args.to:
            raise SystemExit("replay needs --to DIR")
        for name, path in build_set(args.replay_set, Path(args.to)).items():
            print(f"{name}: {path}")
    elif args.step == "retest":
        return retest(root)
    elif args.step == "report":
        report(root)
    elif args.step == "langfuse-seed":
        if not args.project or not args.source:
            raise SystemExit("langfuse-seed needs --project and --from")
        langfuse_seed(args.project, Path(args.source))
    else:
        show_guard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
