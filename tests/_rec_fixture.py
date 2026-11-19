# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the recommendation fixtures through the audit's read-only clients over ``Stubber``.

Not a test module. :func:`stub_recommendation` queues one ``GetRecommendation`` response and the
``GetConfigurationBundleVersion`` responses for the bundle versions it names, so a test can read
a fixture as ``calibrated-agent-evaluation audit recommendation`` does, with no request leaving the
process.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from botocore.stub import Stubber

REPO = Path(__file__).resolve().parents[1]
RECS = REPO / "fixtures" / "raw" / "agentcore" / "recommendations"
ABTEST = REPO / "fixtures" / "raw" / "agentcore" / "abtest"
_BUNDLES = {
    json.loads((ABTEST / n).read_text(encoding="utf-8"))["versionId"]: n
    for n in ("bundle_version_control.json", "bundle_version_treatment.json")
}


def rec(name: str, **overrides: Any) -> dict[str, Any]:
    """The ``GetRecommendation`` fixture ``name``, with top-level ``overrides``."""
    doc = json.loads((RECS / name).read_text(encoding="utf-8"))
    return {**doc, **overrides}


def online_config(name: str) -> dict[str, Any]:
    """The ``GetOnlineEvaluationConfig`` fixture ``name``."""
    return json.loads((RECS / name).read_text(encoding="utf-8"))


def _bundle_refs(doc: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for config in (doc.get("recommendationConfig") or {}).values():
        for source in config.values():
            if isinstance(source, dict) and "configurationBundle" in source:
                refs.append(source["configurationBundle"]["versionId"])
    for result in (doc.get("recommendationResult") or {}).values():
        if "configurationBundle" in result:
            refs.append(result["configurationBundle"]["versionId"])
    return refs


def queue_recommendation(clients: Any, doc: dict[str, Any]) -> list[Stubber]:
    """Queue ``doc`` and its bundle versions on ``clients``; returns the stubbers, not active."""
    data = Stubber(clients.client("bedrock-agentcore"))
    control = Stubber(clients.client("bedrock-agentcore-control"))
    data.add_response("get_recommendation", doc, {"recommendationId": doc["recommendationId"]})
    for version_id in _bundle_refs(doc):
        version = json.loads((ABTEST / _BUNDLES[version_id]).read_text(encoding="utf-8"))
        control.add_response(
            "get_configuration_bundle_version",
            version,
            {"bundleId": version["bundleId"], "versionId": version_id},
        )
    return [data, control]


@contextmanager
def offline_clients() -> Iterator[Any]:
    """The audit's real read-only clients, on fake credentials."""
    import os

    from calibrated_agent_evaluation.agentcore.optimization import audit_clients

    keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.pop("AWS_PROFILE", None)
    os.environ.update({"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing"})
    try:
        with audit_clients("us-west-2") as clients:
            yield clients
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def read_fixture(name: str, **overrides: Any) -> Any:
    """The claim record ``read_recommendation`` makes of the fixture ``name``."""
    from calibrated_agent_evaluation.agentcore.optimization import read_recommendation

    doc = rec(name, **overrides)
    with offline_clients() as clients:
        stubs = queue_recommendation(clients, doc)
        for stub in stubs:
            stub.activate()
        try:
            record = read_recommendation(clients, doc["recommendationId"])
        finally:
            for stub in stubs:
                stub.deactivate()
    return record
