# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary: probe/ imports no vendor SDK.

Contract test — asserts a *public boundary*, not implementation internals (rubric rule 2 blessed).
Uses the ``ast`` module to inspect the ACTUAL import statements (not substrings, which would false-
match the word "inspect_evals" in a docstring). The two allowed transports in ``probe/`` are stdlib
`urllib` (HTTP) and `websockets` (WebSocket — a permissive network-transport library, not a
"vendor SDK"). The sole `inspect_evals` import lives in the sanctioned
`execute/benchmark_tasks.py`. `websockets` is permitted ONLY in `probe/ws_adapter.py`, so a
vendor-SDK leak or a spurious `websockets` import in any other `probe/` module is caught.
"""

from __future__ import annotations

import pathlib

from _imports import imported_roots

_PROBE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation" / "probe"
)
_FORBIDDEN_ROOTS = {"inspect_evals", "inspect_ai", "httpx", "requests"}
# `websockets` is a permissive transport library, allowed under `probe/` only in `ws_adapter.py`
# . Any other file importing it is a boundary leak and this test catches it.
_TRANSPORT_ROOTS = {"websockets"}
_WS_ADAPTER_MODULE = "ws_adapter.py"


def test_probe_imports_no_vendor_sdk() -> None:
    offenders: list[str] = []
    for path in _PROBE_DIR.rglob("*.py"):
        forbidden = imported_roots(path) & _FORBIDDEN_ROOTS
        offenders += [f"{path.name}: imports {mod}" for mod in sorted(forbidden)]
    assert offenders == [], f"vendor imports leaked into probe/: {offenders}"


def test_inspect_evals_confined_to_execute_loader() -> None:
    # The vendor import lives in the sanctioned execution plane, not probe/.
    loader = _PROBE_DIR.parent / "execute" / "benchmark_tasks.py"
    assert "inspect_evals" in loader.read_text(encoding="utf-8")


def test_websockets_confined_to_ws_adapter() -> None:
    # `websockets` is a permissive transport, allowed under `probe/` only in
    # `ws_adapter.py`. Any other probe/ module importing it is a boundary violation.
    offenders: list[str] = []
    for path in _PROBE_DIR.rglob("*.py"):
        if path.name == _WS_ADAPTER_MODULE:
            continue
        forbidden = imported_roots(path) & _TRANSPORT_ROOTS
        offenders += [f"{path.name}: imports {mod}" for mod in sorted(forbidden)]
    assert offenders == [], (
        f"websockets leaked outside ws_adapter.py: {offenders} "
        "(this transport is confined to the WS adapter module)"
    )


def test_policy_modules_import_nothing_from_the_agentcore_plugin() -> None:
    for name in ("policy.py", "policy_battery.py"):
        roots = imported_roots(_PROBE_DIR / name)
        assert {"boto3", "botocore"}.isdisjoint(roots)
        source = (_PROBE_DIR / name).read_text(encoding="utf-8")
        assert "calibrated_agent_evaluation.agentcore" not in source


def test_no_agentcore_attribute_name_appears_in_the_probe_package() -> None:
    offenders = [p.name for p in _PROBE_DIR.glob("*.py") if "aws.agentcore" in p.read_text("utf-8")]
    assert offenders == []


def test_importing_the_policy_modules_loads_no_aws_sdk() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import calibrated_agent_evaluation.probe.policy, "
        "calibrated_agent_evaluation.probe.policy_battery; "
        "sdks = ('boto3', 'botocore', 'bedrock_agentcore'); "
        "loaded = sorted(m for m in sdks if m in sys.modules); "
        "print(','.join(loaded))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == ""
