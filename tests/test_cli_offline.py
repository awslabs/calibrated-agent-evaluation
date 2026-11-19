# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline / graceful-degradation tests.

The default install runs offline with zero third-party runtime deps; an absent extra surfaces a
typed install-hint (never a raw ImportError). A configured backend is not overridden by the local
default.
"""

from __future__ import annotations

import subprocess
import sys

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.backends import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK


def test_importing_cli_pulls_no_third_party():
    """Importing calibrated_agent_evaluation.cli loads no third-party runtime module (clean-sys
    subprocess)."""
    code = (
        "import sys, calibrated_agent_evaluation.cli\n"
        "from calibrated_agent_evaluation.cli import handlers, dispatch, config, parser, errors, "
        "backends\n"
        "bad = [m for m in ('boto3','inspect_ai','autoevals','langfuse','strands','fasthtml',"
        "'botocore','scipy','pyarrow') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "BAD:\n" in result.stdout or result.stdout.strip() == "BAD:", result.stdout


_AGENTCORE_ROOTS = ("boto3", "botocore", "bedrock_agentcore", "strands_evals")


def _loaded_agentcore_roots(code: str) -> list[str]:
    probe = (
        f"{code}\n"
        "import sys\n"
        f"roots = {_AGENTCORE_ROOTS!r}\n"
        "print('LOADED:' + ','.join(r for r in roots if r in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    line = result.stdout.strip().rpartition("LOADED:")[2]
    return [root for root in line.split(",") if root]


def test_importing_calibrated_agent_evaluation_and_the_agentcore_plugin_loads_no_aws_sdk():
    """The plug-in package imports without boto3, botocore, bedrock_agentcore or strands_evals."""
    assert (
        _loaded_agentcore_roots(
            "import calibrated_agent_evaluation, calibrated_agent_evaluation.agentcore, "
            "calibrated_agent_evaluation.cli"
        )
        == []
    )


def test_the_probe_sees_the_sdk_an_adapter_name_loads():
    """Teeth: touching an adapter name does load boto3, and the probe reports it."""
    loaded = _loaded_agentcore_roots(
        "import calibrated_agent_evaluation.agentcore\n"
        "calibrated_agent_evaluation.agentcore.AwsClients"
    )
    assert "boto3" in loaded
    assert "botocore" in loaded


def test_dump_stage_import_stays_offline_with_the_results_table_present():
    """`calibrated-agent-evaluation dump`'s module imports no stats chain; the results table loads
    lazily."""
    code = (
        "import sys, calibrated_agent_evaluation.dumpload, "
        "calibrated_agent_evaluation.cli.stages.dumpload\n"
        "bad = [m for m in ('scipy','statsmodels','pyarrow','inspect_ai') if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
        "from calibrated_agent_evaluation.dumpload import results_table\n"
        "print('LAZY:' + results_table.__module__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "BAD:\n" in result.stdout, result.stdout
    assert "LAZY:calibrated_agent_evaluation.dumpload.results" in result.stdout


def test_no_backend_configured_uses_local(seeded_local_root):
    """With the default (no --backend), the CLI persists+reads through LocalBackend."""
    # `calibrated-agent-evaluation dump` over the seeded local store succeeds — proving the local
    # backend is the default.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        assert main(["dump", d]) == EXIT_OK


def test_configured_backend_not_overridden():
    """Negative (complementary state): resolve_backend keeps an explicitly-passed backend."""
    from calibrated_agent_evaluation.cli.config import resolve_backend

    explicit = LocalBackend(root=":memory:")
    assert resolve_backend(explicit) is explicit


def _fake_probe_absent(*absent: str):
    """A probe_extra replacement reporting the named modules absent, the rest present."""
    from calibrated_agent_evaluation.cli.config import Degradation, extra_for_import
    from calibrated_agent_evaluation.cli.config import probe_extra as real_probe

    def _probe(m: str):
        if m in absent:
            return Degradation(
                capability=m,
                available=False,
                install_hint=install_hint(extra_for_import(m)),
            )
        return real_probe(m)

    return _probe


def test_absent_executor_extra_gives_typed_hint_not_importerror(
    seeded_local_root, monkeypatch, capsys
):
    """Negative: --run with the executor extra absent → typed hint, no ImportError.

    The methodology engine (allpairspy/scipy) is present here; only the strands executor extra is
    absent, so the failure is the executor's typed install-hint (agent resolution happens first, so
    a valid --agent spec is supplied to reach the executor check).
    """
    monkeypatch.setattr(
        "calibrated_agent_evaluation.cli.config.probe_extra", _fake_probe_absent("strands")
    )
    rc = main(["generate", "--run", "--executor", "strands", "--agent", "os:getcwd"])
    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert install_hint("strands") in err
    assert "ImportError" not in err
    assert "Traceback (most recent call last)" not in err


def test_absent_methodology_engine_gives_typed_hint_not_importerror(
    seeded_local_root, monkeypatch, capsys
):
    """Negative: on a bare install `calibrated-agent-evaluation generate` needs [sampling]/[stats] —
    typed hint.

    The sampling/synth engine (allpairspy/scipy) lives in optional extras, not the empty default
    tree. Probing them absent must yield a typed ``CliError`` naming the extra — never a raw
    ``ModuleNotFoundError`` traceback (the gap the convention review caught).
    """
    absent = _fake_probe_absent("allpairspy", "scipy")
    monkeypatch.setattr("calibrated_agent_evaluation.cli.config.probe_extra", absent)
    rc = main(["generate"])
    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert any(install_hint(e) in err for e in ("sampling", "stats"))
    assert "ModuleNotFoundError" not in err
    assert "Traceback (most recent call last)" not in err


def test_run_needs_stats_engine_typed_hint(seeded_local_root, monkeypatch, capsys):
    """Negative: `calibrated-agent-evaluation run` needs the [stats] engine — a bare install gets a
    typed hint."""
    absent = _fake_probe_absent("scipy", "statsmodels")
    monkeypatch.setattr("calibrated_agent_evaluation.cli.config.probe_extra", absent)
    rc = main(["run", "--ci"])
    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert install_hint("stats") in err
    assert "ModuleNotFoundError" not in err
    assert "Traceback (most recent call last)" not in err


def test_offline_stage_completes_with_network_disabled(seeded_local_root, monkeypatch, capsys):
    """Edge: an offline stage completes with socket creation hard-disabled."""
    import socket

    def _no_network(*_a, **_k):
        raise AssertionError("offline stage must not open a socket")

    monkeypatch.setattr(socket, "socket", _no_network)
    # `calibrated-agent-evaluation generate` (offline template renderer, local backend) must
    # complete with no socket.
    rc = main(["generate"])
    assert rc == EXIT_OK
    assert "synthesized" in capsys.readouterr().out


def test_interval_carriers_import_without_the_stats_extra():
    """The record types, the probe and the clusterer import no scipy/statsmodels chain.

    `calibrated_agent_evaluation.calibrate` serves its engine lazily so modules that only carry an
    `Interval` or a `Bound` (probe pass^k, cluster saturation) stay offline-clean.
    """
    code = (
        "import sys, calibrated_agent_evaluation.calibrate.record, "
        "calibrated_agent_evaluation.probe.robustness, "
        "calibrated_agent_evaluation.cluster.saturation\n"
        "bad = [m for m in ('statsmodels',) if m in sys.modules]\n"
        "print('BAD:' + ','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "BAD:", result.stdout
