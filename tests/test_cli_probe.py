# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation probe` black-box probing tests.

`calibrated-agent-evaluation probe --endpoint <url> --authorized` runs the real engine
(`calibrated_agent_evaluation.probe.probe`) against the target, persists the probed traces into the
backend, and exits 0. `--endpoint` is required (a bare `calibrated-agent-evaluation probe` is a
usage error, exit 2). Auth / reachability / missing-extra failures are typed ProbeErrors the CLI
boundary turns into exit 1 (no traceback). The network call is stubbed by patching the engine's
default stdlib transport, so these run hermetically.
"""

from __future__ import annotations

import sys
import urllib.error

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE


def _sse_transport(url, headers, body, timeout):
    """Stub transport mirroring (status, line-iterator); returns a small SSE body."""
    return 200, [
        b'data: {"type": "tool_use", "tool_name": "lookup_order"}',
        b'data: {"response": "I help with orders."}',
    ]


def _install_stub_transport(monkeypatch, transport):
    """Point the engine's default transport at ``transport`` (no real socket)."""
    monkeypatch.setattr("calibrated_agent_evaluation.probe.adapter._urllib_transport", transport)


def test_probe_endpoint_runs_engine_persists_traces(seeded_local_root, monkeypatch, capsys):
    """Positive: probing a reachable endpoint stores probed traces and reports observed tools."""
    _install_stub_transport(monkeypatch, _sse_transport)

    rc = main(["probe", "--endpoint", "https://stub.local/invoke", "--authorized"])

    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "https://stub.local/invoke" in out
    assert "probed trace" in out
    assert "lookup_order" in out  # the observed tool (lower bound)
    assert "LimitsFraming" not in out
    assert "=True" not in out
    assert "a refusal is not proof the agent cannot do something" in out

    # The probed traces are now first-class in the backend for the downstream stages.
    from calibrated_agent_evaluation.backends.local import LocalBackend

    backend = LocalBackend(root=seeded_local_root)
    probed = [t for t in backend.query() if t.ownership == "probed"]
    assert probed, "expected the probed traces to be persisted into the backend"


def test_probe_without_authorization_refuses_exit_one(seeded_local_root, monkeypatch, capsys):
    """Negative: no --authorized → AuthorizationError → exit 1, and nothing is sent."""
    sent = False

    def _tripwire(url, headers, body, timeout):
        nonlocal sent
        sent = True
        return 200, []

    _install_stub_transport(monkeypatch, _tripwire)

    rc = main(["probe", "--endpoint", "https://stub.local/invoke"])

    assert rc == EXIT_DOMAIN_ERROR
    assert sent is False, "the authorization gate must refuse before any request is sent"
    assert "Probing https://" not in capsys.readouterr().err, "no status line before the refusal"


def test_probe_unreachable_endpoint_exits_one(seeded_local_root, monkeypatch):
    """Negative: an unreachable endpoint → EndpointUnreachableError → exit 1 (no traceback)."""

    def _dead(url, headers, body, timeout):
        raise urllib.error.URLError("connection refused")

    _install_stub_transport(monkeypatch, _dead)

    rc = main(["probe", "--endpoint", "https://stub.local/invoke", "--authorized"])

    assert rc == EXIT_DOMAIN_ERROR


def test_probe_no_endpoint_non_tty_refuses_without_hanging(seeded_local_root, monkeypatch):
    """Negative: no --endpoint + non-TTY stdin → typed CliError exit 1, never blocks."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    # If the handler ever called input() here it would hang the test; make that an explicit failure.
    monkeypatch.setattr(
        "builtins.input", lambda *_a, **_k: pytest.fail("must not prompt on a non-TTY")
    )

    rc = main(["probe"])

    assert rc == EXIT_USAGE


def test_probe_wss_without_extra_exits_one(seeded_local_root, monkeypatch):
    """Negative: a wss:// endpoint with [probe-ws] absent → ProbeMissingExtraError → exit 1.

    websockets is installed in the dev env, so the real detection won't fire; simulate the
    absent-extra refusal at its source and assert the CLI surfaces it as a typed exit 1, not a
    silent HTTP fallback.
    """
    from calibrated_agent_evaluation.probe import ProbeMissingExtraError

    def _missing() -> None:
        raise ProbeMissingExtraError(
            "the [probe-ws] extra is required; pip install calibrated-agent-evaluation[probe-ws]"
        )

    monkeypatch.setattr(
        "calibrated_agent_evaluation.probe.harness.require_probe_ws_extra", _missing
    )

    rc = main(["probe", "--endpoint", "wss://stub.local/agent", "--authorized"])

    assert rc == EXIT_DOMAIN_ERROR


def test_probe_conflicting_auth_variants_exit_one(seeded_local_root):
    """Negative: selecting two auth variants → typed CliError → exit 1, not a silent ignore."""
    rc = main(
        [
            "probe",
            "--endpoint",
            "https://stub.local/invoke",
            "--authorized",
            "--auth-bearer-env",
            "TOK_A",
            "--auth-header-env",
            "HDRS_B",
        ]
    )

    assert rc == EXIT_USAGE


def _script(monkeypatch, answers, *, tty=True):
    """Drive interactive mode: fake a TTY and feed scripted input() lines."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(it))


def test_probe_interactive_collects_inputs_and_probes(seeded_local_root, monkeypatch, capsys):
    """Positive: interactive `calibrated-agent-evaluation probe` collects inputs and runs the same
    probe core."""
    _install_stub_transport(monkeypatch, _sse_transport)
    monkeypatch.setenv("PROBE_TOK", "tok-value")
    # endpoint, auth type, bearer env-var NAME, authorized, one seed, blank to start.
    _script(
        monkeypatch,
        ["https://stub.local/invoke", "bearer", "PROBE_TOK", "y", "What can you do?", ""],
    )

    rc = main(["probe"])

    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "probed trace" in out

    from calibrated_agent_evaluation.backends.local import LocalBackend

    backend = LocalBackend(root=seeded_local_root)
    assert [t for t in backend.query() if t.ownership == "probed"]


def test_probe_interactive_secret_not_leaked_into_stored_traces(seeded_local_root, monkeypatch):
    """The credential is referenced by env-var name and never appears in a stored trace."""
    _install_stub_transport(monkeypatch, _sse_transport)
    secret = "super-secret-do-not-store"
    monkeypatch.setenv("PROBE_TOK", secret)
    _script(
        monkeypatch,
        ["https://stub.local/invoke", "bearer", "PROBE_TOK", "y", "hello", ""],
    )

    rc = main(["probe"])
    assert rc == EXIT_OK

    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.schema.serde import to_dict

    backend = LocalBackend(root=seeded_local_root)
    probed = [t for t in backend.query() if t.ownership == "probed"]
    assert probed
    for trace in probed:
        assert secret not in str(to_dict(trace)), "the secret must never appear in a stored trace"


def test_probe_interactive_declining_authorization_sends_nothing(seeded_local_root, monkeypatch):
    """Negative: declining the authorization prompt refuses and sends nothing."""
    sent = False

    def _tripwire(url, headers, body, timeout):
        nonlocal sent
        sent = True
        return 200, []

    _install_stub_transport(monkeypatch, _tripwire)
    # endpoint, auth type none, authorized = "n", blank to end the seed loop.
    _script(monkeypatch, ["https://stub.local/invoke", "none", "n", ""])

    rc = main(["probe"])

    assert rc == EXIT_DOMAIN_ERROR
    assert sent is False

    from calibrated_agent_evaluation.backends.local import LocalBackend

    backend = LocalBackend(root=seeded_local_root)
    assert not [t for t in backend.query() if t.ownership == "probed"]


def test_probe_custom_prompts_are_used(seeded_local_root, monkeypatch, capsys):
    """--prompt overrides the default seeds: two prompts → two probed traces."""
    _install_stub_transport(monkeypatch, _sse_transport)

    rc = main(
        [
            "probe",
            "--endpoint",
            "https://stub.local/invoke",
            "--authorized",
            "--prompt",
            "What is my order status?",
            "--prompt",
            "Cancel order 42",
        ]
    )

    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "2 probed trace(s)" in out


def test_a_url_probe_without_timeout_still_waits_30_seconds(seeded_local_root, monkeypatch):
    """A URL probe given no --timeout waits 30 seconds for each request, as it always has."""
    timeouts: list[float] = []

    def _recording(url, headers, body, timeout):
        timeouts.append(timeout)
        return _sse_transport(url, headers, body, timeout)

    _install_stub_transport(monkeypatch, _recording)

    rc = main(["probe", "--endpoint", "https://stub.local/invoke", "--authorized"])

    assert rc == EXIT_OK
    assert timeouts, "the probe sent nothing"
    assert set(timeouts) == {30.0}


@pytest.mark.parametrize(
    "flag", ["--timeout", "--battery", "--trials", "--allow-use-restricted", "--grader-model"]
)
def test_the_probe_help_says_the_timeout_and_battery_flags_are_for_a_url_probe_only(flag, capsys):
    """An AgentCore probe refuses these flags, so the help says which probe reads them."""
    with pytest.raises(SystemExit) as exit_info:
        main(["probe", "--help"])
    assert exit_info.value.code == EXIT_OK
    out = capsys.readouterr().out
    # Each option's entry starts on a line indented by two spaces; its help wraps below it.
    entries: dict[str, list[str]] = {}
    entry: list[str] = []
    for line in out.splitlines():
        if line.startswith("  -"):
            entry = entries.setdefault(line.split()[0].rstrip(","), [])
            entry.append(line)
        elif line.startswith("   "):
            entry.append(line)
    assert flag in entries, f"{flag} is not in the probe help"
    assert "URL probe only" in " ".join(" ".join(entries[flag]).split())
