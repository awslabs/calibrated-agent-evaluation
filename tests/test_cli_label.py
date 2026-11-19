# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation label` tests.

Default (no --serve) prints the launch command + URL bound to the backend (positive). --serve with
the [ui] extra absent surfaces a typed install-hint (exit 1), never a raw ImportError/traceback.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE


def test_label_prints_launch_command_and_url(seeded_local_root, capsys):
    """Positive: `calibrated-agent-evaluation label` prints the exact launch command + URL for the
    backend."""
    rc = main(["label"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "label --serve" in out
    assert "http://127.0.0.1:5001" in out


def test_label_serve_without_ui_extra_gives_typed_hint(seeded_local_root, monkeypatch, capsys):
    """Edge: --serve with [ui] absent → typed CliError install-hint, exit 1, no traceback."""
    import calibrated_agent_evaluation.cli.stages.label as label_stage
    from calibrated_agent_evaluation.cli.config import Degradation

    # Force the UI extra to look absent regardless of the dev env (which has fasthtml installed).
    monkeypatch.setattr(
        label_stage,
        "probe_extra",
        lambda _m: Degradation(
            capability="fasthtml", available=False, install_hint=install_hint("ui")
        ),
    )

    rc = main(["label", "--serve"])
    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert install_hint("ui") in err
    assert "Traceback (most recent call last)" not in err


def test_label_serve_without_a_rater_is_refused_before_serving(seeded_local_root, capsys):
    """A session must know its rater; it is asked for, never guessed."""
    rc = main(["label", "--serve"])
    assert rc == EXIT_USAGE
    assert "--rater" in capsys.readouterr().err


def test_label_prints_a_launch_command_that_names_the_rater_flag(seeded_local_root, capsys):
    assert main(["label"]) == EXIT_OK
    assert "--rater" in capsys.readouterr().out


def _serve(monkeypatch, root, argv) -> tuple[int, Any]:
    """Run `calibrated-agent-evaluation label --serve ...` with uvicorn stubbed; return (exit code,
    the app it built)."""
    import uvicorn

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **_kw: served.setdefault("app", app))
    monkeypatch.chdir(root)  # the default session log lands under the temp root, not the repo
    rc = main(["label", "--serve", *argv])
    return rc, served.get("app")


def test_label_serve_writes_its_session_log_under_the_local_root_by_default(
    seeded_local_root, monkeypatch, capsys
):
    """The CLI always persists per-label timing."""
    rc, _app = _serve(monkeypatch, seeded_local_root, ["--rater", "alice"])
    assert rc == EXIT_OK
    assert (seeded_local_root / "review-sessions.jsonl").exists()
    assert "review-sessions.jsonl" in capsys.readouterr().err


def test_label_trace_narrows_the_session_to_that_trace(seeded_local_root, monkeypatch):
    """--trace was parsed and then ignored."""
    rc, app = _serve(monkeypatch, seeded_local_root, ["--rater", "alice", "--trace", "CLM-4471"])
    assert rc == EXIT_OK
    assert app.state.review_session.pending == ["CLM-4471"]


def test_label_cell_and_limit_narrow_the_session(seeded_local_root, monkeypatch):
    rc, app = _serve(
        monkeypatch,
        seeded_local_root,
        ["--rater", "alice", "--cell", "feature=claim_adjudication", "--limit", "1"],
    )
    assert rc == EXIT_OK
    assert app.state.review_session.pending == ["CLM-4471"]


def test_label_frame_matching_nothing_is_refused(seeded_local_root, monkeypatch, capsys):
    rc, app = _serve(monkeypatch, seeded_local_root, ["--rater", "alice", "--frame", "discovery"])
    assert rc == EXIT_DOMAIN_ERROR
    assert app is None
    assert "matches the requested scope" in capsys.readouterr().err


def test_label_a_malformed_cell_is_refused(seeded_local_root, monkeypatch, capsys):
    rc, _app = _serve(monkeypatch, seeded_local_root, ["--rater", "alice", "--cell", "feature"])
    assert rc == EXIT_USAGE
    assert "KEY=VALUE" in capsys.readouterr().err


def test_label_resumes_by_default_and_relabel_includes_everything(
    seeded_local_root, monkeypatch, capsys
):
    """The seeded gold labels are rater-1's, so rater-1 has none left."""
    rc, app = _serve(monkeypatch, seeded_local_root, ["--rater", "rater-1"])
    assert rc == EXIT_OK
    assert app.state.review_session.pending == []
    assert "skipped 3 already labeled" in capsys.readouterr().err

    rc, app = _serve(monkeypatch, seeded_local_root, ["--rater", "rater-1", "--relabel"])
    assert sorted(app.state.review_session.pending) == ["CLM-4471", "ORD-1001", "ORD-1002"]


def test_label_serve_without_uvicorn_gives_the_typed_hint(seeded_local_root, monkeypatch, capsys):
    """Fasthtml present but uvicorn absent is still the [ui] hint."""
    import calibrated_agent_evaluation.cli.stages.label as label_stage
    from calibrated_agent_evaluation.cli.config import Degradation

    def _probe(module: str) -> Degradation:
        if module == "uvicorn":
            return Degradation(capability=module, available=False, install_hint=install_hint("ui"))
        return Degradation(capability=module, available=True, install_hint=None)

    monkeypatch.setattr(label_stage, "probe_extra", _probe)
    rc = main(["label", "--serve", "--rater", "alice"])
    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert install_hint("ui") in err
    assert "ModuleNotFoundError" not in err


def _printed_command(out: str) -> list[str]:
    import re
    import shlex

    match = re.search(r"run `([^`]+)`", out)
    assert match, out
    return shlex.split(match.group(1))


def test_the_printed_launch_command_runs_as_printed(seeded_local_root, monkeypatch, capsys):
    """The hint once put `--backend` after the subcommand and exited 2. Run the
    printed argv itself, only filling in the rater placeholder."""
    from calibrated_agent_evaluation.cli.stages.label import RATER_PLACEHOLDER

    assert main(["--backend", "local", "label", "--port", "5077"]) == EXIT_OK
    argv = _printed_command(capsys.readouterr().out)
    assert argv[0] == "calibrated-agent-evaluation"
    assert RATER_PLACEHOLDER in argv
    filled = ["alice" if a == RATER_PLACEHOLDER else a for a in argv[1:]]

    import uvicorn

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(kw))
    assert main(filled) == EXIT_OK
    assert served == {"host": "127.0.0.1", "port": 5077}


def test_the_printed_command_carries_an_explicit_root(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    assert main(["--root", str(root), "label"]) == EXIT_OK
    argv = _printed_command(capsys.readouterr().out)
    assert argv[argv.index("--root") + 1] == str(root)
    assert argv.index("--root") < argv.index("label")


def test_global_options_work_after_the_subcommand(seeded_local_root, capsys):
    """The form the old hint and the offline guide printed now parses too."""
    assert main(["label", "--backend", "local"]) == EXIT_OK


def test_serve_binds_the_requested_host_and_port(seeded_local_root, monkeypatch, capsys):
    rc, _app = _serve_kw(monkeypatch, seeded_local_root, ["--host", "127.0.0.1", "--port", "5099"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "http://127.0.0.1:5099" in out


def _serve_kw(monkeypatch, root, extra) -> tuple[int, dict[str, Any]]:
    import uvicorn

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(kw))
    monkeypatch.chdir(root)
    rc = main(["label", "--serve", "--rater", "alice", *extra])
    return rc, served


def test_a_port_already_in_use_is_a_typed_error_naming_it(seeded_local_root, monkeypatch, capsys):
    import socket

    import uvicorn

    called: list[bool] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: called.append(True))
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        rc = main(["label", "--serve", "--rater", "alice", "--port", str(port)])
    assert rc == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"127.0.0.1:{port}" in err
    assert "--port" in err
    assert "Traceback" not in err
    assert not called


def test_a_host_other_machines_can_reach_is_refused(seeded_local_root, monkeypatch, capsys):
    import uvicorn

    called: list[bool] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: called.append(True))
    for argv in (
        ["label", "--host", "0.0.0.0"],
        ["label", "--serve", "--rater", "alice", "--host", "::"],
    ):
        assert main(argv) == EXIT_DOMAIN_ERROR
        err = capsys.readouterr().err
        assert "ssh -L 5001:127.0.0.1:5001" in err
    assert not called


def test_the_ipv6_loopback_host_prints_a_bracketed_url(seeded_local_root, capsys):
    assert main(["label", "--host", "::1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "http://[::1]:5001" in out
