# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each optional extra installs what the modules it enables import.

The dev environment has every extra and every dependency group installed, so a module importing a
package its extra never declares still imports fine here. That is how ``[strands]`` shipped
without the OTLP encoder ``calibrated-agent-evaluation capture`` needs: ``uv sync --group corpus``
installed it for every developer, and the first clean install outside this repo failed.

Each test runs the import in a subprocess that hides every installed distribution outside the
extra's dependency closure, which is what a fresh
``pip install "calibrated-agent-evaluation[<extra>]"`` sees. The closure is read from the installed
metadata, so a dependency an upstream package declares counts and one it merely happened to pull in
here does not.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from _extras import NEEDS, all_modules, declared_extras, modules_for

# The path finder is swapped for one that resolves each import with the real path finder, then
# reports "not found" when the file belongs to a distribution outside the closure, which is what
# `importlib.util.find_spec` probes for optional dependencies see on a fresh install too. Files no
# distribution owns (the stdlib, the editable calibrated_agent_evaluation source tree) are always
# allowed.
_PROBE = textwrap.dedent(
    """
    import importlib.machinery
    import importlib.metadata as md
    import sys
    from pathlib import Path

    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    extras, mode, rest = set(filter(None, sys.argv[1].split(","))), sys.argv[2], sys.argv[3:]
    extra = ",".join(sorted(extras)) or "default"

    def requirements(dist, extras):
        for raw in md.requires(dist) or []:
            req = Requirement(raw)
            if req.marker is None:
                yield req
                continue
            if any(req.marker.evaluate({"extra": e}) for e in (extras or {""})):
                yield req

    closure = set()
    pending = [("calibrated-agent-evaluation", extras)]
    while pending:
        dist, wanted = pending.pop()
        for req in requirements(dist, wanted):
            name = canonicalize_name(req.name)
            if name not in closure:
                closure.add(name)
                pending.append((req.name, set(req.extras)))

    owner = {}
    for dist in md.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        for file in dist.files or []:
            owner[str(Path(file.locate()).resolve())] = name

    hidden = {}

    class Hide:
        def find_spec(self, fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
                return spec
            dist = owner.get(str(Path(spec.origin).resolve()))
            if dist is not None and dist not in closure and dist != "calibrated-agent-evaluation":
                hidden[fullname] = dist
                return None
            return spec

        def find_distributions(self, *args, **kwargs):
            return importlib.machinery.PathFinder.find_distributions(*args, **kwargs)

        def invalidate_caches(self):
            importlib.machinery.PathFinder.invalidate_caches()

    sys.meta_path[sys.meta_path.index(importlib.machinery.PathFinder)] = Hide()
    if mode == "cli":
        from calibrated_agent_evaluation.cli import main

        sys.exit(main(rest))
    if mode == "exec":
        exec(rest[0])
        sys.exit(0)
    import importlib
    for module in rest:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in hidden:
                raise
            raise ModuleNotFoundError(
                f"{exc.name} (from {hidden[exc.name]}, outside the [{extra}] closure)",
                name=exc.name,
            ) from exc
    print("OK")
    """
)


def _import_with_only(extra: str, *modules: str) -> subprocess.CompletedProcess[str]:
    return _probe(extra, "import", *modules)


def _cli_with_only(extras: str, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run ``calibrated-agent-evaluation *argv`` seeing only the closure of ``extras``
    (comma-separated, maybe empty)."""
    return _probe(extras, "cli", *argv)


def _exec_with_only(extras: str, code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` seeing only the closure of ``extras``."""
    return _probe(extras, "exec", code)


def _probe(extras: str, mode: str, *rest: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PROBE, extras, mode, *rest],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.slow
@pytest.mark.parametrize("extra", ["", *declared_extras()], ids=lambda e: e or "bare")
def test_an_install_with_one_extra_imports_every_module_it_enables(extra: str) -> None:
    """Every bare-install module, plus every module the extra enables, imports in its closure."""
    result = _import_with_only(extra, *modules_for(extra))
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("OK")


def test_the_extras_map_names_real_modules_and_declared_extras() -> None:
    modules = set(all_modules())
    assert set(NEEDS) <= modules, sorted(set(NEEDS) - modules)
    assert set(NEEDS.values()) <= set(declared_extras())


def test_every_public_agentcore_name_loads_with_only_the_agentcore_extra() -> None:
    """Each name the plug-in exports resolves inside the [agentcore] closure.

    Resolving a name imports the module behind it, so an exported name whose module reaches a
    package the extra doesn't install (the sampling stack behind
    ``calibrated_agent_evaluation.synth``, say), or that no module provides, fails here rather than
    on a user's first run.
    """
    import calibrated_agent_evaluation.agentcore

    names = list(calibrated_agent_evaluation.agentcore.__all__)
    code = (
        "import calibrated_agent_evaluation.agentcore\n"
        "for name in calibrated_agent_evaluation.agentcore.__all__:\n"
        "    getattr(calibrated_agent_evaluation.agentcore, name)\n"
        "print('RESOLVED:' + ','.join(calibrated_agent_evaluation.agentcore.__all__))\n"
    )

    result = _exec_with_only("agentcore", code)

    assert names
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"RESOLVED:{','.join(names)}" in result.stdout


def test_the_probe_refuses_a_module_outside_the_closure() -> None:
    """The probe itself has teeth: a module only the [langfuse] extra brings is hidden here."""
    result = _import_with_only("strands", "langfuse")
    assert result.returncode != 0
    assert "outside the [strands] closure" in result.stderr


def test_an_agentcore_adapter_on_a_bare_install_raises_the_typed_missing_extra_error() -> None:
    """Without [agentcore], asking for an adapter names the extra instead of failing an import."""
    code = (
        "from calibrated_agent_evaluation.agentcore import AgentCoreExtraMissing\n"
        "try:\n"
        "    from calibrated_agent_evaluation.agentcore import AwsClients\n"
        "except AgentCoreExtraMissing as exc:\n"
        "    assert exc.extras == ('agentcore',), exc.extras\n"
        "    assert isinstance(exc.__cause__, ModuleNotFoundError)\n"
        "    print('REFUSED:' + str(exc))\n"
        "else:\n"
        "    raise SystemExit('AwsClients imported without the extra')\n"
    )
    result = _exec_with_only("", code)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "REFUSED:" in result.stdout
    assert "[agentcore]" in result.stdout
    assert "Traceback" not in result.stderr


def test_code_interpreter_sandbox_on_a_bare_install_raises_missing_extra() -> None:
    """Selecting the Code Interpreter backend without [agentcore] names the extra; no boto3."""
    code = (
        "from calibrated_agent_evaluation.agentcore import AgentCoreExtraMissing\n"
        "try:\n"
        "    from calibrated_agent_evaluation.agentcore import CodeInterpreterSandbox\n"
        "except AgentCoreExtraMissing as exc:\n"
        "    assert exc.extras == ('agentcore',), exc.extras\n"
        "    assert isinstance(exc.__cause__, ModuleNotFoundError)\n"
        "    print('REFUSED:' + str(exc))\n"
        "else:\n"
        "    raise SystemExit('CodeInterpreterSandbox imported without the extra')\n"
    )
    result = _exec_with_only("", code)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "REFUSED:" in result.stdout
    assert "[agentcore]" in result.stdout
    assert "Traceback" not in result.stderr


def test_doctor_agentcore_on_a_bare_install_prints_the_install_command() -> None:
    result = _cli_with_only("", "doctor", "--agentcore", "--region", "us-west-2")

    assert result.returncode == 1, result.stderr[-2000:]
    assert "Traceback" not in result.stderr
    assert "[agentcore]" in result.stderr
    assert "agentcore" in result.stderr.split("Next:", 1)[-1]


_REPO = Path(__file__).resolve().parent.parent
_JUDGE = _REPO / "fixtures" / "judge-files" / "orderbot-refund-resolved.json"
_CLEAN_JUDGE = _REPO / "fixtures" / "judge-files" / "fixture-clean-judge.json"
_CASSETTES = _REPO / "fixtures" / "raw" / "cassettes"
_BIAS = _REPO / "fixtures" / "bias"


@pytest.fixture
def seeded_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A quickstart store whose judges hold the records `calibrated-agent-evaluation judge` and
    `calibrated-agent-evaluation bias` admit on.

    Seeded with the full dev environment, so only the command under test runs inside the closure.
    Fresh per test: `calibrated-agent-evaluation calibrate` appends a newer REFUSED record that
    `calibrated-agent-evaluation judge` would honor.
    """
    from _flagship_helpers import certified_record, clear_battery
    from calibrated_agent_evaluation.cli import main
    from calibrated_agent_evaluation.cli.judgefile import load_judge_file
    from calibrated_agent_evaluation.cli.records import append_records

    root = tmp_path_factory.mktemp("closure-root")
    assert main(["--root", str(root), "quickstart"]) == 0
    judge = load_judge_file(_JUDGE).judge
    clean = load_judge_file(_CLEAN_JUDGE).judge
    append_records(root, "calibration", [certified_record(judge), certified_record(clean)])
    append_records(root, "bias", [clear_battery(judge)])
    return root


# (extras, argv after `--root ROOT`, expected exit). The extras are the minimal set each command
# documents; a raw ModuleNotFoundError under that set means the docs or an import chain is wrong.
_COMMANDS: list[tuple[str, tuple[str, ...], int]] = [
    ("", ("ingest", "--dry-run", str(_REPO / "fixtures" / "raw" / "otel")), 0),
    ("", ("label",), 0),
    ("stats", ("run", "--verdicts", "human", "--ci"), 1),
    (
        "stats",
        (
            "calibrate",
            "--judge",
            str(_JUDGE),
            "--cassette",
            str(_CASSETTES / "judge_orderbot_resolved.cassette.json"),
            "--cell",
            "feature=refund_handling",
        ),
        1,
    ),
    (
        "stats",
        (
            "bias",
            "--judge",
            str(_CLEAN_JUDGE),
            "--cassette",
            str(_CASSETTES / "bias_fixture_judges_cli.cassette.json"),
            "--probes",
            str(_BIAS / "probes"),
            "--thresholds",
            str(_BIAS / "thresholds.json"),
        ),
        0,
    ),
    (
        "stats",
        (
            "judge",
            "--judge",
            str(_JUDGE),
            "--cassette",
            str(_CASSETTES / "judge_orderbot_resolved.cassette.json"),
            "--trace",
            "ORD-1001",
        ),
        0,
    ),
    ("stats", ("cluster",), 0),
    ("stats", ("dump", "{tmp}/dump"), 0),
    ("sampling", ("define", "--out", "{tmp}/frame.json"), 0),
    (
        "stats",
        ("improve", "--register", "trusted_broken_policy_tool", "--min-reduction", "0.02"),
        0,
    ),
    ("scorers", ("score", "--scorer", "exact", "--mode", "trusted_broken_policy_tool"), 0),
    ("", ("score", "--scorer", "exact", "--mode", "trusted_broken_policy_tool"), 1),
    ("stats", ("judge", "--check", str(_JUDGE)), 0),
    ("", ("judge", "--check", str(_JUDGE)), 1),
    ("stats", ("dump", "--judges-only", "--judge", str(_JUDGE), "{tmp}/kit"), 0),
    ("", ("dump", "--judges-only", "--judge", str(_JUDGE), "{tmp}/kit"), 1),
    ("", ("online", "create-evaluator", "--judge", str(_JUDGE), "--region", "us-east-1"), 1),
    ("", ("audit", "abtest", "--ab-test-id", "ab-1", "--region", "us-east-1"), 1),
    ("", ("audit", "recommendation", "--recommendation-id", "rec-1", "--region", "us-east-1"), 1),
]

# The extra a refused row must name in its install command.
_REFUSED_FOR: dict[tuple[str, str], str] = {
    ("", "score"): "scorers",
    ("", "judge"): "stats",
    ("", "dump"): "stats",
    ("", "online"): "agentcore",
    ("", "audit"): "agentcore",
}


@pytest.mark.parametrize(
    ("extras", "argv", "expected"),
    _COMMANDS,
    ids=[f"{argv[0]}[{extras or 'default'}]" for extras, argv, _ in _COMMANDS],
)
def test_cli_command_runs_with_its_documented_extras(
    seeded_root: Path, tmp_path: Path, extras: str, argv: tuple[str, ...], expected: int
) -> None:
    """Each command reaches its own verdict with only its documented extras installed."""
    args = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    result = _cli_with_only(extras, "--root", str(seeded_root), *args)
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
    assert result.returncode == expected, (result.stdout[-2000:], result.stderr[-2000:])
    refused_for = _REFUSED_FOR.get((extras, argv[0]))
    if refused_for is not None:
        # The install command's shape depends on how this checkout is installed.
        named = (f"[{refused_for}]", f"--extra {refused_for}")
        assert any(n in result.stderr for n in named), result.stderr[-2000:]


def test_a_stage_that_reaches_a_missing_engine_gets_its_install_command() -> None:
    """The dispatch backstop: a stage importing a [stats] module before checking for the extra is
    refused with the install command, not a traceback. The import here is real; only the stage
    that makes it is stood in for."""
    code = (
        "import calibrated_agent_evaluation.cli.dispatch as dispatch\n"
        "def stage(args):\n"
        "    import calibrated_agent_evaluation.stats.drift\n"
        "dispatch._load_handler = lambda command: stage\n"
        "raise SystemExit(dispatch.main(['cluster']))\n"
    )
    result = _exec_with_only("", code)
    assert result.returncode == 1, result.stderr[-2000:]
    assert "Traceback" not in result.stderr, result.stderr[-2000:]
    assert any(n in result.stderr for n in ("[stats]", "--extra stats")), result.stderr


def test_a_missing_module_no_extra_provides_still_propagates() -> None:
    code = (
        "import calibrated_agent_evaluation.cli.dispatch as dispatch\n"
        "def stage(args):\n"
        "    import calibrated_agent_evaluation_no_such_module\n"
        "dispatch._load_handler = lambda command: stage\n"
        "dispatch.main(['cluster'])\n"
    )
    result = _exec_with_only("", code)
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr


@pytest.mark.parametrize("scorer", ["code", "sql"])
def test_an_execution_scorer_needs_no_extra(seeded_root: Path, tmp_path: Path, scorer: str) -> None:
    """`score --scorer code|sql` is stdlib-only: on a bare install it reaches the scorer itself."""
    import json

    oracle = (
        {"tests": "assert True\n"}
        if scorer == "code"
        else {"sql": "SELECT 1", "schema_sql": "CREATE TABLE t (a INTEGER);", "seed_sql": ""}
    )
    expected = tmp_path / "expected.jsonl"
    expected.write_text(
        json.dumps({"trace_id": "ORD-1001", "expected": oracle}) + "\n", encoding="utf-8"
    )
    result = _cli_with_only(
        "",
        *("--root", str(seeded_root), "score", "--scorer", scorer, "--mode", "wrong_answer"),
        *("--expected", str(expected)),
    )
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
    assert "ModuleNotFoundError" not in result.stderr
    assert "extra" not in result.stderr, result.stderr[-2000:]


@pytest.mark.parametrize(
    "extras",
    ["sampling,stats", "bedrock,sampling,stats"],
    ids=["no-aws-sdk", "aws-sdk-from-bedrock"],
)
def test_generate_through_agentcore_without_its_extra_prints_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extras: str
) -> None:
    """With the engines `generate` needs but not [agentcore], the run is refused with the command
    that installs it, before anything is sent. The AWS SDK that [bedrock] brings is not enough:
    it doesn't make the [agentcore] extra installed."""
    from calibrated_agent_evaluation._install import install_hint

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)
    runtime = (
        "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
        "calibrated_agent_evaluation_fixture-AbCdEf1234"
    )
    frame = _REPO / "fixtures" / "raw" / "frames" / "agentcore_CLM-4471.frame.json"
    result = _cli_with_only(
        extras,
        *("--root", str(tmp_path / "root"), "generate", "--run", "--executor", "agentcore"),
        *("--runtime-arn", runtime, "--payload-format", "agentcore_sse", "--frame", str(frame)),
    )
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
    assert result.returncode == 1, (result.stdout[-2000:], result.stderr[-2000:])
    assert "[agentcore]" in result.stderr
    assert install_hint("agentcore") in result.stderr


@pytest.mark.parametrize(
    "extras",
    ["sampling,stats", "bedrock,sampling,stats"],
    ids=["no-aws-sdk", "aws-sdk-from-bedrock"],
)
def test_probe_through_agentcore_without_its_extra_prints_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extras: str
) -> None:
    """Without [agentcore], an authorized AgentCore probe is refused with the command that installs
    it, before its plan is printed and before anything is sent. The AWS SDK that [bedrock] brings
    is not enough. No --yes is passed, so the probe could not reach AWS even if the check moved."""
    from calibrated_agent_evaluation._install import install_hint

    for name in (
        "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime = (
        "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
        "calibrated_agent_evaluation_fixture-AbCdEf1234"
    )
    result = _cli_with_only(
        extras,
        *("--root", str(tmp_path / "root"), "probe", "--executor", "agentcore"),
        *("--runtime-arn", runtime, "--payload-format", "agentcore_sse", "--authorized"),
    )
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
    assert result.returncode == 1, (result.stdout[-2000:], result.stderr[-2000:])
    assert "[agentcore]" in result.stderr
    assert install_hint("agentcore") in result.stderr
    assert "this run can call AWS" not in result.stderr
    assert not (tmp_path / "root" / "records" / "agentcore").exists()


def test_dump_on_a_bare_install_names_the_extra_its_records_need(seeded_root: Path) -> None:
    """Records that need an engine to read give an install hint on a bare install, not a crash."""
    result = _cli_with_only("", "--root", str(seeded_root), "dump", str(seeded_root / "bare-dump"))
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
    assert result.returncode == 1
    assert "[stats]" in result.stderr or "--extra stats" in result.stderr


def test_dump_of_a_store_without_records_needs_no_extra(tmp_path: Path) -> None:
    """A traces-and-labels-only store dumps on a bare install."""
    from calibrated_agent_evaluation.cli import main

    root = tmp_path / "root"
    assert main(["--root", str(root), "quickstart"]) == 0
    result = _cli_with_only("", "--root", str(root), "dump", str(tmp_path / "dump"))
    assert result.returncode == 0, result.stderr[-3000:]
