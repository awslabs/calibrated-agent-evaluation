# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the troubleshooting refusals that need some setup before the failing command.

Each case builds what it needs in a temporary directory, then prints the failing command and its
output. Run from the repository root::

    uv run --all-extras python docs/site/examples/guides/troubleshoot.py schema-version
    uv run --all-extras python docs/site/examples/guides/troubleshoot.py langfuse-unreachable
    uv run --all-extras python docs/site/examples/guides/troubleshoot.py agentcore-needs-yes
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")


def run(args: list[str], cwd: Path, env: dict[str, str], *, shown: str | None = None) -> int:
    done = subprocess.run(  # noqa: S603 - fixed argv
        [CALIBRATED_AGENT_EVALUATION, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if shown is not None:
        print(f"$ {shown}")
        output = (done.stdout + done.stderr).rstrip()
        print(re.sub(r"\[Errno \d+\]", "[Errno N]", output))
        print(f"(exit {done.returncode})")
    return done.returncode


def schema_version(work: Path, env: dict[str, str]) -> None:
    run(["--root", "store", "quickstart"], work, env)
    run(["--root", "store", "dump", "dump"], work, env)
    # Stand in for a dump written by an older Calibrated Agent Evaluation: mark every trace with
    # schema version 0, and record the file's new digest in the manifest as that release would.
    rows_path = work / "dump" / "trace.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    for row in rows:
        row["schema_version"] = "0"
    rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    manifest_path = work / "dump" / "dump-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["trace.json"]["sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shown = "calibrated-agent-evaluation --root restored load dump"
    run(["--root", "restored", "load", "dump"], work, env, shown=shown)


def langfuse_unreachable(work: Path, env: dict[str, str]) -> None:
    env = {
        **env,
        "LANGFUSE_HOST": "http://127.0.0.1:9",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-example",
        "LANGFUSE_SECRET_KEY": "sk-lf-example",
    }
    trace = Path.cwd() / "fixtures" / "raw" / "strands" / "CLM-4471.otlp.json"
    shown = (
        "LANGFUSE_HOST=http://127.0.0.1:9 calibrated-agent-evaluation --backend langfuse ingest "
        "fixtures/raw/strands/CLM-4471.otlp.json"
    )
    run(["--backend", "langfuse", "ingest", str(trace)], work, env, shown=shown)


_AGENTCORE_RUNTIME = (
    "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234"
)
_AGENTCORE_CASSETTE = "fixtures/raw/cassettes/agentcore_runtime_CLM-4471.cassette.json"
_AGENTCORE_FRAME = "fixtures/raw/frames/agentcore_CLM-4471.frame.json"


def _agentcore_auto_run(work: Path, env: dict[str, str], *flags: str) -> None:
    """``generate --run --executor agentcore`` in auto mode, a run that can call AWS."""
    # No credentials, whoever runs this: the refusals shown come before any AWS client is built,
    # and nothing here may reach an account if that ever changes.
    env = {k: v for k, v in env.items() if not k.startswith("AWS_")}
    env.update(
        CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE="auto",
        AWS_PROFILE="calibrated-agent-evaluation-docs-no-such-profile",
        AWS_CONFIG_FILE=os.devnull,
        AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        AWS_EC2_METADATA_DISABLED="true",
    )
    env.pop("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", None)
    options = [
        ("--runtime-arn", _AGENTCORE_RUNTIME),
        ("--payload-format", "agentcore_sse"),
        ("--cassette", _AGENTCORE_CASSETTE),
        ("--frame", _AGENTCORE_FRAME),
    ]
    args = ["generate", "--run", "--executor", "agentcore"]
    for flag, value in options:
        args += [flag, str(Path.cwd() / value) if flag in ("--cassette", "--frame") else value]
    command = " ".join(
        [
            "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=auto calibrated-agent-evaluation generate "
            "--run --executor "
            "agentcore",
            *(f"{flag} {value}" for flag, value in options),
            *flags,
        ]
    )
    # Broken before each option, shell-continued, as the docs show every long command.
    shown = re.sub(r" (?=--)", " \\\\\n    ", command)
    run([*args, *flags], work, env, shown=shown)


def agentcore_needs_yes(work: Path, env: dict[str, str]) -> None:
    _agentcore_auto_run(work, env)


CASES = {
    "schema-version": schema_version,
    "langfuse-unreachable": langfuse_unreachable,
    "agentcore-needs-yes": agentcore_needs_yes,
}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        CASES[sys.argv[1]](Path(tmp), dict(os.environ))
