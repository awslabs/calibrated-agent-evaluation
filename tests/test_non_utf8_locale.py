# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text files are written and read as UTF-8 whatever the process locale.

A file opened without an explicit encoding takes the locale's, so on a POSIX ``C`` locale (or any
non-UTF-8 one) a non-ASCII critique or tool result either failed to write or came back as a
different string. Each check runs in a subprocess with UTF-8 mode off and ``LC_ALL=C``, the setting
under which Python's preferred encoding is ASCII.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

_SCRIPT = textwrap.dedent(
    """
    import json
    import locale
    import sys
    from pathlib import Path

    from calibrated_agent_evaluation._recordstore import atomic_write_text
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.dumpload import dump, load
    from calibrated_agent_evaluation.ingest.core import run_ingest
    from calibrated_agent_evaluation.schema.types import Label

    assert locale.getpreferredencoding(False).lower() not in {"utf-8", "utf8"}, "locale is UTF-8"
    work = Path(sys.argv[1])
    text = "caf\\u00e9 \\u6771\\u4eac"

    target = work / "c.json"
    atomic_write_text(target, json.dumps({"turn": text}, ensure_ascii=False))
    assert target.read_bytes() == json.dumps({"turn": text}, ensure_ascii=False).encode("utf-8")

    source = LocalBackend(":memory:")
    run_ingest([Path(sys.argv[2])], backend=source)
    (trace,) = source.all_traces()
    source.put_score(
        Label(trace_id=trace.trace_id, span_id=None, kind="human", outcome="FAIL", critique=text)
    )
    for fmt in ("csv", "json"):
        out = work / f"dump-{fmt}"
        dump(source, out, fmt)
        restored = LocalBackend(":memory:")
        load(restored, out, fmt)
        assert [label.critique for label in restored.all_scores()] == [text], fmt
    print("ok")
    """
)


def test_writes_and_reads_are_utf8_under_an_ascii_locale(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("LC_ALL does not set the Windows code page")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "PYTHONUTF8"))}
    env.update({"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONIOENCODING": "utf-8"})
    fixture = _REPO / "fixtures" / "raw" / "otel" / "messages" / "swebench_tool_call.otlp.json"
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", _SCRIPT, str(tmp_path), str(fixture)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith("ok")


@pytest.mark.parametrize("locale_name", ["C", "en_US.ISO8859-1"])
def test_cli_output_survives_a_terminal_that_cannot_encode_it(
    tmp_path: Path, locale_name: str
) -> None:
    """The ingest summary prints an arrow, which neither ASCII nor Latin-1 can encode."""
    if os.name == "nt":
        pytest.skip("LC_ALL does not set the Windows code page")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "PYTHON"))}
    env.update({"LC_ALL": locale_name, "LANG": locale_name, "PYTHONUTF8": "0"})
    fixture = _REPO / "fixtures" / "raw" / "otel" / "ORD-1001.events.otlp.json"
    result = subprocess.run(
        [
            *(sys.executable, "-X", "utf8=0", "-m", "calibrated_agent_evaluation"),
            *("--root", str(tmp_path / "store"), "ingest", str(fixture)),
        ],
        capture_output=True,
        env=env,
        cwd=tmp_path,
        check=False,
        timeout=120,
    )
    stderr = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, stderr[-3000:]
    assert "Traceback" not in stderr
    assert b"1 ingested" in result.stdout
    assert b"\\u2192 ORD-1001" in result.stdout
