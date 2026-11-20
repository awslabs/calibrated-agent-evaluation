# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Captured example output for the docs site, and the check that keeps it in step with the code.

Each :class:`Fragment` names a real, offline command: a ``calibrated-agent-evaluation`` invocation
against a fresh local store, an example program under ``docs/site/examples/``, a repo script, or a
generator that renders text (a diagram) from fixtures. Running a fragment captures its output,
scrubs what varies between runs (temp paths, run ids, timestamps, ANSI codes), and checks its exit
code. The result is committed under ``fixtures/expected/docs/`` and pages include it with
``pymdownx.snippets``.

The docs build (``gen_examples.py``) and ``tests/docs`` both regenerate every fragment and fail on
any difference, so a code change that moves a number or a message cannot leave a page stale.
Refresh the committed copies with ``just docs-goldens``.

Every run is offline by construction: cassettes replay only, AWS credentials are removed from the
environment and the profile points at one that does not exist, so a fragment that tries to reach
a model fails instead of spending.

    python docs/site/docs_examples.py            # check every fragment against its golden
    python docs/site/docs_examples.py --update   # rewrite the goldens
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = REPO_ROOT / "fixtures" / "expected" / "docs"
# The goldens the last update wrote; write_goldens only ever removes files listed here.
GOLDEN_MANIFEST = GOLDEN_DIR / "MANIFEST"
_ROOT_PLACEHOLDER = "$CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT"
_TIMEOUT_S = 120

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SCRUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(run|improve|judge|sample)-\d{8}T\d{6}Z-[0-9a-f]{6}"), r"\1-<id>"),
    # A record's id is random; trace ids elsewhere are content hashes and stay.
    (re.compile(r"(recorded )[0-9a-f]{32}\b"), r"\1<id>"),
    (re.compile(r"\b(agreement|codebook)-[0-9a-f]{16}\b"), r"\1-<id>"),
    # The audit's record ids are random; an arms read id is a digest of its settings and stays.
    (re.compile(r"\b(abtest|recommendation)-[0-9a-f]{16}\b"), r"\1-<id>"),
    (re.compile(r"\b(audit|improve)-[0-9a-f]{12}\b"), r"\1-<id>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"), "<time>"),
)
# argparse's invalid-choice error quotes each choice on some CPython patch releases and not on
# others, so the quotes are dropped: a golden then reads the same whichever interpreter made it.
_CHOICES = re.compile(r"\(choose from ([^)]*)\)")
_QUOTED = re.compile(r"'([^']*)'")


def _unquoted_choices(match: re.Match[str]) -> str:
    choices = _QUOTED.sub(r"\1", match.group(1))
    return f"(choose from {choices})"


_WORKDIR_ROOT = "demo"


class FragmentError(Exception):
    """A fragment could not be produced, exited with the wrong code, or differs from its golden."""


@dataclass(frozen=True)
class Step:
    """A command run before a fragment's own, against the same store; its output is not captured."""

    argv: tuple[str, ...]
    expected_exit: int = 0


@dataclass(frozen=True)
class Fragment:
    """One captured example.

    ``argv`` runs with the repo root as the working directory; a leading
    ``"calibrated-agent-evaluation"`` runs the installed CLI and ``"python"`` the current
    interpreter. ``shown`` is the command line printed above the output (``None`` prints none, for
    example programs whose source the page shows). ``seed_quickstart`` seeds the fresh store with
    the bundled corpus first. ``render`` replaces ``argv`` for fragments generated in-process from
    fixtures.

    ``workdir`` runs every command from an empty temp directory with
    ``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=demo``, the way a reader follows the judge kit, so
    paths in the output read ``demo/...``. ``{repo}`` in an argv stands for the repo root. ``setup``
    runs first, in order, against the same store.

    ``no_cassette_mode`` runs without ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE``, as a reader's
    shell has it, for a command that refuses replay outright (an AgentCore probe keeps no cassette).
    Every AWS setting stays offline, so such a fragment must still stop before any AWS call.
    """

    name: str
    argv: tuple[str, ...] = ()
    shown: str | None = None
    expected_exit: int = 0
    seed_quickstart: bool = False
    suffix: str = ".txt"
    render: Callable[[], str] | None = field(default=None, compare=False)
    workdir: bool = False
    setup: tuple[Step, ...] = ()
    no_cassette_mode: bool = False

    @property
    def golden(self) -> Path:
        return GOLDEN_DIR / f"{self.name}{self.suffix}"


# The only variables a fragment inherits from the caller's shell. Everything else (credentials,
# OTEL_*, INSPECT_*, LANGFUSE_*, a developer's CALIBRATED_AGENT_EVALUATION_* settings) is dropped,
# so a build reads the same environment on every machine and cannot reach a live service through it.
_INHERITED_ENV = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "LANG", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT"}
)


def _offline_env(local_root: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _INHERITED_ENV or k.startswith("LC_")}
    env.update(
        AWS_PROFILE="calibrated-agent-evaluation-docs-no-such-profile",
        AWS_CONFIG_FILE=os.devnull,
        AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        # Without this, botocore falls back to the instance metadata service when no profile works.
        AWS_EC2_METADATA_DISABLED="true",
        CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE="replay",
        CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=str(local_root),
        # Fragments show the default, plain output; an --explain fragment passes the flag itself.
        CALIBRATED_AGENT_EVALUATION_EXPLAIN="0",
        NO_COLOR="1",
        PYTHONHASHSEED="0",
        COLUMNS="100",
    )
    return env


def _resolve(argv: Sequence[str]) -> list[str]:
    head, *rest = argv
    if head == "calibrated-agent-evaluation":
        script = Path(sys.executable).parent / "calibrated-agent-evaluation"
        if not script.exists():
            raise FragmentError(
                f"no calibrated-agent-evaluation console script next to {sys.executable}"
            )
        return [str(script), *rest]
    if head == "python":
        return [sys.executable, *rest]
    return list(argv)


def scrub(text: str, local_root: Path) -> str:
    """Remove what varies between runs so the output can be compared byte for byte."""
    text = _ANSI.sub("", text)
    for path in {str(local_root), str(local_root.resolve())}:
        text = text.replace(path, _ROOT_PLACEHOLDER)
    text = text.replace(str(REPO_ROOT) + os.sep, "")
    for pattern, replacement in _SCRUBS:
        text = pattern.sub(replacement, text)
    text = _CHOICES.sub(_unquoted_choices, text)
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def _run(
    argv: Sequence[str], env: dict[str, str], cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    argv = [arg.replace("{repo}", str(REPO_ROOT)) for arg in argv]
    try:
        return subprocess.run(  # noqa: S603 - argv comes from the fixed registry below
            _resolve(argv),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FragmentError(f"{' '.join(argv)} timed out after {_TIMEOUT_S}s") from exc


def produce(fragment: Fragment) -> str:
    """Run one fragment and return its scrubbed output, raising on an unexpected exit code."""
    if fragment.render is not None:
        return fragment.render()
    with tempfile.TemporaryDirectory(prefix="calibrated-agent-evaluation-docs-") as tmp:
        cwd = Path(tmp) if fragment.workdir else REPO_ROOT
        local_root = Path(_WORKDIR_ROOT) if fragment.workdir else Path(tmp) / "store"
        env = _offline_env(local_root)
        if fragment.no_cassette_mode:
            del env["CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"]
        if fragment.seed_quickstart:
            seeded = _run(("calibrated-agent-evaluation", "quickstart"), env, cwd)
            if seeded.returncode != 0:
                raise FragmentError(f"{fragment.name}: seeding failed:\n{seeded.stdout}")
        for step in fragment.setup:
            prior = _run(step.argv, env, cwd)
            if prior.returncode != step.expected_exit:
                raise FragmentError(
                    f"{fragment.name}: setup step {' '.join(step.argv)} exited "
                    f"{prior.returncode}, expected {step.expected_exit}:\n{prior.stdout}"
                )
        done = _run(fragment.argv, env, cwd)
        if done.returncode != fragment.expected_exit:
            raise FragmentError(
                f"{fragment.name}: exited {done.returncode}, expected {fragment.expected_exit}:\n"
                f"{done.stdout}"
            )
        # A workdir store is the relative "demo" the reader sees too; only the temp dir varies.
        body = scrub(done.stdout, Path(tmp) if fragment.workdir else local_root)
    if not fragment.shown:
        return body
    shown = _wrap_command(fragment.shown) if fragment.workdir else fragment.shown
    return f"$ {shown}\n{body}"


def _wrap_command(command: str, width: int = 88) -> str:
    """Break a long command before each option, shell-continued, so a page shows it unscrolled."""
    if len(command) <= width:
        return command
    return re.sub(r" (?=--)", " \\\\\n    ", command)


def crew_span_tree() -> str:
    """A mermaid flowchart of CLM-4471's span tree, coloring the culprit and manifestation spans."""
    trace = json.loads((REPO_ROOT / "fixtures" / "normalized" / "CLM-4471.json").read_text())
    gold = json.loads((REPO_ROOT / "fixtures" / "labels" / "gold.json").read_text())
    outcome = next(g for g in gold if g["trace_id"] == "CLM-4471" and g["span_id"] is None)
    spans = sorted(trace["spans"], key=lambda s: s["seq"])

    def node(span_id: str) -> str:
        return span_id.split("/", 1)[1].replace("-", "_")

    tags = {outcome["culprit_span"]: "culprit", outcome["manifestation_span"]: "manifestation"}
    lines = ["```mermaid", "flowchart TD"]
    for span in spans:
        tag = tags.get(span["span_id"], span["role"])
        lines.append(f'  {node(span["span_id"])}["{span["agent"]}<br/><small>{tag}</small>"]')
    for span in spans:
        if span["parent"]:
            lines.append(f"  {node(span['parent'])} --> {node(span['span_id'])}")
    specialists = [s for s in spans if s["parent"]]
    for before, after in itertools.pairwise(specialists):
        lines.append(f"  {node(before['span_id'])} -. handoff .-> {node(after['span_id'])}")
    lines += [
        "  classDef culprit stroke:#c0392b,stroke-width:3px",
        "  classDef manifestation stroke:#776c59,stroke-width:3px,stroke-dasharray:4 3",
        f"  class {node(outcome['culprit_span'])} culprit",
        f"  class {node(outcome['manifestation_span'])} manifestation",
        "```",
    ]
    return "\n".join(lines) + "\n"


_CERTIFY_KIT = (
    REPO_ROOT / "src" / "calibrated_agent_evaluation" / "_quickstart" / "data" / "certify"
)


def labeled_trace() -> str:
    """One hunted trace from the certify set as a reviewer sees it, with its gold label."""
    from calibrated_agent_evaluation._quickstart import load_certify_set

    traces, labels = load_certify_set()
    label = next(lb for lb in labels if lb.trace_id == "Q-10000")
    trace = next(t for t in traces if t.trace_id == label.trace_id)
    cell = ", ".join(f"{k}={v}" for k, v in trace.taxonomy_cell.items())
    lines = [f"trace {trace.trace_id}  frame: {trace.frame}  cell: {cell}"]
    for span in trace.spans:
        lines.append(f"customer: {span.input['message']}")
        for call in span.tool_calls:
            args = ", ".join(f"{k}={v}" for k, v in call.args.items())
            lines.append(f"tool: {call.name}({args})")
            lines.append(f"  -> {json.dumps(call.result, sort_keys=True)}")
        lines.append("agent's answer:")
        lines += [f"  {line}" for line in str(span.output["answer"]).splitlines() if line]
    lines += [
        "",
        f"gold label: {label.outcome}  failure mode: {label.failure_mode}",
        f"  rater: {label.rater_id}",
        f"  critique: {label.critique}",
    ]
    return "\n".join(lines) + "\n"


def prompt_revision() -> str:
    """The one change between the quickstart judge's v1 and v2 prompts, a sentence per line."""
    v1, v2 = (
        json.loads((_CERTIFY_KIT / f"judge-{v}.json").read_text(encoding="utf-8"))
        for v in ("v1", "v2")
    )

    def sentences(doc: dict[str, str]) -> list[str]:
        return [s + "\n" for s in re.split(r"(?<=\.)\s+", doc["prompt_template"].strip())]

    diff = difflib.unified_diff(sentences(v1), sentences(v2), "judge-v1.json", "judge-v2.json", n=1)
    return "".join(line for line in diff if not line.startswith(("---", "+++", "@@")))


_K = "demo/quickstart"
_CASSETTE = f"--cassette {_K}/judge.cassette.json"

# The judge kit's chain, in order (the lifecycle guide and the calibrate guide include each step):
# (fragment name, command, expected exit code). The tutorial- names predate the guide.
_TUTORIAL: tuple[tuple[str, str, int], ...] = (
    ("tutorial-certify", "calibrated-agent-evaluation quickstart --certify", 0),
    (
        "tutorial-calibrate-v1",
        f"calibrated-agent-evaluation calibrate --judge {_K}/judge-v1.json {_CASSETTE} "
        "--rate-safe --cell label_round=1",
        0,
    ),
    (
        "tutorial-bias-v1",
        f"calibrated-agent-evaluation bias --judge {_K}/judge-v1.json --probes {_K}/bias-probes-1 "
        f"--thresholds {_K}/bias-thresholds.json {_CASSETTE}",
        1,
    ),
    (
        "tutorial-calibrate-v2",
        f"calibrated-agent-evaluation calibrate --judge {_K}/judge-v2.json {_CASSETTE} "
        "--rate-safe --cell label_round=2",
        0,
    ),
    (
        "tutorial-bias-v2",
        f"calibrated-agent-evaluation bias --judge {_K}/judge-v2.json --probes {_K}/bias-probes-2 "
        f"--thresholds {_K}/bias-thresholds.json {_CASSETTE}",
        0,
    ),
    (
        "tutorial-judge-v2",
        f"calibrated-agent-evaluation judge --judge {_K}/judge-v2.json --frame "
        f"production-representative {_CASSETTE}",
        0,
    ),
    ("tutorial-run-ci", f"calibrated-agent-evaluation run --judge {_K}/judge-v2.json --ci", 1),
)


def _tutorial_step(name: str, command: str, *, after: int, exit_code: int = 0) -> Fragment:
    """A kit command run in a store that already holds the first ``after`` steps of the chain."""
    setup = tuple(Step(tuple(shlex.split(cmd)), code) for _, cmd, code in _TUTORIAL[:after])
    argv = tuple(shlex.split(command))
    shown = command if argv[0] == "calibrated-agent-evaluation" else None
    return Fragment(name, argv, shown, exit_code, workdir=True, setup=setup)


def _tutorial_fragments() -> tuple[Fragment, ...]:
    """Every step of the kit chain, plus the variants and renders the guides include."""
    steps = [
        _tutorial_step(name, cmd, after=i, exit_code=code)
        for i, (name, cmd, code) in enumerate(_TUTORIAL)
    ]
    _, bias_v1, _ = _TUTORIAL[2]
    _, run_ci, _ = _TUTORIAL[-1]
    every = len(_TUTORIAL)
    panel = _tutorial_step(
        "home-quickstart-line",
        "python {repo}/docs/site/examples/tutorial/value_panel.py demo",
        after=every,
    )
    judged = _tutorial_step("guide-run-drift", "calibrated-agent-evaluation run", after=6)
    # The baseline is the store as it stood at the last release: a dump, loaded into its own root.
    release = (Step(("calibrated-agent-evaluation", "dump", "release-dump")),)
    release += (
        Step(("calibrated-agent-evaluation", "--root", "baseline", "load", "release-dump")),
    )
    drift_cmd = (
        f"calibrated-agent-evaluation run --judge {_K}/judge-v2.json --drift-baseline baseline "
        "--drift-feature input-chars --fail-on-drift"
    )
    drift = tuple(
        Fragment(
            name,
            tuple(shlex.split(cmd)),
            cmd,
            0,
            workdir=True,
            setup=judged.setup + release,
        )
        for name, cmd in (
            ("guide-run-drift", drift_cmd),
            ("guide-run-drift-explain", f"{drift_cmd} --explain"),
        )
    )
    production_cmd = (
        f"calibrated-agent-evaluation run --judge {_K}/judge-v2.json --ci --drift-baseline "
        "baseline "
        "--drift-feature input-chars --publish local"
    )
    production = Fragment(
        "guide-run-production",
        tuple(shlex.split(production_cmd)),
        production_cmd,
        1,
        workdir=True,
        setup=judged.setup + release,
    )
    v2 = f"{_K}/judge-v2.json"
    edited = "judge-v2-edited.json"
    fingerprint = _tutorial_step(
        "guide-judge-fingerprint",
        f"calibrated-agent-evaluation judge --judge {edited} --frame production-representative "
        f"{_CASSETTE}",
        after=5,
        exit_code=1,
    )
    fingerprint = Fragment(
        fingerprint.name,
        fingerprint.argv,
        fingerprint.shown,
        1,
        workdir=True,
        setup=(
            *fingerprint.setup,
            Step(("python", "{repo}/docs/site/examples/guides/edit_judge_prompt.py", v2, edited)),
        ),
    )
    dump_kit = f"calibrated-agent-evaluation dump kit --judges-only --judge {v2}"
    load_kit = _tutorial_step(
        "guide-load-judges-only", "calibrated-agent-evaluation --root ci load kit", after=5
    )
    load_kit = Fragment(
        load_kit.name,
        load_kit.argv,
        load_kit.shown,
        workdir=True,
        setup=(*load_kit.setup, Step(tuple(shlex.split(dump_kit)))),
    )
    return (
        *steps,
        *drift,
        production,
        fingerprint,
        _tutorial_step(
            "guide-calibrate-check",
            f"calibrated-agent-evaluation calibrate --check --judge {v2} {_CASSETTE}",
            after=5,
            exit_code=1,
        ),
        _tutorial_step("guide-dump-judges-only", dump_kit, after=5),
        load_kit,
        _tutorial_step(
            "guide-results", "calibrated-agent-evaluation results --out results.csv", after=every
        ),
        _tutorial_step("tutorial-bias-v1-explain", f"{bias_v1} --explain", after=2, exit_code=1),
        _tutorial_step("tutorial-run-ci-explain", f"{run_ci} --explain", after=6, exit_code=1),
        _tutorial_step(
            "tutorial-label-command",
            "calibrated-agent-evaluation label --frame discovery --cell label_round=1",
            after=1,
        ),
        _tutorial_step(
            "tutorial-improve-register",
            "calibrated-agent-evaluation improve --register trusted_broken_policy_tool "
            "--min-reduction 0.10",
            after=every,
        ),
        Fragment("tutorial-labeled-trace", render=labeled_trace),
        Fragment("tutorial-prompt-revision", render=prompt_revision),
        Fragment(panel.name, panel.argv, suffix=".md", workdir=True, setup=panel.setup),
    )


_PREFER_DIR = "docs/site/examples/guides/"
_PREFER_ATOMS = _PREFER_DIR + "prefer-atoms.json"
_PREFER_IMPORT = ("--atoms", _PREFER_ATOMS, "--pairs", _PREFER_DIR + "prefer-pairs.jsonl")
_PREFER_IMPORT_SHOWN = ("--atoms", "atoms.json", "--pairs", "pairs.jsonl")


_JUDGE_FILE_NOTES = {
    "judge_id": "The judge's name. Calibration, bias and score records are filed under it.",
    "model": "Any Inspect model string. A local endpoint works the same way, for example "
    "`ollama/llama3.1`.",
    "prompt_version": "Bump this whenever the prompt changes. A record certifies one model and "
    "prompt version, so a new version needs a fresh calibration on labels it has not seen.",
    "level": "`workflow` reads the whole trace. `span` reads every span of one crew agent and then "
    "needs an `agent` field.",
    "prompt_template": "What the judge is asked. The trace is appended to it, and the judge "
    "writes a critique before its PASS or FAIL.",
    "failure_mode": "The one failure mode this binary judge detects. Its verdicts, rate and gate "
    "are reported under this name.",
}


def annotated_judge_file() -> str:
    """The bundled v2 judge file as a JSON block, one Material code annotation per field."""
    path = (
        REPO_ROOT
        / "src"
        / "calibrated_agent_evaluation"
        / "_quickstart"
        / "data"
        / "certify"
        / "judge-v2.json"
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    keys = list(doc)
    lines = ["```json", "{"]
    for i, key in enumerate(keys, start=1):
        comma = "," if i < len(keys) else ""
        lines.append(f"  {json.dumps(key)}: {json.dumps(doc[key])}{comma} // ({i})!")
    lines += ["}", "```", ""]
    lines += [f"{i}. {_JUDGE_FILE_NOTES[key]}" for i, key in enumerate(keys, start=1)]
    return "\n".join(lines) + "\n"


_SMALL_CALIBRATE = (
    "--judge",
    "fixtures/judge-files/orderbot-refund-resolved.json",
    "--cassette",
    "fixtures/raw/cassettes/judge_orderbot_resolved.cassette.json",
    "--cell",
    "feature=refund_handling",
)

# The recorded synthetic runtime, in AWS's documentation account: it can't name a real runtime.
_AGENTCORE_FIXTURE_ARN = (
    "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/"
    "calibrated_agent_evaluation_fixture-AbCdEf1234"
)
# `generate --run --executor agentcore` against the recorded synthetic runtime: its committed
# cassette and the one-row discovery frame it was recorded for.
_AGENTCORE = (
    "generate",
    "--run",
    "--executor",
    "agentcore",
    "--runtime-arn",
    _AGENTCORE_FIXTURE_ARN,
)
_AGENTCORE_FORMAT = ("--payload-format", "agentcore_sse")
_AGENTCORE_CASSETTE = (
    "--cassette",
    "fixtures/raw/cassettes/agentcore_runtime_CLM-4471.cassette.json",
)
_AGENTCORE_FRAME = ("--frame", "fixtures/raw/frames/agentcore_CLM-4471.frame.json")
_AGENTCORE_REPLAY = (*_AGENTCORE, *_AGENTCORE_FORMAT, *_AGENTCORE_CASSETTE, *_AGENTCORE_FRAME)


def _agentcore_shown(args: Sequence[str], *, replay: bool = False) -> str:
    """The command a reader types, with the replay mode the docs build runs in spelled out."""
    mode = "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay " if replay else ""
    return _wrap_command(f"{mode}calibrated-agent-evaluation {' '.join(args)}")


# `probe --executor agentcore` against the synthetic runtime, without --yes: it prints the plan and
# stops before any AWS call.
_AGENTCORE_PROBE_PROMPT = "What can you help me with?"
# The committed policy case file, built from the recorded Gateway run.
_POLICY_CASES = "fixtures/raw/agentcore/policy/cases.json"
_AGENTCORE_PROBE = (
    "probe",
    "--executor",
    "agentcore",
    "--runtime-arn",
    _AGENTCORE_FIXTURE_ARN,
    "--payload-format",
    "agentcore_sse",
    "--authorized",
)


FRAGMENTS: tuple[Fragment, ...] = (
    Fragment(
        "calibrated-agent-evaluation-quickstart",
        ("calibrated-agent-evaluation", "quickstart"),
        shown="calibrated-agent-evaluation quickstart",
    ),
    Fragment(
        "calibrated-agent-evaluation-ingest-dry-run",
        ("calibrated-agent-evaluation", "ingest", "--dry-run", "fixtures/raw/otel"),
        shown="calibrated-agent-evaluation ingest --dry-run fixtures/raw/otel",
    ),
    Fragment(
        "calibrated-agent-evaluation-run-ci",
        ("calibrated-agent-evaluation", "run", "--verdicts", "human", "--ci"),
        shown="calibrated-agent-evaluation run --verdicts human --ci",
        expected_exit=1,
        seed_quickstart=True,
    ),
    Fragment(
        "calibrated-agent-evaluation-run-ci-explain",
        ("calibrated-agent-evaluation", "run", "--verdicts", "human", "--ci", "--explain"),
        shown="calibrated-agent-evaluation run --verdicts human --ci --explain",
        expected_exit=1,
        seed_quickstart=True,
    ),
    Fragment(
        "calibrated-agent-evaluation-cluster",
        ("calibrated-agent-evaluation", "cluster"),
        shown="calibrated-agent-evaluation cluster",
        seed_quickstart=True,
    ),
    Fragment(
        "calibrated-agent-evaluation-cluster-explain",
        ("calibrated-agent-evaluation", "cluster", "--explain"),
        shown="calibrated-agent-evaluation cluster --explain",
        seed_quickstart=True,
    ),
    Fragment("example-rate-correction", ("python", "docs/site/examples/rate_correction.py")),
    Fragment(
        "example-rate-correction-explain",
        ("python", "docs/site/examples/rate_correction.py", "--explain"),
    ),
    Fragment(
        "home-value-panel",
        ("python", "docs/site/examples/rate_correction.py", "--panel"),
        suffix=".md",
    ),
    Fragment("example-calibration-sizing", ("python", "docs/site/examples/calibration_sizing.py")),
    Fragment(
        "example-discipline-refusals", ("python", "docs/site/examples/discipline_refusals.py")
    ),
    Fragment("example-improve-guard", ("python", "docs/site/examples/improve_guard.py")),
    Fragment(
        "example-improve-guard-detail",
        ("python", "docs/site/examples/improve_guard.py", "--detail"),
    ),
    Fragment("example-crew-attribution", ("python", "docs/site/examples/crew_attribution.py")),
    Fragment("crew-span-tree-clm-4471", suffix=".md", render=crew_span_tree),
    Fragment("guide-judge-file", suffix=".md", render=annotated_judge_file),
    *(
        Fragment(
            f"integrate-ingest-{name}",
            ("calibrated-agent-evaluation", "ingest", path),
            shown=f"calibrated-agent-evaluation ingest {path}",
        )
        for name, path in (
            ("strands", "fixtures/raw/strands"),
            ("openinference", "fixtures/raw/otel/openinference/weather_tool_call.otlp.json"),
            ("openllmetry", "fixtures/raw/otel/openllmetry/weather_tool_call.otlp.json"),
            ("messages", "fixtures/raw/otel/messages/swebench_tool_call.otlp.json"),
            ("inspect", "fixtures/raw/inspect/ORD-1001.eval"),
            ("jsonl", "fixtures/raw/hf/swebench-session.jsonl"),
            ("parquet", "fixtures/raw/hf/swebench-slice.parquet"),
        )
    ),
    Fragment(
        "integrate-probe-battery-refusal",
        (
            "calibrated-agent-evaluation",
            "probe",
            "--endpoint",
            "http://127.0.0.1:9/chat",
            "--authorized",
            "--battery",
            "agentdojo",
            "--trials",
            "20",
        ),
        shown=(
            "calibrated-agent-evaluation probe --endpoint http://127.0.0.1:9/chat --authorized "
            "--battery agentdojo --trials 20"
        ),
        expected_exit=1,
    ),
    Fragment(
        "integrate-ingest-sample",
        (
            "calibrated-agent-evaluation",
            "ingest",
            "--sample",
            "3",
            "--seed",
            "7",
            "fixtures/raw/strands",
        ),
        shown="calibrated-agent-evaluation ingest --sample 3 --seed 7 fixtures/raw/strands",
    ),
    Fragment(
        "integrate-ingest-require-owned",
        (
            "calibrated-agent-evaluation",
            "ingest",
            "--require-owned",
            "fixtures/raw/otel/ORD-DEGRADED.otlp.json",
        ),
        shown="calibrated-agent-evaluation ingest --require-owned "
        "fixtures/raw/otel/ORD-DEGRADED.otlp.json",
        expected_exit=1,
    ),
    Fragment("integrate-capture", ("python", "docs/site/examples/integrate/capture_orderbot.py")),
    Fragment(
        "integrate-capture-python",
        ("python", "{repo}/docs/site/examples/integrate/capture_from_python.py"),
        workdir=True,
    ),
    Fragment(
        "integrate-label-hint",
        ("calibrated-agent-evaluation", "--root", "./store", "label"),
        shown="calibrated-agent-evaluation --root ./store label",
        workdir=True,
    ),
    Fragment(
        "integrate-probe-battery",
        ("python", "docs/site/examples/integrate/probe_battery_local.py"),
    ),
    Fragment(
        "integrate-route-strands",
        ("python", "docs/site/examples/integrate/route_strands_replay.py"),
    ),
    Fragment(
        "integrate-route-inspect",
        ("python", "docs/site/examples/integrate/route_inspect_replay.py"),
    ),
    Fragment(
        "integrate-route-agentcore",
        ("calibrated-agent-evaluation", *_AGENTCORE_REPLAY),
        shown=_agentcore_shown(_AGENTCORE_REPLAY, replay=True),
    ),
    Fragment(
        "integrate-route-agentcore-plan",
        ("python", "docs/site/examples/guides/troubleshoot.py", "agentcore-needs-yes"),
    ),
    Fragment(
        "integrate-probe-local", ("python", "docs/site/examples/integrate/probe_local_endpoint.py")
    ),
    Fragment(
        "integrate-probe-agentcore",
        ("calibrated-agent-evaluation", *_AGENTCORE_PROBE, "--prompt", _AGENTCORE_PROBE_PROMPT),
        shown=_wrap_command(
            f"calibrated-agent-evaluation {' '.join(_AGENTCORE_PROBE)} --prompt "
            f'"{_AGENTCORE_PROBE_PROMPT}"'
        ),
        expected_exit=1,
        no_cassette_mode=True,
    ),
    Fragment(
        "integrate-probe-policy",
        ("calibrated-agent-evaluation", *_AGENTCORE_PROBE, "--policy-cases", _POLICY_CASES),
        shown=_wrap_command(
            f"calibrated-agent-evaluation {' '.join(_AGENTCORE_PROBE)} --policy-cases "
            f"{_POLICY_CASES}"
        ),
        expected_exit=1,
        no_cassette_mode=True,
    ),
    Fragment(
        "integrate-gate-sample-size", ("python", "docs/site/examples/integrate/gate_sample_size.py")
    ),
    Fragment(
        "integrate-ci-seed-and-gate",
        ("python", "docs/site/examples/integrate/ci_seed_and_gate.py"),
    ),
    *(
        Fragment(f"guide-{name}", ("python", "docs/site/examples/guides/certify_chain.py", *steps))
        for name, steps in (
            ("certify-seed", ("seed",)),
            ("certify-v1", ("calibrate-v1", "bias-v1")),
            ("certify-v1-bias-explain", ("_calibrate-v1", "bias-v1+explain")),
            ("certify-v2", ("calibrate-v2", "bias-v2", "judge-v2", "run-v2")),
            ("calibrate-v2-explain", ("calibrate-v2+explain",)),
            ("reuse-gold", ("_calibrate-v1", "calibrate-v1")),
            ("rate-too-few", ("_calibrate-v2", "_bias-v2", "judge-v2-one", "run-v2+explain")),
        )
    ),
    *(
        Fragment(
            f"guide-calibrate-not-enough{suffix}",
            ("calibrated-agent-evaluation", "calibrate", *_SMALL_CALIBRATE, *flags),
            shown=" ".join(("calibrated-agent-evaluation calibrate", *_SMALL_CALIBRATE, *flags)),
            expected_exit=1,
            seed_quickstart=True,
        )
        for suffix, flags in (("", ()), ("-explain", ("--explain",)))
    ),
    *(
        Fragment(
            f"recipe-improve{suffix}",
            ("python", "docs/site/examples/guides/improve_request.py", *flags),
            suffix=ext,
        )
        for suffix, flags, ext in (
            ("", (), ".txt"),
            ("-explain", ("--explain",), ".txt"),
            ("-request", ("--print-request",), ".json"),
        )
    ),
    *(
        Fragment(
            f"guide-improve-crew{suffix}",
            ("python", "docs/site/examples/guides/improve_crew.py", *flags),
            suffix=ext,
        )
        for suffix, flags, ext in (
            ("", (), ".txt"),
            ("-explain", ("--explain",), ".txt"),
            ("-file", ("--print-crew",), ".json"),
        )
    ),
    *(
        Fragment(
            name,
            ("calibrated-agent-evaluation", "prefer", *args),
            shown=" ".join(("calibrated-agent-evaluation prefer", *shown)),
            expected_exit=code,
            seed_quickstart=True,
            setup=(Step(("calibrated-agent-evaluation", "prefer", *_PREFER_IMPORT)),)
            if imported
            else (),
        )
        for name, args, shown, code, imported in (
            ("guide-prefer-import", _PREFER_IMPORT, _PREFER_IMPORT_SHOWN, 0, False),
            (
                "guide-prefer-objective",
                ("--atoms", _PREFER_ATOMS, "--pairs", _PREFER_DIR + "prefer-pairs-objective.jsonl"),
                ("--atoms", "atoms.json", "--pairs", "pairs-objective.jsonl"),
                1,
                False,
            ),
            ("guide-prefer-rank", ("--rank", "tone"), ("--rank", "tone"), 0, True),
            (
                "guide-prefer-rank-explain",
                ("--rank", "tone", "--explain"),
                ("--rank", "tone", "--explain"),
                0,
                True,
            ),
        )
    ),
    Fragment(
        "guide-improve-single",
        ("python", "docs/site/examples/guides/improve_crew.py", "--single"),
    ),
    Fragment(
        "guide-define-frame",
        (
            "calibrated-agent-evaluation",
            "define",
            "--taxonomy",
            "taxonomy.json",
            "--out",
            "frame.json",
        ),
        shown="calibrated-agent-evaluation define --taxonomy taxonomy.json --out frame.json",
        workdir=True,
        setup=(Step(("cp", "{repo}/docs/site/examples/guides/taxonomy.json", "taxonomy.json")),),
    ),
    Fragment(
        "walkthrough-generate-template",
        (
            "calibrated-agent-evaluation",
            "generate",
            "--frame",
            "frame.json",
            "--out",
            "inputs.jsonl",
        ),
        shown="calibrated-agent-evaluation generate --frame frame.json --out inputs.jsonl",
        workdir=True,
        setup=(
            Step(("cp", "{repo}/docs/site/examples/guides/taxonomy.json", "taxonomy.json")),
            Step(
                (
                    "calibrated-agent-evaluation",
                    "define",
                    "--taxonomy",
                    "taxonomy.json",
                    "--out",
                    "frame.json",
                )
            ),
        ),
    ),
    Fragment(
        "guide-cluster-ratify",
        (
            "calibrated-agent-evaluation",
            "cluster",
            "--ratify",
            "docs/site/examples/guides/ratify.json",
        ),
        shown="calibrated-agent-evaluation cluster --ratify ratify.json",
        seed_quickstart=True,
    ),
    Fragment(
        "guide-cassette-record",
        ("python", "{repo}/docs/site/examples/guides/record_cassette_local.py"),
        workdir=True,
        setup=(Step(("calibrated-agent-evaluation", "quickstart", "--certify")),),
    ),
    Fragment("recipe-drift-check", ("python", "docs/site/examples/guides/drift_check.py")),
    Fragment("reference-bias-ceiling", ("python", "docs/site/examples/bias_ceiling.py")),
    Fragment("guide-overlap-size", ("python", "docs/site/examples/guides/overlap_size.py")),
    Fragment(
        "guide-plan-gate",
        (
            "calibrated-agent-evaluation",
            "plan",
            "--gate",
            "--threshold",
            "0.8",
            "--expected-pass",
            "0.95",
        ),
        shown="calibrated-agent-evaluation plan --gate --threshold 0.8 --expected-pass 0.95",
    ),
    *(
        Fragment(
            f"guide-score-exact{suffix}",
            (
                "calibrated-agent-evaluation",
                "score",
                "--scorer",
                "exact",
                "--mode",
                "wrong_reply",
                "--key",
                "answer",
                "--expected",
                "docs/site/examples/guides/expected-replies.jsonl",
                *flags,
            ),
            shown=" ".join(
                (
                    "calibrated-agent-evaluation score --scorer exact --mode wrong_reply --key "
                    "answer",
                    "--expected expected-replies.jsonl",
                    *flags,
                )
            ),
            seed_quickstart=True,
        )
        for suffix, flags in (("", ()), ("-explain", ("--explain",)))
    ),
    Fragment(
        "recipe-drift-check-detail",
        ("python", "docs/site/examples/guides/drift_check.py", "--detail"),
    ),
    Fragment(
        "guide-local-judge-check",
        (
            "calibrated-agent-evaluation",
            "judge",
            "--check",
            "docs/site/examples/guides/local-judge.json",
        ),
        shown="calibrated-agent-evaluation judge --check local-judge.json",
    ),
    Fragment(
        "guide-local-judge-endpoint",
        (
            "python",
            "docs/site/examples/guides/local_judge_endpoint.py",
            "docs/site/examples/guides/local-judge.json",
        ),
    ),
    *(
        Fragment(
            f"ts-{name}",
            ("calibrated-agent-evaluation", *args),
            shown=" ".join(("calibrated-agent-evaluation", *args)),
            expected_exit=code,
            seed_quickstart=seed,
        )
        for name, args, code, seed in (
            ("agent-not-found", ("capture", "--agent", "orderbot:run_once"), 1, False),
            ("no-spans", ("capture", "--agent", "os:getcwd"), 1, False),
            ("unknown-trace", ("run", "--trace", "NOPE"), 1, True),
            ("no-calibration", ("judge", "--judge", _SMALL_CALIBRATE[1]), 1, True),
            ("cassette-miss", ("calibrate", *_SMALL_CALIBRATE[:4]), 1, True),
            (
                "live-model-down",
                ("calibrate", *_SMALL_CALIBRATE[:2], *_SMALL_CALIBRATE[4:]),
                1,
                True,
            ),
            ("nothing-to-gate", ("run", "--ci"), 1, True),
            ("usage-error", ("run", "--verdicts", "bogus"), 2, True),
        )
    ),
    *(
        Fragment(
            f"ts-missing-{extra}",
            ("python", "docs/site/examples/guides/missing_extra.py", extra, *args),
            expected_exit=1,
            seed_quickstart=True,
        )
        for extra, args in (
            ("stats", ("run",)),
            ("ui", ("label", "--serve", "--rater", "me")),
            ("langfuse", ("--backend", "langfuse", "ingest", "fixtures/raw/strands")),
            ("agentcore", ("doctor", "--agentcore", "--region", "us-west-2")),
        )
    ),
    Fragment(
        "ts-agentcore-no-credentials",
        ("calibrated-agent-evaluation", "doctor", "--agentcore", "--region", "us-west-2"),
        expected_exit=1,
    ),
    Fragment(
        "integrate-agentcore-dataset-answers",
        ("calibrated-agent-evaluation", "dataset", "answers", "--out", "answer-key.json"),
        shown="calibrated-agent-evaluation dataset answers --out answer-key.json",
        seed_quickstart=True,
        workdir=True,
    ),
    # `calibrated-agent-evaluation audit`, offline: the example serves the fixture A/B test through
    # botocore's Stubber, stores its arms read from the stamped fixture window and builds the H1
    # replay roots.
    Fragment(
        "integrate-agentcore-audit-abtest",
        ("python", "docs/site/examples/integrate/agentcore_audit.py", "abtest"),
        shown="calibrated-agent-evaluation audit abtest --ab-test-id fixture-abtest-stopped "
        "--region us-west-2",
    ),
    Fragment(
        "integrate-agentcore-audit-retest",
        ("python", "docs/site/examples/integrate/agentcore_audit.py", "retest"),
        shown=(
            "calibrated-agent-evaluation audit retest missing_date_range --ab-test-id "
            "fixture-abtest-stopped \\\n"
            "    --before-root replay/control --after-root replay/treatment \\\n"
            "    --development-root replay/development --verdicts human"
        ),
    ),
    Fragment(
        "integrate-agentcore-audit-report",
        (
            "calibrated-agent-evaluation",
            "audit",
            "report",
            "--ab-test-id",
            "fixture-abtest-stopped",
        ),
        shown="calibrated-agent-evaluation audit report --ab-test-id fixture-abtest-stopped",
        setup=(Step(("python", "docs/site/examples/integrate/agentcore_audit.py", "retest")),),
    ),
    Fragment(
        "integrate-agentcore-dataset-build",
        (
            "calibrated-agent-evaluation",
            "dataset",
            "build",
            "--answers",
            "answer-key.json",
            "--out",
            "slice.json",
        ),
        shown="calibrated-agent-evaluation dataset build --answers answer-key.json --out "
        "slice.json",
        seed_quickstart=True,
        workdir=True,
        # The reader's filled-in key: the corpus key, with trusted_broken_policy_tool answered.
        setup=(
            Step(
                (
                    "python",
                    "-c",
                    "import shutil; shutil.copy('{repo}/fixtures/labels/answer-key.json', "
                    "'answer-key.json')",
                )
            ),
        ),
    ),
    Fragment(
        "ts-dataset-publish-no-credentials",
        (
            "calibrated-agent-evaluation",
            "dataset",
            "publish",
            "slice.json",
            "--name",
            "orderbot_regressions",
        ),
        shown="calibrated-agent-evaluation dataset publish slice.json --name orderbot_regressions",
        expected_exit=1,
        seed_quickstart=True,
        workdir=True,
        setup=(Step(("calibrated-agent-evaluation", "dataset", "build", "--out", "slice.json")),),
    ),
    Fragment(
        "ts-dataset-pull-draft",
        (
            "calibrated-agent-evaluation",
            "dataset",
            "pull",
            "--dataset-id",
            "orderbot_regressions-AbCdEf1234",
            "--version",
            "DRAFT",
            "--out",
            "slice.json",
        ),
        shown="calibrated-agent-evaluation dataset pull --dataset-id "
        "orderbot_regressions-AbCdEf1234 --version DRAFT "
        "--out slice.json",
        expected_exit=1,
    ),
    Fragment(
        "ts-missing-agentcore-generate",
        ("python", "docs/site/examples/guides/missing_extra.py", "agentcore", *_AGENTCORE_REPLAY),
        expected_exit=1,
    ),
    *(
        Fragment(
            f"ts-agentcore-{name}",
            ("calibrated-agent-evaluation", *args),
            shown=_agentcore_shown(args, replay=replay),
            expected_exit=code,
        )
        # A flag the command line leaves out or misplaces is a usage error (2); replay without a
        # cassette is a refusal (1).
        for name, args, replay, code in (
            ("no-payload-format", (*_AGENTCORE, *_AGENTCORE_FRAME), False, 2),
            ("flag-without-executor", ("generate", *_AGENTCORE[4:]), False, 2),
            ("replay-no-cassette", (*_AGENTCORE, *_AGENTCORE_FORMAT, *_AGENTCORE_FRAME), True, 1),
        )
    ),
    *(
        Fragment(f"ts-{case}", ("python", "docs/site/examples/guides/troubleshoot.py", case))
        for case in ("schema-version", "langfuse-unreachable")
    ),
    Fragment(
        "integrate-langfuse-no-keys",
        ("calibrated-agent-evaluation", "ingest", "--from-langfuse", "My Project"),
        shown='calibrated-agent-evaluation ingest --from-langfuse "My Project"',
        expected_exit=1,
    ),
    *(
        Fragment(
            f"guide-label-agreement{suffix}",
            (
                "calibrated-agent-evaluation",
                "label",
                "--agreement",
                "--raters",
                "alice,bob",
                "--codebook-version",
                "1",
                *flags,
            ),
            shown=" ".join(
                (
                    "calibrated-agent-evaluation label --agreement --raters alice,bob "
                    "--codebook-version 1",
                    *flags,
                )
            ),
            expected_exit=1,
            workdir=True,
            setup=(Step(("python", "{repo}/docs/site/examples/guides/dual_coded_store.py")),),
        )
        for suffix, flags in (("", ()), ("-explain", ("--explain",)))
    ),
    *_tutorial_fragments(),
)


def produce_all(fragments: Sequence[Fragment] = FRAGMENTS) -> dict[str, str]:
    """Produce every fragment concurrently; raises :class:`FragmentError` on the first failure."""
    with ThreadPoolExecutor(max_workers=min(8, len(fragments))) as pool:
        outputs = list(pool.map(produce, fragments))
    return {f.name: out for f, out in zip(fragments, outputs, strict=True)}


def diff_against_goldens(outputs: dict[str, str]) -> list[str]:
    """One unified diff per fragment whose output differs from (or lacks) its committed golden."""
    by_name = {f.name: f for f in FRAGMENTS}
    problems = []
    for name, actual in outputs.items():
        golden = by_name[name].golden
        expected = golden.read_text(encoding="utf-8") if golden.exists() else ""
        if actual != expected:
            diff = difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=str(golden.relative_to(REPO_ROOT)),
                tofile=f"{name} (regenerated)",
            )
            problems.append("".join(diff))
    return problems


def write_goldens(outputs: dict[str, str]) -> None:
    """Write every golden and remove the ones this generator wrote before but no longer produces.

    The manifest lists the files the last update wrote. Only a file named there and no longer
    produced is removed, so anything else placed in the directory is never touched.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    by_name = {f.name: f for f in FRAGMENTS}
    for name, text in outputs.items():
        by_name[name].golden.write_text(text, encoding="utf-8")
    known = sorted(f.golden.name for f in FRAGMENTS)
    for stale in sorted(_previously_generated() - set(known)):
        (GOLDEN_DIR / stale).unlink(missing_ok=True)
    GOLDEN_MANIFEST.write_text("".join(f"{name}\n" for name in known), encoding="utf-8")


def _previously_generated() -> set[str]:
    if not GOLDEN_MANIFEST.exists():
        return set()
    return {line for line in GOLDEN_MANIFEST.read_text(encoding="utf-8").splitlines() if line}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or refresh the docs example goldens.")
    parser.add_argument("--update", action="store_true", help="rewrite the committed goldens")
    args = parser.parse_args(argv)
    outputs = produce_all()
    if args.update:
        write_goldens(outputs)
        print(f"wrote {len(outputs)} golden(s) to {GOLDEN_DIR.relative_to(REPO_ROOT)}")
        return 0
    problems = diff_against_goldens(outputs)
    for problem in problems:
        print(problem)
    print(f"{len(outputs) - len(problems)} of {len(outputs)} fragment(s) match their goldens")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
