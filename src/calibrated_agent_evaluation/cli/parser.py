# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``calibrated-agent-evaluation`` argparse surface: one subcommand per pipeline stage.

Stdlib ``argparse`` only (no Typer/Click) so the CLI imports with zero third-party deps.
The subcommands are registered in **pipeline order**;
``probe → define → generate → label → cluster → judge → run → improve``; plus ``dump``/``load``,
so ``calibrated-agent-evaluation --help`` presents error analysis (the observe → open-code → cluster
path) before the scoring stages. An unknown/malformed invocation is argparse's own usage error
(exit 2), which the dispatcher surfaces unchanged.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation import __version__
from calibrated_agent_evaluation.cli.args import COMMAND_ARGS, add_trace_arg

# Canonical pipeline stages, in order. Error-analysis stages
# (probe..cluster) precede the scoring stages (judge, run); the ordering IS the workflow-primacy
# invariant. `dump`/`load` are the no-lock-in round-trip pair, listed last.
PIPELINE_STAGES: tuple[str, ...] = (
    "probe",
    "define",
    "generate",
    "label",
    "cluster",
    "judge",
    "score",
    "run",
    "prefer",
    "improve",
)
IO_STAGES: tuple[str, ...] = ("dump", "load")
# Meta-commands: not pipeline stages, but first-class `calibrated-agent-evaluation` subcommands for
# onboarding. `quickstart` runs the bundled sample corpus end-to-end offline; `doctor` reports the
# install.
META_COMMANDS: tuple[str, ...] = (
    "quickstart",
    "doctor",
    "ingest",
    "capture",
    "dataset",
    "audit",
    "online",
)
STAGES: tuple[str, ...] = PIPELINE_STAGES + IO_STAGES
# Planning and reporting commands: they read records or plan sizes, emit no score and read no
# trace, so they sit outside the pipeline order. Listed last in help.
TOOL_COMMANDS: tuple[str, ...] = ("plan", "results")
_TOOL_HELP: dict[str, str] = {
    "plan": (
        "size a sample before labeling it: --gate gives the traces per mode "
        "`calibrated-agent-evaluation run` needs"
    ),
    "results": "write every stored estimate (intervals, bounds, refusals) to one CSV table",
}
_TOOL_DESCRIPTION: dict[str, str] = {
    "plan": "With --gate, the number of production-representative traces each failure mode needs "
    "for `calibrated-agent-evaluation run --threshold T` to pass a suite expected to pass P of "
    "the time, at that size "
    "and every larger one.",
    "results": "Read every record under <root>/records/ and write one CSV row per interval or "
    "bound, with its method, level, sidedness, n and provenance; a refused estimate keeps its "
    "row with the reason.",
}
# The judge-trust gates `calibrated-agent-evaluation judge` admits a judge on, registered just
# before `judge` in help so the order on screen is the order they run: calibrate, then bias, then
# judge. Neither emits a score, so the workflow-primacy order of STAGES is unchanged.
TRUST_COMMANDS: tuple[str, ...] = ("calibrate", "bias")
_TRUST_HELP: dict[str, str] = {
    "calibrate": "certify a judge against the human gold in the store (the first trust gate)",
    "bias": "run a certified judge's bias battery (the second trust gate)",
}
_TRUST_DESCRIPTION: dict[str, str] = {
    "calibrate": "Run the judge over the stored traces that carry human gold for its failure "
    "mode, then certify it: CERTIFIED only when the 95% lower bounds of TPR and TNR both reach "
    "0.80. Reports how many gold labels per class the plan needs and whether the set has them; "
    "--rate-safe plans enough for `calibrated-agent-evaluation run`'s corrected rate at any "
    "observed accuracy; "
    "gold that two raters double-code (--raters 2, or two raters already in it) is planned at "
    "65 per class or more so their agreement can be measured on it. "
    "The record goes to <root>/records/calibration/ either way; a REFUSED judge exits 1.",
    "bias": "Run the CALM bias battery for a judge that holds a certifying calibration record, "
    "through the judge's own model (or --cassette). The record goes to <root>/records/bias/; a "
    "BIASED judge exits 1.",
}

# One line per stage for `calibrated-agent-evaluation --help`, and the paragraph
# `calibrated-agent-evaluation <stage> --help` opens with.
STAGE_HELP: dict[str, str] = {
    "probe": "black-box probe an agent endpoint and store what it shows (needs --authorized)",
    "define": "build the discovery frame (a covering array) from a taxonomy",
    "generate": "synthesize frame-tagged inputs; with --run, route them through your agent",
    "label": "open the review surface where a human labels traces PASS/FAIL with a critique",
    "cluster": "check failure-mode saturation and taxonomy coverage; draft modes with --embedder",
    "judge": "score traces with a judge that has a certifying calibration and a CLEAR bias run",
    "score": "grade traces with a deterministic code scorer (exact, levenshtein, numeric, ...)",
    "run": "gate the stored verdicts, correct each judged mode's rate, record the run",
    "prefer": "import pairwise preferences on subjective dimensions and rank traces (tracked, "
    "never gated)",
    "improve": "pre-register a fix's target, then accept or reject the fix against it",
}
STAGE_DESCRIPTION: dict[str, str] = {
    "probe": "Send prompts to an agent endpoint you are authorized to test, infer its observable "
    "tools and failure hypotheses, and store the probed traces for the later stages.",
    "define": "Turn a taxonomy (a --taxonomy file, or the stored traces' cells) into a discovery "
    "frame in which every pair of dimension values appears, and write it with --out.",
    "generate": "Render one natural-language input per frame row (offline template, or an LLM "
    "with --model). With --run the inputs go through your agent and its traces are stored. "
    "--multi-turn turns each input into a conversation: a simulated person writes the user "
    "turns, your agent answers them, and the last turn's trace is the stored conversation.",
    "label": "Print the command that serves the review surface, or serve it with --serve. Every "
    "label carries the rater id; --trace/--frame/--cell/--limit narrow the work list.",
    "cluster": "Over the human FAIL labels: whether the failure modes are saturated (Good-Turing "
    "coverage lower bound), which taxonomy cells have no failure yet, and with an embedder a "
    "draft codebook for a human to ratify.",
    "judge": "Admit a judge through calibration, the bias gate and the score gate (records under "
    "<root>/records/), then score the traces in scope and store the verdicts. No calibration, "
    "no score. --check FILE only checks a judge file, with no store and no model.",
    "score": "Compare each trace's output with its expected value using a code scorer and "
    "store the PASS/FAIL verdict for --mode, stamped with the scorer and its parameters. Expected "
    "values come from the human FAIL labels for the mode, or --expected FILE. Gate them with "
    "`calibrated-agent-evaluation run --verdicts scorer`.",
    "run": "The CI gate: every failure mode's pass-rate 95% lower bound must clear --threshold "
    "(--ci makes a failure exit 1), over the LLM-judge verdicts or, with --verdicts scorer or "
    "--verdicts human, the code-scorer verdicts or the human labels. With --judge, each judged "
    "mode's rate is corrected for the judge's error. Each run is recorded under "
    "<root>/records/run/. Exit codes: 0 the run finished (and, with --ci, the gate passed); 1 the "
    "gate failed under --ci, a refusal, or a failed --publish; 2 a usage error; 3 with "
    "--fail-on-drift, input drift or a stale judge was flagged while the gate did not fail.",
    "prefer": "--atoms FILE --pairs FILE imports preferences (trace A or B was better on a "
    "subjective dimension), each checked against the atom set: an objective or undeclared "
    "dimension refuses the file. --rank DIMENSION fits Bradley-Terry ratings with 95% intervals "
    "over the stored preferences, per criterion and judge kind, never pooled. The ratings are "
    "tracked, never gated; sequential Elo appears only under --explain.",
    "improve": "--register freezes a target metric and its minimum fail-rate reduction before the "
    "fix; --evaluate pairs the verdicts of the run before the fix and the run after it (two local "
    "stores) per trace, guards every other failure mode, and records the verdict. --verdicts "
    "picks the LLM-judge verdicts (default), the code-scorer verdicts or the human labels. "
    "For a crew metric (--register --crew), --crew-file and --touched-agent add a guard slice "
    "per other agent and per handoff and check that the failures did not just move. "
    "--request takes the pairs from a file instead.",
}
_IO_HELP: dict[str, str] = {
    "dump": "write the store's traces and labels, and the local root's records, to a directory "
    "(--judges-only: just what another store needs to score with given judges)",
    "load": "read a dump directory back into the store and the local root's records",
}
# Stages that read stored traces, and so take --trace. probe reads an endpoint and improve reads a
# request file, so the flag would be accepted and ignored there.
_TRACE_SCOPED = frozenset({"define", "generate", "label", "cluster", "judge", "score", "run"})

# The index in STAGES at and after which a stage may emit a score. Everything before `judge` is an
# error-analysis / discovery stage; the parser asserts no scoring stage is registered ahead of the
# discovery stages (workflow-primacy, checked structurally in the surface test).
_FIRST_SCORING_STAGE = "judge"


_META_HELP: dict[str, str] = {
    "quickstart": "run the bundled sample corpus end-to-end, offline (no setup, no keys)",
    "doctor": "report the install: Python, which optional extras are present, backend reachability",
    "ingest": "normalize trace dumps (OTLP / Inspect .eval / JSONL / parquet) into the store; "
    "refuses lossy ones",
    "capture": "run an instrumented in-process agent, capture its OTel trace, and ingest it",
    "dataset": "turn labeled failures into test scenarios; publish and pull them as AgentCore "
    "dataset versions",
    "audit": "audit an AgentCore A/B test or recommendation: its claim, its arms, a re-test",
    "online": "put a calibrated Calibrated Agent Evaluation judge into AgentCore online evaluation",
}
_META_DESCRIPTION: dict[str, str] = {
    "ingest": "Normalize trace files, directories (walked recursively) or a Langfuse "
    "project into the store. --backend selects the destination and goes before the "
    "subcommand: calibrated-agent-evaluation --backend langfuse ingest exports/. --sample N "
    "--seed S ingests a "
    "seeded random sample of the population and tags it production-representative, which is "
    "what lets a headline rate be computed over it.",
    "dataset": "answers writes a starter answer key listing the failure modes on the human "
    "FAIL labels; build turns each trace every rater labeled FAIL into a single-turn scenario, "
    "with the expected tool order and assertions from the answer key; publish shows its plan, "
    "and with --yes creates or versions an AgentCore dataset (needs the agentcore extra); pull "
    "reads one published version back into a slice file.",
    "online": "create-evaluator creates a workflow-level prompt judge as an AgentCore "
    "LLM-as-judge evaluator at level TRACE (the judge's prompt with AgentCore's trace "
    "placeholders, a 0.0 FAIL / 1.0 PASS scale and the judge's Bedrock model), waits for it to "
    "become ACTIVE and writes an agentcore judge file naming it. Without --yes it shows the exact "
    "request and sends nothing. Needs the agentcore extra.",
}


@dataclass(frozen=True)
class _Command:
    """One subcommand as ``calibrated-agent-evaluation --help`` lists it."""

    name: str
    help: str
    description: str | None = None
    trace_scoped: bool = False


def _commands() -> tuple[_Command, ...]:
    """Every subcommand in help order.

    Onboarding commands first, so ``calibrated-agent-evaluation --help`` leads with the fastest path
    to a result; then the pipeline stages, with the trust gates just before ``judge`` in the order
    they run; then the round-trip pair.
    """
    commands = [_Command(m, _META_HELP[m], _META_DESCRIPTION.get(m)) for m in META_COMMANDS]
    for stage in PIPELINE_STAGES:
        if stage == "judge":
            commands += [_Command(c, _TRUST_HELP[c], _TRUST_DESCRIPTION[c]) for c in TRUST_COMMANDS]
        commands.append(
            _Command(stage, STAGE_HELP[stage], STAGE_DESCRIPTION[stage], stage in _TRACE_SCOPED)
        )
    commands += [_Command(io, _IO_HELP[io]) for io in IO_STAGES]
    commands += [_Command(t, _TOOL_HELP[t], _TOOL_DESCRIPTION[t]) for t in TOOL_COMMANDS]
    return tuple(commands)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``calibrated-agent-evaluation`` parser with a subparser per stage."""
    parser = argparse.ArgumentParser(
        prog="calibrated-agent-evaluation",
        description="Calibrated Agent Evaluation: rigorous, statistically honest evaluation of AI "
        "agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_global_options(parser, top_level=True)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for command in _commands():
        sub = subparsers.add_parser(
            command.name, help=command.help, description=command.description
        )
        if command.trace_scoped:
            add_trace_arg(sub)
        COMMAND_ARGS[command.name](sub)
        # The global options are accepted after the subcommand too
        # (`calibrated-agent-evaluation label --backend langfuse`), so a command copied from help or
        # docs works in either position.
        _add_global_options(sub, top_level=False)
    return parser


def _add_global_options(parser: argparse.ArgumentParser, *, top_level: bool) -> None:
    """``--backend``, ``--root`` and ``--explain``, on the top-level parser and on every subcommand.

    A subcommand's copy defaults to ``SUPPRESS``: argparse copies a subparser's defaults over the
    parent namespace, so a real default there would silently reset a value given before the
    subcommand.
    """
    parser.add_argument(
        "--backend",
        choices=("local", "langfuse"),
        default="local" if top_level else argparse.SUPPRESS,
        help="trace-store backend (default: local, offline)",
    )
    parser.add_argument(
        "--root",
        default=None if top_level else argparse.SUPPRESS,
        metavar="PATH",
        help="local root for the trace store and the stage records (default: "
        "$CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT, else ./.calibrated-agent-evaluation; quickstart "
        "defaults to a temp dir)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=False if top_level else argparse.SUPPRESS,
        help="after the plain summary, print the statistics behind it (bounds, methods, "
        "thresholds, record paths); CALIBRATED_AGENT_EVALUATION_EXPLAIN=1 turns it on for every "
        "command",
    )


def scoring_stage_indices() -> tuple[int, int]:
    """Return (index of the first discovery stage, index of the first scoring stage).

    Used by the surface test to assert no scoring stage is registered before the discovery stages.
    A structural check on the declared order, not a runtime gate.
    """
    return (0, STAGES.index(_FIRST_SCORING_STAGE))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv``; argparse raises SystemExit(2) on a usage error."""
    return build_parser().parse_args(argv)
