# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each ``calibrated-agent-evaluation`` command's own arguments, one function per command.

:mod:`calibrated_agent_evaluation.cli.parser` registers the commands from :data:`COMMAND_ARGS`, in
help order. The declarations live here rather than in the stage modules because those import the
dispatcher, which imports the parser, and some import optional packages; the parser must stay
import-light.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

__all__ = [
    "AGENTCORE_ONLY_FLAGS",
    "AGENTCORE_PAYLOAD_FORMATS",
    "COMMAND_ARGS",
    "add_trace_arg",
]

AGENTCORE_PAYLOAD_FORMATS: tuple[str, ...] = (
    "generic_message",
    "openai",
    "agentcore_sse",
    "nested_agentcore",
)
"""The request bodies ``--payload-format`` names: the probe's four request shapes.

Spelled out rather than read from :mod:`calibrated_agent_evaluation.probe.adapter`, which would load
the probing toolkit every time the parser is built.
"""

AGENTCORE_ONLY_FLAGS: Mapping[str, str] = {
    "runtime_arn": "--runtime-arn",
    "payload_format": "--payload-format",
    "qualifier": "--qualifier",
    "config_bundle_arn": "--config-bundle-arn",
    "config_bundle_version": "--config-bundle-version",
    "read_timeout": "--read-timeout",
    "profile": "--profile",
    "log_dir": "--log-dir",
    "yes": "--yes",
}
"""The flags of ``generate`` and ``probe`` only ``--executor agentcore`` reads, by parsed name.

Either command refuses each of them without ``--executor agentcore``, and ``generate`` without
``--run`` too, since it would otherwise be ignored. ``--log-dir`` among them names the directory of
the run manifest both commands share.
"""


def add_trace_arg(sub: argparse.ArgumentParser) -> None:
    """``--trace``, for the stages that read stored traces."""
    sub.add_argument(
        "--trace",
        dest="trace_id",
        default=None,
        help="operate on one stored trace (default: every stored trace); an unknown id is refused",
    )


def _quickstart(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation quickstart``: the optional certify set."""
    sub.add_argument(
        "--certify",
        action="store_true",
        help="seed a set big enough to certify a judge (real agent runs, synthetic gold) and "
        "write a judge kit whose recorded replies take calibrate, bias, judge and run offline",
    )
    sub.add_argument(
        "--force",
        action="store_true",
        help="seed into a store directory that already holds data (the sample's traces and gold "
        "labels are added to what is there, and the judge kit files are overwritten)",
    )


def _doctor(sub: argparse.ArgumentParser) -> None:
    """``doctor``: the install report, or with ``--agentcore`` an AWS account check."""
    sub.add_argument(
        "--agentcore",
        action="store_true",
        help="check an AWS account is ready for AgentCore (read-only AWS calls; needs [agentcore])",
    )
    sub.add_argument("--region", default=None, help="AWS region to check (default: the profile's)")
    sub.add_argument("--profile", default=None, help="AWS profile to use (default: AWS_PROFILE)")
    sub.add_argument("--runtime", default=None, help="name of an AgentCore Runtime to check")
    sub.add_argument("--omni-domain", default=None, help="name of the CloudWatch Omni domain")


def _ingest(sub: argparse.ArgumentParser) -> None:
    """``ingest``: the sources, their format and convention, and the gates."""
    sub.add_argument("paths", nargs="*", help="trace files or directories to ingest")
    sub.add_argument(
        "--from-langfuse",
        dest="from_langfuse",
        default=None,
        metavar="PROJECT",
        help="pull every trace from a Langfuse project instead of files, including traces "
        "Calibrated Agent Evaluation did not write; PROJECT (name or id) must be the project the "
        "LANGFUSE_* keys "
        "belong to, since Langfuse keys are project-scoped",
    )
    sub.add_argument(
        "--format",
        dest="fmt",
        choices=("auto", "strands", "otel", "inspect", "jsonl", "parquet", "hf"),
        default="auto",
        help="source format (default: auto-detect from the file shape); jsonl / parquet / hf "
        "read a dataset, one trace per record, and hf takes HuggingFace repo ids as the paths",
    )
    sub.add_argument(
        "--convention",
        choices=(
            "auto",
            "gen_ai/latest",
            "gen_ai/legacy",
            "gen_ai/messages",
            "openllmetry",
            "openinference",
        ),
        default="auto",
        help="the OTel GenAI convention to read OTLP sources with (default: detect it); for a "
        "trace whose signals match more than one",
    )
    sub.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be ingested and refused; write nothing to the backend",
    )
    sub.add_argument(
        "--require-owned",
        dest="require_owned",
        action="store_true",
        help="refuse a trace that degrades to outcome-only (ownership='probed'), for a corpus "
        "that will be analyzed step by step",
    )
    sub.add_argument(
        "--frame",
        choices=("discovery", "production-representative"),
        default="discovery",
        help="sampling frame for traces whose spans carry no calibrated_agent_evaluation.frame "
        "attribute (default: "
        "discovery; production-representative is the frame that authorizes a headline rate, so "
        "it is only ever set on purpose)",
    )
    sub.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a stored trace with the same id but different content (default: refuse, "
        "since its labels would be re-attached to the new content)",
    )
    sub.add_argument(
        "--allow-partial-tools",
        dest="allow_partial_tools",
        action="store_true",
        help="ingest a trace that recovered only some of its source tool calls, reporting the "
        "shortfall (default: refuse it)",
    )
    sub.add_argument(
        "--sample",
        dest="sample",
        type=int,
        default=None,
        metavar="N",
        help="ingest a seeded random sample of N traces instead of every one, tagged "
        "production-representative; needs --seed, and records the seed and the population size",
    )
    sub.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="the seed for --sample; the same seed and population always draw the same sample",
    )
    sub.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="with --sample --from-langfuse, draw from the traces whose taxonomy cell has "
        "KEY=VALUE (repeatable); one cell at a time is a stratified sample",
    )
    # --- AgentCore source: reads AWS telemetry into the --backend store (needs [agentcore]) ---
    sub.add_argument(
        "--source",
        choices=("agentcore",),
        default=None,
        help="pull from a managed source instead of files/Langfuse; 'agentcore' reads Amazon "
        "Bedrock AgentCore telemetry into the --backend store (needs the [agentcore] extra)",
    )
    sub.add_argument(
        "--runtime",
        default=None,
        help=(
            "the AgentCore Runtime's OpenTelemetry service.name, which for the default endpoint "
            "is <agent>.DEFAULT, not the runtime id (required with --source agentcore)"
        ),
    )
    sub.add_argument("--region", default=None, help="AWS region (default: the profile's)")
    sub.add_argument("--profile", default=None, help="AWS profile (default: AWS_PROFILE)")
    sub.add_argument(
        "--omni",
        dest="omni",
        action="store_true",
        help="read spans via CloudWatch Omni SQL instead of Logs Insights (needs the extra)",
    )
    sub.add_argument(
        "--since",
        default=None,
        help="start of the read window as ISO-8601 (required with --source agentcore)",
    )
    sub.add_argument(
        "--until",
        default=None,
        help="end of the read window as ISO-8601 (required with --source agentcore)",
    )
    sub.add_argument(
        "--rekey-sessions",
        dest="rekey_sessions",
        action="store_true",
        help="collapse a mixed-session AgentCore trace to the runtime session id "
        "(default: refuse it)",
    )
    sub.add_argument(
        "--with-scores",
        dest="with_scores",
        action="store_true",
        help="also read AgentCore evaluator results as advisory labels",
    )


def _capture(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation capture``: the agent to run and how its trace is stored."""
    sub.add_argument(
        "--agent",
        required=True,
        help="the agent to run, as 'module:callable' (an instrumented, no-argument entrypoint); "
        "the module is imported with the current directory first on the path, as `python -m` "
        "does",
    )
    sub.add_argument("--service", default="agent", help="OTel service.name for the capture")
    sub.add_argument(
        "--trace-id",
        dest="trace_id",
        default=None,
        help="trace id for the captured trace (default: the captured OTel trace id)",
    )
    sub.add_argument(
        "--dry-run", action="store_true", help="capture and report; write nothing to the backend"
    )
    sub.add_argument(
        "--require-owned",
        dest="require_owned",
        action="store_true",
        help="refuse the captured trace if it degrades to outcome-only (ownership='probed')",
    )
    sub.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing trace with the same id (default: refuse, since a capture "
        "cannot be re-run to recover what was replaced)",
    )


def _probe(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation probe``: the endpoint, its auth and the prompts to send."""
    # Probe is the one stage whose input is an EXTERNAL endpoint, not the trace store, so it
    # carries its own connection + auth surface. Without --endpoint it asks for the setup on a
    # terminal and refuses with a typed error otherwise; it never reports over the store.
    sub.add_argument(
        "--endpoint",
        default=None,
        help="target endpoint to black-box probe (http(s):// or ws(s)://); omit for guided "
        "interactive setup when run in a terminal",
    )
    sub.add_argument(
        "--authorized",
        action="store_true",
        help="attest you are authorized to probe the target (the --endpoint, or with --executor "
        "agentcore the --runtime-arn); required before anything is sent",
    )
    sub.add_argument(
        "--auth-bearer-env",
        default=None,
        metavar="ENV_VAR",
        help="name of the env var holding a bearer token (value never inlined/logged)",
    )
    sub.add_argument(
        "--auth-api-key-env",
        default=None,
        metavar="ENV_VAR",
        help="name of the env var holding an API key (paired with --auth-api-key-header)",
    )
    sub.add_argument(
        "--auth-api-key-header",
        default=None,
        metavar="HEADER",
        help="header name to carry the API key (e.g. 'x-api-key')",
    )
    sub.add_argument(
        "--auth-header-env",
        default=None,
        metavar="ENV_VAR",
        help="name of the env var holding a JSON object of raw header name->value pairs",
    )
    sub.add_argument(
        "--auth-subprotocol-env",
        default=None,
        metavar="ENV_VAR",
        help="WebSocket only: env var holding a token sent in Sec-WebSocket-Protocol",
    )
    sub.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        metavar="TEXT",
        help="a prompt to send (repeatable); omit to use the built-in default seeds",
    )
    sub.add_argument(
        "--timeout",
        type=float,
        # None, not 30, so an explicit --timeout can be told apart; the URL probe applies 30.
        default=None,
        help="URL probe only: per-request timeout in seconds (default: 30); an AgentCore probe "
        "takes --read-timeout",
    )
    sub.add_argument(
        "--battery",
        nargs="*",
        default=None,
        metavar="BENCHMARK",
        help="URL probe only: run published safety benchmarks against the endpoint instead of the "
        "seed prompts (strong_reject, xstest; no names runs both). One benchmark sample is one "
        "trial, and the result is an upper bound on the break rate, never a point estimate",
    )
    sub.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="K",
        help="URL probe only: how many samples of each --battery benchmark to run; more trials "
        "tighten the upper bound (0 breaks in 59 trials bounds it at 0.05)",
    )
    sub.add_argument(
        "--allow-use-restricted",
        dest="allow_use_restricted",
        action="store_true",
        help="URL probe only: include --battery benchmarks whose dataset carries a use "
        "restriction (AgentHarm)",
    )
    sub.add_argument(
        "--grader-model",
        dest="grader_model",
        default=None,
        metavar="MODEL",
        help="URL probe only: the model a --battery benchmark grades replies with (its own "
        "default is an OpenAI model); e.g. bedrock/global.anthropic.claude-sonnet-4-6",
    )
    sub.add_argument(
        "--executor",
        choices=("agentcore",),
        default=None,
        help="send the prompts to an agent hosted on Amazon Bedrock AgentCore Runtime, named by "
        "--runtime-arn, instead of an --endpoint URL (agentcore; needs the agentcore extra)",
    )
    _add_agentcore_args(sub)
    sub.add_argument(
        "--policy-cases",
        dest="policy_cases",
        type=Path,
        default=None,
        metavar="PATH",
        help="with --executor agentcore: run the policy cases in PATH (a JSON case file) and score "
        "each case from the Gateway Policy engine's recorded decisions, instead of probing with "
        "--prompt",
    )


def _define(sub: argparse.ArgumentParser) -> None:
    """``define``: the taxonomy, the covering strength and the output file."""
    sub.add_argument(
        "--taxonomy",
        default=None,
        metavar="FILE",
        help="JSON object of dimension -> list of values (default: the stored traces' "
        "taxonomy cells)",
    )
    sub.add_argument(
        "--strength",
        type=int,
        default=2,
        help="cover every combination of this many dimensions (default: 2, pairwise)",
    )
    sub.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="write the frame as JSON, for `calibrated-agent-evaluation generate --frame`",
    )


def _generate(sub: argparse.ArgumentParser) -> None:
    """``generate``: the renderer, the optional agent run and the frame source."""
    sub.add_argument(
        "--run",
        action="store_true",
        help="route synthesized inputs through a real agent (needs an executor)",
    )
    sub.add_argument(
        "--executor",
        choices=("inspect", "strands", "agentcore"),
        default=None,
        help="executor for --run (inspect, strands or agentcore; needs the matching extra)",
    )
    sub.add_argument(
        "--agent",
        default=None,
        help=(
            "the agent under test for --run, as 'module:callable' (an Inspect solver, or a "
            "Strands topology builder); required with --run since the agent is yours, not "
            "the library's (with --executor agentcore the agent is the runtime, so it is refused). "
            "Imported with the current directory first on the path, as `python -m` does"
        ),
    )
    sub.add_argument(
        "--model",
        default=None,
        help="model id for the LLM renderer (needs [bedrock]); omit for the offline template",
    )
    sub.add_argument(
        "--agent-model",
        dest="agent_model",
        default=None,
        metavar="MODEL",
        help="model the agent under test runs on with --run (required for inspect; strands "
        "defaults to its own; refused with agentcore)",
    )
    sub.add_argument(
        "--cassette",
        default=None,
        metavar="PATH",
        help="record/replay the agent's model calls for --run through this cassette file (with "
        "--executor agentcore, the runtime's runs)",
    )
    sub.add_argument(
        "--multi-turn",
        dest="multi_turn",
        action="store_true",
        help="hold a simulated conversation per input instead of one shot: a simulated person "
        "writes each user turn and the agent answers it (needs --run with --executor strands or "
        "agentcore, and the synth extra)",
    )
    sub.add_argument(
        "--max-turns",
        dest="max_turns",
        type=int,
        default=6,
        metavar="N",
        help="stop a --multi-turn conversation after this many agent turns (default: 6)",
    )
    sub.add_argument(
        "--persona",
        default=None,
        metavar="FILE",
        help="JSON persona for --multi-turn (traits, context, actor_goal); the default persona "
        "is built from each input's taxonomy cell",
    )
    add_simulator_args(sub)
    _add_frame_source_args(sub)
    sub.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="write the synthesized inputs as JSONL, one per line (without --run)",
    )
    _add_agentcore_args(sub)


def add_simulator_args(sub: argparse.ArgumentParser) -> None:
    """``--simulator`` and ``--simulator-model``: who plays the customer in a conversation."""
    sub.add_argument(
        "--simulator",
        choices=("strands", "agentcore"),
        default=None,
        help="who plays the customer in each conversation: strands (the strands-agents-evals "
        "actor, the default) or agentcore (AgentCore's actor simulator; needs the agentcore extra)",
    )
    sub.add_argument(
        "--simulator-model",
        dest="simulator_model",
        default=None,
        metavar="MODEL_ID",
        help="the simulated customer's model id (default: the simulator's own)",
    )


def _add_agentcore_args(
    sub: argparse.ArgumentParser, *, skip: frozenset[str] = frozenset()
) -> None:
    """The flags of ``--executor agentcore``, the nine :data:`AGENTCORE_ONLY_FLAGS`.

    Added to each command that can send to an AgentCore Runtime agent; the command refuses each
    of them without ``--executor agentcore``. ``skip`` names flags (by parsed name) the command
    already defines for its own use, such as ``dataset``'s ``--profile`` and ``--yes``.
    """
    sub.add_argument(
        "--runtime-arn",
        dest="runtime_arn",
        default=None,
        metavar="ARN",
        help="with --executor agentcore: the AgentCore Runtime agent to send to, by its runtime "
        "ARN; every AWS call goes to the ARN's region",
    )
    sub.add_argument(
        "--qualifier",
        default=None,
        metavar="NAME",
        help="with --executor agentcore: the runtime endpoint to invoke (default: DEFAULT)",
    )
    sub.add_argument(
        "--payload-format",
        dest="payload_format",
        choices=AGENTCORE_PAYLOAD_FORMATS,
        default=None,
        help="with --executor agentcore: the request body the agent reads (required: it is never "
        "guessed, since each guess would be a billed call)",
    )
    sub.add_argument(
        "--config-bundle-arn",
        dest="config_bundle_arn",
        default=None,
        metavar="ARN",
        help="with --executor agentcore: a configuration bundle for the agent to run with "
        "(needs --config-bundle-version)",
    )
    sub.add_argument(
        "--config-bundle-version",
        dest="config_bundle_version",
        default=None,
        metavar="VERSION",
        help="with --executor agentcore: the version of --config-bundle-arn",
    )
    sub.add_argument(
        "--read-timeout",
        dest="read_timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="with --executor agentcore: how long to wait for each read of the agent's "
        "response, in seconds, not the whole answer (default: 300)",
    )
    if "profile" not in skip:
        sub.add_argument(
            "--profile",
            default=None,
            metavar="NAME",
            help="with --executor agentcore: the AWS profile to call AWS with (default: "
            "AWS_PROFILE)",
        )
    sub.add_argument(
        "--log-dir",
        dest="log_dir",
        default=None,
        metavar="DIR",
        help="with --executor agentcore: where generate and probe keep the run manifest, "
        "agentcore_run_record.json and .csv with one record per run (default: "
        "<root>/records/agentcore, shared by both)",
    )
    if "yes" not in skip:
        sub.add_argument(
            "--yes",
            action="store_true",
            help="with --executor agentcore: go ahead with the billed calls the printed plan "
            "lists (needed for every run that can call AWS; a cassette replay calls none)",
        )


def _label(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation label``: the review server and the work-list scope."""
    sub.add_argument(
        "--serve",
        action="store_true",
        help="launch the review server (needs [ui]); omit to print the launch command + URL",
    )
    sub.add_argument(
        "--rater",
        dest="rater_id",
        default=None,
        help="your rater id, stamped on every label you write (required with --serve)",
    )
    # Work-list scope. --trace (common to every stage) names one trace.
    sub.add_argument(
        "--frame",
        choices=("discovery", "production-representative"),
        default=None,
        help="review only traces in this frame",
    )
    sub.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="review only traces whose taxonomy cell has KEY=VALUE (repeatable)",
    )
    sub.add_argument(
        "--limit",
        type=int,
        default=None,
        help="review at most this many traces still to label",
    )
    sub.add_argument(
        "--relabel",
        action="store_true",
        help="include traces you already labeled (default: resume, skipping them)",
    )
    sub.add_argument(
        "--session-log",
        dest="session_log",
        default=None,
        metavar="PATH",
        help="JSONL file that gets one line per verdict with its time on trace "
        "(default: <root>/review-sessions.jsonl)",
    )
    sub.add_argument(
        "--pairwise",
        action="store_true",
        help="with --serve: serve the A/B page instead, where you pick the better of two traces "
        "on each subjective dimension of --atoms; choices go to <root>/records/preference/",
    )
    sub.add_argument(
        "--atoms",
        default=None,
        metavar="FILE",
        help="with --pairwise: the atom set whose subjective dimensions you compare on",
    )
    sub.add_argument(
        "--host",
        default="127.0.0.1",
        help="loopback interface the review server binds (default: 127.0.0.1; ::1 and "
        "localhost also work). Other addresses are refused: label from another machine over "
        "an SSH port-forward",
    )
    sub.add_argument(
        "--port",
        type=int,
        default=5001,
        help="port the review server listens on (default: 5001)",
    )
    _label_codebook_args(sub)


def _label_codebook_args(sub: argparse.ArgumentParser) -> None:
    """``label``: the codebook version and failure-mode list, and the agreement gate."""
    sub.add_argument(
        "--codebook-version",
        dest="codebook_version",
        default=None,
        metavar="V",
        help="codebook version stamped on every label you write, and the version --agreement "
        "measures (default: none, the unversioned bucket)",
    )
    sub.add_argument(
        "--codebook",
        dest="codebook_file",
        default=None,
        metavar="FILE",
        help='failure-mode list for the review form, JSON {"version": ..., "modes": [{"name": '
        "...}]} (default: the modes `calibrated-agent-evaluation cluster --ratify` stored for "
        "--codebook-version)",
    )
    sub.add_argument(
        "--agreement",
        action="store_true",
        help="measure inter-rater agreement on the gold set (Krippendorff's alpha) instead of "
        "serving; needs --raters",
    )
    sub.add_argument(
        "--raters",
        default=None,
        metavar="A,B[,C]",
        help="with --agreement: the rater ids whose double-coded labels are compared",
    )
    sub.add_argument(
        "--axis",
        choices=("outcome", "failure_mode"),
        default="outcome",
        help="with --agreement: compare PASS/FAIL (default) or the failure mode",
    )


def _cluster(sub: argparse.ArgumentParser) -> None:
    """``cluster``: the coverage grid and the optional draft-codebook embedder."""
    sub.add_argument(
        "--taxonomy",
        default=None,
        metavar="FILE",
        help="grid for the coverage report (default: the stored traces' taxonomy cells)",
    )
    sub.add_argument(
        "--embedder",
        choices=("bedrock",),
        default=None,
        help="embed the FAIL critiques to draft failure modes (Bedrock Titan, needs "
        "[bedrock]; a paid call, so opt-in)",
    )
    sub.add_argument(
        "--embed-model",
        dest="embed_model",
        default=None,
        metavar="ID",
        help="embedding model id for --embedder bedrock (default: Titan text v2)",
    )
    sub.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Ward-linkage distance at which draft clusters are cut (default: 1.0)",
    )
    sub.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="write the draft codebook as JSON (with --embedder)",
    )
    sub.add_argument(
        "--ratify",
        default=None,
        metavar="MAP",
        help="stamp human-ratified failure modes onto the stored FAIL labels from a JSON map "
        '{"codebook_version": ..., "ratified_by": ..., "assign": [{"trace_id": ..., "mode": '
        "...}]} and store the codebook",
    )


def _judge_file_args(sub: argparse.ArgumentParser) -> None:
    """The judge file and its model replies, shared by ``calibrate`` and ``bias``."""
    sub.add_argument(
        "--judge",
        dest="judge_file",
        required=True,
        metavar="FILE",
        help="the judge file, one binary judge as JSON (`calibrated-agent-evaluation judge "
        "--check FILE` validates "
        "one; `calibrated-agent-evaluation quickstart --certify` writes examples)",
    )
    _judge_cassette_args(sub)


def _judge_cassette_args(sub: argparse.ArgumentParser) -> None:
    """``--cassette`` and ``--cassette-mode``: replay judge-model replies, or record them."""
    sub.add_argument(
        "--cassette",
        default=None,
        metavar="PATH",
        help="a cassette of judge-model replies, replayed instead of calling the model or "
        "recorded into with --cassette-mode record (default: the judge's live model, which "
        "needs [inspect])",
    )
    sub.add_argument(
        "--cassette-mode",
        dest="cassette_mode",
        choices=("replay", "record"),
        default="replay",
        help="replay (default): answer from the cassette only, calling no model; record: call "
        "the judge's live model (needs [inspect] and its credentials) for every judge call and "
        "write each reply to the cassette, overwriting a reply it already holds",
    )


def _calibrate(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation calibrate``: the judge file and the gold-set plan."""
    _judge_file_args(sub)
    sub.add_argument(
        "--expected-accuracy",
        dest="expected_accuracy",
        type=float,
        default=None,
        metavar="P",
        help="the judge accuracy the gold-set plan assumes (default: 0.90); with "
        "--expected-tnr it is the expected TPR",
    )
    sub.add_argument(
        "--expected-tnr",
        dest="expected_tnr",
        type=float,
        default=None,
        metavar="P",
        help="the expected TNR, for a judge expected to be better on one class",
    )
    sub.add_argument(
        "--rate-safe",
        dest="rate_safe",
        action="store_true",
        help="plan for a corrected rate: the per-class size at which any accuracy that "
        "certifies leaves the record rate-safe (162 at tau 0.80); without it the "
        "plan only certifies the judge and covers the rate at the expected accuracy",
    )
    sub.add_argument(
        "--raters",
        type=int,
        default=None,
        metavar="N",
        help="how many raters will label this gold (default: the raters already in it). From 2 "
        "the gold is also what they double-code for the agreement check, so the plan is at least "
        "65 per class (130 shared traces)",
    )
    sub.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="calibrate only on traces whose taxonomy cell has KEY=VALUE (repeatable), e.g. "
        "the agent the judge was written for",
    )
    sub.add_argument(
        "--check",
        action="store_true",
        help="re-check a certified judge on fresh gold (traces no calibration of its mode used) "
        "and record the change in TPR and TNR as a judge-check record; never writes or alters a "
        "calibration record. The fresh gold needed per class is derived from the certifying record",
    )
    _agentcore_judge_args(sub, billed=True)


def _bias(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation bias``: the judge file and the probe set."""
    _judge_file_args(sub)
    sub.add_argument(
        "--probes",
        required=True,
        metavar="DIR",
        help="directory of bias probe files, one *.json per bias (`calibrated-agent-evaluation "
        "quickstart --certify` "
        "writes an example set)",
    )
    sub.add_argument(
        "--thresholds",
        required=True,
        metavar="FILE",
        help="per-model bias ceilings as JSON (`calibrated-agent-evaluation quickstart --certify` "
        "writes an example)",
    )
    _agentcore_judge_args(sub, billed=True)
    sub.add_argument(
        "--battery-id",
        dest="battery_id",
        default=None,
        metavar="ID",
        help="id for the battery record (default: a random one)",
    )


def _agentcore_judge_args(sub: argparse.ArgumentParser, *, billed: bool) -> None:
    """The AWS flags a command takes for an AgentCore judge file; refused with a prompt file.

    ``billed`` adds ``--yes``, for a command that makes billed ``Evaluate`` calls.
    """
    if billed:
        sub.add_argument(
            "--yes",
            action="store_true",
            help="an AgentCore judge file: make the billed Evaluate calls; without it the "
            "command shows the estimate and sends nothing",
        )
    sub.add_argument(
        "--profile",
        default=None,
        help="an AgentCore judge file: AWS profile (default: AWS_PROFILE)",
    )
    sub.add_argument(
        "--region",
        default=None,
        help="an AgentCore judge file: AWS region (default: the profile's)",
    )


def _judge(sub: argparse.ArgumentParser) -> None:
    """``judge``: the judge file, its model replies and the frame to score."""
    sub.add_argument(
        "--judge",
        dest="judge_file",
        default=None,
        metavar="FILE",
        help="judge file to score with; it must have a certifying calibration record and a "
        "CLEAR bias battery under <root>/records/",
    )
    sub.add_argument(
        "--check",
        dest="check_file",
        default=None,
        metavar="FILE",
        help="check a judge file the way the scoring stages load it, then exit: reads no "
        "store, calls no model, and prints 'judge file ok' or the error",
    )
    _judge_cassette_args(sub)
    sub.add_argument(
        "--frame",
        choices=("discovery", "production-representative"),
        default=None,
        help="score only traces in this frame (production-representative: the random "
        "sample the rate and the gate are read from)",
    )
    _agentcore_judge_args(sub, billed=True)


def _run(sub: argparse.ArgumentParser) -> None:
    """``run``: the CI gate, the verdict source and the corrected-rate judges."""
    sub.add_argument(
        "--ci",
        action="store_true",
        help="exit non-zero if the CI gate fails (without it a failing gate is reported only)",
    )
    sub.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="the bar each mode's pass-rate 95%% lower bound must clear (default: 0.80)",
    )
    sub.add_argument(
        "--verdicts",
        choices=("judge", "scorer", "human"),
        default="judge",
        help="which verdicts the gate reads: judge (the LLM-judge verdicts, the default), "
        "scorer (the code-scorer verdicts) or human (the human labels); a source with none "
        "in scope is refused, never swapped",
    )
    sub.add_argument(
        "--judge",
        dest="judges",
        action="append",
        default=None,
        metavar="FILE",
        help="judge file whose verdicts get a corrected-rate trend (repeatable; its "
        "calibration is read from <root>/records/calibration)",
    )
    _agentcore_judge_args(sub, billed=False)
    _add_resampling_args(sub)
    sub.add_argument(
        "--publish",
        choices=("local", "langfuse"),
        default=None,
        help="after the run is recorded, publish its rates and drift to a dashboard: local "
        "writes <root>/dashboard/<run_id>/, langfuse also sends them as scores to the "
        "LANGFUSE_* project (whatever --backend is); a failed publish exits 1",
    )
    sub.add_argument(
        "--drift-baseline",
        dest="drift_baseline",
        default=None,
        metavar="STORE",
        help="a local store whose production-representative traces are the baseline; compares "
        "--drift-feature and the taxonomy-cell mix with the traces in scope (reported, never "
        "changes the gate)",
    )
    sub.add_argument(
        "--drift-feature",
        dest="drift_feature",
        choices=("input-chars", "turns", "tool-calls"),
        default=None,
        help="with --drift-baseline: the input feature to compare (input-chars: characters of "
        "the trace input; turns: agent steps; tool-calls: tool calls); required, no default",
    )
    sub.add_argument(
        "--fail-on-drift",
        dest="fail_on_drift",
        action="store_true",
        help="with --drift-baseline: exit 3 when input drift is flagged or an admitted judge "
        "looks stale (a failing gate under --ci still exits 1)",
    )
    sub.add_argument(
        "--atoms",
        dest="atoms_file",
        default=None,
        metavar="FILE",
        help="an AtomSet JSON file; every gated failure mode must be one of its objective "
        "atoms, and a subjective dimension is refused (it is tracked, never gated)",
    )


def _prefer(sub: argparse.ArgumentParser) -> None:
    """``prefer``: import pairwise preferences, or rank one subjective dimension."""
    sub.add_argument(
        "--atoms",
        default=None,
        metavar="FILE",
        help="the atom set (JSON) that declares which dimensions are objective and which "
        "subjective; required with --pairs, narrows --rank to its criterion",
    )
    sub.add_argument(
        "--pairs",
        default=None,
        metavar="FILE",
        help="import preferences from a JSON Lines file (trace_id_a, trace_id_b, dimension, "
        "winner a|b, judge_kind); one bad line refuses the file",
    )
    sub.add_argument(
        "--rank",
        default=None,
        metavar="DIMENSION",
        help="rank the stored preferences on one subjective dimension (Bradley-Terry with 95%% "
        "intervals, tracked, never gated; sequential Elo under --explain)",
    )


def _improve(sub: argparse.ArgumentParser) -> None:
    """``improve``: pre-registration, evaluation and the request-file fallback."""
    sub.add_argument(
        "--register",
        default=None,
        metavar="METRIC",
        help="pre-register the failure mode a fix targets, before the fix exists",
    )
    sub.add_argument(
        "--min-reduction",
        dest="min_reduction",
        type=float,
        default=None,
        metavar="X",
        help="with --register: the absolute fail-rate drop the fix must show, in (0, 1)",
    )
    sub.add_argument(
        "--code-ref",
        dest="code_ref",
        default=None,
        metavar="REF",
        help="with --register: the commit of the agent the target is frozen against",
    )
    sub.add_argument(
        "--crew",
        action="store_true",
        help="with --register: the metric is a crew's workflow outcome rate",
    )
    sub.add_argument(
        "--evaluate",
        default=None,
        metavar="METRIC",
        help="evaluate a fix for a pre-registered metric from the verdicts in --before-root "
        "and --after-root, paired by trace id",
    )
    sub.add_argument(
        "--before-root",
        dest="before_root",
        default=None,
        metavar="PATH",
        help="with --evaluate: the local store of the run before the fix",
    )
    sub.add_argument(
        "--after-root",
        dest="after_root",
        default=None,
        metavar="PATH",
        help="with --evaluate: the local store of the same inputs run after the fix",
    )
    sub.add_argument(
        "--development",
        default=None,
        metavar="FILE",
        help="with --evaluate: the trace ids the fix was developed on, one per line (they "
        "must not be among the paired traces)",
    )
    sub.add_argument(
        "--verdicts",
        choices=("judge", "scorer", "human"),
        default="judge",
        help="with --evaluate: pair the LLM-judge verdicts (default), the code-scorer "
        "verdicts or the human labels, never pooled; with human, a trace whose only label "
        "is a FAIL for another mode is left out of that mode's pairs (the target's and each "
        "guard's) and named",
    )
    sub.add_argument(
        "--crew-file",
        dest="crew_file",
        default=None,
        metavar="CREW.json",
        help="with --evaluate: the crew's agents and handoffs, for a metric registered with "
        "--crew; each other agent and each handoff becomes a guard slice",
    )
    sub.add_argument(
        "--touched-agent",
        dest="touched_agent",
        default=None,
        metavar="NAME",
        help="with --crew-file: the agent the fix changed, whose failures the relocation check "
        "follows",
    )
    sub.add_argument(
        "--request",
        default=None,
        metavar="FILE",
        help="evaluate a fix from a JSON file of paired before/after verdicts and guard "
        "slices measured elsewhere (the fallback to --evaluate)",
    )
    sub.add_argument(
        "--improve-id",
        dest="improve_id",
        default=None,
        metavar="ID",
        help="id for the improve record (default: a timestamped one)",
    )


def _dump(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation dump``: the target directory, or a judges-only export."""
    sub.add_argument("directory", help="target directory for the CSV+JSON artifact set")
    sub.add_argument(
        "--judges-only",
        dest="judges_only",
        action="store_true",
        help="export only what another store needs to admit and score with the --judge files: "
        "each file with its certifying calibration record and CLEAR bias battery (no traces or "
        "labels); `calibrated-agent-evaluation load DIR` restores it",
    )
    sub.add_argument(
        "--judge",
        dest="judge_files",
        action="append",
        default=None,
        metavar="FILE",
        help="a judge file to export with --judges-only (repeatable)",
    )
    _agentcore_judge_args(sub, billed=False)


def _load(sub: argparse.ArgumentParser) -> None:
    """``load``: the source directory and which file of the pair to read."""
    sub.add_argument(
        "directory",
        help="directory written by `calibrated-agent-evaluation dump` (a judges-only dump is "
        "recognized by its "
        "manifest)",
    )
    # dump always writes both files; only load chooses which one to read.
    sub.add_argument(
        "--format",
        dest="fmt",
        choices=("json", "csv"),
        default="json",
        help="which file of the pair to read (default: json, the lossless one; csv is the "
        "human-readable mirror)",
    )


def _score(sub: argparse.ArgumentParser) -> None:
    """``score``: the code scorer, the mode it measures, and where expected values are."""
    sub.add_argument(
        "--scorer",
        required=True,
        choices=("exact", "levenshtein", "numeric", "json-diff", "valid-json", "code", "sql"),
        help="the deterministic scorer. Compare-to-expected: exact match, Levenshtein similarity, "
        "numeric closeness, JSON structural similarity, JSON validity (needs the scorers extra). "
        "Execution: code (run the trace's code against the --expected tests in a sandbox) or sql "
        "(execute the trace's query and the --expected gold query on a seeded SQLite database)",
    )
    sub.add_argument(
        "--mode",
        required=True,
        metavar="NAME",
        help="the failure mode the verdicts measure (a FAIL is stamped with it)",
    )
    sub.add_argument(
        "--key",
        default=None,
        metavar="FIELD",
        help="compare this field of the output and the expected value (default: the whole value)",
    )
    sub.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="X",
        help="similarity at or above which a trace passes, for levenshtein (default 0.9), numeric "
        "(0.99) and json-diff (1.0)",
    )
    sub.add_argument(
        "--expected",
        default=None,
        metavar="FILE",
        help='JSONL of {"trace_id": ..., "expected": {...}} (default: the expected value a human '
        "recorded on a FAIL label for the mode)",
    )
    sub.add_argument(
        "--frame",
        default=None,
        choices=("production-representative", "discovery"),
        help="score only the traces in this frame",
    )
    # Execution-scorer options (--scorer code only; code runs in a sandbox).
    sub.add_argument(
        "--sandbox",
        choices=("local", "agentcore"),
        default="local",
        help="where --scorer code runs the code: local (default, stdlib subprocess, no extra, no "
        "AWS; best-effort isolation, not a security boundary) or agentcore (the Bedrock "
        "AgentCore Code Interpreter, needs [agentcore]; use it for untrusted model output)",
    )
    sub.add_argument(
        "--region",
        default=None,
        help="AWS region for --sandbox agentcore (default: the profile's)",
    )
    sub.add_argument(
        "--profile",
        default=None,
        help="AWS profile for --sandbox agentcore (default: AWS_PROFILE / the default chain)",
    )
    sub.add_argument(
        "--role-arn",
        dest="role_arn",
        default=None,
        metavar="ARN",
        help="require --sandbox agentcore to run as this IAM role, refusing before any call if "
        "the caller is someone else (default: no identity check)",
    )
    sub.add_argument(
        "--max-billed-calls",
        dest="max_billed_calls",
        type=int,
        default=None,
        metavar="N",
        help="cap on billed Code Interpreter calls for --sandbox agentcore; the call past the cap "
        "is refused before it is sent (default: one per scored trace; "
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS also sets it)",
    )


def _results(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation results``: the output file."""
    sub.add_argument(
        "--out",
        required=True,
        metavar="FILE",
        help="CSV file to write: one row per stored estimate, refusals included",
    )


def _plan(sub: argparse.ArgumentParser) -> None:
    """``calibrated-agent-evaluation plan --gate``: the CI gate's per-mode sample size."""
    sub.add_argument(
        "--gate",
        action="store_true",
        help="size the sample `calibrated-agent-evaluation run` gates on: traces per failure mode",
    )
    sub.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        metavar="T",
        help="the gate's threshold, as passed to `calibrated-agent-evaluation run --threshold` "
        "(default: 0.80)",
    )
    sub.add_argument(
        "--expected-pass",
        dest="expected_pass",
        type=float,
        required=True,
        metavar="P",
        help="the pass rate you expect each mode to have, e.g. 0.95",
    )


def _add_frame_source_args(sub: argparse.ArgumentParser) -> None:
    """``--taxonomy`` / ``--frame``: where the stage's taxonomy comes from instead of the store."""
    sub.add_argument(
        "--taxonomy",
        default=None,
        metavar="FILE",
        help="JSON object of dimension -> list of values (default: the stored traces' taxonomy "
        "cells)",
    )
    sub.add_argument(
        "--frame",
        dest="frame_file",
        default=None,
        metavar="FILE",
        help="a frame written by `calibrated-agent-evaluation define --out` (instead of building "
        "one)",
    )


def _add_resampling_args(sub: argparse.ArgumentParser) -> None:
    """``--seed`` / ``--n-resamples`` for a stage whose numbers come from a bootstrap."""
    sub.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the corrected-rate bootstrap, recorded on the run record (default: 0)",
    )
    sub.add_argument(
        "--n-resamples",
        dest="n_resamples",
        type=int,
        default=None,
        help="bootstrap resamples for the corrected-rate interval (default: 9999)",
    )


DATASET_ACTIONS: tuple[str, ...] = ("answers", "build", "publish", "pull", "run")
AUDIT_ACTIONS: tuple[str, ...] = ("abtest", "arms", "recommendation", "retest", "report")


def _dataset(sub: argparse.ArgumentParser) -> None:
    """``dataset``: scenarios from labeled failures, moved to and from AgentCore."""
    sub.add_argument(
        "action",
        choices=DATASET_ACTIONS,
        help="answers: write a starter answer key; build: build a scenario slice from the human "
        "FAIL labels; publish: publish a slice as an AgentCore dataset version; pull: read a "
        "published version back into a slice file; run: hold one simulated conversation per "
        "scenario of a simulated slice and store each as a trace",
    )
    sub.add_argument(
        "slice",
        nargs="?",
        default=None,
        metavar="SLICE",
        help="publish: the slice file to publish; run: the simulated slice to run",
    )
    add_trace_arg(sub)
    sub.add_argument(
        "--out", default=None, metavar="FILE", help="answers, build, pull: file to write"
    )
    sub.add_argument(
        "--force",
        action="store_true",
        help="answers, build, pull: replace --out if it exists (build keeps its published version)",
    )
    sub.add_argument(
        "--answers",
        default=None,
        metavar="FILE",
        help="build: the answer key giving expected tool orders and assertions per failure mode",
    )
    sub.add_argument(
        "--failure-mode",
        dest="failure_mode",
        default=None,
        metavar="NAME",
        help="answers, build: only the human FAIL labels with this failure mode",
    )
    sub.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="publish: the AgentCore dataset name (letters, digits and underscores, starting "
        "with a letter)",
    )
    sub.add_argument(
        "--yes",
        action="store_true",
        help="publish: create or version the dataset; run: make the billed calls its plan lists. "
        "Without it, both only print their plan",
    )
    sub.add_argument(
        "--dataset-id", dest="dataset_id", default=None, metavar="ID", help="pull: the dataset id"
    )
    sub.add_argument(
        "--version",
        dest="dataset_version",
        default=None,
        metavar="N",
        help="pull: the published version number (the editable draft can't be pulled)",
    )
    sub.add_argument(
        "--profile", default=None, help="publish, pull, run: AWS profile (default: AWS_PROFILE)"
    )
    sub.add_argument(
        "--region",
        default=None,
        help="publish, pull, run --judge with --executor strands: AWS region (default: the "
        "profile's; with --executor agentcore the runtime ARN's region is used)",
    )
    sub.add_argument(
        "--executor",
        choices=("strands", "agentcore"),
        default=None,
        help="run: the agent under test, a local Strands agent (the default) or an AgentCore "
        "Runtime agent",
    )
    sub.add_argument(
        "--agent",
        default=None,
        help="run with --executor strands: the agent under test, as 'module:callable' (a Strands "
        "topology builder)",
    )
    sub.add_argument(
        "--agent-model",
        dest="agent_model",
        default=None,
        help="run with --executor strands: the agent's model (default: the executor's)",
    )
    sub.add_argument(
        "--cassette",
        default=None,
        metavar="PATH",
        help="run: record or replay the agent's turns through this cassette, and the simulated "
        "user's turns through <stem>.actor.json beside it",
    )
    sub.add_argument(
        "--judge",
        default=None,
        metavar="FILE",
        help="run: an agentcore judge file with level session; each conversation is judged after "
        "every one has run (a live, billed call)",
    )
    add_simulator_args(sub)
    _add_agentcore_args(sub, skip=frozenset({"profile", "yes"}))


def _audit(sub: argparse.ArgumentParser) -> None:
    """``audit``: AgentCore's A/B claims, and each trace's arm read from its spans."""
    sub.add_argument(
        "action",
        choices=AUDIT_ACTIONS,
        help="abtest: record what AgentCore reports about an A/B test (a vendor claim, not "
        "evidence); arms: read which arm each trace in a window ran in, from its span stamps; "
        "recommendation: record what AgentCore reports about a recommendation; retest: decide "
        "the change on a held-out replay of both bundle versions; report: print the library's "
        "verdict beside AgentCore's claim",
    )
    sub.add_argument(
        "metric",
        nargs="?",
        default=None,
        metavar="METRIC",
        help="retest: the pre-registered target failure mode",
    )
    sub.add_argument(
        "--recommendation-id",
        dest="recommendation_id",
        default=None,
        metavar="ID",
        help="recommendation, retest, report: the recommendation id",
    )
    sub.add_argument(
        "--arms-read",
        dest="arms_read",
        default=None,
        metavar="ID",
        help="retest, report: the stored arms read to use when the test has several",
    )
    sub.add_argument(
        "--audit-id",
        dest="audit_id",
        default=None,
        metavar="ID",
        help="report: the stored re-test to print when the subject has several",
    )
    sub.add_argument(
        "--before-root",
        dest="before_root",
        default=None,
        metavar="PATH",
        help="retest: the replay store under the control (or input) bundle version",
    )
    sub.add_argument(
        "--after-root",
        dest="after_root",
        default=None,
        metavar="PATH",
        help="retest: the replay store under the treatment (or result) bundle version",
    )
    sub.add_argument(
        "--development-root",
        dest="development_root",
        default=None,
        metavar="PATH",
        help="retest: the store of the traces the change was developed on",
    )
    sub.add_argument(
        "--verdicts",
        choices=("judge", "scorer", "human"),
        default=None,
        help="retest: which verdicts decide, as `calibrated-agent-evaluation improve --evaluate` "
        "reads them (default: "
        "judge)",
    )
    sub.add_argument(
        "--ab-test-id", dest="ab_test_id", default=None, metavar="ID", help="the A/B test id"
    )
    sub.add_argument(
        "--claim-record",
        dest="claim_record",
        default=None,
        metavar="ID",
        help="arms, retest, report: the stored claim record to use when the subject has several",
    )
    sub.add_argument(
        "--runtime",
        default=None,
        help="arms: the AgentCore Runtime name to read, as at ingest",
    )
    sub.add_argument(
        "--since", default=None, help="arms: start of the read window as ISO-8601, as at ingest"
    )
    sub.add_argument(
        "--until", default=None, help="arms: end of the read window as ISO-8601, as at ingest"
    )
    sub.add_argument(
        "--omni",
        action="store_true",
        help="arms: read spans through CloudWatch Omni, as at ingest (default: Logs Insights)",
    )
    sub.add_argument(
        "--rekey-sessions",
        dest="rekey_sessions",
        action="store_true",
        help="arms: re-key mixed-session traces to the runtime session id, as at ingest",
    )
    sub.add_argument(
        "--applied-bundle-attribute",
        dest="applied_bundle_attribute",
        default=None,
        metavar="KEY",
        help="arms: a span attribute that must equal the arm's bundle version",
    )
    sub.add_argument(
        "--applied-bundle-require",
        dest="applied_bundle_require",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="arms: a span attribute value every assigned trace must carry (repeatable)",
    )
    sub.add_argument("--profile", default=None, help="AWS profile (default: AWS_PROFILE)")
    sub.add_argument("--region", default=None, help="AWS region (default: the profile's)")


ONLINE_ACTIONS: tuple[str, ...] = ("create-evaluator", "attach", "read", "publish", "cost")


def _online(sub: argparse.ArgumentParser) -> None:
    """``online``: a calibrated Calibrated Agent Evaluation judge in AgentCore online evaluation."""
    sub.add_argument(
        "action",
        choices=ONLINE_ACTIONS,
        help="create-evaluator: create a prompt judge as an AgentCore evaluator and write an "
        "agentcore judge file naming it; attach: attach certified, bias-cleared agentcore judges "
        "to a new online evaluation config on a deployed runtime; read: read the config's "
        "verdicts back into the store and check the judges for staleness (and, when asked, the "
        "traffic for drift); publish: publish each judge's corrected pass rate to CloudWatch; "
        "cost: count and price the run's evaluations",
    )
    sub.add_argument(
        "--judge",
        action="append",
        default=None,
        metavar="FILE",
        help="create-evaluator: the prompt judge file; attach, read, publish: an agentcore judge "
        "file, one per attached evaluator (repeatable)",
    )
    sub.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="create-evaluator: the evaluator name (default: the judge id and prompt version, "
        "for example orderbot_refund_policy_v2); attach: the online config's name",
    )
    sub.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="create-evaluator: the agentcore judge file to write (default: <judge>.agentcore.json "
        "in the current directory); it must not exist",
    )
    sub.add_argument(
        "--runtime-arn",
        dest="runtime_arn",
        default=None,
        metavar="ARN",
        help="attach: the deployed runtime's ARN (default: "
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN)",
    )
    sub.add_argument(
        "--endpoint",
        default=None,
        metavar="NAME",
        help="attach: the runtime endpoint whose sessions are scored (default: DEFAULT)",
    )
    sub.add_argument(
        "--role-arn",
        dest="role_arn",
        default=None,
        metavar="ARN",
        help="attach: the role AgentCore assumes to run the config (default: "
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN)",
    )
    sub.add_argument(
        "--sampling",
        type=float,
        default=None,
        metavar="PCT",
        help="attach: the percentage of sessions scored, 0.01 to 100 (required)",
    )
    sub.add_argument(
        "--session-timeout",
        dest="session_timeout",
        type=int,
        default=None,
        metavar="MINUTES",
        help="attach: how long a session is idle before it is scored, 1 to 1440 (default: 15)",
    )
    sub.add_argument(
        "--filter",
        dest="filters",
        action="append",
        nargs=3,
        default=None,
        metavar=("KEY", "OPERATOR", "VALUE"),
        help="attach: score only sessions whose KEY compares true to VALUE (repeatable, at most "
        "five); VALUE is sent as a number or true/false when it reads as one",
    )
    sub.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SECONDS",
        help="how long to wait for the evaluator (create-evaluator, default 120) or the config "
        "(attach, default 300) to become ACTIVE",
    )
    sub.add_argument(
        "--yes",
        action="store_true",
        help="send the request; without it the command only shows it",
    )
    sub.add_argument(
        "--config",
        default=None,
        metavar="CONFIG_ID",
        help="read, publish, cost: the online config (may be left out when the store holds one)",
    )
    sub.add_argument(
        "--since",
        default=None,
        metavar="ISO",
        help="read: the window's start, with a timezone (default: when the config was created)",
    )
    sub.add_argument(
        "--until",
        default=None,
        metavar="ISO",
        help="read: the window's end, with a timezone (default: now)",
    )
    sub.add_argument(
        "--omni",
        action="store_true",
        help="read: read the traces through CloudWatch Omni instead of Logs Insights",
    )
    sub.add_argument(
        "--wait-minutes",
        dest="wait_minutes",
        type=int,
        default=None,
        metavar="N",
        help="read: how long past a session's timeout its result may still come before the "
        "session counts as unscored (default: 15)",
    )
    sub.add_argument(
        "--drift-baseline",
        dest="drift_baseline",
        default=None,
        metavar="STORE",
        help="read: a local store whose production-representative traces are the baseline; "
        "compares --drift-feature with the traces read back (optional, as on "
        "calibrated-agent-evaluation run)",
    )
    sub.add_argument(
        "--drift-feature",
        dest="drift_feature",
        choices=("input-chars", "turns", "tool-calls"),
        default=None,
        help="read, with --drift-baseline: the input feature to compare (input-chars: characters "
        "of the trace input; turns: agent steps; tool-calls: tool calls)",
    )
    sub.add_argument(
        "--seed",
        type=int,
        default=None,
        help="read: the seed of the drift check's permutation test (default: 0)",
    )
    sub.add_argument("--profile", default=None, help="AWS profile (default: AWS_PROFILE)")
    sub.add_argument("--region", default=None, help="AWS region (default: the profile's)")


COMMAND_ARGS: Mapping[str, Callable[[argparse.ArgumentParser], None]] = {
    "quickstart": _quickstart,
    "doctor": _doctor,
    "ingest": _ingest,
    "capture": _capture,
    "dataset": _dataset,
    "audit": _audit,
    "online": _online,
    "probe": _probe,
    "define": _define,
    "generate": _generate,
    "label": _label,
    "cluster": _cluster,
    "calibrate": _calibrate,
    "bias": _bias,
    "judge": _judge,
    "score": _score,
    "run": _run,
    "prefer": _prefer,
    "improve": _improve,
    "dump": _dump,
    "load": _load,
    "results": _results,
    "plan": _plan,
}
