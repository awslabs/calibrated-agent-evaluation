# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auto-generate one API-reference page per public `calibrated_agent_evaluation` package at build
time.

Run by the mkdocs `gen-files` plugin (configured in mkdocs.yml). For each public package it writes
a virtual page whose body is one mkdocstrings `:::` directive, which renders the names in that
package's `__all__`, plus a `SUMMARY.md` that `literate-nav` turns into the reference navigation.
Submodules get no page of their own: a package's `__all__` is its public surface, so a name a
reader should use is exported there. The pages are materialized into the build only.

This is the mkdocstrings "automatic reference" recipe (https://mkdocstrings.github.io/recipes/),
adapted for the repo's `src/` layout.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mkdocs_gen_files

# This script runs with CWD = docs/site/ (where mkdocs.yml lives). The package is two levels up.
_SITE_DIR = Path(__file__).parent
_SRC = _SITE_DIR.parent.parent / "src"
_PACKAGE = "calibrated_agent_evaluation"

nav = mkdocs_gen_files.Nav()

# Packages that are implementation seams, not API: the CLI's stage handlers (the CLI reference
# documents the commands), the calibration-constant studies, and two vendor-isolation seams: the
# synthesis engine's and the AgentCore plug-in's failure detectors, whose names are documented on
# the `calibrated_agent_evaluation.agentcore` page.
_INTERNAL = frozenset({"agentcore.detectors", "cli.stages", "stats.studies", "synth.borrow"})

# The overview groups packages by plane and names each one's main entry points. The grouping and
# descriptions are written here; the names are checked against each package's `__all__` so a rename
# fails the build instead of leaving a stale entry. A package missing from the table is listed under
# "Other" so a new package is never invisible.
_PLANES: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "Data plane": [
        (
            "schema",
            "The normalized `Trace`, `Span` and `Label` types every stage reads.",
            ("Trace", "Span", "Label", "validate"),
        ),
        (
            "normalize",
            "Turn Inspect logs, Strands sessions and OTel spans into a `Trace`.",
            ("normalize", "OTelSpans", "InspectEvalLog", "StrandsSession"),
        ),
        (
            "ingest",
            "Ingest trace files or Langfuse traces, refusing anything that would lose data.",
            ("run_ingest", "capture_and_ingest", "ingest_langfuse", "verify_no_silent_loss"),
        ),
        (
            "backends",
            "Trace stores: the offline local store and Langfuse.",
            ("LocalBackend", "LangfuseBackend", "get_backend", "TraceStoreBackend"),
        ),
        (
            "dumpload",
            "Dump and load any Calibrated Agent Evaluation record to CSV or JSON.",
            ("dump", "load", "results_table"),
        ),
    ],
    "Execution plane": [
        (
            "execute",
            "Run agents and judge models through Inspect or Strands, with cassette replay.",
            ("InspectExecutor", "StrandsExecutor", "Cassette", "BedrockRenderer"),
        ),
        (
            "synth",
            "Generate test inputs from a taxonomy and optionally run them.",
            ("generate", "generate_and_run", "dedup"),
        ),
        (
            "probe",
            "Evaluate a black-box endpoint you are authorized to test.",
            ("probe", "run_battery", "require_authorized", "pass_hat_k"),
        ),
        (
            "agentcore",
            "Opt-in AWS Bedrock AgentCore plug-in: the Runtime executor behind "
            "`calibrated-agent-evaluation generate --run --executor agentcore`, the read-only "
            "account check behind "
            "`calibrated-agent-evaluation doctor --agentcore`, the window ingest behind "
            "`calibrated-agent-evaluation ingest --source agentcore`, publishing and pulling "
            "evaluation datasets, and "
            "the `strands-agents-evals` failure detector with its failure vocabulary and a draft "
            "namer for failure-mode clusters.",
            (
                "AgentCoreRuntimeExecutor",
                "RuntimeTarget",
                "AgentCoreRuntimeTask",
                "run_checks",
                "DoctorTarget",
                "require_role",
                "ReadOnlyGuard",
                "ingest_agentcore",
                "publish",
                "pull",
                "failure_vocabulary",
                "FailureDetector",
                "DetectorDraftNamer",
            ),
        ),
    ],
    "Methodology plane": [
        (
            "sampling",
            "Build discovery and production-representative sampling frames.",
            ("discovery_frame", "production_frame", "kish_deff"),
        ),
        (
            "review",
            "The trace review surface that records gold labels.",
            ("ReviewSession", "build_label", "build_work_list"),
        ),
        (
            "cluster",
            "Group failures into modes and check saturation and coverage.",
            ("ratify", "saturation_check", "coverage_report", "chao1_interval"),
        ),
        (
            "calibrate",
            "Certify a judge against gold labels and correct its measured rate.",
            ("plan_calibration", "certify", "corrected_rate", "CalibrationRecord"),
        ),
        (
            "bias",
            "The bias battery a judge passes before promotion.",
            ("run_battery", "enforce_bias_gate", "BiasBatteryRecord"),
        ),
        (
            "judges",
            "Run a certified judge over traces or spans and record its scores.",
            ("run_judge", "render_trace", "process_judge", "JudgeDefinition"),
        ),
        (
            "scorers",
            "Deterministic scorers for structured outputs.",
            ("ExactMatchScorer", "JSONDiffScorer", "score_traces"),
        ),
        (
            "scenarios",
            "Test scenarios built from caught failures and an answer key, grouped into slices.",
            ("build_slice", "starter_key", "DatasetSlice", "PredefinedScenario"),
        ),
        (
            "subjective",
            "Pairwise preference ratings for outcomes with no single right answer.",
            ("bradley_terry", "record_preference", "compute_ratings"),
        ),
        (
            "attribution",
            "Find the culprit and manifestation spans in a multi-agent trace.",
            ("validate_walkback", "per_agent_rates", "relocation_check", "CrewDefinition"),
        ),
        (
            "run",
            "The CI gate, trend correction and drift checks.",
            ("run_ci_gate", "trend_report", "input_drift", "judge_drift"),
        ),
        (
            "improve",
            "Decide whether a fix worked, with pre-registration and guard slices.",
            ("evaluate_improvement", "register_prereg", "ImproveRequest", "GuardSlice"),
        ),
        (
            "enforce",
            "Lint judge definitions and refuse scores that skip calibration.",
            ("lint_judge_definition", "enforce_score_gate", "agreement_display"),
        ),
        (
            "stats",
            "The statistics the gates use: bounds, corrections, paired tests.",
            ("wilson_lower", "rogan_gladen", "rg_interval", "mcnemar_paired"),
        ),
    ],
    "Harness": [
        (
            "cli",
            "The `calibrated-agent-evaluation` command. See the CLI reference for every "
            "subcommand.",
            ("main",),
        ),
        (
            "flagship",
            "The end-to-end proof that runs every stage against both reference agents.",
            ("prove", "run_agent_pipeline", "FlagshipReport"),
        ),
    ],
}


def _exports(package: str) -> set[str]:
    """The names in ``calibrated_agent_evaluation.<package>.__all__``, read statically (no import
    side effects).

    ``package`` is a path below ``calibrated-agent-evaluation`` such as ``normalize/conventions``.
    """
    tree = ast.parse((_SRC / _PACKAGE / package / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            elts = getattr(node.value, "elts", [])
            return {
                e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return set()


def _overview() -> str:
    listed: set[str] = set()
    lines = [
        "# API reference\n",
        "The `calibrated_agent_evaluation` package, generated from docstrings. The library is "
        "split into planes: the "
        "data plane stores and normalizes traces, the execution plane runs agents and models, and "
        "the methodology plane holds the statistics and gates. The methodology plane never imports "
        "a trace-store or executor vendor type.\n",
        "Each row names a package's main entry points. Each package has one page listing every "
        "name in its `__all__`: that is the public surface. Anything not listed there is internal "
        "and may change without notice.\n",
    ]
    for plane, rows in _PLANES.items():
        lines += [
            f"## {plane}\n",
            "| Package | What it is for | Start with |",
            "| --- | --- | --- |",
        ]
        for package, purpose, names in rows:
            missing = [n for n in names if n not in _exports(package)]
            if missing:
                msg = (
                    f"gen_ref_pages: calibrated_agent_evaluation.{package}.__all__ "
                    f"has no {missing}; update _PLANES"
                )
                raise SystemExit(msg)
            listed.add(package)
            start = ", ".join(f"`{n}`" for n in names)
            lines.append(
                f"| [`calibrated_agent_evaluation.{package}`]({_PACKAGE}/{package}/index.md) | "
                f"{purpose} | {start} |"
            )
        lines.append("")
    others = sorted(
        p.parent.name
        for p in (_SRC / _PACKAGE).glob("*/__init__.py")
        if not p.parent.name.startswith("_") and p.parent.name not in listed
    )
    if others:
        lines += ["## Other\n"] + [
            f"- [`calibrated_agent_evaluation.{p}`]({_PACKAGE}/{p}/index.md)" for p in others
        ]
    return "\n".join(lines) + "\n"


# An explicit landing page for the reference section (section-index uses index.md as the section's
# own page, and content pages link to `reference/index.md`).
with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.write(_overview())

for init in sorted((_SRC / _PACKAGE).rglob("__init__.py")):
    parts = init.parent.relative_to(_SRC).parts
    dotted = ".".join(parts[1:])
    if len(parts) == 1 or any(part.startswith("_") for part in parts) or dotted in _INTERNAL:
        continue
    exported = _exports(dotted.replace(".", "/"))
    if not exported:
        raise SystemExit(
            f"gen_ref_pages: calibrated_agent_evaluation.{dotted} has no __all__; declare its "
            "public names"
        )

    doc_path = Path(*parts, "index.md")
    nav[parts] = doc_path.as_posix()
    ident = ".".join(parts)
    with mkdocs_gen_files.open(Path("reference", doc_path), "w") as fd:
        fd.write(f"# `{ident}`\n\n::: {ident}\n")
    mkdocs_gen_files.set_edit_path(
        Path("reference", doc_path), init.relative_to(_SITE_DIR.parent.parent)
    )

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    # The landing page first (keeps it in the literate nav so strict mode doesn't flag it), then the
    # per-package tree.
    nav_file.write("* [Overview](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
