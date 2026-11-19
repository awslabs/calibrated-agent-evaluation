# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundational tests: IngestReport exit-code contract and the no-vendor-type
boundary of ``calibrated_agent_evaluation/ingest/``."""

from __future__ import annotations

import ast
import pathlib

from calibrated_agent_evaluation.ingest import IngestReport, IngestResult

_INGEST_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "src"
    / "calibrated_agent_evaluation"
    / "ingest"
)
# Vendor SDKs the ingest plane must never import at module top level — they belong to the
# allowlisted execute/ (OpenTelemetry capture) and backends/ (Langfuse) planes, reached lazily.
_FORBIDDEN_ROOTS = {
    "opentelemetry",
    "google",
    "langfuse",
    "strands",
    "inspect_ai",
    "autoevals",
    "boto3",
    "botocore",
    "fasthtml",
    "pyarrow",
}


def _ingested(source: str) -> IngestResult:
    return IngestResult(source=source, status="ingested", trace_id=source)


def test_exit_code_zero_when_every_source_ingested() -> None:
    """A report where every source ingested exits 0."""
    report = IngestReport(results=[_ingested("a"), _ingested("b")])
    assert report.exit_code == 0
    assert report.ingested == 2


def test_exit_code_nonzero_when_any_source_refused_or_skipped() -> None:
    """Negative: one refused or skipped source makes the process exit non-zero —
    a dropped source is never a silent exit-zero."""
    refused = IngestResult(source="bad", status="refused", reason="0 tools recovered")
    report = IngestReport(results=[_ingested("a"), refused])
    assert report.exit_code != 0

    skipped = IngestResult(source="junk", status="skipped", reason="no adapter")
    assert IngestReport(results=[_ingested("a"), skipped]).exit_code != 0


def test_dry_run_exit_code_is_zero_even_with_refusal() -> None:
    """Edge: a --dry-run preview writes nothing and never fails the process, so exit is 0."""
    refused = IngestResult(source="bad", status="refused", reason="0 tools recovered")
    report = IngestReport(results=[refused], dry_run=True)
    assert report.exit_code == 0


def _top_level_imported_roots(path: pathlib.Path) -> set[str]:
    """Import roots at MODULE TOP LEVEL only — lazy imports inside functions are allowed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:  # module body only, not ast.walk (which catches in-function imports)
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_ingest_package_imports_no_vendor_sdk_at_top_level() -> None:
    """Contract: no module under ``calibrated_agent_evaluation/ingest/`` imports a vendor SDK at top
    level — the OTel/Langfuse SDKs are reached only via lazy imports into execute/ and backends/."""
    offenders: list[str] = []
    for path in _INGEST_DIR.rglob("*.py"):
        forbidden = _top_level_imported_roots(path) & _FORBIDDEN_ROOTS
        rel = path.relative_to(_INGEST_DIR.parent.parent)
        offenders += [f"{rel}: top-level imports {mod}" for mod in sorted(forbidden)]
    assert offenders == [], f"vendor imports leaked into the ingest plane: {offenders}"
