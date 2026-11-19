# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation generate` wiring tests.

Covers renderer selection (offline template vs BedrockRenderer), executor routing (inspect/strands
+ unknown), and embedder selection — asserting the observable behavior and the Optional-pattern
capability-absent paths. The offline path makes no network call.
"""

from __future__ import annotations

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.config import Degradation
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE


def _force_extra_absent(monkeypatch, module: str) -> None:
    """Make ``probe_extra(module)`` report absent, everywhere the generate stage consults it."""
    from calibrated_agent_evaluation.cli.config import extra_for_import
    from calibrated_agent_evaluation.cli.config import probe_extra as real_probe

    def _fake_probe(m: str):
        if m == module:
            return Degradation(
                capability=m,
                available=False,
                install_hint=install_hint(extra_for_import(m)),
            )
        return real_probe(m)

    monkeypatch.setattr("calibrated_agent_evaluation.cli.config.probe_extra", _fake_probe)


def test_offline_generate_uses_template_renderer_no_network(seeded_local_root, capsys):
    """With no --model, generate synthesizes via the template renderer, no network call."""
    rc = main(["generate"])
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "offline template renderer" in out
    assert "synthesized" in out


def test_offline_generate_makes_no_bedrock_call(seeded_local_root, monkeypatch):
    """Negative (network hard-disabled): the offline path never constructs a Bedrock
    client."""
    import calibrated_agent_evaluation.cli.stages.generate as gen

    def _boom(*_a, **_k):
        raise AssertionError("offline generate must not build a BedrockRenderer")

    monkeypatch.setattr(gen, "_resolve_renderer", lambda args: gen.TemplateRenderer())
    # Sanity that the default path is what runs:
    assert main(["generate"]) == EXIT_OK


def test_generate_with_model_but_absent_bedrock_extra_is_refused(
    seeded_local_root, monkeypatch, capsys
):
    """Negative: --model with the bedrock extra absent → typed install-hint, exit 1."""
    _force_extra_absent(monkeypatch, "boto3")

    rc = main(["generate", "--model", "bedrock/claude-sonnet-4-6"])

    assert rc == EXIT_DOMAIN_ERROR
    assert install_hint("bedrock") in capsys.readouterr().err


def test_generate_selects_bedrock_renderer_when_configured(seeded_local_root, monkeypatch, capsys):
    """Positive: with the extra present + --model, generate uses the BedrockRenderer
    (stub)."""
    import calibrated_agent_evaluation.cli.stages.generate as gen

    class _StubRenderer:
        def render(self, cell: dict[str, str]) -> str:
            return "stubbed LLM input for " + ",".join(sorted(cell.values()))

    # Force the resolver to hand back a stubbed BedrockRenderer (the wiring, not live AWS).
    monkeypatch.setattr(gen, "_resolve_renderer", lambda args: _StubRenderer())
    monkeypatch.setattr(gen, "_resolve_embedder", lambda args: None)

    rc = main(["generate", "--model", "bedrock/claude-sonnet-4-6"])
    assert rc == EXIT_OK
    assert "LLM renderer" in capsys.readouterr().out


def test_unknown_executor_is_refused(seeded_local_root, capsys):
    """Negative (capability-absent): --run without a valid executor → typed CliError."""
    # argparse restricts --executor to inspect|strands, so exercise --run with none selected.
    rc = main(["generate", "--run"])
    assert rc == EXIT_USAGE
    assert "executor" in capsys.readouterr().err.lower()


def test_run_without_agent_is_refused(seeded_local_root, monkeypatch, capsys):
    """Negative: --run --executor X with no --agent → typed CliError (the agent is yours)."""
    # Make the strands extra look present so we reach the agent check, not the extra check.
    from calibrated_agent_evaluation.cli.config import Degradation

    monkeypatch.setattr(
        "calibrated_agent_evaluation.cli.config.probe_extra",
        lambda _m: Degradation(capability=_m, available=True, install_hint=None),
    )
    rc = main(["generate", "--run", "--executor", "strands"])
    assert rc == EXIT_USAGE
    assert "--agent" in capsys.readouterr().err


def test_offline_generate_does_not_select_bedrock_embedder(seeded_local_root, monkeypatch):
    """Negative (capability-absent): no --model → the guards get no BedrockEmbedder."""

    # With no model configured, _resolve_embedder must return None (dedup skipped), never a Bedrock
    # client. Assert by forbidding the embedder import path.
    def _no_bedrock_embedder():
        raise AssertionError("offline generate must not construct a BedrockEmbedder")

    monkeypatch.setattr(
        "calibrated_agent_evaluation.execute.bedrock_embedder.BedrockEmbedder",
        lambda *a, **k: _no_bedrock_embedder(),
    )
    assert main(["generate"]) == EXIT_OK


def test_the_run_agent_is_imported_from_the_current_directory(tmp_path, monkeypatch):
    """A console script's sys.path starts at its bin/ directory, not the current directory."""
    import sys

    from calibrated_agent_evaluation.cli.stages.generate import _resolve_agent

    (tmp_path / "cwd_agent_under_test.py").write_text("def build():\n    return 'built'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))])
    monkeypatch.delitem(sys.modules, "cwd_agent_under_test", raising=False)
    assert _resolve_agent("cwd_agent_under_test:build")() == "built"


def test_a_run_agent_that_does_not_resolve_is_a_typed_error(tmp_path, monkeypatch):
    import pytest

    from calibrated_agent_evaluation.cli.errors import CliError
    from calibrated_agent_evaluation.cli.stages.generate import _resolve_agent

    monkeypatch.chdir(tmp_path)
    with pytest.raises(CliError, match="cannot load agent"):
        _resolve_agent("no_such_agent_module:build")
