# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CLI's install hints and summaries say what a user can act on."""

from __future__ import annotations

import re

import pytest

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.config import Degradation, extra_for_import, require_extras
from calibrated_agent_evaluation.cli.errors import CliError

_INTERNAL_ID = re.compile(r"\b(F-\d{3}|SC-\d{3}|NFR-\d{3}|FR-\d{3}|ENF-\d)\b")


@pytest.mark.parametrize(
    ("module", "extra"),
    [
        ("websockets", "probe-ws"),
        ("opentelemetry", "otel"),
        ("inspect_evals", "probe"),
        ("strands_evals", "synth"),
        ("huggingface_hub", "hf"),
    ],
)
def test_every_optional_module_maps_to_its_real_extra(module, extra):
    assert extra_for_import(module) == extra


def test_the_extras_hints_name_real_extras():
    from importlib.metadata import metadata

    from calibrated_agent_evaluation.cli.config import _MODULE_TO_EXTRA

    declared = set(metadata("calibrated-agent-evaluation").get_all("Provides-Extra") or [])
    assert set(_MODULE_TO_EXTRA.values()) <= declared


def test_require_extras_names_every_missing_extra_at_once(monkeypatch):
    def _absent(module: str) -> Degradation:
        return Degradation(
            module, False, f"pip install calibrated-agent-evaluation[{extra_for_import(module)}]"
        )

    monkeypatch.setattr("calibrated_agent_evaluation.cli.config.probe_extra", _absent)
    with pytest.raises(CliError) as exc:
        require_extras("allpairspy", "scipy")
    message = str(exc.value)
    assert install_hint("sampling") in message
    assert install_hint("stats") in message
    assert install_hint("sampling", "stats") in message


def test_require_extras_with_one_missing_names_just_that_one(monkeypatch):
    from calibrated_agent_evaluation.cli.config import probe_extra as real

    def _probe(module: str) -> Degradation:
        if module == "scipy":
            return Degradation(module, False, "pip install calibrated-agent-evaluation[stats]")
        return real(module)

    monkeypatch.setattr("calibrated_agent_evaluation.cli.config.probe_extra", _probe)
    with pytest.raises(CliError) as exc:
        require_extras("allpairspy", "scipy")
    assert str(exc.value).endswith(f"isn't installed: {install_hint('stats')}")


@pytest.mark.parametrize(
    "argv",
    [
        ["define"],
        ["generate"],
        ["label"],
        ["cluster"],
        ["run"],
        ["doctor"],
    ],
)
def test_stage_summaries_carry_no_internal_spec_ids(argv, seeded_local_root, capsys):
    main(argv)
    captured = capsys.readouterr()
    assert not _INTERNAL_ID.search(captured.out), captured.out
