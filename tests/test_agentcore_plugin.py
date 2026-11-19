# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore plug-in loads lazily, names its missing extra, and is never a default."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import calibrated_agent_evaluation.agentcore
from calibrated_agent_evaluation.agentcore import AgentCoreExtraMissing
from calibrated_agent_evaluation.schema.errors import MissingExtraError


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=True
    )
    return result.stdout


def test_an_adapter_module_loads_on_first_access_and_not_before() -> None:
    out = _run(
        """
        import sys
        import calibrated_agent_evaluation.agentcore
        print("before:", "calibrated_agent_evaluation.agentcore.session" in sys.modules)
        calibrated_agent_evaluation.agentcore.AwsClients
        print("after:", "calibrated_agent_evaluation.agentcore.session" in sys.modules)
        """
    )
    assert "before: False" in out
    assert "after: True" in out


def test_an_unknown_name_is_an_attribute_error_not_a_missing_extra() -> None:
    with pytest.raises(AttributeError, match="NoSuchAdapter"):
        _ = calibrated_agent_evaluation.agentcore.NoSuchAdapter


@pytest.fixture
def unloaded_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget any cached load of the session adapter so the next access imports it again."""
    # Checked through vars(): hasattr() would run the lazy loader and import the adapter.
    if "AwsClients" in vars(calibrated_agent_evaluation.agentcore):
        monkeypatch.delattr(calibrated_agent_evaluation.agentcore, "AwsClients")
    monkeypatch.delitem(sys.modules, "calibrated_agent_evaluation.agentcore.session", raising=False)


@pytest.mark.usefixtures("unloaded_session")
def test_a_missing_aws_sdk_raises_the_typed_missing_extra_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(AgentCoreExtraMissing) as raised:
        _ = calibrated_agent_evaluation.agentcore.AwsClients

    assert isinstance(raised.value, MissingExtraError)
    assert raised.value.extras == ("agentcore",)
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)
    assert "[agentcore]" in str(raised.value)


def test_an_unrelated_missing_module_is_not_disguised_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "agentcore_broken_adapter.py").write_text("import a_module_that_does_not_exist\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        calibrated_agent_evaluation.agentcore._LAZY, "BrokenAdapter", "agentcore_broken_adapter"
    )

    with pytest.raises(ModuleNotFoundError) as raised:
        _ = calibrated_agent_evaluation.agentcore.BrokenAdapter

    assert not isinstance(raised.value, MissingExtraError)
    assert raised.value.name == "a_module_that_does_not_exist"


requires_agentcore = pytest.mark.skipif(
    importlib.util.find_spec("bedrock_agentcore") is None, reason="needs the [agentcore] extra"
)

_AWS_SDK_ROOTS = frozenset({"boto3", "botocore", "bedrock_agentcore", "strands_evals"})
# The modules of the runtime executor and the runtime probe adapter. The CLI already loads the
# execution plane for its error table, so these, not the execution plane, are what a bare CLI
# import must leave unloaded.
_RUNTIME_MODULES = frozenset(
    {
        "calibrated_agent_evaluation.agentcore.run_manifest",
        "calibrated_agent_evaluation.agentcore.run_spans",
        "calibrated_agent_evaluation.agentcore.runtime",
        "calibrated_agent_evaluation.agentcore.runtime_boundary",
        "calibrated_agent_evaluation.agentcore.runtime_cassette",
        "calibrated_agent_evaluation.agentcore.runtime_errors",
        "calibrated_agent_evaluation.agentcore.runtime_executor",
        "calibrated_agent_evaluation.agentcore.runtime_probe",
        "calibrated_agent_evaluation.agentcore.runtime_response",
    }
)


def _loaded_before_and_after(first: str, then: str) -> tuple[set[str], set[str]]:
    """The modules a fresh interpreter holds after running ``first``, and after also ``then``."""
    out = _run(
        f"""
        import json, sys
        {first}
        print(json.dumps(sorted(sys.modules)))
        {then}
        print(json.dumps(sorted(sys.modules)))
        """
    )
    before, after = out.strip().splitlines()[-2:]
    return set(json.loads(before)), set(json.loads(after))


def _aws_sdks(modules: set[str]) -> set[str]:
    return {name for name in modules if name.partition(".")[0] in _AWS_SDK_ROOTS}


def _execution_plane(modules: set[str]) -> set[str]:
    return {
        name
        for name in modules
        if name == "calibrated_agent_evaluation.execute"
        or name.startswith("calibrated_agent_evaluation.execute.")
    }


@requires_agentcore
def test_importing_the_plugin_loads_no_aws_sdk_and_no_execution_plane() -> None:
    """``import calibrated_agent_evaluation.agentcore`` stays light until a name is asked for;
    asking loads both."""
    before, after = _loaded_before_and_after(
        "import calibrated_agent_evaluation.agentcore",
        "[getattr(calibrated_agent_evaluation.agentcore, name) for name in "
        "calibrated_agent_evaluation.agentcore.__all__]",
    )

    assert _aws_sdks(before) == set()
    assert _execution_plane(before) == set()
    assert {"boto3", "botocore"} <= _aws_sdks(after)
    assert "calibrated_agent_evaluation.execute" in _execution_plane(after)


@requires_agentcore
def test_importing_the_cli_loads_no_aws_sdk_and_no_runtime_executor_module() -> None:
    """``import calibrated_agent_evaluation.cli`` loads nothing of the runtime executor or the probe
    adapter until something asks for them."""
    before, after = _loaded_before_and_after(
        "import calibrated_agent_evaluation.cli",
        "import calibrated_agent_evaluation.agentcore; "
        "calibrated_agent_evaluation.agentcore.AgentCoreRuntimeExecutor; "
        "calibrated_agent_evaluation.agentcore.RuntimeProbeAdapter",
    )

    assert _aws_sdks(before) == set()
    assert before & _RUNTIME_MODULES == set()
    assert {"boto3", "botocore"} <= _aws_sdks(after)
    assert after >= _RUNTIME_MODULES


requires_strands_evals = pytest.mark.skipif(
    importlib.util.find_spec("strands_evals") is None, reason="needs the [agentcore] extra"
)


@requires_strands_evals
def test_detector_names_load_on_first_access_and_not_before() -> None:
    out = _run(
        """
        import sys
        import calibrated_agent_evaluation.agentcore
        def loaded():
            return sorted(
                m for m in sys.modules
                if m.startswith("calibrated_agent_evaluation.agentcore.detectors")
                or m.partition(".")[0] in ("strands", "strands_evals")
            )
        print("before:", loaded())
        vocabulary = calibrated_agent_evaluation.agentcore.failure_vocabulary
        module = sys.modules.get("calibrated_agent_evaluation.agentcore.detectors.vocabulary")
        print("after:", module is not None)
        print("same:", vocabulary is getattr(module, "failure_vocabulary", None))
        """
    )
    assert "before: []" in out
    assert "after: True" in out
    assert "same: True" in out


# A None entry in sys.modules makes every import under that root fail, but only in a fresh
# process: once a submodule such as strands.models.model is cached, importing it skips the root.
_ACCESS_WITH_ROOT_BLOCKED = """
import json
import sys
sys.modules[sys.argv[1]] = None
import calibrated_agent_evaluation.agentcore
try:
    getattr(calibrated_agent_evaluation.agentcore, sys.argv[2])
except Exception as exc:
    cause = exc.__cause__
    print(json.dumps({
        "raised": type(exc).__name__,
        "extras": list(getattr(exc, "extras", ())),
        "message": str(exc),
        "cause": type(cause).__name__ if cause is not None else None,
        "missing": getattr(cause, "name", None),
    }))
else:
    print(json.dumps({"raised": None}))
"""


def _access_with_root_blocked(root: str, name: str) -> dict[str, object]:
    done = subprocess.run(
        [sys.executable, "-c", _ACCESS_WITH_ROOT_BLOCKED, root, name],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    ("root", "name"),
    [("strands_evals", "failure_vocabulary"), ("strands", "detector_bedrock_model")],
)
def test_a_missing_strands_sdk_raises_the_typed_missing_extra_error(root: str, name: str) -> None:
    raised = _access_with_root_blocked(root, name)

    assert raised["raised"] == "AgentCoreExtraMissing"
    assert raised["extras"] == ["agentcore"]
    assert f"calibrated_agent_evaluation.agentcore.{name} needs the [agentcore] extra" in str(
        raised["message"]
    )
    assert raised["cause"] == "ModuleNotFoundError"
    # The blocked root, or a submodule under it when the root is cached as None.
    assert str(raised["missing"]).partition(".")[0] == root


def test_the_detector_types_import_without_the_vendor_sdks() -> None:
    out = _run(
        """
        import sys
        roots = ("strands", "strands_evals", "boto3", "botocore", "bedrock_agentcore")
        for root in roots:
            sys.modules[root] = None
        from calibrated_agent_evaluation.agentcore.detectors import types
        from calibrated_agent_evaluation.agentcore import (
            DetectorCallError,
            DetectorCallLimitExceeded,
            DetectorConfigError,
            DetectorError,
            DetectorOutputError,
            DetectorSessionTooLarge,
            DraftNamingError,
            SessionConversionError,
            VocabularyParseError,
        )
        print("types:", sorted(types.__all__))
        loaded = [m for m, module in sys.modules.items() if module is not None]
        print("vendor:", sorted(m for m in loaded if m.partition(".")[0] in roots))
        """
    )
    assert (
        "types: ['Confidence', 'DetectedFailure', 'Detection', 'DetectorProvenance', "
        "'FailureVocabulary', 'NamingResult', 'VocabularyEntry']"
    ) in out
    assert "vendor: []" in out


@requires_agentcore
def test_the_default_trace_store_stays_local_with_the_extra_installed(tmp_path: Path) -> None:
    from calibrated_agent_evaluation.backends import LocalBackend
    from calibrated_agent_evaluation.cli.config import resolve_backend

    assert isinstance(resolve_backend(local_root=tmp_path), LocalBackend)


@requires_agentcore
def test_no_executor_is_chosen_without_asking() -> None:
    from calibrated_agent_evaluation.cli.args import COMMAND_ARGS

    for command in ("generate", "probe"):
        parser = argparse.ArgumentParser()
        COMMAND_ARGS[command](parser)
        executor = next(a for a in parser._actions if "--executor" in a.option_strings)

        # AgentCore is offered, but only a run or a probe that names it goes there.
        assert executor.default is None, command
        assert executor.choices is not None, command
        assert "agentcore" in executor.choices, command


@requires_agentcore
def test_the_default_judge_model_runs_through_inspect_not_agentcore() -> None:
    pytest.importorskip("inspect_ai")
    from calibrated_agent_evaluation.cli.judging import model_backend
    from calibrated_agent_evaluation.execute import InspectJudgeModel

    judge = argparse.Namespace(model="bedrock/anthropic.claude-test", temperature=0.0)

    backend = model_backend(argparse.Namespace(cassette=None), judge)

    assert isinstance(backend, InspectJudgeModel)
    assert not type(backend).__module__.startswith("calibrated_agent_evaluation.agentcore")
