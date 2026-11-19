# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The methodology-plane boundary check.

No module under ``src/calibrated_agent_evaluation/`` — other than the ``backends/`` and ``execute/``
adapters — may import a vendor SDK type. The methodology plane speaks the normalized schema only; a
vendor type leaking into it is the master-spec Risk-6 tripwire and breaks both-worlds portability.

Two complementary checks:
  * a **static AST scan** of every methodology-plane module's source asserts it contains no
    direct ``import langfuse`` / ``import inspect_ai`` (catches the leak at the source line); and
  * a **runtime reachability** check imports the whole methodology plane in a subprocess and
    asserts no vendor module landed in ``sys.modules`` — catching a *transitive* leak (a
    methodology module importing another methodology module that imports a vendor SDK) the
    per-file source scan alone would miss.

The negative companion injects a deliberate ``import langfuse`` into a fixture module and
asserts the static scan catches it — proving the guard actually bites.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import calibrated_agent_evaluation
from _imports import imported_roots

_VENDOR_ROOTS = {"langfuse", "inspect_ai", "strands", "boto3", "botocore", "bedrock_agentcore"}
# Adapter packages allowed to import vendor SDKs (they are the isolation seam).
# adds `calibrated_agent_evaluation.normalize` — the trajectory-normalizer adapter package (its
# inspect/otel adapters import inspect_ai / opentelemetry, lazily) — to the same allowlist.
# adds `strands` to _VENDOR_ROOTS: StrandsExecutor imports it, but ONLY under the already
# allowlisted `calibrated_agent_evaluation.execute` package — so the methodology-plane scan actively
# guards against a stray `import strands` anywhere else (e.g. leaking into `synth/`) while execute/
# stays exempt. calibrated_agent_evaluation.agentcore is the single home of the AgentCore plug-in;
# boto3/botocore stay in the adapter packages, and bedrock_agentcore is confined more tightly, to
# calibrated_agent_evaluation.agentcore alone.
_ADAPTER_PACKAGES = {
    "calibrated_agent_evaluation.backends",
    "calibrated_agent_evaluation.execute",
    "calibrated_agent_evaluation.normalize",
    "calibrated_agent_evaluation.agentcore",
}


def _under(module_name: str, package: str) -> bool:
    return module_name == package or module_name.startswith(package + ".")


def _methodology_modules() -> list[str]:
    """Every importable module under calibrated_agent_evaluation/ that is NOT a backends/ or
    execute/ adapter."""
    names: list[str] = []
    for info in pkgutil.walk_packages(
        calibrated_agent_evaluation.__path__, prefix="calibrated_agent_evaluation."
    ):
        name = info.name
        if any(name == pkg or name.startswith(pkg + ".") for pkg in _ADAPTER_PACKAGES):
            continue
        names.append(name)
    return names


def _vendor_imports_in_source(path: Path) -> set[str]:
    """Return the vendor root modules imported by the Python source at ``path`` (static scan)."""
    return imported_roots(path) & _VENDOR_ROOTS


# --- positive: the real methodology plane is vendor-free -------------------


def _source_files(module_names: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for mod_name in module_names:
        mod_file = getattr(importlib.import_module(mod_name), "__file__", None)
        if mod_file is not None:
            files[mod_name] = Path(mod_file)
    return files


def _vendor_offenders(files: dict[str, Path]) -> dict[str, set[str]]:
    offenders: dict[str, set[str]] = {}
    for mod_name, path in files.items():
        vendor = _vendor_imports_in_source(path)
        if vendor:
            offenders[mod_name] = vendor
    return offenders


def _offender_message(offenders: dict[str, set[str]]) -> str:
    return f"vendor SDK imported in methodology-plane modules: {offenders}"


def test_methodology_plane_imports_no_vendor_module() -> None:
    offenders = _vendor_offenders(_source_files(_methodology_modules()))
    assert not offenders, _offender_message(offenders)


def test_a_planted_aws_sdk_import_is_reported_with_its_module_and_root(tmp_path: Path) -> None:
    planted = tmp_path / "judges_leak.py"
    planted.write_text("import botocore.session\n\nX = 1\n")

    offenders = _vendor_offenders({"calibrated_agent_evaluation.judges.leak": planted})

    assert offenders == {"calibrated_agent_evaluation.judges.leak": {"botocore"}}
    message = _offender_message(offenders)
    assert "calibrated_agent_evaluation.judges.leak" in message
    assert "botocore" in message


def test_schema_and_dumpload_packages_are_covered_by_the_scan() -> None:
    # Guard against a false-clean scan: the methodology-plane module list must actually include
    # the packages ships there (schema, dumpload), or the check above proves nothing.
    scanned = set(_methodology_modules())
    assert "calibrated_agent_evaluation.schema.types" in scanned
    assert "calibrated_agent_evaluation.schema.validate" in scanned
    assert "calibrated_agent_evaluation.dumpload.io" in scanned
    # ...and must EXCLUDE the adapter packages (which are allowed to import vendors).
    assert "calibrated_agent_evaluation.backends.langfuse" not in scanned


def test_normalize_is_allowlisted_but_other_normalize_free_modules_stay_clean() -> None:
    # normalize/ is the ONLY new methodology-plane package permitted to import a
    # vendor SDK, so it must be excluded from the scanned methodology modules (allowlisted)...
    scanned = set(_methodology_modules())
    assert "calibrated_agent_evaluation.normalize.inspect_adapter" not in scanned
    assert "calibrated_agent_evaluation.normalize.otel_adapter" not in scanned
    # ...while the vendor-free members of normalize/ (sources/tree/ownership/errors) genuinely
    # import no vendor SDK — proving the allowlist widening didn't blanket-excuse real leaks. These
    # are excluded from the scan by package, so assert their source is clean directly.
    import calibrated_agent_evaluation.normalize.ownership as own
    import calibrated_agent_evaluation.normalize.sources as srcs
    import calibrated_agent_evaluation.normalize.tree as tree

    for mod in (srcs, tree, own):
        mod_file = mod.__file__
        assert mod_file is not None
        assert _vendor_imports_in_source(Path(mod_file)) == set(), mod.__name__


def test_no_vendor_module_reachable_after_importing_methodology_plane() -> None:
    # Transitive guard the per-file AST scan can't give: import EVERY methodology-plane module
    # in a clean subprocess and assert no vendor module ended up in sys.modules. This catches a
    # leak where a methodology module imports another methodology module that imports a vendor
    # SDK. Subprocess so the test process's own (legitimate) vendor imports don't pollute it.
    program = (
        "import sys, pkgutil, importlib, calibrated_agent_evaluation\n"
        f"ADAPTERS = {tuple(sorted(_ADAPTER_PACKAGES))!r}\n"
        "for info in pkgutil.walk_packages(calibrated_agent_evaluation.__path__, "
        "'calibrated_agent_evaluation.'):\n"
        "    n = info.name\n"
        "    if any(n == p or n.startswith(p + '.') for p in ADAPTERS):\n"
        "        continue\n"
        "    importlib.import_module(n)\n"
        f"roots = {tuple(sorted(_VENDOR_ROOTS))!r}\n"
        "leaked = [m for m in sys.modules if m.split('.')[0] in roots]\n"
        "print('LEAKED:' + ','.join(sorted(leaked)))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"vendor module reachable after importing the methodology plane: {result.stdout.strip()}"
    )


@pytest.mark.parametrize(
    "module",
    [
        "calibrated_agent_evaluation.agentcore.runtime_executor",
        "calibrated_agent_evaluation.agentcore.simulated_user",
    ],
)
def test_agentcore_conversation_modules_do_not_load_the_sampling_stack(module: str) -> None:
    # The [agentcore] extra does not install what calibrated_agent_evaluation.sampling needs, and
    # anything under calibrated_agent_evaluation.synth loads calibrated_agent_evaluation.sampling
    # through the package's __init__. A clean subprocess, so the test process's own imports do not
    # hide a leak.
    program = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[:2] in "
        "(['calibrated-agent-evaluation', 'sampling'], ['calibrated-agent-evaluation', 'synth']))\n"
        "print('LOADED:' + ','.join(loaded))\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{module} loaded {result.stdout.strip()} {result.stderr}"


# --- negative: the check catches a deliberately-planted vendor import --------------


def test_check_catches_an_injected_vendor_import(tmp_path: Path) -> None:
    # Write a fake methodology-plane module that imports a vendor SDK; the static scan must flag
    # it. This proves the guard bites (a guard that never fails is not a guard).
    planted = tmp_path / "fake_methodology_module.py"
    planted.write_text("import langfuse\n\nX = 1\n")
    assert _vendor_imports_in_source(planted) == {"langfuse"}


def test_allowlist_widening_does_not_mask_a_leak_in_a_non_allowlisted_module() -> None:
    # widening the allowlist to include normalize/ must NOT excuse a vendor import in a
    # different (non-adapter) methodology module. Simulate the exact classification the real scan
    # uses: a leak in calibrated_agent_evaluation.schema.* is still flagged; a normalize.* module is
    # exempt.
    def is_allowlisted(module_name: str) -> bool:
        return any(
            module_name == pkg or module_name.startswith(pkg + ".") for pkg in _ADAPTER_PACKAGES
        )

    # A non-allowlisted methodology module that imported inspect_ai would still be caught...
    assert not is_allowlisted("calibrated_agent_evaluation.schema.validate")
    # ...while the newly-allowlisted normalizer adapter is (legitimately) exempt.
    assert is_allowlisted("calibrated_agent_evaluation.normalize.inspect_adapter")
    assert "inspect_ai" in _VENDOR_ROOTS  # the very import the widened package is allowed to make


def test_check_passes_on_a_clean_module(tmp_path: Path) -> None:
    clean = tmp_path / "clean_module.py"
    clean.write_text("from calibrated_agent_evaluation.schema.types import Trace\n\nY = 2\n")
    assert _vendor_imports_in_source(clean) == set()


@pytest.mark.parametrize("vendor", sorted(_VENDOR_ROOTS))
def test_check_catches_from_import_form(tmp_path: Path, vendor: str) -> None:
    planted = tmp_path / f"from_{vendor}.py"
    planted.write_text(f"from {vendor}.client import Thing\n")
    assert _vendor_imports_in_source(planted) == {vendor}


# --- the review surface is inside the vendor-free methodology plane ----


def test_review_surface_is_scanned_and_vendor_free() -> None:
    # review/ is a methodology-plane package (NOT an adapter) — it must be covered by the
    # boundary scan and import no Langfuse/Inspect SDK. (fasthtml is the UI framework, not a
    # data-plane vendor, so it is not in _VENDOR_ROOTS.)
    scanned = set(_methodology_modules())
    review_modules = {
        m
        for m in scanned
        if m == "calibrated_agent_evaluation.review"
        or m.startswith("calibrated_agent_evaluation.review.")
    }
    assert review_modules, "review/ must be covered by the methodology-plane boundary scan"
    for name in review_modules:
        module = importlib.import_module(name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is not None:
            assert _vendor_imports_in_source(Path(mod_file)) == set(), name


def test_scan_catches_a_vendor_import_planted_in_a_review_module(tmp_path: Path) -> None:
    # negative: a vendor import planted in a review-style module is flagged by the scan.
    planted = tmp_path / "review_leak.py"
    planted.write_text("from inspect_ai import Task\n\nWIDGET = 1\n")
    assert _vendor_imports_in_source(planted) == {"inspect_ai"}


# --- the dataset loader SDKs are confined to normalize.datasets --------------

# The loader SDKs adds (pyarrow for parquet, huggingface_hub for remote resolve). Like the
# vendor roots, these may appear ONLY under the allowlisted normalize/ package — and, more
# tightly, only under its `datasets` subpackage (the convention adapters read plain OTLP-JSON and
# must stay loader-SDK-free).
_LOADER_ROOTS = {"pyarrow", "huggingface_hub"}


def _loader_imports_in_source(path: Path) -> set[str]:
    return imported_roots(path) & _LOADER_ROOTS


def _all_calibrated_agent_evaluation_modules() -> list[str]:
    names: list[str] = []
    for info in pkgutil.walk_packages(
        calibrated_agent_evaluation.__path__, prefix="calibrated_agent_evaluation."
    ):
        names.append(info.name)
    return names


def test_loader_sdks_are_imported_only_under_normalize_datasets() -> None:
    # Every calibrated_agent_evaluation module that imports pyarrow/huggingface_hub must live under
    # calibrated_agent_evaluation.normalize.datasets — a loader import anywhere else (incl. the
    # convention adapters or the dataset __init__ public surface) is a boundary violation.
    offenders: dict[str, set[str]] = {}
    for mod_name in _all_calibrated_agent_evaluation_modules():
        module = importlib.import_module(mod_name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is None:
            continue
        loaders = _loader_imports_in_source(Path(mod_file))
        if loaders and not mod_name.startswith("calibrated_agent_evaluation.normalize.datasets"):
            offenders[mod_name] = loaders
    assert not offenders, f"loader SDK imported outside normalize.datasets: {offenders}"


def test_convention_adapters_are_loader_sdk_free() -> None:
    # The convention layer reconstructs from plain OTLP-JSON dicts — it must never import a loader
    # SDK (that is the datasets readers' job). Assert every conventions/* module source is clean.
    import calibrated_agent_evaluation.normalize.conventions as conv_pkg

    conv_modules = [
        info.name
        for info in pkgutil.walk_packages(
            conv_pkg.__path__, prefix="calibrated_agent_evaluation.normalize.conventions."
        )
    ]
    assert conv_modules, "conventions/ package must expose modules to scan"
    for name in conv_modules:
        module = importlib.import_module(name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is not None:
            assert _loader_imports_in_source(Path(mod_file)) == set(), name


def test_public_dataset_descriptors_expose_no_loader_type() -> None:
    # The public source descriptors are vendor-free dataclasses over stdlib types — importing them
    # must not drag a loader SDK into sys.modules, and their fields are stdlib/normalized types.
    import calibrated_agent_evaluation.normalize.sources as srcs

    mod_file = srcs.__file__
    assert mod_file is not None
    assert _loader_imports_in_source(Path(mod_file)) == set()
    # The descriptors exist and are frozen dataclasses (public input surface).
    for name in ("ParquetDataset", "JSONLDataset", "HFDataset", "SessionSpans"):
        assert hasattr(srcs, name), name


# --- the synthesis borrow SDK is confined to calibrated_agent_evaluation.synth.borrow ----------

# The synthesis engine BORROWS strands-agents-evals (`strands_evals`). It lives in a SEPARATE set
# (NOT _VENDOR_ROOTS): synth/ is a methodology-plane package, so it is scanned by the
# langfuse/inspect_ai/strands check like any other, and `strands_evals` is confined MORE tightly to
# the one isolation sub-package `synth/borrow/`. Adding it to _VENDOR_ROOTS instead would force
# adding synth.borrow to _ADAPTER_PACKAGES (exempting it from the vendor scan) — the separate-set
# approach keeps borrow/ inside every other scan.
# NOTE: only `strands_evals` (the synthesis-specific borrow) is confined here — plain `strands` is a
# legitimate execute/-adapter vendor for the StrandsExecutor (governed by _VENDOR_ROOTS +
# _ADAPTER_PACKAGES), so it must NOT be in this synth-confinement set.
_SYNTH_ROOTS = {"strands_evals"}
# calibrated_agent_evaluation.agentcore may import strands_evals too: simulated AgentCore sessions
# and failure detection run strands_evals over AgentCore traces, and that code belongs with the rest
# of the plug-in.
_SYNTH_PACKAGES = (
    "calibrated_agent_evaluation.synth.borrow",
    "calibrated_agent_evaluation.agentcore",
)


def _synth_sdk_allowed(module_name: str) -> bool:
    return any(_under(module_name, package) for package in _SYNTH_PACKAGES)


def _synth_imports_in_source(path: Path) -> set[str]:
    return imported_roots(path) & _SYNTH_ROOTS


def test_synth_sdk_imported_only_under_synth_borrow() -> None:
    # Every calibrated_agent_evaluation module that imports strands_evals/strands must live under
    # calibrated_agent_evaluation.synth.borrow — a synth-SDK import anywhere else (incl. the synth
    # core generate/guards/risk/types or the synth __init__ public surface) is a boundary violation.
    offenders: dict[str, set[str]] = {}
    for mod_name in _all_calibrated_agent_evaluation_modules():
        module = importlib.import_module(mod_name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is None:
            continue
        synth_sdks = _synth_imports_in_source(Path(mod_file))
        if synth_sdks and not _synth_sdk_allowed(mod_name):
            offenders[mod_name] = synth_sdks
    assert not offenders, f"synth SDK imported outside {_SYNTH_PACKAGES}: {offenders}"


def test_synth_sdk_allowlist_matches_whole_package_names() -> None:
    assert _synth_sdk_allowed("calibrated_agent_evaluation.synth.borrow.simulator")
    assert _synth_sdk_allowed("calibrated_agent_evaluation.agentcore.detectors")
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.synth.generate")
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.synth.borrowed")
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.agentcorex")
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.stats")


def test_scan_flags_a_detectors_import_planted_outside_the_allowed_packages(
    tmp_path: Path,
) -> None:
    planted = tmp_path / "detectors_leak.py"
    planted.write_text("from strands_evals import detectors\n")
    assert _synth_imports_in_source(planted) == {"strands_evals"}
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.synth.generate")


def test_a_detector_import_planted_in_cluster_is_flagged(tmp_path: Path) -> None:
    # The detector and its namer live in calibrated_agent_evaluation.agentcore. The cluster module
    # the namer feeds, and the review module's Prefill seam, reach detector output only through
    # protocols, so neither may import the vendor SDKs.
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.review.prefill")
    assert not _synth_sdk_allowed("calibrated_agent_evaluation.cluster.embed")
    assert {
        "calibrated_agent_evaluation.cluster.embed",
        "calibrated_agent_evaluation.review.prefill",
    } <= set(_methodology_modules())
    planted = tmp_path / "cluster_namer_leak.py"
    planted.write_text("from strands.models.model import Model\n\nNAMER = Model\n")

    offenders = _vendor_offenders({"calibrated_agent_evaluation.cluster.namer_leak": planted})

    assert offenders == {"calibrated_agent_evaluation.cluster.namer_leak": {"strands"}}


def test_synth_core_modules_are_sdk_free() -> None:
    # The synth core reconstructs everything from normalized types — it must never import the
    # borrowed SDK (that is borrow/simulator.py's job). Assert every synth core module is clean.
    import calibrated_agent_evaluation.synth as synth_pkg

    core_modules = [
        info.name
        for info in pkgutil.walk_packages(
            synth_pkg.__path__, prefix="calibrated_agent_evaluation.synth."
        )
        if not info.name.startswith("calibrated_agent_evaluation.synth.borrow")
    ]
    assert core_modules, "synth/ package must expose core modules to scan"
    for name in core_modules:
        module = importlib.import_module(name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is not None:
            assert _synth_imports_in_source(Path(mod_file)) == set(), name


def test_scan_catches_a_synth_sdk_import_planted_in_a_core_module(tmp_path: Path) -> None:
    # Negative: a strands import planted in a synth-core-style module is flagged by the scan
    # (a guard that never fails is not a guard).
    planted = tmp_path / "synth_core_leak.py"
    planted.write_text("from strands_evals import ActorSimulator\n\nX = 1\n")
    assert _synth_imports_in_source(planted) == {"strands_evals"}


# --- the AgentCore SDK is confined to calibrated_agent_evaluation.agentcore
# ------------------------

_AGENTCORE_ONLY_ROOTS = {"bedrock_agentcore"}


def _agentcore_sdk_offenders(files: dict[str, Path]) -> dict[str, set[str]]:
    offenders: dict[str, set[str]] = {}
    for mod_name, path in files.items():
        if _under(mod_name, "calibrated_agent_evaluation.agentcore"):
            continue
        roots = _vendor_imports_in_source(path) & _AGENTCORE_ONLY_ROOTS
        if roots:
            offenders[mod_name] = roots
    return offenders


def test_agentcore_sdk_imported_only_under_calibrated_agent_evaluation_agentcore() -> None:
    offenders = _agentcore_sdk_offenders(_source_files(_all_calibrated_agent_evaluation_modules()))
    assert not offenders, (
        f"bedrock_agentcore imported outside calibrated_agent_evaluation.agentcore: {offenders}"
    )


def test_a_bedrock_agentcore_import_in_another_adapter_package_is_flagged(tmp_path: Path) -> None:
    planted = tmp_path / "execute_leak.py"
    planted.write_text("from bedrock_agentcore.runtime import BedrockAgentCoreApp\n")
    allowed = tmp_path / "agentcore_ok.py"
    allowed.write_text("import bedrock_agentcore\n")

    offenders = _agentcore_sdk_offenders(
        {
            "calibrated_agent_evaluation.execute.leak": planted,
            "calibrated_agent_evaluation.agentcore.runtime": allowed,
        }
    )

    assert offenders == {"calibrated_agent_evaluation.execute.leak": {"bedrock_agentcore"}}


# --- the scenario model is methodology plane --------------------------------


def test_scenario_package_may_not_import_aws_sdks(tmp_path: Path) -> None:
    # The scenario package is scanned like any methodology module and is vendor-free today...
    scanned = set(_methodology_modules())
    assert "calibrated_agent_evaluation.scenarios.types" in scanned
    clean = _vendor_offenders(
        _source_files(
            sorted(m for m in scanned if _under(m, "calibrated_agent_evaluation.scenarios"))
        )
    )
    assert clean == {}
    # ...and a boto3 import planted in one of its modules is reported.
    planted = tmp_path / "scenarios_leak.py"
    planted.write_text("import boto3\n\nCLIENT = None\n")
    assert _vendor_offenders({"calibrated_agent_evaluation.scenarios.leak": planted}) == {
        "calibrated_agent_evaluation.scenarios.leak": {"boto3"}
    }


def test_audit_modules_other_than_the_vendor_seam_load_no_aws_sdk() -> None:
    # The A/B audit's records, arm reader and report live in the AgentCore plug-in, which the
    # scan above skips as an adapter package; only its optimization module may reach AWS.
    assert _audit_import_check().returncode == 0


def _audit_import_check(plant: str = "") -> subprocess.CompletedProcess[str]:
    """Import the vendor-free audit modules in a fresh interpreter, with ``plant`` added to arms."""
    program = (
        "import sys, importlib, importlib.util\n"
        f"plant = {plant!r}\n"
        "if plant:\n"
        "    spec = importlib.util.find_spec('calibrated_agent_evaluation.agentcore.arms')\n"
        "    source = open(spec.origin, encoding='utf-8').read() + '\\n' + plant + '\\n'\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules['calibrated_agent_evaluation.agentcore.arms'] = module\n"
        "    exec(compile(source, spec.origin, 'exec'), module.__dict__)\n"
        "for name in ('audit_types', 'arms', 'replay', 'audit_report'):\n"
        "    importlib.import_module('calibrated_agent_evaluation.agentcore.' + name)\n"
        "roots = ('boto3', 'botocore', 'bedrock_agentcore')\n"
        "leaked = [m for m in sys.modules if m.split('.')[0] in roots]\n"
        "print('LEAKED:' + ','.join(sorted(leaked)))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )


def test_the_audit_boundary_check_catches_a_planted_aws_import() -> None:
    pytest.importorskip("boto3")
    result = _audit_import_check("import boto3")
    assert result.returncode == 1
    assert "boto3" in result.stdout


def test_conversation_engine_may_not_import_simulator_sdks(tmp_path: Path) -> None:
    # The conversation engine and its user cassette are methodology-plane modules: scanned, and
    # free of every vendor SDK and of the simulator SDKs today...
    engine = [
        "calibrated_agent_evaluation.synth.conversation",
        "calibrated_agent_evaluation.synth.conversation_cassette",
    ]
    scanned = set(_methodology_modules())
    present = [m for m in engine if m in scanned]
    assert "calibrated_agent_evaluation.synth.conversation" in present
    files = _source_files(present)
    assert _vendor_offenders(files) == {}
    assert {
        m: _synth_imports_in_source(p) for m, p in files.items() if _synth_imports_in_source(p)
    } == {}
    # ...and a simulator SDK planted in an engine-style module is reported by both scans.
    planted = tmp_path / "conversation_leak.py"
    planted.write_text(
        "from strands_evals import ActorSimulator\n"
        "from bedrock_agentcore.evaluation.runner import AgentCoreActorSimulator\n"
    )
    assert _synth_imports_in_source(planted) == {"strands_evals"}
    assert _agentcore_sdk_offenders(
        {"calibrated_agent_evaluation.synth.conversation_leak": planted}
    ) == {"calibrated_agent_evaluation.synth.conversation_leak": {"bedrock_agentcore"}}
