# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI config-resolver tests.

The resolver reports whether an optional extra is available and, when it is not, the exact
uv install hint. Backend resolution defaults to the offline
local backend and never overrides a configured one.
"""

from __future__ import annotations

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.backends import LocalBackend
from calibrated_agent_evaluation.cli.config import (
    Degradation,
    extra_for_import,
    probe_extra,
    resolve_backend,
)


def test_present_module_reports_available_no_hint():
    """Positive: an installed module resolves with available=True and no hint."""
    # `json` is stdlib — always importable — standing in for "an installed capability".
    result = probe_extra("json")

    assert isinstance(result, Degradation)
    assert result.available is True
    assert result.install_hint is None


def test_absent_module_reports_exact_install_hint():
    """Negative: an absent module maps to the exact pip install hint for its extra."""
    result = probe_extra("boto3_not_installed_xyz")

    assert result.available is False
    assert result.install_hint is not None
    assert result.install_hint == install_hint(extra_for_import(result.capability))


def test_known_module_maps_to_its_named_extra():
    """A known vendor module maps to its declared extra name (e.g. boto3 → bedrock)."""
    assert extra_for_import("boto3") == "bedrock"
    assert extra_for_import("inspect_ai") == "inspect"
    # An unknown module falls back to its own name rather than a wrong extra.
    assert extra_for_import("some_unknown_pkg") == "some_unknown_pkg"


def test_probe_does_not_import_the_module(monkeypatch):
    """Probing availability must not import the module (no load side-effect)."""
    import sys

    # boto3 is available in the dev env (corpus/bedrock), but a mere probe must not load it.
    sys.modules.pop("boto3", None)
    probe_extra("boto3")
    assert "boto3" not in sys.modules, "probe_extra must use find_spec, not import"


def test_no_backend_configured_defaults_to_local(tmp_path):
    """With nothing configured, the resolver returns a LocalBackend (offline default)."""
    backend = resolve_backend(None, local_root=tmp_path / "store")

    assert isinstance(backend, LocalBackend)


def test_configured_backend_is_not_overridden(tmp_path):
    """Negative (complementary state): a configured backend is returned unchanged."""
    configured = LocalBackend(root=tmp_path / "explicit")

    resolved = resolve_backend(configured, local_root=tmp_path / "other")

    assert resolved is configured
