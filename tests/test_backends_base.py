# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the backend-abstraction interface.

The interface is defined *solely* against the normalized schema. This test introspects
every method of both protocols and asserts no parameter or return annotation resolves to a
vendor (``langfuse`` / ``inspect`` / ``inspect_ai``) type — the structural guarantee that
the methodology plane never sees a vendor type through the interface.
"""

from __future__ import annotations

import inspect as _inspect  # stdlib introspection — not the eval vendor `inspect_ai`
import typing

import pytest

from calibrated_agent_evaluation.backends.base import LabelBackend, TraceStoreBackend

_VENDOR_PREFIXES = ("langfuse", "inspect_ai", "inspect.")
_PROTOCOL_METHODS = [
    (TraceStoreBackend, name)
    for name in ("put_trace", "get_trace", "query", "put_score", "get_scores", "retract_score")
] + [(LabelBackend, name) for name in ("next_item", "submit")]


def _type_module_names(tp: object) -> set[str]:
    """Collect the defining module of ``tp`` and of every type nested inside it."""
    names: set[str] = set()
    module = getattr(tp, "__module__", None)
    if isinstance(module, str):
        names.add(module)
    for arg in typing.get_args(tp):
        names |= _type_module_names(arg)
    return names


@pytest.mark.parametrize(("protocol", "method_name"), _PROTOCOL_METHODS)
def test_no_vendor_type_in_interface_signatures(protocol: type, method_name: str) -> None:
    method = getattr(protocol, method_name)
    hints = typing.get_type_hints(method)
    assert hints, f"{protocol.__name__}.{method_name} should carry resolved type hints"
    offending: set[str] = set()
    for annotation in hints.values():
        for module_name in _type_module_names(annotation):
            if any(module_name.startswith(prefix) for prefix in _VENDOR_PREFIXES):
                offending.add(module_name)
    assert not offending, (
        f"{protocol.__name__}.{method_name} exposes a vendor type from {sorted(offending)}; "
        "the interface must speak calibrated_agent_evaluation.schema types only"
    )


def test_every_interface_annotation_resolves_to_calibrated_agent_evaluation_or_builtin() -> None:
    # Positive companion: confirm the hints come from calibrated_agent_evaluation.schema / builtins
    # / typing — proving the parametrized check above is exercising real, resolvable annotations.
    # "types" covers stdlib types.UnionType (the runtime form of `X | None`).
    allowed_roots = ("calibrated_agent_evaluation", "builtins", "typing", "collections", "types")
    for protocol, method_name in _PROTOCOL_METHODS:
        method = getattr(protocol, method_name)
        for annotation in typing.get_type_hints(method).values():
            for module_name in _type_module_names(annotation):
                root = module_name.split(".")[0]
                assert root in allowed_roots, (
                    f"{protocol.__name__}.{method_name} annotation from unexpected module "
                    f"{module_name!r}"
                )


def test_concrete_signatures_have_no_vendor_default_objects() -> None:
    # Defaults are part of the signature surface; assert none is a vendor object.
    for protocol, method_name in _PROTOCOL_METHODS:
        sig = _inspect.signature(getattr(protocol, method_name))
        for param in sig.parameters.values():
            if param.default is _inspect.Parameter.empty:
                continue
            default_module = type(param.default).__module__
            assert not default_module.startswith(_VENDOR_PREFIXES), (
                f"{protocol.__name__}.{method_name} has a vendor default object"
            )
