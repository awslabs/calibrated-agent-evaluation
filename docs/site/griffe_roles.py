# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Griffe extension that rewrites Sphinx cross-reference roles into mkdocstrings autorefs.

The `calibrated_agent_evaluation` docstrings use RST-style cross-reference roles —
``:func:`certify```, ``:mod:`calibrated_agent_evaluation.stats```, ``:class:`CalibrationRecord```,
because they were authored against a Sphinx-shaped convention. mkdocstrings/Griffe does not
understand those roles, so without this they render literally as ``:func:`certify``` text. This
extension runs at docstring-parse time and rewrites each role into mkdocstrings' autoref link
syntax, so they become real cross-links (and degrade to plain code spans when the target can't be
resolved).

Registered via the `extensions:` option of the mkdocstrings Python handler (see mkdocs.yml), by
file path — so it needs no install and no `sys.path` juggling.

Each role becomes an optional autoref: the target is qualified against the docstring's own scope,
the leaf name (or the explicit "title <target>" title) is shown as code, and a target outside the
published API renders as plain code rather than failing the strict build.

The same mapping applies to every reference role Griffe won't otherwise handle: func, meth, class,
exc, mod, obj, data, attr, const. Literal/formatting roles are left untouched.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import griffe

if TYPE_CHECKING:
    from griffe import Attribute, Class, Function, GriffeLoader, Module, Object

# The reference roles whose target is a Python identifier we can hand to an autoref. Presentation
# roles (``:ref:``, ``:doc:``, ``:term:``…) are intentionally excluded — they don't map to API
# objects, so rewriting them to an autoref would create a dangling link.
_REF_ROLES = frozenset(
    {"func", "meth", "class", "exc", "mod", "obj", "data", "attr", "const", "py:func", "py:class"}
)

# `:role:`payload`` — capture the role name and the backtick payload. Payload may be a bare dotted
# name, or the Sphinx "title <target>" form.
_ROLE_RE = re.compile(r":(?P<role>[\w:]+):`(?P<payload>[^`]+)`")
# The "title <target>" payload form.
_TITLED_RE = re.compile(r"^(?P<title>.*?)\s*<(?P<target>[^>]+)>$")


def _resolve(scope: Object | None, target: str) -> str:
    """Qualify ``target`` against ``scope`` (its imports and parents), or return it unchanged."""
    if scope is None:
        return target
    head, _, rest = target.partition(".")
    try:
        resolved = scope.resolve(head)
    except Exception:  # noqa: BLE001 - an unresolvable name renders as plain code, not an error
        return target
    return f"{resolved}.{rest}" if rest else resolved


def _autoref(title: str, target: str) -> str:
    # `optional` makes autorefs render an unresolvable target as plain code instead of failing the
    # strict build, which is right for names that sit outside the published API surface.
    return f'<autoref identifier="{target}" optional><code>{title}</code></autoref>'


def _rewrite_payload(payload: str, scope: Object | None) -> str:
    """Turn a role payload into an optional mkdocstrings autoref."""
    titled = _TITLED_RE.match(payload.strip())
    if titled:
        title = titled.group("title").strip()
        target = titled.group("target").strip().lstrip("~")
        return _autoref(title, _resolve(scope, target))
    # A leading `~` is Sphinx for "show only the last component"; the leaf is shown either way.
    target = payload.strip().lstrip("~")
    return _autoref(target.rsplit(".", 1)[-1], _resolve(scope, target))


def _rewrite(text: str, scope: Object | None = None) -> str:
    """Rewrite every recognized reference role in ``text``; leave others verbatim."""

    def _sub(match: re.Match[str]) -> str:
        role = match.group("role").removeprefix("py:")
        if role not in _REF_ROLES and match.group("role") not in _REF_ROLES:
            return match.group(0)
        return _rewrite_payload(match.group("payload"), scope)

    return _ROLE_RE.sub(_sub, text)


class SphinxCrossrefRoles(griffe.Extension):
    """Rewrite Sphinx cross-reference roles in every docstring into mkdocstrings autorefs."""

    def _apply(self, obj: Object) -> None:
        if obj.docstring is not None and obj.docstring.value:
            obj.docstring.value = _rewrite(obj.docstring.value, obj)

    def on_module(self, *, mod: Module, loader: GriffeLoader, **kwargs: object) -> None:
        self._apply(mod)

    def on_class(self, *, cls: Class, loader: GriffeLoader, **kwargs: object) -> None:
        self._apply(cls)

    def on_function(self, *, func: Function, loader: GriffeLoader, **kwargs: object) -> None:
        self._apply(func)

    def on_attribute(self, *, attr: Attribute, loader: GriffeLoader, **kwargs: object) -> None:
        self._apply(attr)
