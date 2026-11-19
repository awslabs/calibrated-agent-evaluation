# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host names and URLs for the review server, with IPv6 literals handled.

Kept free of the ``[ui]`` extra so the CLI can check a bind address before probing for it.
"""

from __future__ import annotations

import ipaddress

__all__ = ["host_of", "is_loopback", "surface_url"]


def is_loopback(host: str) -> bool:
    """Whether ``host`` names this machine only (``localhost`` or a loopback address).

    Args:
        host: A bind address or host name, IPv6 with or without brackets.
    """
    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def surface_url(host: str, port: int) -> str:
    """The ``http://`` URL a browser opens for a server bound to ``host``:``port``.

    An IPv6 literal is bracketed (``http://[::1]:5001``); a bare ``::1:5001`` is not a URL.
    """
    name = host.strip("[]")
    if ":" in name:
        name = f"[{name}]"
    return f"http://{name}:{port}"


def host_of(host_header: str) -> str:
    """The host name in a ``Host`` header, without its port; IPv6 literals lose their brackets.

    Args:
        host_header: The raw header, e.g. ``127.0.0.1:5001`` or ``[::1]:5001``.

    Returns:
        The lower-cased name, or ``""`` for a malformed header (which no allowlist matches).
    """
    value = host_header.strip().lower()
    if value.startswith("["):
        name, sep, rest = value[1:].partition("]")
        return name if sep and (not rest or rest.startswith(":")) else ""
    return value.rsplit(":", 1)[0] if ":" in value else value
