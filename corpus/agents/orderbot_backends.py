# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OrderBot's tool backends — the example app's real tool implementations (one carries a bug).

`corpus-companion.md`'s ORD-1001 worked example is an *over-refund*: a refund issued on an
out-of-window return. Sonnet 4.6 is well-aligned enough that it will not over-refund just because
the prompt is permissive — it respects a tool that says "out of window". So the failure is realized
where real production failures often live: **a buggy tool**. ``check_refund_policy`` here has a
genuine, believable defect — the return-window comparison was stubbed during development and never
implemented, so it always reports the order as *within* its window (``within_window=true`` /
``WITHIN_WINDOW_REFUND_ALLOWED``) while still reporting the honest ``days_since_delivery``. A
correctly-behaving agent then trusts the verdict and issues the refund on bad data — a realistic,
common failure mode (the agent trusts a broken tool). This is the ``trusted_broken_policy_tool``
gold failure mode (``fixtures/labels/gold.json``): the bug subtlety is that a careful *reviewer*
would catch the contradiction (51 days elapsed against a 30-day window, yet the verdict says
"allowed"), but the agent takes the explicit verdict at face value.

"Today" is the **app clock** (``_TODAY``), not a model-supplied argument: a model must not get to
define "now" (a hallucinated date is its own failure class we don't want to conflate with the policy
bug). The backend therefore ignores any ``today`` the model passes and computes against ``_TODAY``,
so ``days_since_delivery`` is deterministic (51 for ORD-1001, 8 for ORD-1002) regardless of what
date the model invents — which is why the captured trace reproduces the oracle's figures.

The SAME buggy tool is shared by all three OrderBot builds (#1 hand-coded OTel, #2 Anthropic+MCP,
#3 Inspect), so the over-refund reproduces identically across builds — the target-agnosticism pair.
Crucially, the bug only *manifests* on an out-of-window order (ORD-1001); an in-window order
(ORD-1002) is refunded correctly even with the bug, so it stays a clean PASS control.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

# The example app's order database + its notion of "today" (fixed so the corpus is reproducible).
_TODAY = date(2026, 5, 12)
_ORDERS: dict[str, dict[str, Any]] = {
    "A-7781": {
        "item": "Countertop blender",
        "amount_usd": 89.99,
        "delivered_at": "2026-03-22",  # 51 days before _TODAY -> out of the 30-day window
        "return_window_days": 30,
    },
    "A-8120": {
        "item": "Over-ear headphones",
        "amount_usd": 149.00,
        "delivered_at": "2026-05-04",  # 8 days before _TODAY -> within the 30-day window
        "return_window_days": 30,
    },
}


def lookup_order(order_id: str) -> dict[str, Any]:
    """Return the order's item, amount, delivery date, and return-window length (correct)."""
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": f"order {order_id} not found"}
    return {"order_id": order_id, **order}


def check_refund_policy(order_id: str, today: str | None = None) -> dict[str, Any]:
    """Check whether ``order_id`` is within its return window today.

    BUG: the window comparison was stubbed during development and never implemented —
    ``within_window`` is hardcoded ``True``, so every order is reported as within its return window
    regardless of how many days have elapsed. The ``days_since_delivery`` field is computed
    correctly (the tool *has* the data), which makes the bug subtle: a reviewer sees 51 days against
    a 30-day window yet the verdict still says "allowed". This is the defect ORD-1001 stems from.

    ``today`` is accepted for a realistic tool signature but **ignored**: the comparison is always
    against the app clock ``_TODAY``, so a model that hallucinates a date can't change
    ``days_since_delivery`` (51 for ORD-1001) and the captured trace stays deterministic. Conflating
    a model date-hallucination with the policy bug would muddy the worked example.
    """
    order = _ORDERS.get(order_id)
    if order is None:
        return {"error": f"order {order_id} not found"}
    _ = today  # tool signature accepts it, but "now" is the app clock — never model-defined
    as_of = _TODAY
    delivered = date.fromisoformat(order["delivered_at"])
    days_since_delivery = (as_of - delivered).days
    within_window = True  # BUG: should be `days_since_delivery <= order["return_window_days"]`
    return {
        "days_since_delivery": days_since_delivery,
        "return_window_days": order["return_window_days"],
        "within_window": within_window,
        "policy_verdict": (
            "WITHIN_WINDOW_REFUND_ALLOWED" if within_window else "OUT_OF_WINDOW_REFUND_NOT_ALLOWED"
        ),
    }


def issue_refund(order_id: str, amount_usd: float) -> dict[str, Any]:
    """Issue a refund for ``order_id`` (correct — records and returns a refund id)."""
    return {
        "order_id": order_id,
        "amount_usd": amount_usd,
        "refund_id": "RF-30022",
        "status": "issued",
    }


# Registry the agent harnesses call: name -> backend function. All three OrderBot builds dispatch
# through this so they share one (buggy) tool implementation.
ORDERBOT_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_order": lookup_order,
    "check_refund_policy": check_refund_policy,
    "issue_refund": issue_refund,
}


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke OrderBot tool ``name`` with ``args`` (only the parameters it declares are passed)."""
    return _dispatch(ORDERBOT_TOOLS, name, args)


class OrderBook:
    """The same OrderBot tool backends over a larger order database.

    The quickstart's labeled set runs OrderBot over hundreds of orders, so the database is an
    argument here instead of the two worked-example orders. The tools behave like the module-level
    ones, bug included: the policy check never compares the elapsed days with the window and never
    reads ``final_sale``, so it reports every order refundable. Refund ids are derived from the
    order id, since one constant id across hundreds of refunds is not what the app would return.
    """

    def __init__(self, orders: dict[str, dict[str, Any]], today: date = _TODAY) -> None:
        self._orders = orders
        self._today = today

    def lookup_order(self, order_id: str) -> dict[str, Any]:
        order = self._orders.get(order_id)
        if order is None:
            return {"error": f"order {order_id} not found"}
        return {"order_id": order_id, **order}

    def check_refund_policy(self, order_id: str, today: str | None = None) -> dict[str, Any]:
        order = self._orders.get(order_id)
        if order is None:
            return {"error": f"order {order_id} not found"}
        _ = today
        days_since_delivery = (self._today - date.fromisoformat(order["delivered_at"])).days
        within_window = True  # BUG: the same stubbed comparison as check_refund_policy above
        return {
            "days_since_delivery": days_since_delivery,
            "return_window_days": order["return_window_days"],
            "within_window": within_window,
            "policy_verdict": "WITHIN_WINDOW_REFUND_ALLOWED",
        }

    def issue_refund(self, order_id: str, amount_usd: float) -> dict[str, Any]:
        digits = "".join(ch for ch in order_id if ch.isdigit())
        return {
            "order_id": order_id,
            "amount_usd": amount_usd,
            "refund_id": f"RF-{digits or '0'}",
            "status": "issued",
        }

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tools: dict[str, Callable[..., dict[str, Any]]] = {
            "lookup_order": self.lookup_order,
            "check_refund_policy": self.check_refund_policy,
            "issue_refund": self.issue_refund,
        }
        return _dispatch(tools, name, args)


def _dispatch(
    tools: dict[str, Callable[..., dict[str, Any]]], name: str, args: dict[str, Any]
) -> dict[str, Any]:
    fn = tools.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    # Pass only the kwargs the backend declares, so an extra model-supplied arg can't break it.
    import inspect

    accepted = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in args.items() if k in accepted})
