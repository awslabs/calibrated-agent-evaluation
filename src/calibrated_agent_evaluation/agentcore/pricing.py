# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price an AgentCore evaluation run from the AWS Price List API.

Before a billed run the caller sees an estimate and confirms it. The rates come from
``pricing:GetProducts`` (service code ``AmazonBedrockAgentCore``), not from constants that would go
stale: a built-in evaluator is priced per input and output token, with its model cost included; a
custom evaluator is priced per evaluation, and its own model usage (or Lambda) is billed to the
account separately. Third-party evaluators have no listed rate.

Tokens are estimated before the run, roughly by construction: per call, one token per four
characters of the trace's content (message text, tool arguments and results), plus an allowance
for the evaluator's own instructions, and a fixed number of output tokens. The usage AgentCore
reports on each result after the run is what to compare the estimate with.

Vendor boundary: the boto3 ``pricing`` call lives here, inside the AgentCore package; the rates and
the estimate are plain values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "INSTRUCTION_TOKENS",
    "OUTPUT_TOKENS_PER_CALL",
    "EvaluationRates",
    "RunEstimate",
    "estimate_run",
    "lookup_evaluation_rates",
    "unpriced_reason",
    "usage_cost",
]

# The Price List API answers only from a few endpoint regions; it is filtered by the account's
# region code, whichever endpoint answers.
_PRICING_REGION = "us-east-1"
_SERVICE_CODE = "AmazonBedrockAgentCore"
# Usage types end with these, after a region prefix (``USE1-`` for us-east-1). Only Tier1 is read:
# the Tier2 rate listed beside it is not documented as the batch rate or as a volume tier.
_BUILTIN_INPUT = "Evaluations:Consumption-based:BuiltIn-Input:Tier1"
_BUILTIN_OUTPUT = "Evaluations:Consumption-based:BuiltIn-Output:Tier1"
_CUSTOM = "Evaluations:Consumption-based:CustomEvaluators:Tier1"
# The unit each used price is quoted in, as the Price List API listed them live.
_UNITS = {
    _BUILTIN_INPUT: "1M Input Tokens",
    _BUILTIN_OUTPUT: "1M Output Tokens",
    _CUSTOM: "Evaluations",
}
_ONE_MILLION = Decimal(1_000_000)

#: Input tokens allowed per call for the evaluator's own instructions.
INSTRUCTION_TOKENS = 500
#: Output tokens assumed per call: the figure AWS's own pricing example uses.
OUTPUT_TOKENS_PER_CALL = 300
_CHARACTERS_PER_TOKEN = 4
_CONTENT_ATTRIBUTES = (
    "gen_ai.task.input",
    "gen_ai.task.output",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
)
_PER_TOKEN_TYPES = frozenset({"Builtin"})
_PER_EVALUATION_TYPES = frozenset({"Custom", "CustomDerived", "CustomCode"})


@dataclass(frozen=True)
class EvaluationRates:
    """The AgentCore Evaluations rates the Price List API lists for one region, in USD.

    A rate is ``None`` when the region lists none.
    """

    region: str
    builtin_input_per_million: Decimal | None = None
    builtin_output_per_million: Decimal | None = None
    custom_per_evaluation: Decimal | None = None


@dataclass(frozen=True)
class RunEstimate:
    """What a billed run is expected to cost, shown before the first billed call.

    ``cost_usd`` is ``None`` when no rate applies, and ``no_rate_reason`` then says why.
    """

    evaluator_id: str
    evaluator_type: str
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None
    no_rate_reason: str | None
    region: str | None

    def describe(self) -> str:
        """The estimate as one paragraph for a confirmation prompt."""
        text = (
            f"This run makes {self.calls} billed AgentCore Evaluate call(s) (one per trace) with "
            f"{self.evaluator_id}: about {self.input_tokens:,} input and {self.output_tokens:,} "
            "output tokens (a rough estimate)."
        )
        if self.cost_usd is None:
            return f"{text} There is no cost estimate: {self.no_rate_reason}."
        if self.evaluator_type in _PER_EVALUATION_TYPES:
            return (
                f"{text} Estimated cost: ${self.cost_usd:.4f}, priced per evaluation at the "
                f"{self.region} rate the AWS Price List API publishes; the evaluator's own model "
                "usage (or Lambda) is charged separately and is not included."
            )
        return (
            f"{text} Estimated cost: ${self.cost_usd:.4f} at the {self.region} per-token rates "
            "the AWS Price List API publishes (model cost included)."
        )


def lookup_evaluation_rates(clients: AwsClients) -> EvaluationRates:
    """Read the clients' region's AgentCore Evaluations rates from the AWS Price List API.

    Raises :class:`AgentCoreError` if the clients have no region, the call fails (for example it
    is denied), an entry or its usage type cannot be read, or an Evaluations rate the estimate
    uses does not parse. Entries for other AgentCore products, and entries that list no usage
    type, are skipped without validating their prices. A region that lists no Evaluations rate
    gets :class:`EvaluationRates` with every rate ``None``.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    region = clients.region
    if not region:
        raise AgentCoreError(
            "the AWS clients have no region, so there is no region to look Evaluations rates up for"
        )
    client = clients.client("pricing", region=_PRICING_REGION)
    filters = [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}]
    found: dict[str, Decimal] = {}
    try:
        pages = client.get_paginator("get_products").paginate(
            ServiceCode=_SERVICE_CODE, Filters=filters, FormatVersion="aws_v1"
        )
        for page in pages:
            for entry in page.get("PriceList", []):
                # The usage type is read first, so only the three rates used are validated: a
                # malformed entry for another AgentCore product cannot fail the lookup.
                product, usage_type = _parse_usage_type(entry)
                suffix = next(
                    (
                        s
                        for s in (_BUILTIN_INPUT, _BUILTIN_OUTPUT, _CUSTOM)
                        if usage_type is not None and usage_type.endswith(s)
                    ),
                    None,
                )
                if suffix is None:
                    continue
                dimensions = _price_dimensions(product, usage_type)
                price = _one_price(usage_type, dimensions, unit=_UNITS[suffix])
                if suffix in found and found[suffix] != price:
                    raise AgentCoreError(
                        f"the AWS Price List lists two prices for {usage_type} ({found[suffix]} "
                        f"and {price}), so which one applies is not known"
                    )
                found[suffix] = price
    except (BotoCoreError, ClientError) as exc:
        raise AgentCoreError(f"the AWS Price List lookup for {region} failed: {exc}") from exc
    return EvaluationRates(
        region=region,
        builtin_input_per_million=found.get(_BUILTIN_INPUT),
        builtin_output_per_million=found.get(_BUILTIN_OUTPUT),
        custom_per_evaluation=found.get(_CUSTOM),
    )


def _one_price(usage_type: str | None, dimensions: list[Any], *, unit: str) -> Decimal:
    """The one USD price of a used Price List entry, which must be quoted in ``unit``.

    Raises :class:`AgentCoreError` when the entry has no price or more than one, so which applies
    is not known, its unit is not the one the estimate multiplies it by, or the price is not a
    finite non-negative number.
    """
    if len(dimensions) != 1:
        raise AgentCoreError(
            f"the AWS Price List entry for {usage_type} lists {len(dimensions)} prices, not one"
        )
    (dimension,) = dimensions
    listed = dimension.get("unit")
    if listed != unit:
        raise AgentCoreError(
            f"the AWS Price List prices {usage_type} per {listed!r}, not per {unit!r}, which the "
            "estimate is computed in"
        )
    try:
        price = Decimal(dimension["pricePerUnit"]["USD"])
    except (KeyError, TypeError, InvalidOperation) as exc:
        raise AgentCoreError(
            f"the AWS Price List entry for {usage_type} has no USD price: {exc!r}"
        ) from exc
    if not price.is_finite() or price < 0:
        raise AgentCoreError(
            f"the AWS Price List entry for {usage_type} has the price {price}, which is not a "
            "finite non-negative number"
        )
    return price


def _parse_usage_type(entry: Any) -> tuple[Any, str | None]:
    """An entry's parsed product and its usage type.

    Raises :class:`AgentCoreError` when the entry does not parse or its usage type is not text:
    such an entry may be one of the rates the estimate uses, so it is refused rather than skipped.
    An entry that lists no usage type gets ``None`` and is treated as another product.
    """
    try:
        product = json.loads(entry) if isinstance(entry, str) else entry
        usage_type = product["product"]["attributes"].get("usagetype")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise AgentCoreError(f"an AWS Price List entry did not parse: {exc!r}") from exc
    if usage_type is not None and not isinstance(usage_type, str):
        raise AgentCoreError(
            f"an AWS Price List entry has a usage type that is not text: {usage_type!r}"
        )
    return product, usage_type


def _price_dimensions(product: Any, usage_type: str | None) -> list[Any]:
    """A used entry's on-demand price dimensions; raises if they do not parse."""
    try:
        dimensions = [
            dimension
            for term in product.get("terms", {}).get("OnDemand", {}).values()
            for dimension in term["priceDimensions"].values()
        ]
    except (KeyError, TypeError, AttributeError) as exc:
        raise AgentCoreError(
            f"the AWS Price List entry for {usage_type} did not parse: {exc!r}"
        ) from exc
    if not all(isinstance(dimension, dict) for dimension in dimensions):
        raise AgentCoreError(
            f"the AWS Price List entry for {usage_type} has a price that is not an object"
        )
    return dimensions


def _content_characters(documents: Sequence[dict[str, Any]]) -> int:
    return sum(
        len(value)
        for document in documents
        for key in _CONTENT_ATTRIBUTES
        if isinstance(value := (document.get("attributes") or {}).get(key), str)
    )


def estimate_run(
    *,
    evaluator_id: str,
    evaluator_type: str,
    documents_per_trace: Sequence[Sequence[dict[str, Any]]],
    rates: EvaluationRates | None,
    no_rate_reason: str | None,
) -> RunEstimate:
    """Estimate a run of one call per trace from its serialized span documents.

    The token counts are a rough estimate: each call's input is its span documents' content
    characters divided by four, rounded up, plus ``INSTRUCTION_TOKENS`` for the evaluator's own
    instructions, and its output is ``OUTPUT_TOKENS_PER_CALL``. The logged actual usage after a
    run is what to compare them with.

    The cost uses the rates :func:`lookup_evaluation_rates` read from the AWS Price List API (the
    Tier1 rates; the Tier2 rate is not used). A built-in evaluator is priced per estimated input and
    output token, with model cost included. A custom evaluator (``Custom``, ``CustomDerived`` or
    ``CustomCode``) is priced per evaluation, one per call, and the model usage (or Lambda) it is
    billed for on top is not included. A third-party or unknown evaluator type has no rate. Without
    a cost, the estimate still names the calls and tokens and gives the reason: the one passed as
    ``no_rate_reason`` (the lookup failed or was not made), or the one :func:`usage_cost` returns
    (the region lists no rate, or the evaluator type has none).

    Exactly one of ``rates`` and ``no_rate_reason`` is given; ``ValueError`` otherwise.
    """
    if (rates is None) == (no_rate_reason is None):
        raise ValueError("give exactly one of rates and no_rate_reason")
    calls = len(documents_per_trace)
    input_tokens = sum(
        math.ceil(_content_characters(documents) / _CHARACTERS_PER_TOKEN) + INSTRUCTION_TOKENS
        for documents in documents_per_trace
    )
    output_tokens = OUTPUT_TOKENS_PER_CALL * calls
    cost: Decimal | None = None
    if rates is not None:
        cost, no_rate_reason = usage_cost(
            evaluator_type, rates, input_tokens, output_tokens, evaluations=calls
        )
    return RunEstimate(
        evaluator_id=evaluator_id,
        evaluator_type=evaluator_type,
        calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        no_rate_reason=no_rate_reason,
        region=rates.region if rates is not None else None,
    )


def unpriced_reason(evaluator_type: str) -> str | None:
    """Why an evaluator type has no rate to look up, or ``None`` when it has one."""
    if evaluator_type == "ThirdParty":
        return "a third-party evaluator has no published AgentCore rate"
    if evaluator_type not in _PER_TOKEN_TYPES | _PER_EVALUATION_TYPES:
        return f"evaluator type {evaluator_type!r} has no known AgentCore pricing"
    return None


def usage_cost(
    evaluator_type: str,
    rates: EvaluationRates,
    input_tokens: int,
    output_tokens: int,
    *,
    evaluations: int,
) -> tuple[Decimal | None, str | None]:
    """The cost of the given usage at ``rates``, or ``None`` with the reason there is none."""
    reason = unpriced_reason(evaluator_type)
    if reason is not None:
        return None, reason
    unlisted = (
        f"no AgentCore Evaluations rate is listed for {rates.region} in the AWS Price List API"
    )
    if evaluator_type in _PER_EVALUATION_TYPES:
        if rates.custom_per_evaluation is None:
            return None, unlisted
        return rates.custom_per_evaluation * evaluations, None
    if rates.builtin_input_per_million is None or rates.builtin_output_per_million is None:
        return None, unlisted
    cost = (
        Decimal(input_tokens) * rates.builtin_input_per_million
        + Decimal(output_tokens) * rates.builtin_output_per_million
    ) / _ONE_MILLION
    return cost, None
