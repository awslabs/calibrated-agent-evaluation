# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Looking up AgentCore Evaluations rates from the AWS Price List API."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.pricing import (
    EvaluationRates,
    estimate_run,
    lookup_evaluation_rates,
)
from calibrated_agent_evaluation.agentcore.session import AwsClients

_PREFIX = "USE1-Evaluations:Consumption-based:"


def _entry(usagetype: str, unit: str, usd: str) -> str:
    """One PriceList entry, in the shape the Price List API returned live."""
    dimension = {"unit": unit, "pricePerUnit": {"USD": usd}}
    return json.dumps(
        {
            "product": {"attributes": {"usagetype": usagetype, "regionCode": "us-east-1"}},
            "terms": {"OnDemand": {"T1": {"priceDimensions": {"D1": dimension}}}},
        }
    )


_LISTING = [
    _entry(f"{_PREFIX}BuiltIn-Input:Tier1", "1M Input Tokens", "2.4000000000"),
    _entry(f"{_PREFIX}BuiltIn-Output:Tier1", "1M Output Tokens", "12.0000000000"),
    _entry(f"{_PREFIX}BuiltIn-Input:Tier2", "1M Input Tokens", "1.8000000000"),
    _entry(f"{_PREFIX}CustomEvaluators:Tier1", "Evaluations", "0.0015000000"),
    _entry("USE1-Runtime:vCPU-Hours", "Hours", "0.0895000000"),
]

_REQUEST = {
    "ServiceCode": "AmazonBedrockAgentCore",
    "Filters": [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": "us-east-1"}],
    "FormatVersion": "aws_v1",
}


def _clients(region: str | None = "us-east-1") -> AwsClients:
    session = boto3.Session(
        region_name=region, aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def _listing(clients: AwsClients, price_list: list[str]) -> Stubber:
    stub = Stubber(clients.client("pricing", region="us-east-1"))
    stub.add_response(
        "get_products", {"PriceList": price_list, "FormatVersion": "aws_v1"}, _REQUEST
    )
    return stub


def test_the_regions_evaluation_rates_are_read_from_the_price_list() -> None:
    clients = _clients()
    with _listing(clients, _LISTING):
        rates = lookup_evaluation_rates(clients)
    assert rates.region == "us-east-1"
    # Tier1 only: the Tier2 rate listed alongside is not used.
    assert rates.builtin_input_per_million == Decimal("2.4000000000")
    assert rates.builtin_output_per_million == Decimal("12.0000000000")
    assert rates.custom_per_evaluation == Decimal("0.0015000000")


def test_a_region_with_no_evaluations_listing_yields_no_rates() -> None:
    clients = _clients()
    with _listing(clients, [_LISTING[-1]]):
        rates = lookup_evaluation_rates(clients)
    assert rates.builtin_input_per_million is None
    assert rates.custom_per_evaluation is None


def test_a_denied_rate_lookup_raises_a_typed_error() -> None:
    clients = _clients()
    stub = Stubber(clients.client("pricing", region="us-east-1"))
    stub.add_client_error("get_products", service_error_code="AccessDeniedException")
    with stub, pytest.raises(AgentCoreError, match="AccessDeniedException"):
        lookup_evaluation_rates(clients)


def test_a_malformed_entry_for_another_product_does_not_hide_the_evaluations_rates() -> None:
    # Only the three Evaluations rates are validated; entries for other AgentCore products are
    # skipped without validating their prices.
    bad_dimension = json.dumps(
        {
            "product": {"attributes": {"usagetype": "USE1-Runtime:vCPU-Hours"}},
            "terms": {"OnDemand": {"T1": {"priceDimensions": {"D1": "not an object"}}}},
        }
    )
    no_terms = json.dumps({"product": {"attributes": {"usagetype": "USE1-Runtime:GB-Hours"}}})
    clients = _clients()
    with _listing(clients, [*_LISTING, bad_dimension, no_terms]):
        rates = lookup_evaluation_rates(clients)
    assert rates.builtin_input_per_million == Decimal("2.4000000000")
    assert rates.builtin_output_per_million == Decimal("12.0000000000")
    assert rates.custom_per_evaluation == Decimal("0.0015000000")


@pytest.mark.parametrize(
    "entry",
    [
        "not json",
        json.dumps({"product": {"attributes": {"usagetype": f"{_PREFIX}BuiltIn-Input:Tier1"}}}),
    ],
    ids=["not-json", "no-price"],
)
def test_a_price_entry_that_does_not_parse_raises_a_typed_error(entry: str) -> None:
    clients = _clients()
    with _listing(clients, [entry]), pytest.raises(AgentCoreError, match="Price List"):
        lookup_evaluation_rates(clients)


def test_clients_with_no_region_cannot_be_priced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No region anywhere: not passed, not in the environment, not in a config file.
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-config"))
    clients = _clients(region=None)
    assert clients.region is None
    with pytest.raises(AgentCoreError, match="no region"):
        lookup_evaluation_rates(clients)


def test_the_rates_are_read_across_every_page_of_the_listing() -> None:
    clients = _clients()
    stub = Stubber(clients.client("pricing", region="us-east-1"))
    first = {"PriceList": _LISTING[:2], "FormatVersion": "aws_v1", "NextToken": "page-2"}
    stub.add_response("get_products", first, _REQUEST)
    stub.add_response(
        "get_products",
        {"PriceList": _LISTING[2:], "FormatVersion": "aws_v1"},
        {**_REQUEST, "NextToken": "page-2"},
    )
    with stub:
        rates = lookup_evaluation_rates(clients)
    # The custom rate is only on the second page.
    assert rates.custom_per_evaluation == Decimal("0.0015000000")
    stub.assert_no_pending_responses()


@pytest.mark.parametrize(
    ("rates", "reason"),
    [(EvaluationRates(region="us-east-1"), "why"), (None, None)],
    ids=["both-given", "neither-given"],
)
def test_an_estimate_takes_exactly_one_of_rates_and_a_reason(
    rates: EvaluationRates | None, reason: str | None
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        estimate_run(
            evaluator_id="e",
            evaluator_type="Builtin",
            documents_per_trace=[[]],
            rates=rates,
            no_rate_reason=reason,
        )


@pytest.mark.parametrize(
    "listing",
    [
        # The input rate listed per million output tokens.
        [_entry(f"{_PREFIX}BuiltIn-Input:Tier1", "1M Output Tokens", "2.4000000000")],
        # Two different prices for the one Tier1 input rate.
        [
            _entry(f"{_PREFIX}BuiltIn-Input:Tier1", "1M Input Tokens", "2.4000000000"),
            _entry(f"{_PREFIX}BuiltIn-Input:Tier1", "1M Input Tokens", "3.0000000000"),
        ],
    ],
    ids=["wrong-token-unit", "two-prices"],
)
def test_rates_that_cannot_be_read_unambiguously_are_refused_with_the_reason(
    listing: list[str],
) -> None:
    clients = _clients()
    with _listing(clients, listing), pytest.raises(AgentCoreError, match="BuiltIn-Input"):
        lookup_evaluation_rates(clients)


def _raw_entry(usagetype: object, dimension: object) -> str:
    return json.dumps(
        {
            "product": {"attributes": {"usagetype": usagetype, "regionCode": "us-east-1"}},
            "terms": {"OnDemand": {"T1": {"priceDimensions": {"D1": dimension}}}},
        }
    )


@pytest.mark.parametrize(
    "entry",
    [
        _raw_entry(7, {"unit": "1M Input Tokens", "pricePerUnit": {"USD": "2.4"}}),
        _raw_entry(f"{_PREFIX}BuiltIn-Input:Tier1", "not an object"),
        _entry(f"{_PREFIX}BuiltIn-Input:Tier1", "1M Input Tokens", "NaN"),
        _entry(f"{_PREFIX}BuiltIn-Input:Tier1", "1M Input Tokens", "-1"),
    ],
    ids=["usage-type-not-text", "dimension-not-an-object", "nan-price", "negative-price"],
)
def test_a_malformed_or_impossible_price_entry_is_refused_with_a_typed_error(entry: str) -> None:
    clients = _clients()
    with _listing(clients, [entry]), pytest.raises(AgentCoreError):
        lookup_evaluation_rates(clients)
