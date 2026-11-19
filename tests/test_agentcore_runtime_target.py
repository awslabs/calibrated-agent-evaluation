# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The runtime target refuses what can't be invoked, and every run gets fresh IDs and headers."""

from __future__ import annotations

import importlib.metadata
import re
import secrets
from urllib.parse import unquote

import pytest

pytest.importorskip("boto3")

import botocore.session

from _agentcore_runtime_fakes import BUNDLE_ARN, BUNDLE_VERSION, RUNTIME_ARN
from calibrated_agent_evaluation.agentcore import (
    RuntimeInvocationError,
    RuntimeTarget,
    runtime,
    runtime_executor,
)
from calibrated_agent_evaluation.agentcore.runtime import (
    AGENT_RUNTIME_ARN,
    CONFIGURATION_BUNDLE_ARN,
    CONFIGURATION_BUNDLE_VERSION,
    ENDPOINT_NAME,
    PARENT_SPAN_ID_PATTERN,
    SESSION_ID_PATTERN,
    W3C_TRACE_ID_PATTERN,
    RunIds,
    baggage,
    new_run_ids,
    traceparent,
)
from calibrated_agent_evaluation.probe.adapter import PAYLOAD_FORMATS

_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-01")
_IDS = {
    "session_id": "calibrated-agent-evaluation-4bf92f3577b34da6a3ce929d0e0e4736",
    "w3c_trace_id": "0af7651916cd43dd8448eb211c80319c",
    "parent_span_id": "b7ad6b7169203331",
}


def _target(**overrides: object) -> RuntimeTarget:
    fields: dict[str, object] = {"runtime_arn": RUNTIME_ARN, "payload_format": "agentcore_sse"}
    fields.update(overrides)
    return RuntimeTarget(**fields)  # pyright: ignore[reportArgumentType]


def _baggage_entries(header: str) -> dict[str, str]:
    """The W3C baggage members of ``header``, values percent-decoded."""
    entries: dict[str, str] = {}
    for member in header.split(","):
        key, _, value = member.strip().partition("=")
        entries[key.strip()] = unquote(value.split(";")[0].strip())
    return entries


def test_runtime_target_reads_region_and_runtime_id_from_the_arn() -> None:
    target = _target()

    assert target.region == "us-west-2"
    assert target.runtime_id == "calibrated_agent_evaluation_fixture-AbCdEf1234"
    assert target.qualifier == "DEFAULT"

    gov = _target(
        runtime_arn=(
            "arn:aws-us-gov:bedrock-agentcore:us-gov-west-1:111122223333:runtime/a-0123456789"
        )
    )
    assert gov.region == "us-gov-west-1"
    assert gov.runtime_id == "a-0123456789"


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws:s3:::calibrated-agent-evaluation-fixture-bucket",
        "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/calibrated_agent_evaluation_fixture",
        f"{RUNTIME_ARN} trailing",
        f"{RUNTIME_ARN}/runtime-endpoint/DEFAULT",
    ],
    ids=["s3-arn", "no-suffix", "trailing-text", "endpoint-arn"],
)
def test_a_non_runtime_arn_is_refused_naming_it(arn: str) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _target(runtime_arn=arn)

    assert arn in str(refused.value)


def test_an_unknown_payload_format_is_refused_listing_the_accepted_names() -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _target(payload_format="sse")

    message = str(refused.value)
    assert "'sse'" in message
    for name in [*(fmt.name for fmt in PAYLOAD_FORMATS), "custom"]:
        assert name in message


def test_custom_is_an_accepted_format_name() -> None:
    assert _target(payload_format="custom").payload_format == "custom"


def test_the_payload_builder_type_is_one_public_alias_in_both_runtime_modules() -> None:
    # Defined where request bodies are built, and still importable from the executor's module.
    assert "PayloadBuilder" in runtime.__all__
    assert "PayloadBuilder" in runtime_executor.__all__
    assert runtime_executor.PayloadBuilder is runtime.PayloadBuilder


@pytest.mark.parametrize(
    ("given", "named", "missing"),
    [
        ({"config_bundle_arn": BUNDLE_ARN}, "config_bundle_arn", "config_bundle_version"),
        ({"config_bundle_version": BUNDLE_VERSION}, "config_bundle_version", "config_bundle_arn"),
        # An empty part counts as missing.
        (
            {"config_bundle_arn": BUNDLE_ARN, "config_bundle_version": ""},
            "config_bundle_arn",
            "config_bundle_version",
        ),
        (
            {"config_bundle_arn": "", "config_bundle_version": BUNDLE_VERSION},
            "config_bundle_version",
            "config_bundle_arn",
        ),
    ],
    ids=["arn-only", "version-only", "empty-version", "empty-arn"],
)
def test_half_a_configuration_bundle_is_refused_naming_the_missing_part(
    given: dict[str, str], named: str, missing: str
) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _target(**given)

    # The part that is set and the part that is missing are both named.
    assert f"{named} is set" in str(refused.value)
    assert f"{missing} is missing" in str(refused.value)


def test_a_whole_well_formed_bundle_or_none_is_accepted() -> None:
    gov_bundle = (
        "arn:aws-us-gov:bedrock-agentcore:us-gov-west-1:111122223333:"
        "configuration-bundle/calibrated-agent-evaluation-bundle_2-0123456789"
    )
    both = _target(config_bundle_arn=BUNDLE_ARN, config_bundle_version=BUNDLE_VERSION)
    gov = _target(config_bundle_arn=gov_bundle, config_bundle_version=BUNDLE_VERSION)
    neither = _target()
    both_empty = _target(config_bundle_arn="", config_bundle_version="")

    assert (both.config_bundle_arn, both.config_bundle_version) == (BUNDLE_ARN, BUNDLE_VERSION)
    assert (gov.config_bundle_arn, gov.config_bundle_version) == (gov_bundle, BUNDLE_VERSION)
    assert (neither.config_bundle_arn, neither.config_bundle_version) == (None, None)
    assert (both_empty.config_bundle_arn, both_empty.config_bundle_version) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("config_bundle_arn", RUNTIME_ARN, id="runtime-arn"),
        pytest.param("config_bundle_arn", f"{BUNDLE_ARN}/extra", id="arn-with-trailing-path"),
        pytest.param(
            "config_bundle_arn",
            "calibrated_agent_evaluation_bundle-AbCdEf1234",
            id="bare-bundle-id",
        ),
        pytest.param("config_bundle_version", "v1", id="not-a-uuid"),
        pytest.param("config_bundle_version", BUNDLE_VERSION.upper(), id="uppercase-uuid"),
        pytest.param("config_bundle_version", f"{BUNDLE_VERSION}0", id="uuid-with-trailing-digit"),
    ],
)
def test_a_malformed_bundle_arn_or_version_is_refused_naming_it(field: str, value: str) -> None:
    bundle = {
        "config_bundle_arn": BUNDLE_ARN,
        "config_bundle_version": BUNDLE_VERSION,
        field: value,
    }

    with pytest.raises(RuntimeInvocationError) as refused:
        _target(**bundle)

    assert value in str(refused.value)


def test_the_arn_and_bundle_patterns_match_the_installed_service_models() -> None:
    model = botocore.session.get_session().get_service_model("bedrock-agentcore-control")

    def pattern(shape: str) -> str:
        return model.shape_for(shape).metadata["pattern"]

    assert pattern("AgentRuntimeArn") == AGENT_RUNTIME_ARN
    assert pattern("ConfigurationBundleArn") == CONFIGURATION_BUNDLE_ARN
    assert pattern("ConfigurationBundleVersion") == CONFIGURATION_BUNDLE_VERSION
    assert pattern("EndpointName") == ENDPOINT_NAME


@pytest.mark.parametrize(
    "qualifier",
    ["", "live/1", "a" * 49, "1live"],
    ids=["empty", "slashed", "49-characters", "leading-digit"],
)
def test_a_malformed_qualifier_is_refused_naming_it(qualifier: str) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        _target(qualifier=qualifier)

    message = str(refused.value)
    assert "qualifier" in message
    assert repr(qualifier) in message


def test_default_and_an_underscored_qualifier_are_accepted() -> None:
    longest = "L" + "x" * 47

    assert _target().qualifier == "DEFAULT"
    assert _target(qualifier="DEFAULT").qualifier == "DEFAULT"
    assert _target(qualifier="live_v2").qualifier == "live_v2"
    assert _target(qualifier=longest).qualifier == longest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("session_id", "4bf92f3577b34da6a3ce929d0e0e4736", id="unprefixed-session"),
        pytest.param(
            "session_id",
            "calibrated-agent-evaluation-4BF92F3577B34DA6A3CE929D0E0E4736",
            id="uppercase-session",
        ),
        pytest.param(
            "session_id", "calibrated-agent-evaluation-" + "0" * 32, id="all-zero-session"
        ),
        pytest.param("w3c_trace_id", "0af7651916cd43dd8448eb211c80319", id="31-character-trace"),
        pytest.param("w3c_trace_id", "0" * 32, id="all-zero-trace"),
        pytest.param("parent_span_id", "0" * 16, id="all-zero-parent-span"),
    ],
)
def test_run_ids_refuse_a_malformed_or_all_zero_id(field: str, value: str) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        RunIds(**{**_IDS, field: value})

    # The field and the form it needs, never the value, which may have been read from a file.
    assert field in str(refused.value)
    assert value not in str(refused.value)


def test_new_run_ids_are_distinct_valid_and_carry_a_prefixed_hex_session_id() -> None:
    runs = [new_run_ids() for _ in range(1000)]

    for ids in runs:
        assert re.fullmatch(r"calibrated-agent-evaluation-[0-9a-f]{32}", ids.session_id)
        assert len(ids.session_id) == 60
        assert RunIds.model_validate(ids.model_dump()) == ids
        assert re.fullmatch(SESSION_ID_PATTERN, ids.session_id)
        assert re.fullmatch(W3C_TRACE_ID_PATTERN, ids.w3c_trace_id)
        assert re.fullmatch(PARENT_SPAN_ID_PATTERN, ids.parent_span_id)
    assert len({ids.session_id for ids in runs}) == 1000
    assert len({ids.w3c_trace_id for ids in runs}) == 1000


def test_bundle_baggage_round_trips_through_the_agentcore_sdk_parser() -> None:
    """The bundle keys and their percent-encoding are what the AgentCore SDK reads, through the
    private ``_extract_baggage`` and ``_parse_config_bundle_baggage`` of
    ``bedrock_agentcore.config_bundle.baggage``: the SDK keeps its baggage keys as literals there.
    Written against bedrock-agentcore 1.24.2; a failure names the installed one."""
    pytest.importorskip("bedrock_agentcore")
    from bedrock_agentcore.config_bundle.baggage import (
        _extract_baggage,  # pyright: ignore[reportPrivateUsage]
        _parse_config_bundle_baggage,  # pyright: ignore[reportPrivateUsage]
    )

    sdk = importlib.metadata.version("bedrock-agentcore")
    assert ":" in BUNDLE_ARN
    assert "/" in BUNDLE_ARN
    ids = new_run_ids()
    target = _target(config_bundle_arn=BUNDLE_ARN, config_bundle_version=BUNDLE_VERSION)

    parsed = _extract_baggage([("baggage", baggage(target, ids))])
    bundle = _parse_config_bundle_baggage(parsed)

    assert bundle is not None, f"bedrock-agentcore {sdk} read no bundle from the baggage"
    assert bundle.bundle_arn == BUNDLE_ARN, f"bedrock-agentcore {sdk}"
    assert bundle.bundle_version == BUNDLE_VERSION, f"bedrock-agentcore {sdk}"
    assert parsed["session.id"] == [ids.session_id], f"bedrock-agentcore {sdk}"


def test_traceparent_is_a_w3c_version_00_header() -> None:
    ids = new_run_ids()

    match = _TRACEPARENT.fullmatch(traceparent(ids))

    assert match is not None
    assert match.groups() == (ids.w3c_trace_id, ids.parent_span_id)


@pytest.mark.parametrize("with_bundle", [True, False], ids=["with-bundle", "without-bundle"])
def test_baggage_carries_only_the_session_and_bundle_keys(with_bundle: bool) -> None:
    ids = new_run_ids()
    bundle = {"config_bundle_arn": BUNDLE_ARN, "config_bundle_version": BUNDLE_VERSION}
    target = _target(**bundle) if with_bundle else _target()

    entries = _baggage_entries(baggage(target, ids))

    expected = {"session.id": ids.session_id}
    if with_bundle:
        expected["aws.agentcore.configbundle_arn"] = BUNDLE_ARN
        expected["aws.agentcore.configbundle_version"] = BUNDLE_VERSION
    assert entries == expected


def test_an_all_zero_trace_or_parent_id_is_redrawn(monkeypatch: pytest.MonkeyPatch) -> None:
    real_token_hex = secrets.token_hex
    zeroed: set[int | None] = set()

    def zeros_once_per_size(nbytes: int | None = None) -> str:
        if nbytes not in zeroed:
            zeroed.add(nbytes)
            return "00" * (nbytes or 32)
        return real_token_hex(nbytes)

    monkeypatch.setattr(secrets, "token_hex", zeros_once_per_size)

    ids = new_run_ids()

    assert zeroed, "the IDs were not drawn from secrets.token_hex, so no zero draw was tried"
    assert re.fullmatch(r"[0-9a-f]{32}", ids.w3c_trace_id)
    assert re.fullmatch(r"[0-9a-f]{16}", ids.parent_span_id)
    assert ids.w3c_trace_id.strip("0")
    assert ids.parent_span_id.strip("0")
    assert _TRACEPARENT.fullmatch(traceparent(ids))


def test_one_runs_headers_are_identical_each_time_they_are_built() -> None:
    ids = new_run_ids()
    target = _target(config_bundle_arn=BUNDLE_ARN, config_bundle_version=BUNDLE_VERSION)

    assert traceparent(ids) == traceparent(ids)
    assert baggage(target, ids) == baggage(target, ids)
    assert traceparent(new_run_ids()) != traceparent(ids)
