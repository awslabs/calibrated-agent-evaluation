# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached boto3 clients for one AWS profile and region, with call handlers attached."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Self

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    ProfileNotFound,
    SSOError,
    TokenRetrievalError,
)

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AwsCredentialsMissing,
    WrongAwsIdentity,
)
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget

__all__ = [
    "DEFAULT_CLIENT_CONFIG",
    "AwsClients",
    "BeforeCallHandler",
    "caller_is_role",
    "require_role",
]

# Explicit, so a dead endpoint fails in seconds and a botocore default change can't lengthen it.
DEFAULT_CLIENT_CONFIG = Config(
    connect_timeout=10, read_timeout=30, retries={"mode": "standard", "max_attempts": 3}
)

BeforeCallHandler = Callable[..., Any]
"""A botocore ``before-call`` handler; raising from it aborts the call before it is sent."""


class AwsClients:
    """Builds and caches one client per AWS service from an injected boto3 session.

    Every handler in ``handlers`` runs before each API call on every client this builds, ahead
    of any other ``before-call`` handler, including botocore's ``Stubber``. Use it as a context
    manager so the clients' connection pools are closed. Not safe to share across threads; the
    clients it hands out are.
    """

    def __init__(
        self,
        session: boto3.Session,
        *,
        handlers: Sequence[BeforeCallHandler] = (),
        config: Config = DEFAULT_CLIENT_CONFIG,
    ) -> None:
        self._session = session
        self._handlers = tuple(handlers)
        self._config = config
        self._clients: dict[tuple[str, str | None], BaseClient] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close every client built so far; later ``client()`` calls build fresh ones."""
        clients, self._clients = self._clients, {}
        for built in clients.values():
            built.close()

    @classmethod
    def from_profile(
        cls,
        *,
        region: str | None = None,
        profile: str | None = None,
        handlers: Sequence[BeforeCallHandler] = (),
        config: Config = DEFAULT_CLIENT_CONFIG,
    ) -> AwsClients:
        """Clients for a named profile (or the default credential chain) in ``region``.

        Without ``region``, the profile's configured region is used.
        """
        try:
            session = boto3.Session(profile_name=profile, region_name=region)
        except ProfileNotFound as exc:
            raise AwsCredentialsMissing(
                f"AWS profile {profile!r} doesn't exist; set AWS_PROFILE or pass --profile"
            ) from exc
        except BotoCoreError as exc:
            raise AgentCoreError(f"could not open AWS profile {profile!r}: {exc}") from exc
        return cls(session, handlers=handlers, config=config)

    def with_config(self, config: Config) -> Self:
        """A second set of clients with ``config`` merged over this one's settings.

        The new set shares this one's class, boto3 session and handlers, so every guard installed
        here runs on its calls too, and a billed-call budget counts both sets' calls together.
        Only the settings ``config`` names change; the rest (timeouts it leaves out, proxies, the
        user agent) carry over. This set's clients are left as they are. The new set builds and
        caches its own clients, and closing one set leaves the other's open, so close each.

        Args:
            config: The settings to change, such as a single-attempt retry policy.

        Returns:
            The new set of clients, with no client built yet.
        """
        # A shallow copy, so a subclass keeps its own state too; only the settings and the client
        # cache differ from this set.
        view = copy.copy(self)
        view._config = self._config.merge(config)
        view._clients = {}
        return view

    @property
    def region(self) -> str | None:
        """The region every client is built for."""
        return self._session.region_name

    @property
    def boto_session(self) -> boto3.Session:
        """The underlying boto3 session, for an adapter that must build its own vendor client.

        Named ``boto_session`` (not ``session``) to stay clear of the unrelated review ``session``
        and of subclasses that keep their own ``session`` attribute. The Code Interpreter SDK
        (``code_session``) constructs its own ``bedrock-agentcore`` client from a session rather
        than taking one of ours, so it needs the session itself; identity and billed-call limits are
        enforced by the adapter around it (``require_role`` + the budget).
        """
        return self._session

    @property
    def billed_call_budget(self) -> BilledCallBudget | None:
        """The first billed-call budget among the handlers, or ``None`` (no limit).

        Reading it charges nothing, so a caller can check ``remaining`` before a run.
        """
        return next((h for h in self._handlers if isinstance(h, BilledCallBudget)), None)

    def client(self, service_name: str, *, region: str | None = None) -> BaseClient:
        """The cached client for ``service_name``, built on first use.

        ``region`` builds the client in another region (for an API that answers only from a few
        endpoint regions, such as the Price List API), with the same handlers and settings; it is
        cached per service and region. Without it, the client is built in this set's region.
        """
        key = (service_name, region)
        cached = self._clients.get(key)
        if cached is not None:
            return cached
        try:
            built = self._session.client(  # pyright: ignore[reportUnknownMemberType]
                service_name, region_name=region, config=self._config
            )
        except BotoCoreError as exc:
            raise AgentCoreError(
                f"could not build an AWS client for {service_name!r} in region "
                f"{region or self._session.region_name!r}: {exc}"
            ) from exc
        for handler in self._handlers:
            built.meta.events.register_first("before-call.*.*", handler)
        self._clients[key] = built
        return built

    def caller_identity(self) -> str:
        """The caller's ARN, which also proves credentials resolve before any other call."""
        try:
            identity = self.client("sts").get_caller_identity()  # pyright: ignore[reportAttributeAccessIssue]
        except NoCredentialsError as exc:
            raise AwsCredentialsMissing(
                "no AWS credentials found; set AWS_PROFILE or pass --profile"
            ) from exc
        except (SSOError, TokenRetrievalError) as exc:
            raise AwsCredentialsMissing(_REFRESH.format(exc=exc)) from exc
        except BotoCoreError as exc:
            raise AgentCoreError(f"could not confirm the AWS identity: {exc}") from exc
        except ClientError as exc:
            # Not a BotoCoreError: an expired or revoked token comes back from STS as a service
            # error, and used to escape untyped from
            # `calibrated-agent-evaluation doctor --agentcore`.
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in _EXPIRED_OR_INVALID:
                raise AwsCredentialsMissing(_REFRESH.format(exc=exc)) from exc
            raise AgentCoreError(f"could not confirm the AWS identity: {exc}") from exc
        return str(identity["Arn"])


_EXPIRED_OR_INVALID = frozenset(
    {
        "ExpiredToken",
        "ExpiredTokenException",
        "InvalidClientTokenId",
        "UnrecognizedClientException",
    }
)
_REFRESH = (
    "the AWS credentials are expired or invalid ({exc}); refresh them (for an SSO profile, "
    "`aws sso login --profile <name>`) and run again"
)


def caller_is_role(caller_arn: str, role_arn: str) -> bool:
    """Whether ``caller_arn`` is a session of the IAM role ``role_arn`` (same account and name).

    STS reports a role session as ``arn:aws:sts::<account>:assumed-role/<name>/<session>``; the
    role itself is ``arn:aws:iam::<account>:role/<path/><name>``.
    """
    caller = caller_arn.split(":", 5)
    role = role_arn.split(":", 5)
    if len(caller) != 6 or len(role) != 6:
        return False
    if caller[2] != "sts" or role[2] != "iam" or caller[4] != role[4] or caller[1] != role[1]:
        return False
    kind, _, rest = caller[5].partition("/")
    role_kind, _, role_path = role[5].partition("/")
    if kind != "assumed-role" or role_kind != "role":
        return False
    return rest.partition("/")[0] == role_path.rpartition("/")[2]


def require_role(clients: AwsClients, role_arn: str) -> str:
    """The caller's ARN, after confirming it is a session of ``role_arn``.

    Only ``sts:GetCallerIdentity`` is called, so nothing else reaches the account under the wrong
    identity.
    """
    caller = clients.caller_identity()
    if not caller_is_role(caller, role_arn):
        raise WrongAwsIdentity(
            f"running as {caller}, but these calls must run as the role {role_arn}; "
            "switch to a profile that assumes it"
        )
    return caller
