# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation label``: launch or point-to the review surface.

The Label stage is inherently interactive (a human reads + labels traces via the FastHTML hotkey
surface), so it cannot run headless in a pipeline. Default behavior **prints the exact
launch command + URL** bound to the current backend; ``--serve`` actually launches the server
(needs the ``[ui]`` extra). Reuses the shipped ``build_app``; the ``[ui]`` extra absent
surfaces as a typed ``CliError`` with the install hint, never a raw ``ImportError``.

The surface is served on this machine only: a ``--host`` other machines can reach is refused, and
labeling from another machine goes through an SSH port-forward.
"""

from __future__ import annotations

import argparse
import shlex
import socket
import sys

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli.config import probe_extra
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5001
# The placeholder the printed launch command carries; shell-safe, unlike `<your-id>`.
RATER_PLACEHOLDER = "RATER_ID"


def launch_command(args: argparse.Namespace) -> list[str]:
    """The ``label --serve`` argv that serves the surface these ``args`` describe.

    Global options go before the subcommand (the form ``calibrated-agent-evaluation --help``
    documents); ``--root`` is carried when one was chosen so the server opens the same store. The
    rater is the placeholder.
    """
    from calibrated_agent_evaluation.cli.backends import explicit_local_root

    argv = ["calibrated-agent-evaluation", "--backend", getattr(args, "backend", "local")]
    root = explicit_local_root(args)
    if root is not None:
        argv += ["--root", str(root)]
    argv += ["label", "--serve", "--rater", RATER_PLACEHOLDER]
    host = getattr(args, "host", DEFAULT_HOST)
    port = getattr(args, "port", DEFAULT_PORT)
    if host != DEFAULT_HOST:
        argv += ["--host", host]
    if port != DEFAULT_PORT:
        argv += ["--port", str(port)]
    if getattr(args, "codebook_file", None):
        argv += ["--codebook", str(args.codebook_file)]
    if getattr(args, "codebook_version", None) is not None:
        argv += ["--codebook-version", args.codebook_version]
    return argv


def label_handler(args: argparse.Namespace) -> StageOutcome:
    """Launch (``--serve``) or print the launch command + URL for the review surface.

    ``--agreement`` instead measures inter-rater agreement (``cli.stages.agreement``).
    """
    if getattr(args, "agreement", False):
        from calibrated_agent_evaluation.cli.stages.agreement import agreement_handler

        return agreement_handler(args)
    from calibrated_agent_evaluation.review.addresses import is_loopback, surface_url

    backend_flag = getattr(args, "backend", "local")
    host = getattr(args, "host", DEFAULT_HOST)
    port = getattr(args, "port", DEFAULT_PORT)
    if not is_loopback(host):
        raise CliError(_remote_refusal(host, port))
    url = surface_url(host, port)

    if getattr(args, "pairwise", False) and not getattr(args, "serve", False):
        raise UsageError(
            "--pairwise is a mode of --serve: run `calibrated-agent-evaluation label --serve "
            "--pairwise ...`"
        )
    if not getattr(args, "serve", False):
        # Default: print the exact command + URL a human runs to open the surface. No server is
        # started (headless-safe); this is the observable output the acceptance check cites.
        cmd = shlex.join(launch_command(args))
        return StageOutcome(
            stage="label",
            exit_code=EXIT_OK,
            summary=(
                f"review surface for the {backend_flag!r} backend: run `{cmd}` with "
                f"{RATER_PLACEHOLDER} replaced by your rater id, then open {url} (needs the UI "
                f"extra: {install_hint('ui')})"
            ),
        )

    # --serve: actually launch. Needs the [ui] extra (FastHTML + uvicorn).
    # Both halves of the [ui] extra: the app framework and the server that runs it. Probing only
    # fasthtml let a missing uvicorn surface as a raw ModuleNotFoundError at `import uvicorn`.
    for module in ("fasthtml", "uvicorn"):
        probe = probe_extra(module)
        if not probe.available:
            raise CliError(f"the review surface needs the UI extra; {probe.install_hint}")

    # Every label is stamped with its rater so double-coded traces stay separable for agreement.
    # Asked for, never guessed: an OS-login default merges two raters who share a machine.
    rater_id = (getattr(args, "rater_id", None) or "").strip()
    if not rater_id:
        raise UsageError(
            "`calibrated-agent-evaluation label --serve` needs --rater <your-id>: every label you "
            "write carries it"
        )

    if getattr(args, "pairwise", False):
        from calibrated_agent_evaluation.cli.stages.label_pairwise import serve_pairwise

        return serve_pairwise(args, host=host, port=port, rater_id=rater_id)

    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.scope import parse_cells
    from calibrated_agent_evaluation.cli.stages.codebook import resolve_session_codebook
    from calibrated_agent_evaluation.review import build_app
    from calibrated_agent_evaluation.review.guards import allowed_hosts_for
    from calibrated_agent_evaluation.review.session import WorkScope

    scope = WorkScope(
        frame=getattr(args, "frame", None),
        cell=parse_cells(getattr(args, "cell", None)),
        trace_id=getattr(args, "trace_id", None),
        limit=getattr(args, "limit", None),
        relabel=bool(getattr(args, "relabel", False)),
    )
    # Per-label timing is persisted as it happens, so a crashed session keeps it.
    session_log = getattr(args, "session_log", None) or str(
        local_root(args) / "review-sessions.jsonl"
    )
    codebook = resolve_session_codebook(args, local_root(args))
    from calibrated_agent_evaluation.cli.proposals import RecordsProposalStore

    proposals = RecordsProposalStore(local_root(args))
    with open_backend(args) as backend:
        app = build_app(
            backend,
            rater_id=rater_id,
            scope=scope,
            session_log=session_log,
            allowed_hosts=allowed_hosts_for(host),
            codebook_version=codebook.version,
            failure_modes=codebook.modes,
            proposals=proposals,
        )
        pending = len(app.state.review_session.pending)
        skipped = len(app.state.already_labeled)
        resumed = (
            f"; skipped {skipped} already labeled (--relabel includes them)" if skipped else ""
        )
        # Printed before serving: the resume count is the labeler's only cue that earlier work
        # was kept.
        print(
            f"{pending} trace(s) to label for rater {rater_id!r}{resumed}; timings go to "
            f"{session_log}",
            file=sys.stderr,
        )
        print(codebook.note, file=sys.stderr)
        if codebook.modes and codebook.version is not None:
            waiting = proposals.pending(codebook.version)
            if waiting:
                print(
                    f"{len(waiting)} proposed failure mode(s) wait for the codebook owner and are "
                    f"in the picker: {', '.join(p.name for p in waiting)}",
                    file=sys.stderr,
                )
        from calibrated_agent_evaluation.backends import TraceScan
        from calibrated_agent_evaluation.cli.scope import skip_lines

        for line in skip_lines(
            TraceScan(traces=[], skipped=tuple(app.state.not_calibrated_agent_evaluation))
        ):
            print(line.strip(), file=sys.stderr)
        _require_bindable(host, port)

        import uvicorn

        # Blocks until Ctrl-C. Reaching here means the surface is being served at `url`.
        try:
            uvicorn.run(app, host=host, port=port)
        except OSError as exc:
            raise CliError(_bind_message(host, port, exc)) from exc
    return StageOutcome(stage="label", exit_code=EXIT_OK, summary=f"review surface served at {url}")


def _remote_refusal(host: str, port: int) -> str:
    return (
        f"refusing to serve the review surface on {host!r}: that address is reachable from other "
        "machines, and the surface shows trace content and writes gold labels. To label from "
        f"another machine, keep the default host and forward the port over SSH "
        f"(`ssh -L {port}:127.0.0.1:{port} <this-machine>`, then open "
        f"http://127.0.0.1:{port})"
    )


def _require_bindable(host: str, port: int) -> None:
    """Refuse a host/port the server cannot bind, before uvicorn logs and exits on it.

    uvicorn reports a bind failure by logging and calling ``sys.exit(1)``, which reaches the user
    as a log line and a bare exit code rather than the CLI's typed diagnostic.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        family, socktype, proto, _canon, addr = infos[0]
        with socket.socket(family, socktype, proto) as probe:
            # The same option uvicorn sets, so a port in TIME_WAIT is not refused here and then
            # bound fine by the server; a port with a live listener still fails.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(addr)
    except OSError as exc:
        raise CliError(_bind_message(host, port, exc)) from exc


def _bind_message(host: str, port: int, exc: OSError) -> str:
    return (
        f"cannot serve the review surface on {host}:{port} ({exc.strerror or exc}); pick another "
        "with --port (or --host)"
    )
