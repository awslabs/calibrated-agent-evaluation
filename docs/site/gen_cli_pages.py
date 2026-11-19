# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the `calibrated-agent-evaluation` CLI reference from the real argparse parser at build
time.

Run by the mkdocs `gen-files` plugin. Imports
`calibrated_agent_evaluation.cli.parser.build_parser()`, the same parser the shipped CLI uses, and
writes one page per subcommand under `cli/` plus an index that lists them with their one-line help.
A `cli/SUMMARY.md` feeds literate-nav, so a new subcommand appears in the navigation on the next
build with no hand-editing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mkdocs_gen_files

# Resolve the repo `src/` so `import calibrated_agent_evaluation` works during the docs build (no
# editable install).
_SRC = Path(__file__).parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from calibrated_agent_evaluation.cli.parser import (
    build_parser,  # import after the sys.path insert above (E402)
)


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:  # argparse exposes no public accessor for its actions
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _one_line_help(action: argparse._SubParsersAction) -> dict[str, str]:
    """Each subcommand's `help=` string, which argparse keeps on the pseudo-actions only."""
    return {c.dest: (c.help or "").replace("|", "\\|") for c in action._choices_actions}


_HELP_WIDTH = "100"


def _help_block(parser: argparse.ArgumentParser) -> str:
    # argparse wraps to the terminal width, so pin it: `just docs` in a wide terminal and a piped
    # build must render the same page.
    saved = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = _HELP_WIDTH
    try:
        text = parser.format_help().rstrip()
    finally:
        if saved is None:
            del os.environ["COLUMNS"]
        else:
            os.environ["COLUMNS"] = saved
    return f"```text\n{text}\n```\n"


parser = build_parser()
action = _subparsers_action(parser)
commands = dict(action.choices) if action is not None else {}
summaries = _one_line_help(action) if action is not None else {}

index = [
    "# CLI reference\n",
    "The `calibrated-agent-evaluation` command runs the whole evaluation pipeline. These pages "
    "are generated from the "
    "argparse parser when the docs are built, so they always match the installed CLI.\n",
    "| Command | What it does |",
    "| ------- | ------------ |",
]
# Declaration order: onboarding, ingest, then the pipeline in workflow order, then dump/load.
index.extend(
    f"| [`calibrated-agent-evaluation {name}`]({name}.md) | {summaries.get(name, '')} |"
    for name in commands
)
index += [
    "",
    "## `calibrated-agent-evaluation`\n",
    "Global options apply to every subcommand.\n",
    _help_block(parser),
]

with mkdocs_gen_files.open("cli/index.md", "w") as fd:
    fd.write("\n".join(index))

for name, sub in commands.items():
    with mkdocs_gen_files.open(f"cli/{name}.md", "w") as fd:
        summary = summaries.get(name, "")
        fd.write(f"# `calibrated-agent-evaluation {name}`\n\n{summary}\n\n{_help_block(sub)}")

with mkdocs_gen_files.open("cli/SUMMARY.md", "w") as fd:
    fd.write("* [Overview](index.md)\n")
    fd.writelines(f"* [calibrated-agent-evaluation {name}]({name}.md)\n" for name in commands)
