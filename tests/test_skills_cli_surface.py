# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every `calibrated-agent-evaluation` command a skill tells the user to run exists in the parser.

The skills hand off to the `calibrated-agent-evaluation` CLI, so a skill that names a subcommand, a
flag or a flag value the parser does not have sends an agent to a usage error, or to a stage that
does something else. This test reads the commands out of the code spans and fenced blocks of every
``skills/**/SKILL.md``, ``skills/README.md``, the plugin manifests under ``.claude-plugin/`` and
the README's skills section, and checks each one against
:func:`calibrated_agent_evaluation.cli.parser.build_parser`:

  * the subcommand after ``calibrated-agent-evaluation`` (and its global options) is registered;
  * every ``--flag`` is one that subcommand, or the top-level parser, accepts;
  * a value given to a flag with fixed choices is one of them (``a|b`` names two);
  * a command written with any flag carries every option the subcommand requires, and every option
    whose help says it is "required with --X" when ``--X`` is given (read from the parser, so the
    rule cannot drift from it);
  * a bare ``--flag`` span belongs to a subcommand named in the same paragraph.

It also checks that every ``calibrated_agent_evaluation.<module>`` attribute a skill names imports,
so a skill cannot point at a library entry point that was renamed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli.parser import build_parser

_REPO = Path(__file__).resolve().parent.parent
_SEPARATORS = {";", "&&", "||", "|", "then", "and", "or"}


@dataclass(frozen=True)
class Mention:
    """One code span or fenced-block line, with the paragraph it came from."""

    source: str
    text: str
    paragraph: int


def _skill_sources() -> list[Path]:
    paths = sorted((_REPO / "skills").glob("**/SKILL.md"))
    paths.append(_REPO / "skills" / "README.md")
    paths.extend(sorted((_REPO / ".claude-plugin").glob("*.json")))
    return paths


def _readme_skills_section() -> str:
    text = (_REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Skills plugin\n(.*?)(?=^## )", text, re.DOTALL | re.MULTILINE)
    assert m, "README.md has no '## Skills plugin' section"
    return m.group(1)


def _json_strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _json_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _json_strings(value)


def _mentions(source: str, text: str) -> list[Mention]:
    """Code spans and fenced-block lines, each tagged with its paragraph (a blank line or a new
    list item starts one)."""
    found: list[Mention] = []
    paragraph = 0
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            paragraph += 1
            continue
        if in_fence:
            if stripped and not stripped.startswith("#"):
                found.append(Mention(source, stripped, paragraph))
            continue
        if not stripped or re.match(r"^(?:[-*|]|\d+\.)\s", stripped):
            paragraph += 1
        found.extend(Mention(source, span, paragraph) for span in re.findall(r"`([^`]+)`", line))
    return found


def _all_mentions() -> list[Mention]:
    mentions: list[Mention] = []
    for path in _skill_sources():
        rel = str(path.relative_to(_REPO))
        if path.suffix == ".json":
            for i, s in enumerate(_json_strings(json.loads(path.read_text(encoding="utf-8")))):
                mentions.extend(
                    Mention(rel, m.text, i * 1000 + m.paragraph) for m in _mentions(rel, s)
                )
        else:
            mentions.extend(_mentions(rel, path.read_text(encoding="utf-8")))
    mentions.extend(_mentions("README.md#skills-plugin", _readme_skills_section()))
    return mentions


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("calibrated-agent-evaluation parser has no subcommands")


def _options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {opt: a for a in parser._actions for opt in a.option_strings}


def _tokens(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def check_command(tokens: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """Problems with the ``calibrated-agent-evaluation ...`` invocation in ``tokens`` (which start
    with ``calibrated-agent-evaluation``)."""
    subs = _subparsers(parser)
    top = _options(parser)
    problems: list[str] = []
    i = 1
    # Global options before the subcommand.
    while i < len(tokens) and tokens[i].startswith("-"):
        flag = tokens[i].split("=", 1)[0]
        action = top.get(flag)
        if action is None:
            return [f"unknown global option {flag!r}"]
        i += 1 if "=" in tokens[i] or action.nargs == 0 else 2
    if i >= len(tokens) or tokens[i] in _SEPARATORS:
        return (
            problems  # bare `calibrated-agent-evaluation` or `calibrated-agent-evaluation --help`
        )
    name = tokens[i]
    if name not in subs:
        return [f"unknown subcommand {name!r}"]
    options = {**top, **_options(subs[name])}
    given: set[str] = set()
    i += 1
    while i < len(tokens) and tokens[i] not in _SEPARATORS:
        tok = tokens[i]
        i += 1
        if not tok.startswith("--"):
            continue
        flag, eq, inline = tok.partition("=")
        action = options.get(flag)
        if action is None:
            problems.append(f"`calibrated-agent-evaluation {name}` has no option {flag!r}")
            continue
        given.update(action.option_strings)
        if action.nargs == 0:
            continue
        value = inline if eq else (tokens[i] if i < len(tokens) else None)
        if not eq:
            i += 1
        if action.choices is not None and value is not None and not value.startswith("<"):
            bad = [v for v in value.split("|") if v not in action.choices]
            if bad:
                problems.append(
                    f"`calibrated-agent-evaluation {name} {flag}` has no choice "
                    f"{', '.join(map(repr, bad))} "
                    f"(choices: {', '.join(action.choices)})"
                )
    if given:
        problems.extend(_missing_required(name, subs[name], given))
    return problems


def _missing_required(name: str, sub: argparse.ArgumentParser, given: set[str]) -> list[str]:
    missing: list[str] = []
    for action in sub._actions:
        if not action.option_strings or given & set(action.option_strings):
            continue
        flag = action.option_strings[0]
        if action.required:
            missing.append(f"`calibrated-agent-evaluation {name}` needs {flag}")
            continue
        trigger = re.search(r"required with (--[\w-]+)", action.help or "")
        if trigger and trigger.group(1) in given:
            missing.append(f"`calibrated-agent-evaluation {name} {trigger.group(1)}` needs {flag}")
    return missing


def _invocations(tokens: list[str]) -> Iterator[list[str]]:
    for i, tok in enumerate(tokens):
        if tok == "calibrated-agent-evaluation":
            yield tokens[i:]


def check_mentions(mentions: list[Mention], parser: argparse.ArgumentParser) -> list[str]:
    """Every problem across ``mentions``, each prefixed with its source and the span."""
    subs = _subparsers(parser)
    top = _options(parser)
    named: dict[tuple[str, int], set[str]] = {}
    problems: list[str] = []
    for m in mentions:
        for inv in _invocations(_tokens(m.text)):
            problems.extend(f"{m.source}: `{m.text}`: {p}" for p in check_command(inv, parser))
            sub = next((t for t in inv[1:] if not t.startswith("-") and t in subs), None)
            if sub:
                named.setdefault((m.source, m.paragraph), set()).add(sub)
    for m in mentions:
        tokens = _tokens(m.text)
        if not tokens or not tokens[0].startswith("--") or "calibrated-agent-evaluation" in tokens:
            continue
        flag = tokens[0].split("=", 1)[0]
        here = named.get((m.source, m.paragraph), set())
        owners = here or set(subs)
        if flag in top or any(flag in _options(subs[s]) for s in owners):
            continue
        where = (
            f"`calibrated-agent-evaluation {'`/`calibrated-agent-evaluation '.join(sorted(here))}`"
            if here
            else "any `calibrated-agent-evaluation` subcommand"
        )
        problems.append(f"{m.source}: `{m.text}`: {where} has no option {flag!r}")
    return problems


def check_library_names(mentions: list[Mention]) -> list[str]:
    """Every ``from calibrated_agent_evaluation.x import y`` and ``calibrated_agent_evaluation.x.y``
    a skill names must import."""
    problems: list[str] = []
    for m in mentions:
        for module, names in re.findall(
            r"from (calibrated_agent_evaluation[\w.]*) import ([\w, ]+)", m.text
        ):
            for attr in (n.strip() for n in names.split(",")):
                if not hasattr(importlib.import_module(module), attr):
                    problems.append(f"{m.source}: `{m.text}`: {module} has no {attr!r}")
        for dotted in re.findall(r"\b(calibrated_agent_evaluation(?:\.\w+)+)", m.text):
            if f"from {dotted} import" in m.text:
                continue
            parts = dotted.split(".")
            for cut in range(len(parts), 0, -1):
                try:
                    obj: object = importlib.import_module(".".join(parts[:cut]))
                except ImportError:
                    continue
                for attr in parts[cut:]:
                    if not hasattr(obj, attr):
                        problems.append(f"{m.source}: `{m.text}`: {dotted} does not resolve")
                        break
                    obj = getattr(obj, attr)
                break
    return problems


def test_every_calibrated_agent_evaluation_command_a_skill_names_exists_in_the_parser():
    mentions = _all_mentions()
    assert any("calibrated-agent-evaluation" in _tokens(m.text) for m in mentions), (
        "no calibrated-agent-evaluation command found to check"
    )
    problems = check_mentions(mentions, build_parser())
    assert not problems, (
        "skills name calibrated-agent-evaluation commands the CLI lacks:\n" + "\n".join(problems)
    )


def test_every_calibrated_agent_evaluation_name_a_skill_names_imports():
    problems = check_library_names(_all_mentions())
    assert not problems, "skills name library entry points that do not exist:\n" + "\n".join(
        problems
    )


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        ("calibrated-agent-evaluation calibrate --judge J --rate-safe", []),
        ("calibrated-agent-evaluation --backend local label --serve --rater R", []),
        ("calibrated-agent-evaluation --root R run --ci --verdicts human", []),
        ("calibrated-agent-evaluation judge", []),
        ("calibrated-agent-evaluation judge --check judge.json", []),
        (
            "calibrated-agent-evaluation calibrate --rate-safe",
            ["`calibrated-agent-evaluation calibrate` needs --judge"],
        ),
        (
            "calibrated-agent-evaluation label --serve",
            ["`calibrated-agent-evaluation label --serve` needs --rater"],
        ),
        (
            "calibrated-agent-evaluation generate --run --executor strands",
            ["`calibrated-agent-evaluation generate --run` needs --agent"],
        ),
        ("calibrated-agent-evaluation lint J", ["unknown subcommand 'lint'"]),
        (
            "calibrated-agent-evaluation judge --definition J",
            ["`calibrated-agent-evaluation judge` has no option '--definition'"],
        ),
        (
            "calibrated-agent-evaluation label --serve --rater R --frame x",
            ["`calibrated-agent-evaluation label --frame` has no choice 'x'"],
        ),
        (
            "calibrated-agent-evaluation generate --run --agent m:f --executor inspect|bogus",
            ["has no choice 'bogus'"],
        ),
        ("calibrated-agent-evaluation --verbose run", ["unknown global option '--verbose'"]),
    ],
)
def test_the_checker_catches_what_the_parser_lacks(span, expected):
    problems = check_command(_tokens(span), build_parser())
    assert len(problems) == len(expected)
    for got, want in zip(problems, expected, strict=True):
        assert want in got


def test_a_bare_flag_is_checked_against_the_subcommand_in_its_paragraph():
    text = (
        "Run `calibrated-agent-evaluation label` and pass `--serve`.\n\nThen `--rate-safe`.\n\n"
        "And `calibrated-agent-evaluation run`, `--serve`."
    )
    problems = check_mentions(_mentions("x.md", text), build_parser())
    assert problems == [
        "x.md: `--serve`: `calibrated-agent-evaluation run` has no option '--serve'"
    ]


def test_a_library_name_that_does_not_import_is_caught():
    mentions = _mentions(
        "x.md",
        "`from calibrated_agent_evaluation.calibrate import plan_calibration_size, "
        "no_such_planner` and "
        "`calibrated_agent_evaluation.stats.rg_interval` and "
        "`calibrated_agent_evaluation.stats.wilson_bogus`",
    )
    problems = check_library_names(mentions)
    assert len(problems) == 2
    assert "no_such_planner" in problems[0]
    assert "calibrated_agent_evaluation.stats.wilson_bogus" in problems[1]


def test_the_annotation_skill_serves_the_surface_the_cli_prints(tmp_path, capsys):
    """The skill's launch command and install line are the ones `calibrated-agent-evaluation label`
    prints, for an installed package (no `uv run`, global options before the subcommand)."""
    from calibrated_agent_evaluation.cli import main

    assert main(["label"]) == 0
    printed = capsys.readouterr().out
    assert main(["--root", str(tmp_path), "label"]) == 0
    assert f"--root {tmp_path} label --serve" in capsys.readouterr().out
    from calibrated_agent_evaluation._install import install_hint

    cli_cmd = re.search(r"run `([^`]+)`", printed)
    assert cli_cmd
    assert f"needs the UI extra: {install_hint('ui')}" in printed
    text = (_REPO / "skills" / "build-annotation-ui" / "SKILL.md").read_text(encoding="utf-8")
    assert "the install command `calibrated-agent-evaluation doctor` prints" in text
    assert "uv run" not in text

    parser = build_parser()
    skill_cmds = [
        m.text
        for m in _mentions("build-annotation-ui", text)
        if m.text.startswith("calibrated-agent-evaluation ") and "--serve" in m.text
    ]
    assert skill_cmds, "the skill names no `calibrated-agent-evaluation ... label --serve` command"
    want = parser.parse_args(_tokens(cli_cmd.group(1))[1:])
    for cmd in skill_cmds:
        got = parser.parse_args(_tokens(cmd)[1:])
        assert (got.command, got.serve, got.rater_id) == (want.command, want.serve, want.rater_id)
