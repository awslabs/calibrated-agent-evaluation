# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Skills-plugin tests.

Scans the repo-root ``skills/`` plugin: all five canon skills present + discoverable (valid
frontmatter), each drives ``calibrated-agent-evaluation``/``calibrated-agent-evaluation`` with no
re-implemented statistics or second language, and each recommends the enforced command rather than
bypassing a gate.
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex
import tomllib

import pytest

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SKILLS_DIR = _REPO / "skills"
_PLUGIN = _REPO / ".claude-plugin" / "plugin.json"
_MARKETPLACE = _REPO / ".claude-plugin" / "marketplace.json"
# The manifest fields this plugin may use: those of the plugins Claude Code loads on the
# maintainer's machine (frontend-design, mattpocock-skills), in which `claude plugin validate
# --strict` reports no unknown field. A field outside these is a guess at the schema.
_PLUGIN_FIELDS = {
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
}
_MARKETPLACE_FIELDS = {"name", "owner", "description", "plugins"}
_MARKETPLACE_ENTRY_FIELDS = {"name", "source", "description", "category", "keywords"}
_CANON = [
    "audit-eval-pipeline",
    "run-error-analysis",
    "build-binary-judge",
    "validate-judge",
    "build-annotation-ui",
]


def _skill_md(name: str) -> pathlib.Path:
    return _SKILLS_DIR / name / "SKILL.md"


def _frontmatter(text: str) -> dict[str, str]:
    """Parse the leading ``--- ... ---`` YAML-ish frontmatter into a dict (name/description)."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return {}
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
    return fields


def test_all_five_canon_skills_present_and_discoverable():
    """The five canon skills exist as SKILL.md with a valid name+description frontmatter."""
    for name in _CANON:
        path = _skill_md(name)
        assert path.exists(), f"missing skill: {name}"
        fm = _frontmatter(path.read_text(encoding="utf-8"))
        assert fm.get("name") == name, f"{name}: frontmatter name must match dir"
        assert fm.get("description"), f"{name}: needs a non-empty description"


def _pyproject() -> dict:
    return tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def _manifest(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_repo_is_a_plugin_whose_default_skills_root_holds_the_canon():
    """Claude Code loads a plugin's skills from ``<plugin root>/skills/``, and a
    plugin root is a directory with ``.claude-plugin/plugin.json``. Without the manifest the five
    skills sat in no directory an agent runtime loads."""
    manifest = _manifest(_PLUGIN)
    assert set(manifest) <= _PLUGIN_FIELDS, f"unverified field(s): {set(manifest) - _PLUGIN_FIELDS}"
    root = _PLUGIN.parent.parent
    for name in _CANON:
        assert (root / "skills" / name / "SKILL.md").is_file(), name
    project = _pyproject()
    assert manifest["name"] == "calibrated-agent-evaluation"
    assert manifest["version"] == project["version"]
    assert manifest["license"] == project["license"]


def test_the_marketplace_lists_the_plugin_at_the_repo_root():
    doc = _manifest(_MARKETPLACE)
    assert set(doc) <= _MARKETPLACE_FIELDS
    (entry,) = doc["plugins"]
    assert set(entry) <= _MARKETPLACE_ENTRY_FIELDS
    assert entry["name"] == _manifest(_PLUGIN)["name"]
    assert (_REPO / entry["source"] / ".claude-plugin" / "plugin.json").resolve() == _PLUGIN


def _install_docs() -> dict[str, str]:
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^## Skills plugin\n(.*?)(?=^## )", readme, re.DOTALL | re.MULTILINE)
    assert m, "README.md has no '## Skills plugin' section"
    return {
        "README.md#skills-plugin": m.group(1),
        "skills/README.md": (_SKILLS_DIR / "README.md").read_text(encoding="utf-8"),
    }


def test_the_install_docs_name_the_local_routes_and_the_plugin():
    """The docs install the plugin from a checkout, either loaded for one session or added as a
    local marketplace."""
    marketplace = _manifest(_MARKETPLACE)["name"]
    plugin = _manifest(_PLUGIN)["name"]
    for doc, text in _install_docs().items():
        assert "claude --plugin-dir <path-to-checkout>" in text, doc
        assert "/plugin marketplace add <path-to-checkout>" in text, doc
        assert f"/plugin install {plugin}@{marketplace}" in text, doc


_PROJECT_REPO = "github.com/awslabs/calibrated-agent-evaluation"


def test_the_plugin_and_its_docs_point_at_no_repo_but_the_project_s_own():
    """A GitHub URL in a manifest or an install doc names the project's repository, never a guessed
    one."""
    texts = {str(path): path.read_text(encoding="utf-8") for path in (_PLUGIN, _MARKETPLACE)}
    texts |= _install_docs()
    for where, text in texts.items():
        for url in re.findall(r"github\.com/[\w.-]+/[\w.-]+", text):
            assert url.removesuffix(".git") == _PROJECT_REPO, (where, url)


def test_a_missing_or_malformed_skill_is_detectable():
    """Negative: a skill dir without a valid SKILL.md is caught (guards the discovery
    check)."""
    # A non-canon dir with no SKILL.md must not be picked up as a valid skill.
    assert not _skill_md("not-a-real-skill").exists()
    # Malformed frontmatter (no name) yields an empty parse — the positive test would fail on it.
    assert _frontmatter("no frontmatter here") == {}


@pytest.mark.parametrize("name", _CANON)
def test_skill_drives_calibrated_agent_evaluation_and_has_no_stats_or_second_language(name):
    """Each skill invokes the calibrated-agent-evaluation CLI or library and embeds no statistics /
    no non-Python code fence."""
    text = _skill_md(name).read_text(encoding="utf-8")
    assert re.search(r"\bcalibrated-agent-evaluation\b|\bcalibrated_agent_evaluation\b", text), (
        f"{name}: must drive the calibrated-agent-evaluation CLI or library"
    )

    # No re-implemented statistics: the skill must not itself compute a bound/correction/test.
    forbidden_stats = ["import scipy", "import statsmodels", "wilson_lower", "def bootstrap"]
    for token in forbidden_stats:
        assert token not in text, f"{name}: must not re-implement statistics ({token})"

    # No second implementation language: only ```python / ```bash / plain fences allowed.
    fences = re.findall(r"```(\w+)", text)
    allowed = {"python", "bash", "sh", "text", "json", ""}
    bad = [f for f in fences if f.lower() not in allowed]
    assert not bad, f"{name}: non-Python/second-language code fence(s): {bad}"


@pytest.mark.parametrize("name", _CANON)
def test_skill_recommends_and_does_not_bypass_gates(name):
    """Each skill recommends the enforced calibrated-agent-evaluation path and disclaims direct
    gate/label bypass."""
    text = _skill_md(name).read_text(encoding="utf-8").lower()

    # Recommends deferring to the enforced pipeline.
    assert "recommend" in text, f"{name}: must recommend the enforced command"

    # Has an explicit no-bypass disclaimer: it does NOT itself score/gate/write-gold.
    assert "does not" in text or "never" in text, f"{name}: must disclaim direct scoring/gating"


def test_the_audit_skill_quotes_the_committed_coverage_map():
    """The audit skill's coverage figures are the committed study's, read here
    from the file it cites, and it no longer credits `calibrated-agent-evaluation run` with BY or
    Holm."""
    text = _skill_md("audit-eval-pipeline").read_text(encoding="utf-8")
    doc = json.loads(
        (_REPO / "fixtures" / "expected" / "coverage" / "rg_interval_coverage.json").read_text()
    )
    (cell,) = [
        c for c in doc["cells"] if (c["p_hat"], c["prevalence"], c["n_eval"]) == (0.90, 0.06, 100)
    ]
    assert cell["n_class"] == 55
    naive = cell["arms"]["naive_wilson"]["coverage"]
    corrected = cell["arms"]["rg_interval"]["coverage"]
    assert f"covers the true 6% rate in {naive} of {doc['replicates']} replicates" in text
    assert f"the same cell covers {corrected}" in text
    n_under = sum(c["arms"]["naive_wilson"]["verdict"] == "under" for c in doc["cells"])
    assert f"Across that map's {len(doc['cells'])} planner-reachable cells" in text
    assert f"the raw Wilson interval undercovers in {n_under}" in text
    # The CI gate is an intersection-union test with no correction; BY is the library's flag rule.
    assert "BY / Holm" not in text
    assert "`calibrated-agent-evaluation run` corrects its drift flags" not in text
    assert "`calibrated-agent-evaluation run` computes no drift flags" in text


def _fenced_commands(name: str, prefix: str) -> list[str]:
    text = _skill_md(name).read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:bash|sh)\n(.*?)```", text, re.DOTALL)
    return [ln.strip() for b in blocks for ln in b.splitlines() if ln.strip().startswith(prefix)]


def test_build_binary_judge_check_refuses_the_files_calibrated_agent_evaluation_would_refuse(
    tmp_path, monkeypatch, capsys
):
    """The skill checks its judge file with `calibrated-agent-evaluation judge --check`, the
    load and schema check `calibrated-agent-evaluation calibrate` and
    `calibrated-agent-evaluation judge` apply. It passes a valid file, exits 1 on a verdict-first or
    unknown-field file, and writes nothing."""
    from calibrated_agent_evaluation.cli import main

    assert not _fenced_commands("build-binary-judge", "python")
    (cmd,) = _fenced_commands("build-binary-judge", "calibrated-agent-evaluation judge --check")
    argv = shlex.split(cmd)
    assert argv == ["calibrated-agent-evaluation", "judge", "--check", "judge.json"]
    good = _REPO / "fixtures" / "judge-files" / "orderbot-refund-resolved.json"
    doc = json.loads(good.read_text(encoding="utf-8"))
    verdict_first = tmp_path / "verdict-first.json"
    verdict_first.write_text(json.dumps({**doc, "output_fields": ["verdict", "critique"]}))
    scored = tmp_path / "scored.json"
    scored.write_text(json.dumps({**doc, "scale": "1-5"}))
    monkeypatch.chdir(tmp_path)

    def run(path: pathlib.Path) -> tuple[int, str, str]:
        rc = main([*argv[1:-1], str(path)])
        out, err = capsys.readouterr()
        return rc, out, err

    rc, out, err = run(good)
    assert rc == 0, err
    assert out.strip() == "judge file ok"
    rc, _, err = run(verdict_first)
    assert rc == 1
    assert "must place a non-empty 'critique' before 'verdict'" in err
    rc, _, err = run(scored)
    assert rc == 1
    assert "Likert" in err
    assert set(tmp_path.iterdir()) == {verdict_first, scored}
