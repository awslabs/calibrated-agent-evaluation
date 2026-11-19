# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for environments and
[just](https://github.com/casey/just) as the task runner.

```bash
# Every extra plus the dev and corpus groups (installs Python 3.13 if needed). The whole suite,
# corpus integrity included, needs all of them, and pyright resolves the corpus/ imports.
just install

# Run the CLI from the working tree
uv run calibrated-agent-evaluation --help
```

A plain `uv sync --group dev` leaves out the extras, so the tests for them skip or fail. Node
(pinned in `mise.toml`) runs the review UI's hotkey tests, which skip without it.

Install the pre-commit hooks:

```bash
just hooks
```

The hooks run Ruff, Pyright, the license scan (when a dependency manifest changes), a check for
the copyright and SPDX header on Python files, and a `gitleaks` secrets scan. The secrets scan
runs a locally installed `gitleaks` binary, so install it first (`brew install gitleaks`).
Commits fail until it is on your `PATH`.

### Live tests

Tests marked `integration` talk to real services and are deselected by default. Each one is
opt-in:

- `just langfuse-up` starts a self-hosted Langfuse stack bound to `127.0.0.1`, and
  `just test-integration` runs the suite against it. `just langfuse-down` removes the stack and
  its volumes.
- The Bedrock tests need AWS credentials with Bedrock access and an opt-in variable, for example
  `CALIBRATED_AGENT_EVALUATION_JUDGE_LIVE=1 uv run pytest -m integration -k judges_live`. They make
  billed model calls.
- `just test-agentcore-live` runs the AgentCore suite against a real account. See
  `tests/integration/agentcore/` for the role and variables it needs.

The variables are listed under [Environment variables](#environment-variables).

## The checks

Run the full local gate before opening a pull request:

```bash
just check      # lint, format check, type check, tests, license scan, id scan
```

Individually:

| Command          | What it runs                                             |
| ---------------- | -------------------------------------------------------- |
| `just test`      | the test suite (pytest)                                  |
| `just lint`      | Ruff lint                                                |
| `just fix`       | Ruff autofix + format                                    |
| `just typecheck` | Pyright (standard mode)                                  |
| `just fmt-check` | Ruff format check, no changes                            |
| `just license-scan` | fail on any copyleft node in the resolved tree        |
| `just no-ids`    | fail on a requirement or ticket id in code, tests, scripts, corpus, fixture prose, skills or config |
| `just matrix`    | the full test matrix across Python 3.11–3.14 (nox)       |
| `just test-extras` | install each extra alone from the built wheel and import what it enables |
| `just test-integration` | the opt-in suite against a running Langfuse (`just langfuse-up` first) |
| `just docs-build`| strict docs-site build                                   |

## Conventions that matter

- **Typed functional core, no silent failure.** Boundaries wrap errors in a typed error family; a
  refusal message is a feature: it cites *why* the operation was refused. Don't swallow exceptions.
- **The runtime dependency tree stays permissive.** No copyleft (GPL/LGPL/AGPL) runtime dependency,
  including per-optional-extra, and no weak-copyleft one beyond the reviewed list. Reimplement
  rather than take a copyleft dependency. See [Dependencies](#dependencies).
- **Vendor isolation.** Third-party SDKs (Langfuse, Inspect, Strands, Bedrock) are confined to adapter
  packages; the methodology plane imports no vendor type.
- **Verify a vendor surface before writing against it.** Install the real package and check its API;
  never guess method names. Pin the verified floor.
- **Statistical honesty.** No headline number without its confidence treatment; no score from an
  uncalibrated judge.

## Tests

- Every new behavior ships with at least one negative-path or refusal test.
- Tests assert behavior, not internals; every change keeps `just check` green.

## Dependencies

A new runtime dependency, including one added only to an optional extra, must carry a permissive
license such as MIT, BSD or Apache-2.0. Copyleft licenses (GPL, LGPL, AGPL) are not accepted
anywhere in the shipped tree. A weak-copyleft dependency (MPL, EPL, CDDL) or one with no declared
license needs a review entry in `scripts/license_scan.py` (`REVIEWED_WEAK_COPYLEFT` or
`REVIEWED_UNKNOWN`) naming the license string and why it is acceptable; certifi and tqdm, both
MPL-2.0 and used unmodified, are the reviewed weak-copyleft entries today. Two checks enforce this:

- `just license-scan` resolves the default install and each extra and fails on any copyleft
  node, or any unreviewed weak-copyleft or unknown-license node. It runs in `just check`, and in
  the pre-commit hook whenever `pyproject.toml` or `uv.lock` changes.
- The dependency review workflow checks the dependency changes in every pull request.

Declare the lowest version whose API you checked as the floor, and cap the next major. Code or
data copied into the repository from another project needs an entry in
[THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES) with its copyright lines and full license text.

## Copyright notices

Python files start with the project's copyright and license header (JavaScript and CSS files carry
the same two lines as `//` and `/* */` comments):

```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
```

Keep that header on files you add; a pre-commit hook checks it. If your employer requires its
copyright to be recorded, add a line to [NOTICE](NOTICE) in your pull request, in this form:

```text
Portions Copyright <year> <Company>.
```

## Environment variables

Every `CALIBRATED_AGENT_EVALUATION_*` variable the package, the scripts and the tests read. Variables a test or script
uses only as the name of an auth token (`CALIBRATED_AGENT_EVALUATION_TOKEN`, `CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN`,
`CALIBRATED_AGENT_EVALUATION_PROBE_WS_TOKEN`) are left out: any name works there.

| Variable | Read by | What it does |
| --- | --- | --- |
| `CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT` | CLI | Root directory of the local store when `--root` is not given. |
| `CALIBRATED_AGENT_EVALUATION_EXPLAIN` | CLI | `1` turns on `--explain` for every command. |
| `CALIBRATED_AGENT_EVALUATION_READ_WAIT_SECONDS` | CLI | How long a command waits for earlier Langfuse writes to become readable (default 30). |
| `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE` | package, corpus | `replay`, `record` or `auto` for model-call cassettes. |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS` | package | Cap on billed AgentCore calls in one process. |
| `CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID` | corpus, flagship script, live tests | Bare Bedrock model id for live runs (default `global.anthropic.claude-sonnet-4-6`). Inspect callers add the `bedrock/` prefix themselves; an id that already has it is refused, and the retired `CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL` is refused rather than ignored. |
| `CALIBRATED_AGENT_EVALUATION_JUDGE_LIVE` | live tests | `1` runs the live judge test. |
| `CALIBRATED_AGENT_EVALUATION_DETECTOR_LIVE` | live tests | `1` runs the live failure-detector test, one billed Bedrock call through `AWS_PROFILE` (`tests/integration/test_detectors_live.py`). |
| `CALIBRATED_AGENT_EVALUATION_PROBE_ENDPOINT` | live tests | HTTP endpoint for the live probe test. |
| `CALIBRATED_AGENT_EVALUATION_PROBE_WS_ENDPOINT` | live tests | WebSocket endpoint for the live WebSocket probe test. |
| `CALIBRATED_AGENT_EVALUATION_PROBE_WS_AUTH` | live tests | `subprotocol` (default) or `bearer` for the live WebSocket probe test. |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE` | live tests | `1` opts in to the AgentCore live suite. |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN` | live tests | Role the AgentCore live suite runs as. |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_REGION` | live tests | Region for the AgentCore live suite. |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME` | live tests | AgentCore runtime name for the live doctor test. |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_OMNI_DOMAIN` | live tests | Domain for the live doctor test. |
| `CALIBRATED_AGENT_EVALUATION_DOCS_SKIP_EXAMPLES` | docs build | `1` skips the captured-example check (`just docs` sets it). |
| `CALIBRATED_AGENT_EVALUATION_DOCS_SITE_URL` | docs build | Base URL the site is built for (default the GitHub Pages site). |
| `CALIBRATED_AGENT_EVALUATION_DOCS_REPO_URL` | docs build | Repository the header links to (default the GitHub repository). |
| `CALIBRATED_AGENT_EVALUATION_DOCS_REPO_NAME` | docs build | Label of that link (default `awslabs/calibrated-agent-evaluation`). |

## Contributing to the docs

The documentation site's pages are Markdown under `docs/site/content/`, and the navigation is the
`nav` block in `docs/site/mkdocs.yml`. A new page needs a nav entry, or the strict build fails. The
API and CLI reference pages are generated from docstrings and from the `calibrated-agent-evaluation` parser at build time,
so change the code, not a reference page.

```bash
just docs        # serve the site with live reload at http://127.0.0.1:8000
just docs-build  # the strict build: a broken link or anchor fails it
```

Run `just docs-build` before you open a pull request that touches the docs. Prose on the site follows
[`docs/site/STYLE.md`](docs/site/STYLE.md). In short: example first, real command output pasted
rather than paraphrased, no em dashes, bold only for a term being defined, and every statistical
method cited in [`docs/site/content/concepts/methodology.md`](docs/site/content/concepts/methodology.md).
`docs/site/README.md` covers the toolchain and hosting.

## Working with coding agents

[`AGENTS.md`](AGENTS.md) is a short version of this guide for coding agents: setup, the checks and
the code conventions.

## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
