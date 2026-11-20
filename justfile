# Calibrated Agent Evaluation task catalog. Run `just` (or `just --list`) to see recipes.
# `just --list` shows the last comment line above each recipe, so each block ends with its summary.

# list available recipes
default:
    @just --list

# Every extra plus the dev and corpus groups: the whole suite, corpus integrity included, runs, and
# pyright resolves the corpus/ imports.
# create / refresh the dev environment
install:
    uv sync --all-extras --group dev --group corpus

# run the test suite on the current interpreter
test *args:
    uv run pytest -n auto --dist loadfile {{args}}

# run tests with coverage
cov:
    uv run pytest --cov --cov-report=term-missing

# Needs uv and a package index or a warm uv cache.
# install each extra alone from the built wheel into a fresh venv and import what it enables
test-extras *args:
    uv run pytest -m wheel tests/test_extras_wheel.py tests/test_check_pypi_readme.py {{args}}

# run the opt-in integration suite against a running Langfuse stack (see `just langfuse-up`)
test-integration *args:
    LANGFUSE_HOST=${LANGFUSE_HOST:-http://localhost:3000} \
    LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-pk-lf-calibrated-agent-evaluation-dev} \
    LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-sk-lf-calibrated-agent-evaluation-dev} \
    uv run pytest -m integration {{args}}

# lint with Ruff
lint:
    uv run ruff check .

# auto-fix lint issues and format the code
fix:
    uv run ruff check --fix .
    uv run ruff format .

# verify formatting without modifying files
fmt-check:
    uv run ruff format --check .

# static type-check with Pyright
typecheck:
    uv run pyright

# build the wheel and sdist
build:
    uv build

# validate the shared fixture corpus against the normalized schema (stdlib only, no install)
validate-fixture:
    python3 fixtures/validate_corpus.py

# The compose file holds no credential; the dev values come from docker/langfuse/.env when you
# create one, else from the committed loopback-only dev defaults in .env.example.
langfuse_env := if path_exists("docker/langfuse/.env") == "true" { "docker/langfuse/.env" } else { "docker/langfuse/.env.example" }

# bring up the self-hosted Langfuse v3 OSS stack on 127.0.0.1 (the Langfuse backend; telemetry off)
langfuse-up:
    docker compose --env-file {{langfuse_env}} -f docker/langfuse/docker-compose.yml up -d

# tear down the Langfuse stack and remove its volumes
langfuse-down:
    docker compose --env-file {{langfuse_env}} -f docker/langfuse/docker-compose.yml down -v

# seed the shared fixture corpus into a backend: `just seed-fixture local` | `just seed-fixture langfuse`
seed-fixture backend="local":
    uv run python scripts/seed_fixture.py --backend {{backend}}

# Both agents end-to-end through all 8 stages, calibrated and gated judges, on the local store and Langfuse.
# Cassette replay and local by default; add `--live` (AWS creds) for the Bedrock sign-off leg, drop
# `--local-only` for the Langfuse leg (start it with `just langfuse-up`).
# e.g. `just flagship --agent both` | `just flagship --live`
# run the flagship proof
flagship *args:
    uv run --all-extras python scripts/flagship/run.py {{args}}

# run the full test matrix across Python versions
# nox and pre-commit build their own environments, so they run as pinned uv tools rather than
# project dependencies; that keeps their virtualenv out of uv.lock.
matrix:
    uvx nox@2026.10.20 -s tests

# GIT_CONFIG=/dev/null makes pre-commit ignore a system-level core.hooksPath set in /etc/gitconfig,
# which otherwise makes it refuse to install. No-op when none is set.
# install the pre-commit hooks
hooks:
    uv tool install --quiet pre-commit==4.7.0
    GIT_CONFIG=/dev/null pre-commit install

# Covers the default install and every optional extra; the scanner (pip-licenses) is itself MIT.
# fail on any GPL/LGPL/AGPL node in the resolved dependency tree
license-scan *args:
    uv run python scripts/license_scan.py {{args}}

# Material reads NO_MKDOCS_2_WARNING and properdocs reads DISABLE_MKDOCS_2_WARNING; both banners are
# about MkDocs 2 and do not apply to this site.

# Needs the `docs` group. --all-extras so mkdocstrings can import optional-dep modules (review/,
# execute/, ...) for autodoc. Skips the captured-example check so live reload stays fast;
# docs-build and `just test` run it.
# serve the documentation site with live reload at http://127.0.0.1:8000
docs *args:
    CALIBRATED_AGENT_EVALUATION_DOCS_SKIP_EXAMPLES=1 NO_MKDOCS_2_WARNING=true DISABLE_MKDOCS_2_WARNING=true uv run --group docs --all-extras mkdocs serve -f docs/site/mkdocs.yml {{args}}

# Fails on broken links, bad refs and captured example output that no longer matches the code.
# strict static build of the docs site into docs/site/_site/
docs-build *args:
    NO_MKDOCS_2_WARNING=true DISABLE_MKDOCS_2_WARNING=true uv run --group docs --all-extras mkdocs build --strict -f docs/site/mkdocs.yml {{args}}

# Review the diff before committing: a changed golden is a changed page.
# re-run every captured docs example offline and rewrite its golden under fixtures/expected/docs/
docs-goldens:
    uv run --all-extras python docs/site/docs_examples.py --update

# Needs CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE=1, CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN and an AWS_PROFILE that assumes it;
# CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS optionally caps billed calls. See tests/integration/agentcore/.
# opt-in live AgentCore tests against a real account, as the scoped live-test role only
test-agentcore-live *args:
    uv run --extra agentcore pytest -m "integration and agentcore" tests/integration/agentcore {{args}}

# lint the CloudFormation templates under infra/ (cfn-lint runs through uvx; not a dependency)
lint-infra:
    uvx cfn-lint infra/*.yaml

# run every local check (lint, format, types, tests, license scan, ids)
check: lint fmt-check typecheck test license-scan no-ids

# Repo-local git config: rebase on pull (autostashing local edits) and reuse recorded conflict
# resolutions.
# configure git for this clone (run once after cloning)
setup:
    git config pull.rebase true
    git config rebase.autoStash true
    git config rerere.enabled true
    git config rerere.autoUpdate true

# fail if a requirement or ticket id appears in shipped code, tests, scripts, corpus, fixture prose, docker, skills or config
no-ids:
    uv run python scripts/check_no_ids.py
