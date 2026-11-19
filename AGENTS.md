# Working on Calibrated Agent Evaluation

This file is for coding agents and contributors. It covers setup, the checks a change must pass,
and the conventions the code follows. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the longer version.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just).

```bash
just install   # every extra plus the dev and corpus groups
just hooks     # the pre-commit hooks (the secrets scan needs a local `gitleaks` binary)
```

Node, pinned in `mise.toml`, runs the review UI's hotkey tests; they skip without it.

## Test and lint

```bash
just test               # the test suite; pass pytest arguments through, e.g. `just test -k drift`
just lint               # Ruff lint
just fix                # Ruff autofix, then format
just fmt-check          # Ruff format check
just typecheck          # Pyright
just check              # lint, format check, types, tests, license scan and id scan
just docs-build         # strict docs build; a broken link or a stale captured example fails it
just validate-fixture   # check the fixture corpus against the trace schema (stdlib only)
```

Run `just check` before you commit. `just --list` shows every recipe. The opt-in suites
(`just test-integration`, `just test-agentcore-live`, `just test-extras`) need a running Langfuse,
an AWS account or a package index.

## Conventions

- Fail loudly. Wrap third-party errors at the boundary in the package's typed error family, and
  make each refusal say why it refused and what to do next. Never swallow an exception or return a
  partial result as if it were whole.
- Data that crosses a boundary (traces, labels, records, settings) is a Pydantic v2 model.
  Behavioral seams are `typing.Protocol` ports, passed in through the constructor.
- Vendor SDKs (Langfuse, Inspect, Strands, boto3) stay in their adapter packages, imported lazily.
  The methodology code imports no vendor type.
- Check a vendor API against the installed package before you write code against it, and pin the
  version you checked.
- No number without its uncertainty, and no score from a judge that has not been calibrated
  against human labels.
- No copyleft dependency, in the default install or in any extra (`just license-scan`).
- Fixtures are real recorded output. Don't hand-write a trace or a cassette; record it, and redact
  account ids and credentials before committing.
- Tests describe the behavior they check, cover the refusal path, and run against real
  components where the suite can. Code, tests and docs carry no ticket or requirement ids
  (`just no-ids`).
- Prose on the docs site follows [`docs/site/STYLE.md`](docs/site/STYLE.md).
