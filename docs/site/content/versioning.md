# Versioning and stability

The current release is 0.1.0, published on PyPI as `calibrated-agent-evaluation`.

```console
$ calibrated-agent-evaluation --version
calibrated-agent-evaluation 0.1.0
```

Every release is listed in the [changelog](changelog.md).

## What pre-1.0 means here

- The project follows [Semantic Versioning](https://semver.org/). Before 1.0, a minor release
  (0.1 to 0.2) may break the Python API, the `calibrated-agent-evaluation` command line, or a
  persisted record format, and there is no deprecation window yet. Patch releases do not break
  them.
- Breaking changes are listed in the changelog. Read it before you upgrade if you depend on
  anything the library writes to disk.
- If you build on it now, pin the minor version you tested against
  (`calibrated-agent-evaluation~=0.1.0`).

## Persisted formats

Every record the library writes carries a schema version, and every reader refuses a record whose
version differs from the one it was built for. Nothing is converted silently, so an old store or
dump fails loudly after an upgrade instead of being misread. The probe record is the exception: it
is stamped with its version, but no reader checks it yet.

| Record | Defined in | Schema version |
| ------ | ---------- | -------------- |
| `Trace`, `Span`, `Label` (the trace store and `calibrated-agent-evaluation dump`) | `calibrated_agent_evaluation.schema` | 1 |
| Calibration record | `calibrated_agent_evaluation.calibrate` | 7 |
| Bias battery record | `calibrated_agent_evaluation.bias` | 4 |
| Run record | `calibrated_agent_evaluation.run` | 7 |
| Improve record | `calibrated_agent_evaluation.improve` | 6 |
| Flagship report | `calibrated_agent_evaluation.flagship` | 4 |
| Judge definitions and enforcement decisions | `calibrated_agent_evaluation.enforce` | 1 |
| Judge run records and the verdict cache | `calibrated_agent_evaluation.judges` | 1 |
| Probe record | `calibrated_agent_evaluation.probe` | 1 |
| Dump manifest (`dump-manifest.json`) | `calibrated_agent_evaluation.dumpload` | 1 |

These values are the `*_SCHEMA_VERSION` constants in the code at the time of writing:
`calibrated_agent_evaluation.schema.SCHEMA_VERSION`, `calibrated_agent_evaluation.calibrate.record.CALIB_SCHEMA_VERSION`, and so on.
The dump manifest's is `calibrated_agent_evaluation.dumpload.io.DUMP_FORMAT_VERSION`.

A store or dump written under an older version has to be regenerated with the release that
reads it. There is no migration tool: `calibrated-agent-evaluation load` refuses a dump whose version differs, as
[Troubleshooting](guides/troubleshooting.md) shows.

The fixture corpus under `fixtures/` carries its own `VERSION` file, which must equal the trace
schema version the code runs.

## Dependency pins

Most extras pin a floor at the version whose API was checked when the adapter was written, and cap
the next major version. `datasets` and `hf` keep older floors than the versions they were checked
against. Three are narrower:

| Extra | Pin | Why |
| ----- | --- | --- |
| `ui` | `python-fasthtml>=0.14,<0.15` | FastHTML is pre-1.0, so a minor release may break. The cap stops one arriving unreviewed. |
| `inspect` | `inspect-ai>=0.3.242,<0.4` | Same reason: a 0.x package, capped at the next minor. The `bedrock` extra carries the same pin. |
| `probe` | `inspect_evals>=0.14.3,<0.15`, `inspect-ai>=0.3.242,<0.3.288` | Also 0.x; the benchmark battery reads task names that a minor release can rename. inspect_evals 0.14 imports a symbol inspect-ai 0.3.288 removed. |
| `synth`, `agentcore` | `strands-agents-evals>=1.6.0,<1.7` | The AgentCore failure detectors are verified and recorded against the 1.6 detector prompt and failure vocabulary; 1.10 changed both. |

Raising any of these means re-checking the adapter against the new version first.

The checkout's `uv.lock` pins the whole development tree. The committed coverage figures under
`fixtures/expected/coverage/` record the numpy, scipy and statsmodels versions they were measured
with, so `uv sync --locked` from the same commit reproduces the environment behind them.

## The documentation

This site is built from the `main` branch and describes that version only. There is no version
selector yet.
