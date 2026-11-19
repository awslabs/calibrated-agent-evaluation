# The extras model

Calibrated Agent Evaluation's base install has one runtime dependency: pydantic, which validates
every record and trace that crosses a boundary.

## Why the core is small

```toml
dependencies = ["pydantic>=2.8,<3"]
```

Beyond pydantic (and its own small dependencies, `pydantic-core`, `typing-extensions` and
`annotated-types`, all MIT or BSD), the default `calibrated-agent-evaluation` install uses only the
standard library. The offline local backend is stdlib `sqlite3`; the CLI is stdlib `argparse`. This
means:

- It installs fast and small, and works on an air-gapped laptop out of the box.
- A CI gate needs only `[stats]`; the UI, Langfuse and cloud SDKs are separate extras.

## How capability degrades gracefully

Because a capability might not be installed, the library never lets a bare `ImportError` escape. When a
stage needs an extra you don't have, it raises a typed error carrying the exact fix, written
for how you installed it (here, a uv-managed checkout):

```text
--8<-- "fixtures/expected/docs/ts-missing-stats.txt"
```

A stage checks for its engine with `importlib.util.find_spec`, so the check never imports the vendor
SDK. `calibrated-agent-evaluation doctor` lists every extra the installed package declares, read
from its metadata, as installed or not, with the install command for any that are missing. Those
commands work once the package is installed from a checkout or a wheel; see
[Installation](../getting-started/installation.md). [Troubleshooting](../guides/troubleshooting.md)
lists the messages.

## The license guarantee

Every extra is license-scanned before it can ship. The default install is permissive (BSD / MIT /
Apache). The optional extras are permissive apart from two weak-copyleft dependencies, used
unmodified: certifi (MPL-2.0, required by httpx and requests) and tqdm (MPL-2.0 and MIT, required
by huggingface-hub and datasets). Nothing in the tree is copyleft (GPL / LGPL / AGPL). Where the
only option would be a copyleft dependency, the piece is reimplemented instead. An automated scan
enforces this: it fails on copyleft, on a weak-copyleft dependency that has not been reviewed, and
on a dependency with no declared license.

## Vendor isolation

Extras that wrap a third-party SDK (Langfuse, Inspect, Strands, Bedrock, AgentCore) are confined to adapter
packages at the edge. The methodology plane (the statistics, the gates, the schema) imports no
vendor type. This lets the library swap trace stores or executors without touching the part that
computes the numbers.

## Next

- [Installation](../getting-started/installation.md): the full extras table.
- [Run fully offline](../guides/offline.md): which extras stay offline and which reach the network.
