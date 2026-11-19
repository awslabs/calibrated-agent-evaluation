# Quickstart

Two commands, both offline with no API keys. The first shows the front of the pipeline on six
traces. The second seeds 848 traces and a judge kit for a full judge-trust chain you run in six
commands.

## 1. Install

From the root of a checkout ([Installation](installation.md) has the clone command and the other
install methods):

```bash
uv sync --inexact --extra stats
source .venv/bin/activate    # puts `calibrated-agent-evaluation` on your PATH
```

The `stats` extra is needed only for step 3. The base install runs step 2.

## 2. The sample corpus

```bash
calibrated-agent-evaluation quickstart
```

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/calibrated-agent-evaluation-quickstart.txt"
```

</div>

The six traces come from two reference agents, a single-agent refund bot (OrderBot) and a
multi-agent claims crew (ClaimsCrew), and their labels were written by people. The four failure
modes are the ones those people named. Without `--root`, the store goes in a temp directory, so
there's nothing to clean up.

## 3. A judge you can trust, or not

```bash
export CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=demo
calibrated-agent-evaluation quickstart --certify
```

This seeds 848 real OrderBot runs (two labeled rounds of 324 and an unlabeled production sample of
200) with synthetic labels and writes a judge kit, then prints six commands. Run them in order and you see:

1. Judge v1 is certified against the labels, then refused because it changes its verdict when
   answers get longer and goes along with claims made in the conversation.
2. Judge v2, one prompt revision later, is certified on fresh labels and passes the bias check.
3. v2 scores 200 production traces, and `calibrated-agent-evaluation run --ci` reports the failure rate with its range
   and fails the gate.

The judge's model replies replay from a recorded file, so none of it needs credentials.
[Calibrate a judge](../guides/calibrate-a-judge.md) walks through each command and its output.

## 4. Check your install

```bash
calibrated-agent-evaluation doctor
```

It reports your Python version, which optional extras are installed, and the install command for
each one that isn't.

## Next

- [From first trace to production](../guides/lifecycle/index.md): the same stages on your own
  agent, phase by phase.
- [Choose your path](../integrate/index.md): get your own agent's traces in.
- [Calibrate a judge](../guides/calibrate-a-judge.md): the checks a judge passes before it scores.
- [How the numbers work](../concepts/how-the-numbers-work/index.md): why the range and the
  refusals matter.
