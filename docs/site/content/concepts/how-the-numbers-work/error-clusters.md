# Error clusters and when to stop labeling

Fifty labeled traces are enough if your agent fails in two ways and you have seen both many times.
They are not enough if every new batch turns up a failure you had not seen before.
`calibrated-agent-evaluation cluster` groups your labeled failures into named modes and tells you
which of those two situations you are in.

```text
--8<-- "fixtures/expected/docs/calibrated-agent-evaluation-cluster.txt"
```

That is the bundled corpus: 7 failures in 4 modes, and the answer is to keep labeling.

## Have you seen most of the failure modes?

The question to ask is: if the agent failed again right now, how likely is it to fail in a way you
have already named? I. J. Good published an estimate for this, crediting the idea to Alan Turing
(Good 1953). Count the modes you have seen exactly once. If many of your
modes are one-offs, you are still finding new ones, and the next failure has a good chance of being
another new one.

In the corpus, 2 of the 7 failures are in modes seen only once. The estimate is that a new failure
falls into a known mode with probability 1 - 2/7 = 0.714. With only 7 failures that estimate is
loose, and its 95% lower bound is 0.279 (Esty 1983).

Calibrated Agent Evaluation calls labeling **saturated** when all three of these hold:

- the lower bound on that probability is at least 0.95,
- at least 30 failures have been labeled,
- at least two different modes were seen. One mode holding every failure gives a probability of 1.0
  and says nothing about modes you have not met.

## How many modes are out there?

The same counts give an estimate of the total number of failure modes, including ones you have not
seen yet: Chao1 (Chao 1984), with a 95% range from Chao (1987). In the corpus it estimates 6.0
modes, between 4.2 and 26.1. Four have been seen. A range that wide is another way of saying keep
labeling.

## Where you have not looked

The last part of the output is coverage of your input space. If you defined a grid of situations
(feature, persona, scenario), each combination is a cell, and a cell with no observed failure is a
blind spot: either the agent is safe there or nobody has tested it. The corpus has failures in 3 of
24 cells, so 21 cells are unexamined.

## Naming the modes

Clustering drafts the groups from the written critiques, and a human ratifies them into a codebook
before they can tag anything. Drafting needs an embedding model (the `--embedder bedrock` option,
from the `[bedrock]` extra), which is why the output above says the draft step was skipped. The
statistics in this page work from the ratified modes either way.
