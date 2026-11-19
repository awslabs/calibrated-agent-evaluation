# Reviewer notes: known limits of the corrected-rate interval

This page is for readers checking the statistical claims. It records where the corrected-rate
interval was measured to fall short of its stated 95% coverage, and what Calibrated Agent Evaluation
does about it. The numbers come from the committed files under `fixtures/expected/coverage/`.

Every method the library ships, its code location and its primary source are listed on
[Methods and citations](methodology.md). The learner-facing explanation is
[How the numbers work](how-the-numbers-work/index.md).

## Two sizes gate a corrected rate

A corrected rate is computed only when both samples are large enough.

- **Gold labels per class** must reach the larger of the coverage floor at the planned accuracy and
  at the accuracy the judge showed (`COVERAGE_FLOOR` in
  `calibrated_agent_evaluation.calibrate.plan`): 28 at 0.85 and 0.88, 35 at 0.90, 80 at 0.92 and
  0.95, 130 at 0.98, 162 at 0.99, and 11 at 0.995, 0.999 and 1.0. Between two measured accuracies
  the larger floor applies, so 0.993 needs 162.
- **Rate-sample traces** must number at least 150 (`MIN_RATE_SAMPLE`). Clustered verdicts count
  distinct traces.

Below either size, `corrected_rate` refuses and says how many more labels or traces are needed.

## How the coverage floor was validated, and what a stronger test found

The floor table was derived at 1,000 replicates per cell (one seed, 20261014) by
`calibrated_agent_evaluation.stats.studies.planner_floor`. A row passed when Holm across its 42
cells (prevalence 0.02 to 0.50, rate samples of 100 to 100,000 traces) rejected no one-sided test
against the band 0.9405 to 0.9595.

The same grid was rerun at 3,000 replicates per cell, pooling seeds 20261014, 20261015 and 20261016
(`planner_floor_uniform.json`). At that power the interval under-covers wherever a class shows
about 2 to 3 errors:

| Accuracy | Gold per class | Expected errors per class | Lowest coverage | Cells rejected (Holm) | Admitted by the shipped floor |
|---|---|---|---|---|---|
| 0.85 | 20 | 3.0 | 0.922 | 1 of 42 | no |
| 0.88 | 20 | 2.4 | 0.910 | 11 of 42 | no |
| 0.90 | 28 | 2.8 | 0.919 | 6 of 42 | no |
| 0.92 | 35 | 2.8 | 0.914 | 9 of 42 | no |
| 0.95 | 61 | 3.1 | 0.923 | 2 of 42 | no |
| 0.98 | 130 | 2.6 | 0.919 | 4 of 42 | yes |
| 0.99 | 260 | 2.6 | 0.917 | 2 of 42 | yes |

The 1,000-replicate scan lacked the power to see these dips. The interval without the boundary
rule shows the same dips on the same data, so they are not caused by it.

Two of these rows are inside the shipped floor. A judge at 0.98 accuracy certified on 130 gold per
class, or at 0.99 on 260, can receive an interval that covers less often than 95%. Over every
symmetric cell the floor admits (a judge equally accurate on both classes; 2,900 cells, Holm across
all of them), the under-coverage that more traces cannot fix is confined to those two rows at
prevalence 0.02 and 0.05 with 100,000-trace rate samples. Asymmetric judges are a separate known
limit, below.

| Accuracy | Gold per class | Prevalence | Rate-sample traces | Coverage |
|---|---|---|---|---|
| 0.98 | 130 | 0.02 | 100,000 | 0.919 |
| 0.98 | 130 | 0.05 | 100,000 | 0.921 |
| 0.99 | 260 | 0.02 | 100,000 | 0.917 |
| 0.99 | 260 | 0.05 | 100,000 | 0.920 |

The lowest coverage among the remaining admitted cells at 150 or more traces is 0.923 (0.98, 130
per class, prevalence 0.06, 100,000 traces), not rejected after Holm.

### Asymmetric judges

The floor is read at the higher of the judge's two rates, so a judge with TPR 0.95 and TNR 0.99
uses the 0.99 floor of 162 gold per class. The 3,000-replicate rerun scanned that pairing from 11
to 800 per class (`planner_floor_uniform.json`, the same seeds and 42-cell grid):

| Gold per class | Lowest coverage | Under-coverage rejected (Holm) | Admitted by the shipped floor |
|---|---|---|---|
| 162 to 200 | 0.934 | 0 of 42 | yes |
| 260 to 350 | 0.928 | 0 of 42 | yes |
| 450 | 0.926 | 1 of 42 | yes |
| 600 | 0.940 | 0 of 42 | yes |
| 800 | 0.919 | 1 of 42 | yes |

At 450 and 800 per class the interval was measured to under-cover, and the floor admits both sizes.
At 200 or fewer per class the rows fail only on over-coverage, which makes the interval wider than
needed, not less honest. The rate-sample minimum below was derived from symmetric cells only, so it
is not validated for asymmetric judges. Pairings other than TPR 0.95 with TNR 0.99 to 1.0 were not
scanned at 3,000 replicates.

### Why the table stays

The floor table is kept at its 1,000-replicate derivation. Shipping the 3,000-replicate table would
leave 0.99 with no qualifying size, so no accurate judge could ever report a corrected rate, and it
would still not remove the dips. The fix that would remove them is a different interval at low
error counts, which has not been built.

## The boundary rows: 0.995, 0.999 and 1.0

A judge that makes no errors on a class (or almost none) puts the false-positive or miss rate at
the edge of its range, where the interval's posterior draws never reach zero and the corrected rate
was biased. The interval now applies the modified Jeffreys boundary rule (Brown, Cai and DasGupta
2001): when a count sits at its boundary, the matching interval limit is set to the boundary. Away
from the boundary the interval is unchanged.

The floors at 0.995, 0.999 and 1.0 were derived separately, by
`calibrated_agent_evaluation.stats.studies.planner_floor` over `planner_floor_boundary_scan.json`:

- configurations: symmetric 0.995, 0.999 and 1.0, and TPR 0.95 with TNR 0.995, 0.999 and 1.0, at
  11 to 800 gold per class, over the same prevalences and rate-sample sizes as the interior scan
- seeds 20261014, 20261015 and 20261016, 1,000 replicates each, pooled to 3,000 per cell
  (4,284 cells)
- criterion: Holm across each row's 42 cells of a one-sided exact binomial test for coverage
  below 0.9405. Over-coverage is not tested in these rows, because the boundary rule widens the
  interval there on purpose.

Every row passes at every scanned size, so each floor is 11, the smallest size scanned. The
intervals there are conservative: median cell coverage is 0.969 and some cells cover every
replicate. They are not uniformly at or above 95%. The lowest cell covers 0.927 (1.0 accuracy, 600
per class, prevalence 0.05, 100 traces), and 45 of the 4,284 cells sit below 0.94 on their point
estimate, none of them rejected after Holm within its row. These rows were scanned at 3,000
replicates, so they are not subject to the low-power problem of the 1,000-replicate interior table.

## Where the rate-sample minimum comes from

`calibrated_agent_evaluation.stats.studies.rate_sample_minimum` pools the admitted symmetric cells
across the three seeds (3,000 replicates each) and adds cells at 150, 200, 300 and 400 traces for
prevalence 0.02 and 0.05. It runs a one-sided exact binomial test for coverage below 0.9405 on each
of the 2,900 cells and applies Holm across all of them.

- One configuration is eval-limited: 0.99 accuracy, 600 per class, prevalence 0.02, which covers
  0.917 with 100 traces and passes from 150 traces up. The minimum is 150.
- The four configurations above are calibration-limited: still rejected at the largest rate sample,
  so more traces do not help them.

Prevalence below 0.02 was not measured, and the minimum is not validated there.

## Files

- `fixtures/expected/coverage/planner_floor.json`: the shipped floor and its 1,000-replicate rows.
- `fixtures/expected/coverage/planner_floor_uniform.json`: the 3,000-replicate rerun.
- `fixtures/expected/coverage/planner_floor_boundary_scan.json` and `planner_floor_boundary.json`:
  the pooled boundary cells and the floor rows derived from them.
- `fixtures/expected/coverage/rate_sample_minimum.json`: the pooled cell counts and the derived
  minimum. `tests/test_rate_sample_minimum.py` re-derives it from those counts.
