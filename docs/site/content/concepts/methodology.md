# Methods and citations

This page is for reviewers. It lists every statistical method Calibrated Agent Evaluation ships, the
function that implements it, and the source it comes from. None of the statistics are new. For an
explanation of each one from first principles, read [How the numbers
work](how-the-numbers-work/index.md). Where the corrected-rate interval falls short of its stated
coverage is on [Reviewer notes](reviewer-notes.md).

Each step of the pipeline, and the published work it follows:

- `calibrate` certifies a judge when the Wilson (1927) lower bounds on its TPR and TNR both clear
  the threshold, an intersection-union test (Berger 1982).
- `run --judge` corrects the judge's flag rate
  (`calibrated_agent_evaluation.calibrate.corrected_rate`) with Rogan and Gladen (1978), the
  correction Lee et al. (2026) work out for LLM judges. The interval around it comes from Jeffreys
  posteriors (Brown, Cai and DasGupta 2001), not from Lee et al.'s construction.
- `bias` checks the CALM biases of Ye et al. (2025), each bounded with Newcombe's (1998) paired
  interval.
- `cluster` reports saturation as Good–Turing coverage (Good 1953; Esty 1983) and Chao1 richness
  (Chao 1984, 1987).
- `run --ci` gates each failure mode on a Wilson lower bound, again an intersection-union test.
- `improve` tests a fix with McNemar's (1947) paired test under Holm's (1979) correction.

Calibrated Agent Evaluation does not implement the hypothesis test of Feng et al. (2026) or the
prediction-powered estimators studied by Chen et al. (2026) and packaged in GLIDE (Martinon, Merad
and Raki 2026). [Prior work](#prior-work-on-correcting-llm-judge-rates) says how each differs.

## Methods by stage

| Stage | Method | Where it lives | Source |
| ----- | ------ | -------------- | ------ |
| `calibrate` | Wilson score lower bound on TPR and TNR, one-sided 95% | `calibrated_agent_evaluation.stats.wilson_lower`, `calibrated_agent_evaluation.calibrate.certify` | Wilson (1927) |
| `calibrate` | Certification as an intersection-union test over the two bounds | `calibrated_agent_evaluation.calibrate.certify` | Berger (1982) |
| `calibrate` | Sample-size planner: the smallest per-class size from which the Wilson bound at the planned accuracy (correct count rounded down) clears the gate at every larger size, then the measured coverage floor | `calibrated_agent_evaluation.calibrate.plan_calibration` | Wilson (1927); the coverage study below |
| `calibrate`, `run` | Minimum rate sample of 150 traces for a corrected rate | `calibrated_agent_evaluation.calibrate.plan.MIN_RATE_SAMPLE` | the coverage study below; see [Reviewer notes](reviewer-notes.md#where-the-rate-sample-minimum-comes-from) |
| `calibrate` | Youden's J, with the corrected point refused below 0.5 | `calibrated_agent_evaluation.stats.rogan_gladen.youden_j` | Youden (1950) |
| `calibrate`, `run` | Rogan–Gladen prevalence correction | `calibrated_agent_evaluation.stats.rogan_gladen.rogan_gladen` | Rogan and Gladen (1978) |
| `calibrate`, `run` | Corrected-rate interval from Jeffreys posteriors on TPR, 1 - TNR and the measured rate, with the modified Jeffreys rule at boundary counts | `calibrated_agent_evaluation.stats.rogan_gladen.rg_interval` | Brown, Cai and DasGupta (2001) for the Jeffreys interval and its modification |
| `calibrate`, `run` | Design effect for verdicts clustered within a trace | `calibrated_agent_evaluation.stats.rogan_gladen.cluster_design_effect` | Kish (1965); Korn and Graubard (1998) |
| `calibrate --explain` | Cohen's kappa with its large-sample interval and the confusion cells, for judge-versus-gold agreement on imbalanced gold (raw agreement with its Wilson interval otherwise) | `calibrated_agent_evaluation.stats.kappa.cohen_kappa_interval`, `calibrated_agent_evaluation.enforce.display.agreement_display` | Cohen (1960); Fleiss, Cohen and Everitt (1969) |
| `bias` | Excess of each probe's disagreement rate over its placebo's, bounded by the one-sided 95% upper limit of Newcombe's paired interval, against a per-model threshold; for length and sycophancy, the same bound on directional moves toward the favored verdict over the placebo's, against 0.20; positional bias on pairwise judges only | `calibrated_agent_evaluation.bias.run_battery` | Newcombe (1998); the CALM bias set of Ye et al. (2025) |
| `label`, `calibrate` | Krippendorff's alpha, Fleiss' kappa and per-class specific agreement, each with a unit-bootstrap interval; `calibrate` refuses gold from overlapping raters unless at least 130 traces are double-coded and alpha's lower bound reaches 0.80 | `calibrated_agent_evaluation.stats.agreement`, `calibrated_agent_evaluation.review.agreement` | Krippendorff (2004, 2011); Hayes and Krippendorff (2007); Fleiss (1971) |
| `cluster` | Good–Turing sample coverage with Esty's one-sided 95% lower bound | `calibrated_agent_evaluation.cluster.saturation.good_turing_coverage_bound` | Good (1953); Esty (1983) |
| `cluster` | Chao1 richness with Chao's log-transformed 95% interval | `calibrated_agent_evaluation.cluster.saturation.chao1_interval` | Chao (1984) for the estimator; Chao (1987) for the interval |
| `run` | CI gate: every mode's pass-rate Wilson lower bound must clear its threshold | `calibrated_agent_evaluation.run.run_ci_gate` | Wilson (1927); Berger (1982) |
| `run --drift-baseline` | Benjamini–Yekutieli FDR at q = 0.10 over the two drift flags (PSI and Wasserstein) | `calibrated_agent_evaluation.run.flag_slices`, called by `calibrated_agent_evaluation.run.drift_report.input_drift` | Benjamini and Yekutieli (2001) |
| library only | Benjamini–Yekutieli FDR at q = 0.10 over per-slice flags (a flag per taxonomy cell); `calibrated-agent-evaluation run` raises none until a per-slice test is specified | `calibrated_agent_evaluation.run.flag_slices` | Benjamini and Yekutieli (2001) |
| `run --drift-baseline` | Input drift on one `--drift-feature`: PSI on baseline-quantile bins and IQR-normalized Wasserstein distance, each with a permutation p-value, BY-FDR over the two; Jensen–Shannon divergence on the taxonomy-cell counts (the library default is the PSI bins) is reported but never flags | `calibrated_agent_evaluation.stats.drift`, `calibrated_agent_evaluation.run.drift_report.input_drift`, `calibrated_agent_evaluation.run.features` | Lin (1991) for JSD; Yurdakul and Naranjo (2020) for PSI |
| `run --drift-baseline` | Staleness warning for each judge admitted with `--judge` when input drift is flagged or its calibration no longer certifies it; never changes the gate, and `--fail-on-drift` gives it exit code 3 | `calibrated_agent_evaluation.run.check_staleness` | a rule over the drift flags above, no further statistic |
| library only | Judge drift: paired change in TPR or TNR with Newcombe's hybrid-score interval; no command computes it | `calibrated_agent_evaluation.run.drift_report.judge_drift_paired`, `calibrated_agent_evaluation.stats.proportions.newcombe_paired_diff` | Newcombe (1998) |
| `calibrate --check`, `calibrate --explain` | Judge drift between two calibrations on independent gold: change in TPR or TNR with Newcombe's hybrid-score interval for independent proportions; the re-check's size is the smallest from which a judge showing 0.80 is flagged at every larger size | `calibrated_agent_evaluation.run.judge_drift`, `calibrated_agent_evaluation.run.judge_check.check_size` | Newcombe (1998, independent proportions) |
| `plan --gate` | The gate's sample size: the smallest size from which the Wilson lower bound at the expected pass rate (count rounded down) clears the threshold at every larger size | `calibrated_agent_evaluation.calibrate.gate_set_size` | Wilson (1927) |
| `improve` | McNemar's paired test: mid-p below 25 discordant pairs, uncorrected asymptotic above | `calibrated_agent_evaluation.stats.mcnemar.mcnemar_paired` | McNemar (1947); Fagerland, Lydersen and Laake (2013) |
| `improve` | Holm correction over the guard panel at a family-wise alpha of 0.05 | `calibrated_agent_evaluation.improve.panel.guard_panel` | Holm (1979) |
| `prefer` | Bradley–Terry ratings fitted by Hunter's MM iteration, 95% Wald intervals from the Fisher information under a sum-to-zero constraint; refused when the win graph is not strongly connected | `calibrated_agent_evaluation.subjective.bradley_terry` | Bradley and Terry (1952); Hunter (2004); Ford (1957) |
| `probe` | pass^k over repeated trials, and an exact Clopper–Pearson interval on the per-trial rate | `calibrated_agent_evaluation.probe.pass_hat_k`, `calibrated_agent_evaluation.probe.clopper_pearson` | Yao et al. (2024); Clopper and Pearson (1934) |

The Wasserstein distance is `scipy.stats.wasserstein_distance` divided by the pooled sample's
interquartile range. The project cites no primary source for it beyond the SciPy implementation.
A significant Wasserstein distance flags only above 0.10 (a tenth of the interquartile range),
so a negligible shift on a very large sample is not flagged. The null flag rate and the power at
four shift sizes are in `fixtures/expected/drift/null_calibration.txt`.

## Measured coverage of the corrected-rate interval

The corrected-rate interval's coverage was measured by Monte Carlo rather than assumed. The study
is `calibrated_agent_evaluation.stats.studies.coverage`, and its results are committed under
`fixtures/expected/coverage/`. `tests/test_coverage_map.py` re-measures named cells so the files
cannot drift from the shipped estimator.

| File | Cells | Replicates per cell | `rg_interval` coverage, lowest to highest | Cells off 0.95 after Holm |
| ---- | ----- | ------------------- | ----------------------------------------- | ------------------------- |
| `rg_interval_coverage.json` (TPR = TNR) | 168 | 2000 | 0.932 to 0.962 | 0 of 168 |
| `rg_interval_coverage_asymmetric.json` (TPR differs from TNR) | 840 | 2000 | 0.9295 to 0.965 | 0 of 840 |

The grid crosses expected judge accuracy 0.85 to 0.98, the per-class sizes the planner returns,
prevalence 0.02 to 0.50, and rate-sample sizes 100 to 5000. A cell is off nominal when an exact
binomial test of its covered count against 0.95 rejects after Holm across the file.

These figures come from 2,000 replicates per cell. A later rerun of the planner's floor grid at
3,000 replicates per cell found under-coverage where a class shows about two to three judge errors,
including two configurations the shipped floor admits. Those cells, and the 150-trace rate-sample
minimum they led to, are on [Reviewer notes](reviewer-notes.md).

The same files record a naive Wilson interval on the uncorrected rate as a comparison arm. Its
coverage of the true rate ranges from 0.0 to about 0.96 over the symmetric grid, because it estimates
the judge's flag rate rather than the failure rate.

The planner's coverage floor (`plan_calibration`'s `coverage_floor`) comes from a separate scan,
`planner_floor_scan.json`, summarized in `planner_floor.json`:

| Expected accuracy | 0.85 | 0.88 | 0.90 | 0.92 | 0.95 | 0.98 | 0.99 | 0.995 | 0.999 | 1.00 |
| ----------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- | ---- |
| Floor, per class  | 28   | 28   | 35   | 80   | 80   | 130  | 162  | 11    | 11    | 11   |

The rows up to 0.99 were derived at 1,000 replicates per cell. The rows at 0.995, 0.999 and 1.0
were scanned at 3,000 replicates per cell with the boundary rule in place and pass on coverage at
least nominal; an accuracy between 0.99 and 0.995 takes the larger floor, 162.
[Reviewer notes](reviewer-notes.md) give their provenance.

## Prior work on correcting LLM-judge rates

Correcting a judge-labeled rate with a small human-labeled sample was worked out for the LLM-judge
case in 2026. This library implements a version of that correction and does not claim a new estimator.

- Lee et al. (2026) build intervals for the Rogan–Gladen-corrected judge rate that carry
  uncertainty from both the test set and the human-labeled calibration set, and add adaptive
  allocation of the calibration sample.
- Lang and Reiczigel (2014) derived confidence limits for prevalence adjusted for estimated
  sensitivity and specificity, the same problem in veterinary epidemiology.
- Feng et al. (2026) treat the question as a hypothesis test with finite-sample Type-I error control
  under calibration uncertainty. The gate here thresholds a Wilson bound instead and does not have
  that guarantee.
- Chen et al. (2026) characterize the conditions under which a prediction-powered estimator
  (Angelopoulos et al. 2023) has strictly smaller asymptotic variance than a measurement-error
  correction. The result is conditional. The Rogan–Gladen interval here is not variance-optimal,
  and the library ships no prediction-powered estimator.
- Martinon, Merad and Raki (2026) package prediction-powered estimators and samplers as a library,
  GLIDE.
- Dorner, Nastl and Hardt (2025) show that when the judge is no more accurate than the model under
  evaluation, no debiasing scheme halves the human labels required.

## References

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I. and Zrnic, T. (2023). Prediction-powered inference. *Science* 382(6671):669–674. [doi:10.1126/science.adi6000](https://doi.org/10.1126/science.adi6000)
- Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics* 29(4):1165–1188. [doi:10.1214/aos/1013699998](https://doi.org/10.1214/aos/1013699998)
- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. *Technometrics* 24(4):295–300. [doi:10.2307/1267823](https://doi.org/10.2307/1267823)
- Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika* 39(3/4):324–345. [doi:10.2307/2334029](https://doi.org/10.2307/2334029)
- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science* 16(2):101–133. [doi:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)
- Chao, A. (1984). Nonparametric estimation of the number of classes in a population. *Scandinavian Journal of Statistics* 11(4):265–270.
- Chao, A. (1987). Estimating the population size for capture-recapture data with unequal catchability. *Biometrics* 43(4):783–791. [doi:10.2307/2531532](https://doi.org/10.2307/2531532)
- Chen, Y. T., Lu, S., Li, S., Guo, M. and Li, S. (2026). Efficient inference for noisy LLM-as-a-judge evaluation. arXiv:2601.05420.
- Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika* 26(4):404–413. [doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404)
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement* 20(1):37–46. [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104)
- Dorner, F. E., Nastl, V. Y. and Hardt, M. (2025). Limits to scalable evaluation at the frontier: LLM as judge won't beat twice the data. ICLR 2025. arXiv:2410.13341.
- Esty, W. W. (1983). A normal limit law for a nonparametric estimator of the coverage of a random sample. *The Annals of Statistics* 11(3):905–912. [doi:10.1214/aos/1176346256](https://doi.org/10.1214/aos/1176346256)
- Fagerland, M. W., Lydersen, S. and Laake, P. (2013). The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. *BMC Medical Research Methodology* 13:91. [doi:10.1186/1471-2288-13-91](https://doi.org/10.1186/1471-2288-13-91)
- Feng, C., Shen, M., Balashankar, A., Gerner-Beuerle, C. and Rodrigues, M. R. D. (2026). Noisy but valid: robust statistical evaluation of LLMs with imperfect judges. ICLR 2026. arXiv:2601.20913. <!-- writing-standards: ignore -->
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin* 76(5):378–382. [doi:10.1037/h0031619](https://doi.org/10.1037/h0031619)
- Fleiss, J. L., Cohen, J. and Everitt, B. S. (1969). Large sample standard errors of kappa and weighted kappa. *Psychological Bulletin* 72(5):323–327. [doi:10.1037/h0028106](https://doi.org/10.1037/h0028106)
- Ford, L. R. Jr. (1957). Solution of a ranking problem from binary comparisons. *The American Mathematical Monthly* 64(8):28–33. [doi:10.2307/2308513](https://doi.org/10.2307/2308513)
- Good, I. J. (1953). The population frequencies of species and the estimation of population parameters. *Biometrika* 40(3–4):237–264. [doi:10.1093/biomet/40.3-4.237](https://doi.org/10.1093/biomet/40.3-4.237)
- Hayes, A. F. and Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. *Communication Methods and Measures* 1(1):77–89. [doi:10.1080/19312450709336664](https://doi.org/10.1080/19312450709336664)
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6:65–70.
- Hunter, D. R. (2004). MM algorithms for generalized Bradley–Terry models. *The Annals of Statistics* 32(1):384–406. [doi:10.1214/aos/1079120141](https://doi.org/10.1214/aos/1079120141)
- Kish, L. (1965). *Survey Sampling*. Wiley.
- Korn, E. L. and Graubard, B. I. (1998). Confidence intervals for proportions with small expected number of positive counts estimated from survey data. *Survey Methodology* 24(2):193–201.
- Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology*, 2nd ed. Sage.
- Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. Annenberg School for Communication, University of Pennsylvania, departmental paper.
- Lang, Z. and Reiczigel, J. (2014). Confidence limits for prevalence of disease adjusted for estimated sensitivity and specificity. *Preventive Veterinary Medicine* 113(1):13–22. [doi:10.1016/j.prevetmed.2013.09.015](https://doi.org/10.1016/j.prevetmed.2013.09.015)
- Lee, C., Zeng, T., Jeong, J., Sohn, J. and Lee, K. (2026). How to correctly report LLM-as-a-judge evaluations. ICML 2026. arXiv:2511.21140.
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory* 37(1):145–151. [doi:10.1109/18.61115](https://doi.org/10.1109/18.61115)
- Martinon, G., Merad, I. and Raki, M. (2026). Industrializing prediction-powered inference: the GLIDE library for reliable GenAI and agentic systems evaluation. ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems. arXiv:2605.31278.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2):153–157. [doi:10.1007/BF02295996](https://doi.org/10.1007/BF02295996)
- Newcombe, R. G. (1998). Improved confidence intervals for the difference between binomial proportions based on paired data. *Statistics in Medicine* 17(22):2635–2650.
- Newcombe, R. G. (1998). Interval estimation for the difference between independent proportions: comparison of eleven methods. *Statistics in Medicine* 17(8):873–890.
- Rogan, W. J. and Gladen, B. (1978). Estimating prevalence from the results of a screening test. *American Journal of Epidemiology* 107(1):71–76.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158):209–212. [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
- Yao, S., Shinn, N., Razavi, P. and Narasimhan, K. (2024). τ-bench: a benchmark for tool-agent-user interaction in real-world domains. arXiv:2406.12045.
- Ye, J., Wang, Y., Huang, Y. et al. (2025). Justice or prejudice? Quantifying biases in LLM-as-a-judge. ICLR 2025. arXiv:2410.02736.
- Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer* 3(1):32–35.
- Yurdakul, B. and Naranjo, J. (2020). Statistical properties of the population stability index. *The Journal of Risk Model Validation*. [doi:10.21314/JRMV.2020.227](https://doi.org/10.21314/JRMV.2020.227)
