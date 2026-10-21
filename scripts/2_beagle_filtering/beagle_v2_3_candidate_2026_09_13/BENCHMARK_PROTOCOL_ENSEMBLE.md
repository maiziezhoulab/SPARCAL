# Prespecified Beagle v2.3 posterior-ensemble extension

Frozen before the first clean full-genome screen result was available. This is
a development-set protocol. Matched P4/P6 WES is used only to evaluate fixed
rules and is never a production input.

## Question

Beagle 4.1 uses a random seed and the output posterior varies slightly between
runs. The clean workflow therefore evaluates whether a small fixed ensemble can
replace a single direct Beagle run with measurably better accuracy or stability.
This extension does not request any additional run beyond the three seeds
already required for the final stability assessment.

## Fixed rules

For each exact allele, collect `P(nonref)=GP(0/1)+GP(1/1)` from seeds
`20260915`, `20260916`, and `20260917` under one otherwise identical
configuration. Compare:

1. **single-seed comparator:** seed 20260915, `P(nonref)>=0.90`;
2. **mean-posterior ensemble:** arithmetic mean P(nonref) across all three seeds
   is at least 0.90;
3. **median-posterior ensemble:** median P(nonref) across all three seeds is at
   least 0.90, equivalent to at least two seeds lying above the threshold when
   no value is exactly on the boundary.

No threshold is tuned and missing alleles are forbidden: the v2.3 marker
contract must establish an identical exact-allele universe in all three runs.
Panel-absent records remain panel-absent and are never converted into anchors.

## Advancement and cost rule

An ensemble may replace the eligible single-run configuration only if:

1. WES-somatic anchor leakage does not increase in either P4 or P6;
2. mean P4/P6 normal-WES anchor sensitivity is no more than 0.005 lower;
3. mean selected-anchor normal-WES support is no more than 0.005 lower;
4. it improves mean sensitivity or support by at least 0.002, or reduces
   somatic leakage; and
5. the three leave-one-seed-out ensemble anchor sets have minimum within-sample
   Jaccard at least 0.995.

If neither fixed ensemble passes, production remains one fixed seed. If both
pass, prioritize lower somatic leakage, then sensitivity, support, leave-one-out
stability, and finally the mean-posterior rule. The 3-fold runtime penalty must
be reported; an ensemble with no material gain is rejected even if numerically
stable.

Downstream `all_panel_present` matrices are compared exactly first. Posterior
aggregation should not change panel presence, but the final selected module
must still undergo the one-time end-to-end DLPFC STAGATE followed by mclust
gate in `BENCHMARK_PROTOCOL_DLPFC_FINAL.md`, even when its matrices are exactly
equal to the accepted matrix.
