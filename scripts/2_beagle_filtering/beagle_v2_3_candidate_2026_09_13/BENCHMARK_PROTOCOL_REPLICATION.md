# Prespecified Beagle v2.3 three-seed selection gate

Frozen before seeds `20260916` and `20260917` were run. This is a
development-set selection protocol: matched P4/P6 WES is used only by the
assessor and is never passed to Beagle or the production router.

## Eligible inputs

Only configurations admitted by the checksum-verified factorial, modelscale,
and maxlr seed-`20260915` screens are replicated. The comparator
`modelscale=2,niterations=0,maxlr=5000` is mandatory. Each
configuration is evaluated at seeds `20260915`, `20260916`, and `20260917` in
P4 and P6. Raw input, executable, reference ledger, non-seed parameters, exact
allele universe, and routing code must match across runs.

## Replicated eligibility

The comparator is the fail-safe default. An alternative may replace it only if
all of the following hold after pooling the fixed 3-seed x 2-sample grid:

1. minimum within-setting, within-sample anchor Jaccard is at least 0.99;
2. minimum route agreement is at least 0.998;
3. median anchor Jaccard is no more than 0.002 below the comparator;
4. median P(nonref) MAE is no more than 0.002 above the comparator;
5. WES-somatic anchor leakage does not increase in any matched seed/sample;
6. mean normal-WES anchor sensitivity is no more than 0.005 below the
   comparator;
7. mean selected-anchor normal-WES support is no more than 0.005 below the
   comparator; and
8. the alternative improves mean sensitivity or mean support by at least
   0.002, or reduces total WES-somatic anchor leakage.

The last three rules reapply the seed-`20260915` no-harm/material-gain logic to
the full replicated grid. They prevent an arm that passed once but degrades at
the other seeds from winning on leakage alone.

Among eligible alternatives and the comparator, choose lower total somatic
leakage, then higher normal-WES sensitivity, higher selected-anchor support,
higher seed stability, lower runtime, and the simpler configuration. If no
alternative is eligible, retain the comparator.

P4/P6 WES participates in development-set selection, so these values are not
described as independent or prospective validation in the paper.

## Mandatory downstream clustering gate

Run DLPFC STAGATE followed by mclust once for the final selected Beagle
configuration, after all Beagle and ensemble decisions are frozen. Compare it
with the accepted old Beagle downstream matrix under the fixed clustering
protocol and require non-inferiority. This gate is required even when the
selected and old `all_panel_present` matrices are byte-identical; in that case
the identity is reported and the new run serves as an end-to-end regression.
Failed screen arms do not trigger STAGATE runs.
