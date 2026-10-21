# Frozen Beagle v2.3 selection protocol

Frozen before the clean factorial screen. This is a development-set protocol,
not a prospective preregistration: P4/P6 WES evidence and the earlier mixed-code
iteration screen informed the endpoints and margins. WES is evaluation-only;
no production command or routing decision receives WES data.

## Fixed design

- Input: the same frozen P4/P6 `baseQ0mapQ0` raw candidate VCFs.
- Reference: the bound 88-artifact ledger
  `beagle_reference_panel_checksums_20260913_v3_frozen`.
- Evidence rule: `P(nonref) >= 0.90` is the only hard germline anchor.
- Screen seed: `20260915`.
- Factorial configurations:
  - `modelscale=2`, `niterations=0` (current SPARCAL/Monopogen parameters);
  - `modelscale=2`, `niterations=5` (Beagle default iterations);
  - `modelscale=0.8`, `niterations=0` (Beagle default modelscale);
  - `modelscale=0.8`, `niterations=5` (both Beagle defaults).
- `impute=false`, `maxlr=5000`, `gprobs=true`, one Beagle process per task,
  and no genetic map are fixed. Imputation and map ablations are excluded
  because this GL-refinement pass does not add reference-only markers.
- The Monopogen second `gt=` pass remains an external phase-orientation
  comparison, not a candidate route: its first-pass GP and second-pass record
  universes/unordered genotype classes were already shown to be identical.

## Fail-closed run eligibility

Every sample/configuration must pass the v2.3 semantic validator, including:

- all 22 contig states accounted for;
- exact `raw intersect panel == raw intersect Beagle` allele identity;
- raw-record conservation across the four routes;
- usable tabix indexes;
- full output, source, executable, input, and reference-ledger identity checks;
- no fallback and no WES production input.

An ineligible run is retained for audit but contributes no performance result.

## Screen and replication gates

The `modelscale=2,niterations=0` arm is the comparator. An alternative advances
to seeds `20260916` and `20260917` only if, at seed `20260915`:

1. WES-somatic anchor leakage does not increase in either P4 or P6;
2. mean P4/P6 normal-WES anchor sensitivity is no more than 0.005 below the
   comparator;
3. mean selected-anchor normal-WES support is no more than 0.005 below the
   comparator; and
4. it improves either mean sensitivity or mean support by at least 0.002, or
   reduces somatic leakage, unless it is required as the one-factor ablation.

The comparator is always replicated. A replicated alternative is eligible as
the default only if:

- minimum within-setting, within-sample anchor Jaccard is at least 0.99;
- minimum route agreement is at least 0.998;
- its median anchor Jaccard is no more than 0.002 below the comparator;
- its median P(nonref) MAE is no more than 0.002 above the comparator; and
- no seed/sample increases WES-somatic anchor leakage relative to the matched
  comparator.

If no alternative passes, retain `modelscale=2,niterations=0`. If more than one
passes, prioritize lower somatic leakage, then normal-WES sensitivity, selected
anchor support, seed stability, runtime, and finally the simpler/faster arm.

## Clustering gate

Beagle posterior parameters should not change the `all_panel_present` feature
universe when the exact marker contract passes. First compare downstream
matrices exactly. If they are identical to the accepted frozen matrix, the
existing 120-restart-per-arm DLPFC regression is inherited and STAGATE is not
rerun. If matrices differ, run the strict 120-restart clustering gate once for
the final selected configuration; do not run it for failed screen arms.
