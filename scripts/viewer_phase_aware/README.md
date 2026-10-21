# Phase-aware tumor evidence for SPARCALviewer

This is an additive, label-free sidecar. It does not modify the shipped
spatial filter and it never reads WES, silver-standard, or other outcome
labels. The output is a frozen per-spot posterior consumed by the viewer's
experimental `Phase-aware tumor posterior` method.

The model separates four natural-log evidence terms:

1. `phase_cnv_log_bf`: phased allele counts under CNV + purity versus balanced
   normal tissue;
2. `purity_log_prior`: CalicoST tumor fraction as an explicit occupancy prior;
3. `somatic_vaf_log_bf`: candidate VAF under genotype + CNV + purity;
4. `molecular_log_bf`: strand/start/UMI evidence;
5. `spatial_log_prior`: label-free spatial continuity.

All coefficients are one because the inputs are log likelihood ratios/log
odds, not arbitrary features. Fixed symmetric caps only prevent one correlated
source from dominating. `information` shrinks evidence toward the stated prior.
Missing components contribute zero log evidence. Missing purity and CNV in the
phase likelihood are marginalized over explicit priors, never replaced with
purity 0 or diploid CN.

## Phase safety

Use `--phase-cn-orientation aligned` only for counts and copy numbers sharing
the same upstream phase convention, such as CalicoST native B counts with
CalicoST A/B CN. Otherwise retain the default `marginalize`, which integrates
over the two A/B orientations.

SPARCAL's confident candidate phase is candidate-to-local-anchor molecular
phase. It is valid for that pair only. Beagle 4.1 versus 5.4 chromosome-wide
orientation concordance is approximately chance, so the chromosome-valued
`PS_BLOCK` must not be used to combine candidates attached to different
anchors. The current confident set covers only 897/128,872 candidates (0.696%).

## Build

The spot table must have one unique `barcode` row. It may additionally contain
the four evidence columns, `information`, and the boolean masks `valid`,
`phase_available`, and `purity_available`.

An optional long phase-observation table has columns:

`barcode, B, TOT, purity, copy_a, copy_b`

Multiple observations per spot are robustly aggregated with an effective
observation cap because neighboring genomic bins are correlated.

```bash
python scripts/viewer_phase_aware/build_tumor_evidence.py \
  spots.tsv NEW_PROFILE/tumor_evidence.npz \
  --phase-observations phase_bins.tsv \
  --phase-cn-orientation aligned \
  --model-name p6_rep1_phase_cnv_purity_v1 \
  --confirm-label-free
```

The command refuses to overwrite an existing output. Use a new profile/output
name for every ablation. Ground truth remains evaluation-only after the profile
has been frozen.

For a CalicoST-native phased-BAF/CNV/purity arm, build the viewer bundle in one
command (select the intended ploidy-specific CNV file explicitly):

```bash
python scripts/viewer_phase_aware/build_from_calicost.py \
  RUN/calicost/parsed_inputs \
  RUN/estimate_tumor_prop/loh_estimator_tumor_prop.tsv \
  RUN/calicost/clone3_rectangle0_w1.0/clone_labels.tsv \
  RUN/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv \
  NEW_PROFILE/tumor_evidence.npz \
  --model-name p6_rep1_calicost_phase_cnv_purity_v1 \
  --confirm-label-free --acknowledge-in-sample-calicost-evidence
```

This arm uses CalicoST BAF data together with CN/purity inferred from that same
run. It is appropriate for tumor selection and visualization, but its
phase/CNV likelihood is in-sample descriptive evidence, not independent
validation of CalicoST.
