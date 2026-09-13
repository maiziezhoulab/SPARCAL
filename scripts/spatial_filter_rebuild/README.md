# SPARCAL spatial-filter rebuild v1

This is a sidecar experiment. It does not import labels into scoring and does
not modify the production spatial filter, current call sets, matrices, or viewer
profiles.

## Frozen hypotheses

- `baseline`: current production somatic membership.
- `molecular`: AD/DP, strand balance, base/map quality, caller real-variant
  probability, local mismatch density, editing/xMHC risk, and replicate evidence.
- `spatial`: molecular plus coverage-conditioned join-count enrichment.
- `calicost`: molecular plus CNV/purity-aware VAF and clone enrichment.
- `integrated`: molecular plus both spatial and CalicoST evidence.
- `consensus`: robust rank consensus of the four preceding scores.

Every model computes expectations for artifact, germline heterozygous,
germline homozygous-ALT, somatic subclonal, and somatic clonal states. Population
panel membership is a soft prior, not a pre-filter. Missing CNV/purity is
marginalized by the existing mechanistic model; low information is shrunk toward
unresolved probability rather than called normal.

For each model, `matched` selects the same number of denovo SNPs as the frozen
baseline; `natural` uses the pre-registered posterior/information thresholds.
Viewer profiles use `matched` so set size cannot explain clustering differences.

## Additive reliability-v2 follow-up

The frozen v1 effective score shrinks weak five-state posteriors toward 0.5.
Although 0.5 is neutral in a binary problem, it is above the declared denovo
somatic prior (0.10) here. When a large matched quota is required, this causes
low-information loci to outrank well-measured non-somatic loci.

`score_reliability.py` tests a non-fitted correction without changing v1:
each component model is ranked by
`P(somatic | evidence) * observation_information`, followed by the same robust
rank consensus. New model names end in `_reliability`, so selected sets,
matrices, viewer studies, manifests, and evaluations remain collision-safe.

`evaluate_full.py` is also additive. It includes low-depth, xMHC, and
burden--UMI metrics, Wilson confidence intervals, hypergeometric enrichment,
and exact paired silver-standard tests. It refuses to write into a non-empty
evaluation directory.

`score_quality_balanced.py` adds the final label-free experiment. It maximally
retains each frozen CalicoST/integrated/consensus matched set while enforcing
an xMHC veto and a section-specific DP<3 ceiling equal to the frozen baseline.
Removed loci are filled with the corresponding reliability score. The separate
`score_quality_balanced_ablation.py` supplies molecular/spatial controls under
the identical constraints.


## Leakage boundary

All build/feature/score/select modules have no WES or pathology input.
`evaluate.py` and `evaluate_full.py` are the only stages that read either. The
label-swapping unit test verifies production somatic/UPV/ambiguous labels cannot
change any new-model score.

## Run

```bash
bash run_slurm/spatial_filter_rebuild/submit_all.sh
```

The dependency chain is cache/features -> scoring -> matrices/viewer -> external
evaluation. Outputs are namespaced under
`data/spatial_filter_rebuild_2026-09-12/multistate_v1_20260912/`; viewer studies
are under `viewer/dist/studies/sf_multistate_v1_20260912/` in the benchmarking
repository. No stage overwrites an existing result.

