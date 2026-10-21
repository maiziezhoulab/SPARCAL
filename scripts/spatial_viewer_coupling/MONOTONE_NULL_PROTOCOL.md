# Monotone coverage correction — 2026-09-16

This is an explicitly **post hoc correction prompted by the user's independent
review**, specified before running the corrected evaluation. DESIGN.md, the
inference bundles, evaluation_v1 and legacy evaluate.py stay byte-for-byte intact.
No inference, selected calls, detector parameters, GT or viewer profile is changed.
The public evaluate_registered.slurm entry now runs evaluate_monotone.py. For a
new run it first generates the legacy frozen score cache if absent, then always
applies this correction. An existing cache must have a valid seal, payload hashes,
code hashes and the same run ID; a partial cache fails closed.

## Nulls and processing

Let H be the existing viewer's one-hop coverage smoother, u the frozen UMI vector,
c=H(u), and R(x)=x-OLS([1,u],x), with the existing missing-UMI convention.

* rank_coverage = scipy.stats.rankdata(c, method='average').
* spline_coverage: x=log1p(c); PCHIP interpolates the empirical x quantiles
  at probabilities (0,.10,.25,.50,.75,.90,1) to those fixed probabilities.
  Duplicate x knots are merged using the mean of their probability ordinates;
  a constant input returns a constant. No extrapolation is needed. Knots use
  coverage alone, never labels, residuals or detector performance. This provides
  a smooth monotone empirical-CDF approximation, distinct from exact ranks.
  Both transforms must preserve all finite coverage ranks, including ties.
* Posterior arms, smooth_coverage, rank_coverage and spline_coverage have exactly
  the same detector paths: raw=x; coverage_residual=H(R(x)). In particular, the
  previously missing post-residual H for smooth_coverage is restored.
* Frozen comparator and matched-call burdens retain the historical paths
  raw=H(burden), coverage_residual=H(R(burden)). Their native input is an unsmoothed
  ALT-positive burden rather than a spatial posterior. Every row exports its
  native-score definition, raw chain and residual chain; this explicit distinction
  avoids silently changing the previously registered comparator endpoints.
* Native-score and raw-detector-score Spearman versus c are exported for every
  arm; Pearson, maximum absolute rank difference and number of eligible spots are
  also exported. Correlations on the full lattice and valid-only lattice are
  distinguished.

The fixed default detector remains (90,80,5,.4). Both modes of **every** score arm,
including intermediate k1/k2 and both nulls, run all 81 registered grid settings.
There is no GT-based model, knot, null, K or detector selection. Comparisons are
reported separately against both nulls, not a winner selected from their results.

## Statistics and audit

All default-detector delta F1, paired 95% percentile block-bootstrap intervals
(2,000 draws), one-sided improvement permutation p and two-sided permutation p
(999 draws) use rank_coverage or spline_coverage as reference. The same resampling
draws are shared across arms. Spatial cells have side four nearest-neighbor pitches;
bootstrap draws are stratified by above/below median positive fraction as in v1.
The spatial null permutes whole equal-size 32-spot blocks ordered by spatial cells,
with the final incomplete block fixed. It preserves labels bijectively but can
break across-block clusters. These are section-conditional, unadjusted p-values,
not familywise evidence across the grid or multiple arms.

Every grid point also exports paired F1 differences versus both nulls; grid
sensitivity reports the fraction positive and min/median/max differences. It is
sensitivity analysis, not permission to select a winning grid point. Confidence
intervals and permutation tests concern the fixed default detector only.

P6-minus-mean(P4,DCIS) model-by-context F1 interactions are recomputed versus each
null for all shared arms, using independent section bootstrap draws. P4 is still
copyKAT silver, not pathology; P6_rep2 has no pathology metric. Replicate, WES,
intrinsic QC, call burden–UMI and frozen iteration replay results are unchanged
because no inference/call set changes. The original 264 label-swap replays are
verified by hash and referenced, not represented as newly rerun inference.
New coverage transforms and score processing undergo true/complement/permuted
label-context replay with numerical file I/O forbidden, including a positive
control for that guard. The transform API accepts no labels.

All consumed score caches, B masks, graph, labels' source provenance, legacy and
new code, this protocol, spline knots, processed maps, prediction masks and
statistical tables are recorded in a new sealed evaluation directory. Original
GT inputs are still read only by evaluators. New Beagle versions require fresh
run/cache IDs; no version-specific score values are baked into the nulls.

## Interpretation and falsification

The k0 coverage prior is an affine transform of coverage ranks. Exact rank
agreement and matching rank-null detector masks would explain its residual F1
without new tumor biology. Linear residualization against raw UMI is not invariant
to monotone score transformations; it is not a test of coverage independence.
An apparent gain that disappears against these nulls is withdrawn as precision
evidence. Even improvement against both particular nulls would not establish
independence from all functions of coverage. Contract fidelity, bounded feedback,
frozen replay and measurable circularity are separate mechanism properties.

The report must answer whether any of the six frozen comparators beats either
null at the fixed detector, and distinguish that answer from isolated grid wins.
No conclusion in report/09_18_record.md is edited.
