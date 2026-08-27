# DLPFC spatial-clustering signal mechanism

**Status:** COMPLETE. All completion gates passed; final source tables and figure inputs are
frozen and non-overwriting.

**Analysis date:** 2026-08-26
**Cohort:** 12 spatialLIBD DLPFC sections; four serial sections from each of three donors
**Biological replicate:** donor (n=3), not section or optimizer seed
**Primary question:** Why can a normalized panel-locus matrix approach gene-expression ARI when
Beagle-corrected pseudobulk genotype does not improve clustering?

## Executive answer

The DLPFC score is primarily a **spatially registered relative detection/expression profile**,
not a spot-level genotype signal and not a one-number coverage effect.

The conclusion is supported by five independent observations:

1. The preceding corrected-genotype experiment found mean ARI 0.351 for corrected dosage,
   0.347 for depth-stratified shuffled dosage, 0.365 for matched presence, and 0.412 for gene
   expression. Corrected genotype therefore added essentially nothing beyond the observation
   matrix.
2. In the present decomposition, projected expression on the identical 250-kb panel-bin basis
   reaches or exceeds the panel-depth representation, while REF and ALT counts both retain
   structure and spot-level allelic fractions have little layer or spatial association.
3. Panel detection breadth and depth are extremely correlated with RNA capture complexity
   across spots, but equalizing every spot's matrix norm retains most of the STAGATE score.
   Therefore magnitude contributes but is not required; the relative high-dimensional profile
   is the important part.
4. A feature-only embedding retains modest information, especially for expression, but the
   correctly registered tissue graph adds a large amount. Graph topology without molecular
   features is weak (donor-aware mean ARI 0.092), and reassigning coordinates while leaving each
   feature vector and layer label on its original spot collapses ARI to approximately zero.
5. Post hoc bin attribution shows diffuse detectability structure rather than a small corrected-
   genotype marker set. The highest 1% of total-depth bins contain about 20% of its layer-
   association mass, and the highest 10% contain about 53%.

The paper-safe statement is that panel-locus observation matrices preserve cortical-layer-
decodable anatomical information. The results do **not** validate individual germline calls,
show that inherited genotype differs across cortical layers, or establish genotype as the cause
of clustering.

## Relationship to the corrected-genotype experiment

This analysis extends, rather than replaces, the completed report at:

`/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc_genotype_aware_2026-08-24/RESULTS.md`

That experiment established three facts needed here:

- Beagle GT is section-level pseudobulk. Every spot receives the same corrected GT at a locus;
  GT can select or weight spot evidence but cannot create true spot-varying inherited genotype.
- The production matrix carries allele observation/detection, not corrected GT. Beagle-corrected
  genotype does not flow through the production combined VCF into the matrix.
- Corrected dosage failed the predeclared comparisons with matched presence, shuffled dosage,
  and gene expression. Relative REF+ALT depth was the strongest SNV-derived representation.

STAGATE is therefore used here as a **spatial decodability assay**. Computationally it accepts
any aligned numeric spot-by-feature matrix in `AnnData.X`; biologically, its output can only be
interpreted in terms of the matrix supplied. A high ARI from counts or presence is not evidence
that STAGATE received genotype.

## Cohort and fixed model

Sections 151507--151510 are donor 1, 151669--151672 donor 2, and 151673--151676 donor 3. Donors
1 and 3 have seven scored classes (six cortical layers plus white matter); donor 2 has five
observed label classes. The executable benchmark fits the number of observed classes, not seven
for every section.

Each new clustering cell uses five optimizer seeds. The fixed model uses a radius-150 spatial
graph, hidden dimensions `[512, 30]`, 1,000 epochs, learning rate 0.001, weight decay 0.0001,
gradient clipping 5, and mclust. Ordinary nonnegative matrices use per-spot
`normalize_total(1e4) + log1p`; signed residual and explicitly normalized controls use no
additional preprocessing. ARI is calculated only after fitting and is not used to construct a
matrix, graph, residual, or normalization.

## Representations and what each tests

| Representation | Construction | Mechanism tested |
|---|---|---|
| Gene expression | Ordinary Visium gene counts | Reference anatomical modality |
| Matched presence | Exact-locus binary observation, binned at 250 kb | Detection/zero pattern without count magnitude |
| Total depth | REF+ALT depth at matched loci, binned | Relative locus-coverage profile |
| REF depth | Total minus ALT | Whether ALT evidence is required |
| ALT depth | Alternate-read count | ALT detection/coverage, not GT dosage |
| Projected expression | Gene counts from genes overlapping matched loci, distributed to the same bins | Expression on the identical panel-bin basis |
| Expression-adjusted depth | Signed standardized residual after total UMI, detected genes, and same-bin projected expression | Remaining relative depth structure; label blind |
| Equal-row-norm profile | Unit-L2 row, then one matrix-wide `sqrt(p)` constant | Removes all spot-specific row magnitude while preserving direction |
| RNA capture scalars | Total UMI and detected genes | Whether RNA complexity alone is sufficient |
| Panel capture scalars | Panel depth, observed loci, nonzero bins, and post-log row norm | Whether panel complexity alone is sufficient |
| Combined capture scalars | RNA plus panel scalars | Whether six global capture variables are sufficient |

Projected-expression mapping is conservative: a matched locus is assigned to overlapping GRCh38
gene bodies; multi-gene overlaps split weight, and each gene's counts are distributed among its
panel bins in proportion to matched-locus count while preserving that gene's total count. Across
sections, approximately 73.3% of matched loci map to at least one gene body.

## Result 1: section 151507 mechanism pilot

The pilot was run before scaling the predeclared primary matrices. Values are means over five
optimizer seeds; `failure` means no ARI was invented for a rank-degenerate or numerically invalid
embedding.

| Representation | Correct-graph ARI | Equal-row-norm ARI | Feature-only ARI |
|---|---:|---:|---:|
| Gene expression | 0.482 | -- | 0.367 |
| Projected expression | 0.535 | 0.544 | 0.316; direction-only 0.291 |
| Total depth | 0.397 | 0.355 | 0.114; direction-only 0.140 |
| ALT depth | 0.379 | -- | 0.051 |
| REF depth | 0.315 | -- | 0.037 |
| Matched presence | 0.329 | 0.303 | 0.039; direction-only 0.043 |
| Expression-adjusted total depth | failure (5/5) | 0.184 | 0.092; direction-only 0.010 |

Two details matter. First, literal norm-one input across about 8,800 features was numerically too
small: many STAGATE embeddings had rank at least seven, but both EEE and mclust model search
failed. Multiplying every row by the same `sqrt(p)` constant changes neither direction nor equal
row norms and restored stable fits. The unscaled failures are retained as attempted cells; they
are not interpreted as loss of biological information.

Second, constant scaling is not responsible for the feature-only result. Scaled and unscaled
unit-norm SVD controls were nearly identical: total depth 0.140 versus 0.140, projected expression
0.291 versus 0.290, presence 0.043 versus 0.036, and residual 0.010 versus 0.011. The scale rescue
is specific to neural-network/mclust numerical conditioning.

The first held-out section, 151508, reproduced the central result before the remaining cohort was
examined: total depth 0.316 equal-row-norm versus 0.373 original, presence 0.312 versus 0.271,
projected expression 0.472 versus 0.456, and adjusted residual 0.207 versus an original 5/5
rank failure.

## Result 2: donor-wide primary and equal-row-norm grids

All six non-residual primary representations completed 60/60 fits. Means are first calculated
within section and then within donor; the SD is across the three donor means.

| Representation | Donor-aware mean ARI | SD across donors | Section range | Successful/attempted fits |
|---|---:|---:|---:|---:|
| Projected expression | 0.509 | 0.018 | 0.405--0.596 | 60/60 |
| Gene expression | 0.448 | 0.011 | 0.308--0.595 | 60/60 |
| Total depth | 0.408 | 0.058 | 0.296--0.585 | 60/60 |
| ALT depth | 0.400 | 0.088 | 0.308--0.543 | 60/60 |
| Matched presence | 0.360 | 0.078 | 0.257--0.554 | 60/60 |
| REF depth | 0.340 | 0.092 | 0.230--0.515 | 60/60 |

Projected expression exceeds ordinary gene expression in every donor (mean paired difference
+0.061 ARI), although with three donors the two-sided exact Wilcoxon P value can only be 0.25.
Total depth is below gene expression by mean 0.040 and is donor heterogeneous. ALT nearly matches
total depth, while REF and presence remain clearly structured. Together with the allelic-fraction
negative control, this ranking favors RNA detection/coverage anatomy over genotype dosage.

The equal-row-norm comparison removes all spot-specific row magnitude while preserving the
relative feature direction:

| Profile | Original ARI | Equal-row-norm ARI | Paired mean change | Donors higher after equalization |
|---|---:|---:|---:|---:|
| Projected expression | 0.509 | 0.511 | +0.002 | 1/3 |
| Total depth | 0.408 | 0.350 | -0.058 | 0/3 |
| Matched presence | 0.360 | 0.321 | -0.040 | 2/3 |

Thus equalization retains approximately 100%, 86%, and 89% of the projected-expression, total-
depth, and presence cohort means, respectively. Magnitude matters for total depth, but it is not
required for a substantial score. The profile direction itself carries most of the information.
At n=3 donors, none of the exact paired tests can establish a small effect; the donor means and
section points are therefore shown rather than promoted as high-powered inference.

The original signed residual is not a complete cohort endpoint: it succeeds in 35/60 attempts
across seven sections and fails all 25 attempts in the other five. The successful-section mean is
0.335 (range 0.215--0.593), but donor 1 has no valid original residual section, so its nominal
donor mean must not be compared with complete modalities. The scaled equal-row-norm residual
succeeds 60/60 with donor-aware mean 0.211 (SD 0.060; section range 0.054--0.293). This confirms
heterogeneous remaining spatial structure while showing that the largest original residual scores
partly depend on magnitude and numerical rank.

Post hoc score correlations reinforce the same interpretation. Across the available section-by-
modality combinations, ARI correlates with breadth-versus-total-UMI Spearman rho (rho=0.572,
P=5.15e-7; n=66) and breadth layer eta-squared (rho=0.543, P=2.42e-6; n=66). Mean within-section
rank correlations are 0.565 and 0.521, respectively. These are descriptive, partly dependent
comparisons, and eta-squared uses labels; they show what tracks score rather than establishing a
causal coefficient.

The equal-row-norm cohort extension rule was written down after the first valid 151507 presence
and total-depth cells, before any result from another section: run all four scaled unit-norm
representations in all remaining sections regardless of ARI or failure rate. This avoids choosing
only a favorable control.

## Result 3: capture complexity is informative but insufficient

Across sections, the number of detected panel bins/features is almost a monotone proxy for RNA
capture:

| Matrix breadth | Mean Spearman rho with total UMI | Mean layer eta-squared | Mean Moran's I |
|---|---:|---:|---:|
| Projected expression | 0.991 | 0.375 | 0.637 |
| Matched presence | 0.976 | 0.330 | 0.591 |
| Total depth | 0.975 | 0.320 | 0.579 |
| ALT depth | 0.971 | 0.315 | 0.572 |
| REF depth | 0.962 | 0.301 | 0.556 |

This confirms a major capture/detection confound, but the scalar controls show it is not the full
explanation. In 151507, four panel-capture variables produce five valid fits with mean ARI 0.134;
the six combined RNA+panel variables produce five valid fits with mean ARI 0.153. The two RNA-
only variables yield rank-5--6 embeddings and fail in all five seeds. A cubic-spline sensitivity
gives panel capture one valid seed at ARI 0.138 and combined capture four valid seeds at mean
0.177; RNA-only still fails 5/5. Because the nonlinear graph encoder can expand input rank, the
initial concern that fewer than seven input variables must fail was incorrect and is explicitly
corrected in the experiment log.

Scalar capture ARI is clearly above the coordinate-permutation null but well below full projected
expression and below total/presence profiles. Capture magnitude and tissue morphology are useful
proxies; they are not sufficient to explain the high-dimensional score.

The allelic-fraction negative control points in the same direction. Across sections, spot-level
REF/ALT fractions have only mean layer eta-squared 0.015 and mean Moran's I 0.018, far below the
breadth diagnostics above. Counts of alternate reads can still cluster because they carry a
detection/coverage profile; their fraction of allelic reads carries little layer structure.

## Result 4: projected expression and residualization

Approximately 17.6% of total-depth variance per bin is explained by same-bin projected expression
after adjusting total UMI and detected genes (section range approximately 13.9--19.8%). The
analogous REF-depth value is approximately 9.9%. Total depth and projected expression have only
modest median per-bin correlation (about 0.091) and median spot cosine similarity (about 0.365),
but their layer-centroid profiles are substantially more similar (about 0.541). Thus the matrices
are not numerically interchangeable, yet both align with the same anatomy.

The signed total-depth residual has low layer-centroid correlation with projected expression
(about 0.080) but its row magnitude remains strongly capture-associated. Original residual
STAGATE can collapse when the requested number of groups exceeds effective embedding rank.
Equal-row normalization changes that behavior and reveals heterogeneous residual spatial signal:
strong in some sections, weak or near random in others. This is evidence for remaining structured
relative profiles, not proof of genotype. Residualization is linear, expression projection is
incomplete, and unmeasured transcription, mapping, morphology, cell density, and sampling remain.

## Result 5: what the spatial graph adds

In 151507, the correct graph, a 30-dimensional feature-only SVD, and five coordinate-permuted
graphs were compared for the same representations:

| Representation | Correct graph | Feature only | Graph topology only | Coordinate-permuted graph |
|---|---:|---:|---:|---:|
| Gene expression | 0.482 | 0.367 | 0.082 | 0.015 |
| Projected expression | 0.535 | 0.316 | 0.082 | 0.005 |
| Total depth | 0.397 | 0.114 | 0.082 | 0.001 |
| Matched presence | 0.329 | 0.039 | 0.082 | 0.002 |

The coordinate control is often loosely called a spot permutation, but the exact operation is
more informative. For each of five replicates, coordinates are randomly reassigned among spots
before rebuilding the radius-150 graph. The feature vector and manual layer label remain attached
to their original spot. The same coordinate permutation is used across modalities and the model
seed is fixed at zero. Therefore the control changes only which feature rows are treated as
neighbors. Its collapse means arbitrary graph smoothing is not sufficient; mixing unrelated
layer profiles destroys the registered signal. It does not mean that the graph alone was tested,
and it does not identify genotype as the useful feature.

The graph-topology-only control was added adaptively to test what the coordinate permutation
cannot: whether real tissue adjacency is sufficient without molecular features. It uses the same
radius-150 graph, a fixed 30-dimensional spectral embedding, and five tied-covariance Gaussian-
mixture seeds; labels choose the number of groups and calculate ARI but do not enter either fit.
After the temporary 151507 pilot returned mean ARI 0.082, all 12 sections were committed before
examining another section. All 60 fits converged. The donor-aware mean was 0.092 (SD 0.017 across
three donors; section range 0.037--0.128). This is above the mismatched-graph null but far below
feature-plus-correct-graph expression scores. Because it uses a spectral embedding rather than
STAGATE, it is a topology decodability reference, not an architecture-matched ablation.

The feature-only control was extended without modality selection to all 12 sections. Its final
donor-aware differences are reported in the completion-gated source tables below.

| Feature-only representation | Donor-aware mean ARI | SD across donors | Successful/attempted fits |
|---|---:|---:|---:|
| Gene expression | 0.427 | 0.057 | 60/60 |
| Projected expression | 0.350 | 0.047 | 60/60 |
| Total depth | 0.134 | 0.048 | 60/60 |
| ALT depth | 0.105 | 0.060 | 60/60 |
| Matched presence | 0.089 | 0.046 | 60/60 |
| Expression-adjusted total depth | 0.073 | 0.019 | 60/60 |
| REF depth | 0.066 | 0.037 | 60/60 |

Feature vectors alone therefore retain substantial layer information for ordinary and
panel-projected expression, but much less for allele-observation matrices. This makes the graph
gain especially important for total depth and matched presence. All listed feature-only fits are
valid; their lower values are not a failure-handling artifact.

## Result 6: post hoc bin attribution

This diagnostic uses manual labels through per-bin eta-squared and is explicitly post hoc. It
does not select features or refit clustering.

| Total-depth comparator | Binwise eta-squared rho | Moran's-I rho | Top-1% eta-bin overlap | Top-10% overlap |
|---|---:|---:|---:|---:|
| ALT depth | 0.913 | 0.771 | 0.788 | 0.830 |
| Matched presence | 0.714 | 0.557 | 0.644 | 0.746 |
| REF depth | 0.550 | 0.374 | 0.623 | 0.602 |
| Projected expression | 0.340 | 0.182 | 0.225 | 0.361 |
| Expression-adjusted residual | -0.098 | 0.504 | 0.041 | 0.040 |

Total-depth feature eta-squared correlates with mean abundance/detectability (mean Spearman
approximately 0.559). Its layer-association mass is moderately concentrated but still diffuse:
the top 1%, 5%, and 10% of bins contain about 20.2%, 41.7%, and 53.5% of total eta-squared mass.
Projected expression has stronger per-bin layer association overall, and its spatial/layer rank
is only partly shared with total depth. ALT depth closely tracks total depth because ALT counts
at transcribed candidate loci are themselves a coverage/detection channel; weak allelic-fraction
diagnostics argue against interpreting this concordance as genotype specificity.

## Failure accounting and adaptive decisions

No failed cell is replaced by zero or omitted from its attempted denominator.

- Original expression-adjusted residual is rank-degenerate in all five 151507 seeds but succeeds
  in some five-class donor-2 sections; failure therefore depends on embedding rank and requested G.
- Literal norm-one high-dimensional inputs are numerically invalid for the current
  STAGATE-to-mclust scale. Constant `sqrt(p)` rescaling is label blind and preserves the intended
  equal-row-norm ablation.
- RNA-only scalar embeddings fail 5/5 in raw and spline forms. Panel and combined scalar controls
  have valid raw fits, correcting the pre-pilot assumption that low input dimension necessarily
  forces low embedding rank.
- REF-residual and positive-residual pilot sensitivities remain failed 5/5 at ranks two and three
  and were not promoted under their predeclared stability rule.

## Exact interpretation for the paper

### Supported

- Panel-supported allele-observation matrices contain cortical-layer-decodable spatial
  information.
- Much of that information is preserved after removing all spot-specific row magnitude.
- Relative genomic-bin profiles, detection breadth, expression/capture, and correct graph
  registration all contribute.
- The strongest scores require alignment between informative feature profiles and real
  neighborhoods; neither feature-only embeddings nor graph topology alone reaches them.
- A modest, heterogeneous expression-adjusted relative-depth component remains, especially after
  equal-row normalization.

### Not supported

- Beagle-corrected genotype improves DLPFC clustering.
- Germline genotype differs by cortical layer within a donor.
- DLPFC layer ARI validates individual germline variants.
- Scalar coverage alone explains the result.
- STAGATE or the spatial graph alone creates the observed ARI.
- Residual ARI is a uniquely genetic or causal biological signal.

### Recommended SI framing

Suggested subsection title:

> Spatially registered detection profiles, rather than corrected genotype, explain DLPFC layer
> decodability.

Suggested result logic: corrected pseudobulk genotype does not improve matched observation
matrices; projected expression on the same feature basis recapitulates the strongest signal;
equal-row-norm profiles retain it; global capture scalars are weaker; and coordinate-permuted
graphs collapse. This is a useful, honest SI mechanism result. In the main paper, DLPFC should be
described as a positive control for spatial information preservation, while germline call
validity should rest on orthogonal DNA concordance (P1-1), not clustering.

## Limitations

- There are three biological donors; 12 sections and five seeds are not 60 biological
  replicates.
- Manual layer labels are used for ARI and post hoc eta-squared attribution.
- Projected expression covers genes intersected by matched loci and does not measure every
  transcriptional or technical factor affecting RNA pileups.
- RNA allelic counts are affected by expression, allelic expression, sampling, mapping bias,
  morphology, and cell density.
- Feature-only SVD is a distinct reference, not an architecture-matched STAGATE ablation.
- Graph-only spectral embedding plus tied-covariance mixture is also a distinct adaptive
  reference, not a featureless STAGATE model.
- Coordinate permutation tests graph registration, not a featureless graph-only model.
- STAGATE stochasticity is summarized over seeds but GPU execution is not bit-reproducible.

## Source tables and code

Analysis root:

`/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc_spatial_signal_mechanism_2026-08-26/`

Final completion-gated tables are under `summary/`, including run-level attempts, section and
donor means, paired direction-only effects, graph/feature controls, diagnostic-score
correlations, failure counts, matrix audits, and the exact four panel source tables. Post hoc bin
tables are already under `feature_attribution/`.

Code root:

`/data/maiziezhou_lab/leiy4/snv_calling/clustering_benchmark/mechanism_2026_08_26/`

Key files:

- `build_mechanism_matrices.py`: REF/ALT, projected-expression, and residual matrices.
- `build_second_stage_controls.py`: unscaled unit-norm and scalar capture controls.
- `build_scaled_unitnorm_controls.py`: label-blind constant-rescaled direction-only controls.
- `build_scalar_spline_controls.py`: scalar basis sensitivity.
- `run_graph_feature_controls.py`: feature-only and coordinate-permutation controls.
- `compute_graph_topology_control.py`: adaptive molecular-feature-free graph spectral control.
- `compute_signal_diagnostics.py` and `compute_second_stage_diagnostics.py`: feature, spot, and
  spatial diagnostics.
- `compute_feature_attribution.py`: post hoc bin concordance and concentration.
- `summarize_mechanism.py`: donor-aware aggregation and completion gates.
- `plot_mechanism.py`: exact source tables and v5 figure assets.
- `../SPARCAL_clustering.py`: common STAGATE and guarded mclust driver.

The report figure dossier is:

`/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figures/fig-dlpfc-spatial-signal-mechanism.md`

The final figure assets are written at:

`/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v5/fig_dlpfc_spatial_signal_mechanism.{pdf,png}`

## Reproduction

From `/data/maiziezhou_lab/leiy4/snv_calling`:

```bash
python clustering_benchmark/mechanism_2026_08_26/summarize_mechanism.py
python clustering_benchmark/mechanism_2026_08_26/plot_mechanism.py
```

Both final steps refuse incomplete inputs; the summarizer and figure generator refuse to replace
nonempty/final outputs. Production matrices and earlier experiments were read-only inputs.

## Completion and scheduler audit

| Purpose | Slurm job(s) | Final state |
|---|---|---|
| 151507 primary pilot | 13607334 | completed, exit 0 |
| 151508--151676 primary cohort | 13608906 | 11/11 array tasks completed, exit 0 |
| 151507 coordinate permutations | 13607336 | completed, exit 0 |
| Feature-only pilot and cohort | 13607339, 13608914 | 12/12 sections completed, exit 0 |
| Raw and spline scalar capture | 13608952, 13608936 | completed, exit 0 |
| Unscaled/scaled feature-only direction checks | 13608963, 13608996 | completed, exit 0 |
| Scaled equal-row-norm pilot and cohort | 13608995, 13609006, 13609007 | 12/12 sections completed, exit 0 |
| Adaptive topology-only cohort | 13609259 | 12/12 array tasks completed, exit 0 |

The frozen aggregate contains 720 STAGATE attempts, 495 feature/permutation controls, and 60
topology-only fits. All six non-residual primary matrices have 12 sections x 5 successful fits;
all four scaled direction-only matrices have 12 x 5 successful fits. Original signed residual has
35 successes and 25 explicit rank failures. Graph topology has 60/60 converged fits.

SHA-256 hashes at final aggregation:

- `SPARCAL_clustering.py`: `00302cbac9689ca9530e41166e1f856cdb8c353a6f3a9e9d59d59395b9aa95a9`
- `mechanism_config.json`: `0f093ef4fed46ca9f534d8876e28fc197056bb2eff16ddfff5085cb5e673e401`
- `summarize_mechanism.py`: `aeaa8716b2da35fe63f3d73c986a1becf8c49c51659eaab94fee816d0dd63826`
- `plot_mechanism.py`: `e7e10eb7e9c8e7715cd72b23466c686483256109b3f9a2cd203b3df62988682c`
