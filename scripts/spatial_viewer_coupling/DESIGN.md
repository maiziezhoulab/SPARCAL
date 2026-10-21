# Spatial filter ↔ viewer coupling v1 — frozen design, 2026-09-16

## Scope and freeze

This additive experiment preserves allele evidence and makes feedback explicit and bounded.
It does not change production, Beagle, existing studies, matrices or call sets. Code is under
`scripts/spatial_viewer_coupling/`; all generated files, including logs, are under
`data/spatial_viewer_coupling_2026-09-16/`. This file is written before implementation and
will not be retrospectively edited. Corrections/deviations go into separate dated addenda.
No filter is rerun. Read-only imports reuse the viewer NumPy kernels and watershed.
The production `run_spatial_to_viewer.py` is neither executed nor copied.

## Contract A: `SPARCAL_variant_evidence_v1`

A directory contains `manifest.json`, `variants.npz`, `spots.npz`, `ref.npz`, `alt.npz`,
`record_presence.npz`, `graph.npz`, and optional `phase.npz`. All NPZ files disallow pickle.
Arrays in variants have length V; spot arrays have length S; allele CSR matrices have shape
(S,V). IDs are unique Unicode strings, exact `chrom_pos_ref_alt` (1-based position), never
position-only keys. Genome build is mandatory. CSR counts are uint32, sorted unique columns,
nonnegative; indptr/indices are integer. Omitted CSR entries mean zero observed reads, never
homozygous reference or negative tissue. A separate per-spot `ad_available` bool distinguishes
missing VCF from a measured all-zero row. `record_presence` uint8 is a compatibility field
whose 1 means a VCF record exists, including AD=0; it is never used for inference.

`variants.npz` required fields:
- `variant_key` U, `chrom` U, `pos` int64, `ref_allele` U, `alt_allele` U, `race` U;
- `posterior` float64[V,5], in ordered states artifact, germline_het, germline_hom_alt,
  somatic_subclonal, somatic_clonal; finite, nonnegative, sum=1;
- `information`, `reliability`, `weight` float64[V], each [0,1], weight=information*reliability;
- `class_id` uint8[V] = posterior argmax (0..4), `selected` bool[V], `eligible` bool[V];
- `qc_flags` uint16[V]: bit0 xMHC veto, bit1 editing-compatible substitution,
  bit2 >1 candidates within ±10bp, bit3 total AD depth<3, bit4 one ALT-positive spot;
- `base_logits` float64[V,5], `feedback_log_evidence` float64[V];
- `channels` float64[V,3,5]: bounded molecular, frozen spatial and CalicoST increments.

`spots.npz`: `barcode` U[S], `xy` float64[S,2], `umi` float64[S],
`ad_available` bool[S]. UMI missing values are explicitly NaN and median-imputed only inside
coverage computations; counts themselves are never imputed. Missing AD makes a spot ineligible
for feedback. Optional phase/CN/purity is declared by `optional_evidence` in manifest:
status `absent` plus reason, or `present` plus hashed sidecar and orientation/missing policy.
Unknown purity/CN is not zero/diploid. A frozen CalicoST channel contributes zero where no
context reliability exists; available contextual posterior ratios are descriptive evidence.
The phase adapter uses the existing sidecar's explicit purity/CN prior marginalization.

## Contract B: `SPARCAL_spatial_prior_v1`

Directory: `manifest.json`, `prior.npz`, `graph.npz`.
`prior.npz`: `barcode` U[S], `tumor_prior` float64[S] in [.01,.99], `valid` bool[S],
`region_id` int32[S] (-1 outside regions), `anchor_prior` float64[S],
`spot_evidence` float64[S]. The graph is a zero-diagonal CSR uint8[S,S] adjacency;
barcode order must agree exactly with A. `source` is one of `coverage-only prior`,
`phase-aware posterior`, `none`, `user-curated`; user-curated requires explicit label-free
attestation and remains disallowed in this registered experiment. Region membership is an
explanation/output, never a training label or a feedback gate.

Both manifests require: schema, section, genome_build, iteration, arm, created_utc,
`inputs` [{path:absolute,sha256:64hex,role}], `files` {relative_path:sha256}, parameters,
`provenance` {gt_free:true, gt_used:false, evidence_mode, calicost_participated:bool,
calicost_in_sample_descriptive_evidence:bool, independent_validation:false,
input_version_independent:true}, optional_evidence, elapsed_seconds, source_code hashes,
and design_sha256. Missing fields, inconsistent booleans, unknown sources, changed hashes,
invalid arrays or inconsistent ordering fail closed. Manifest itself receives a SHA256
completion marker, written last; existing bundle directories are never overwritten.
Input files are checked before and after each run. New upstream versions require new run IDs.

## Input adapters

P4/P6 frozen multistate tables contain all five-state posteriors. They also contain an
ALT/depth cache whose depth is max(DP,sum(AD)); subtracting ALT would not give exact REF.
Therefore exact REF/ALT and record presence are read directly from existing spot VCFs, keyed
by allele, without calling/filtering variants. Frozen tables supply only evidence channels,
quality annotations and matched quotas. Existing all-class caches establish ordering.

DCIS1 has exact AD cache but no frozen multistate output in the supplied four-section tree.
Its explicitly named `dcis_ad_adapter` uses the existing pre-filter race/nn metadata and
fixed beta-binomial state marginal likelihoods (q grids: artifact .001/.005/.01/.02/.05;
het .3/.5/.7; hom .9/.97/.99; subclonal .02/.05/.1/.2; clonal .2/.3/.4/.5;
kappa 20/8/12/8/8), with existing fixed panel/denovo state priors. It has no CalicoST
channel; its controls cannot be described as the historical P4/P6 QB models. Production
membership defines only quota and a QC budget. This availability deviation is declared
before evaluation. For all sections xMHC is excluded from new matched sets and DP<3 count
cannot exceed the frozen baseline budget. Impossible quota is an error, not silent shrinkage.

## Bounded update and fixed point

Let p_v^mol, p_v^sp, p_v^cal be frozen state posteriors and q_v the panel/denovo state prior
([.08,.72,.16,.03,.01] / [.72,.16,.02,.08,.02]). Let c=(0,0,0,1,1).
Each posterior is floored at 1e-12 before logarithms. Channels are centered across states:
M=clip(center(log(p_mol/q)),-4,4), S=clip(center(log(p_sp/p_mol)),-1,1),
C=clip(center(log(p_cal/p_mol)),-2,2). Let h=min(number of covered spots,8)/8.
Frozen base L=log(q)+h*(M+S+switch_C*C). Context-unavailable C=0.
This bounds every state channel even if the frozen model was overconfident. No multiplication
by read depth or raw spot count occurs after h reaches one. These are generalized evidence
increments, not independent likelihood ratios; independence/calibration is not claimed.

Information I=(1-exp(-total_AD/10))*min(ALT-positive spots/3,1).
Reliability R=1-.5*editing-.25*dense, clipped [.25,1], and R=0 in xMHC.
For each variant, a_v is the uniform distribution over ALT-positive observed spots and d_v
over AD-covered spots. With zero support, the corresponding channel is zero.
E_v(pi)=lambda*h_v*tanh((a_v-d_v)^T(2*pi-1)), lambda=.5.
P_v(pi)=softmax(L_v+c*E_v(pi)). This expresses soft spatial enrichment conditional on
callability. pi is never thresholded into a tumor label. Each update rebases on L, rather
than accumulating its own previous posterior as new evidence.

The viewer computes t_s = sum_v O_sv*w_v*P_v(somatic) / sum_v O_sv*w_v,
where O=1[ALT>0], w=I*R fixed at ingestion. All eligible denovo SNPs contribute;
matched selection does not gate the loop, avoiding discontinuous top-N feedback.
J_s=min(sum O*w,8)/8; zero-denominator t_s=.5 and J_s=0.
Using the viewer's `coverage_prior` NumPy kernel as a row-stochastic neighbor smoother H,
F(pi)=sigmoid(logit(pi0)+2*H[J*(2*t-1)]). pi0 is the fixed anchor:
coverage: .1+.8*within-section midrank(H[UMI]); none: .5;
phase: existing phase-aware posterior clipped [.1,.9], missing spots neutral .5 and invalid.
Phase evidence already uses the sidecar's effective observation cap 8; phase/CNV/purity from
one fit remain in-sample descriptive and are never represented as independent validation.

pi_next=(1-d)*pi+d*F(pi), d=.5. For fixed input weights and valid mask,
||a-d||_1<=2, logistic derivative<=1/4 imply Lipschitz(F)<=lambda=.5.
The damped map therefore contracts by at most .75 in sup norm, has a unique fixed point,
and its remaining error is bounded by ||pi_next-pi||/(1-.75). Hard matching and watershed
are downstream diagnostics and are excluded from this proof. This guarantee does not imply
biological accuracy. Prior-off lambda=0 makes the map independent of pi, but anchor influence
remains explicit; the `none` arm also removes the anchor.

Freeze iteration 0 (E=0, no feedback) and iterations 1..3 (each consumes the previous frozen
B). K=3 is the evaluation point, never chosen from labels. Record sup change, posterior
change and matched Jaccard each step; convergence tolerance 1e-4 is diagnostic, with no
label-dependent stopping. Even if unconverged at K, report K as a bounded finite run.

Hyperparameters are scale/guardrail choices: caps 4/1/2, n_eff=8 to limit correlated counts;
lambda=.5 to guarantee contraction; damping=.5 to temper transient changes; anchor range
.1..9 avoids saturated priors; viewer cap2 limits extra odds to exp(2); K3 bounds cost.
None is fitted to GT. Identity-destroyed control applies a fixed seeded permutation of
variant identities to posterior/weight channels relative to allele columns, preserving depth
and graph; it is a control, not another fitted model.

## Runs and evaluation frozen before label access

Sections P6_rep1 (primary), P6_rep2, P4_rep1, DCIS1. Arms: coverage_Coff (primary),
coverage_Con, none_Coff, phase_Con (where native phase available), coverage_Coff_prioroff,
coverage_Coff_identity. Full factorial phase+Coff is not biologically a CalicoST-off arm
because native phase construction itself uses CalicoST; provenance must expose that.

Frozen comparators: production baseline, molecular-QB, spatial-QB, CalicoST-QB,
integrated-QB, consensus-QB; smooth coverage mandatory. Reproduce historical binary-burden
comparisons separately from new AD-weighted spot/posterior outputs. Same evaluator universe,
viewer graph and detector for every method, with explicit missing-spot handling.
Detector grid seed 85/90/95 × grow 70/80/90 × size 3/5/10 × split .3/.4/.5 (243 cells).
Default 90/80/5/.4. No best-cell selection. Raw and linear-UMI-residual then smoothed F1
are reported (historical viewer convention); smooth coverage residual is computed by
residualizing its already-smoothed score against UMI, explicitly identified.

Paired block bootstrap: 2,000 replicates, square blocks of side four nearest-neighbor pitches;
resample blocks within above/below section-median GT-fraction strata to stabilize composition.
Spatial null: 999 whole-map translations on the observed Visium lattice, with nearest
coordinate wrapping only at domain bounds; preserve the label cluster pattern and report
mapping collisions/missingness. If irregular coordinates prevent a bijective spatial shift,
use equal-length contiguous spatial-block score permutations and explicitly report this
null approximation. No independent spot permutations. Primary contrast K3 versus coverage;
secondary K3 versus iteration0, production→consensus and molecular→CalicoST.

Support: exact-allele replicate Jaccard, callable-WES eligible union Wilson CI, QC fractions,
Spearman burden–UMI. P4/DCIS are context controls. Interaction is a paired difference of
model effects (P6 minus mean P4/DCIS) with section-stratified spatial block bootstrap;
P4's available copyKAT labels are silver and not interchangeable with pathology. Also test
interaction in change of absolute burden–UMI correlation with the same blocks. Three
contexts do not support population-level generalization. WES eligible union n=5 is supporting.

## Leakage and label-swap

Scoring/adapter modules have no evaluator imports or GT arguments. They read allowlisted
numeric/model inputs only; no StudyData constructor (which may load tumor_groups) is used.
Viewer methods are imported and called as pure kernels. Evaluator reads P6 pathology, WES
and DCIS GT only after completion markers exist. Each iteration is replayed with true,
complemented and seeded-permuted labels in an evaluator-owned context; replay calls receive
identical contract inputs, never labels. Compare numeric array hashes for posterior, selected
alleles, spot evidence, priors and regions. Record label-context hashes and verify evaluator
metrics change where labels change. A runtime file-open guard denies GT files during replay;
provenance and strict schemas reject undeclared label fields. This is a software leakage test,
not proof that legacy CalicoST inputs were independent of their own data.

## Falsification and decision

Drop implementation if any input/hash/schema violation is accepted, label-swap changes
inference, quotas drift, channel caps fail, missing CN becomes zero/diploid, or contraction
bounds fail. Reject localization improvement if K3 fails to beat smooth coverage on paired
raw and residual F1 with CIs supporting positive differences and grid stability. Reject an
identity-specific mechanism claim if identity destruction retains the gain. Reject context
specificity without a direct interaction estimate. Keep the contract/audit mechanism if
round-trip evidence is lossless, controls are operational and bounded replay is reproducible,
even if accuracy does not improve; do not promote a accuracy-neutral loop into production.
