# SPARCAL step 7 redesign: per-variant posterior over generative hypotheses

Status: DRAFT v0 (2026-10-08), for review. Target: design frozen by 2026-10-11.

## 1. Why replace the current step 7

Current step 7 (`scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py`, `classify_variants`):
- Stage 1: a de novo variant with `spatial_uniformity > 0.5` becomes UPV (germline_denovo).
- Stage 2: among the rest, each feature (delta = purity correlation, epsilon = clustering x purity,
  zeta = spatial clustering, theta = CNV consistency) gives one vote to its top 20%; the top 10% of
  non-germline candidates by vote count become somatic and the rest become unresolved.

Observed failures:
1. The somatic class size is a fixed 10% quota, not a data-driven quantity.
2. Ranks discard magnitude; the 20% and 10% cut-offs are arbitrary.
3. Massive vote ties. Legacy code broke ties by set iteration order: two runs on the same input
   (P4 rep2, 2026-09-07) share only 1,598 of 8,065 somatic calls. The current deterministic
   tie-break fixes reproducibility but not the arbitrariness.
4. Features use presence (ALT >= 1) only. A covered spot with REF reads and no ALT is evidence of
   absence, but is treated like an uncovered spot, so expression level is confounded with spatial
   clustering.
5. theta has a documented scoring-direction bias (copy-neutral fixed at 0.5), which penalises calls in
   CNV/LOH segments.
6. No per-call uncertainty.

External evidence (P4/P6, `SPARCAL_Benchmarking/analysis/caller_panel_somatic_2026-10-08/`):
adding CalicoST to the vote improves no external metric and lowers rep1-to-rep2 recurrence.

## 2. Data available per variant (no upstream rerun needed)

For each de novo candidate v (non-1KGP, passed steps 1-6) and each in-tissue spot s:
- a_vs, n_vs: ALT and total UMI-deduplicated read counts from per-spot `AD` (already loaded into
  `snv_alt_counts`, `snv_ref_counts`). Very sparse: most spots have n_vs = 0.
- Spot coordinates and the hexagonal neighbour graph.

From CalicoST (hg19 rerun, `/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906/{P4,P6}_sec{1,2}`):
- pi_s: tumor proportion per spot (`estimate_tumor_prop/loh_estimator_tumor_prop.tsv`). Missing for
  P4 rep2 (619/648 NaN, 29 zero); `normal_candidate_barcodes.txt` exists there.
- Clone labels and soft clone posteriors (`posterior_clone_probability.npy`).
- Allele-specific copy number (A_k, B_k) per clone and segment (`cnv_seglevel.tsv`).
- Realised clones: 1 in P4 rep1, P6 rep1, P6 rep2; 2 in P4 rep2.

Labelled reference sets for fitting nuisance parameters (no somatic labels exist):
- 1KGP-resolved germline heterozygous sites (germline_defined): positives for the germline model.
- Sequence-error class from step 4: positives for the artifact model.
- Published matched-WES somatic SNVs (683 P4 / 386 P6, few observable in RNA): evaluation only.

## 3. Proposed model

For each variant, compare hypotheses by marginal likelihood over spots; spot terms are
beta-binomial(n_vs, f_vs, rho) with a global overdispersion rho, fitted on the reference sets, to absorb
allele-specific expression and UMI noise.

Expected ALT fraction f_vs by hypothesis, where CN_s = pi_s (A_k + B_k) + 2 (1 - pi_s) is total copy number
in the spot mixture:

- **H_G, private germline heterozygous (UPV).** Allele on haplotype h in {A, B}, prior 1/2 each:
  f = [ (1 - pi_s) + pi_s h_k ] / CN_s, where h_k is that haplotype's copy number in clone k(s).
  LOH falls out naturally (h_k = 0 or 2+). Present everywhere, scaled by copy number.
- **H_S, somatic in tumor cells.** Cancer cell fraction c in (0, 1] on a grid, multiplicity m = 1:
  f = pi_s c m / CN_s. Near zero in normal spots, rising with purity.
- **H_S_local, spatially restricted somatic.** As H_S, but present only in a patch P from a fixed patch
  dictionary (multi-scale neighbourhoods on the spot graph): f = pi_s c 1[s in P] / CN_s, marginalised
  over P with a uniform prior. Needed because CalicoST resolves only one clone in three of four
  sections, so subclonal, spatially focal mutations are not represented by clone labels.
- **H_E, artifact or RNA editing.** A site-level fraction e ~ Beta(alpha_E, beta_E), the same in all
  spots and independent of purity: f = e. The prior is fitted on the sequence-error class; editing
  (often 5-40%) is expected to fall here rather than H_G because it does not track copy number and sits far from 0.5.

Output per variant: posteriors P(G), P(S) = P(H_S) + P(H_S_local), P(E), using class priors estimated
by EM over all de novo candidates in the section. Assign a class when its posterior is >= tau (tau = 0.8,
to be calibrated); otherwise the variant is unresolved. There is no quota.

Low-information variants (for example a single ALT read in a single spot) give posteriors close to the
prior and therefore stay unresolved. This is the intended behaviour.

## 4. Purity-missing sections (P4 rep2)

Do not build our own latent-tumor HMRF. CalicoST already infers clones with a hidden Markov random
field over spots (with an HMM along the genome), so a second spatial latent layer would duplicate it and
inherit the same dependence on the same data. Proposed fallback: pi_s = 0 for
`normal_candidate_barcodes`, otherwise pi_s = pi_hat with pi_hat integrated over a Beta prior fitted to
the other three sections. Alternative to consider: derive pi_s from germline allelic imbalance in LOH
segments using our own 1KGP-defined heterozygous sites.

## 5. Computation

- Per variant: sum over covered spots only (sparse); grid over c (~20 points) x haplotype (2) x patches
  (|P| ~ 100-300 for H_S_local). With sparse matrices and log-sum-exp this is ~10^9 cheap operations for
  P6 rep1 (~0.6-0.9 M de novo candidates): minutes in vectorised numpy, no GPU.
- Global parameters (rho, alpha_E, beta_E, class priors): a few EM iterations.
- Step 7 alone is rerun; steps 1-6 are unchanged. Current step 7 takes about 8 minutes per section.

## 6. Validation plan (no somatic ground truth)

1. Calibration on labelled sets held out from fitting: 1KGP germline sites should get P(G) high;
   sequence-error sites should get P(E) high. Report reliability curves.
2. Simulation: spike synthetic variants into the real coverage of real spots (real n_vs, real pi_s),
   drawn from each hypothesis including focal patches at several sizes and CCFs; measure recall,
   precision and calibration of the posterior, versus the current vote.
3. Sensitivity to CalicoST: perturb pi_s (noise, shrinkage toward the mean), swap hard for soft clone
   labels, and drop CN (all diploid); report how many class assignments change.
4. External metrics (not used for ranking where they reward germline content): matched-WES alleles,
   fraction at RNA-editing catalogue sites, 1KGP leakage, COSMIC, rep1-to-rep2 recurrence.

## 7. Open questions for review

Q1. Is the hypothesis set sufficient? In particular, do we need a separate editing hypothesis, or can
    H_E with a broad Beta prior absorb both sequencing artifacts and editing? Editing levels vary by cell
    type, so editing could look spatially structured and mimic H_S_local.
Q2. Is a fixed multi-scale patch dictionary a reasonable approximation for focal somatic variants, or is
    there a better tractable alternative (scan statistic, per-variant Ising prior, Gaussian-process
    intensity)? How should the patch prior be set so H_S_local does not win by sheer model flexibility
    (Occam penalty)?
Q3. Allele-specific expression makes germline heterozygous ALT fractions deviate from the expected value.
    Is a single global beta-binomial rho enough, or do we need gene-level overdispersion?
Q4. With extremely sparse data (median covered spots per de novo candidate is small), is a marginal-
    likelihood comparison identifiable, or will priors dominate almost everywhere? How should tau be
    calibrated?
Q5. UMI-level correlation: reads from one molecule are already collapsed by UMI dedup, but fragments of one
    molecule at different start positions remain (Space Ranger duplicate flags are excluded by mpileup).
    Does this break the binomial independence assumption?
Q6. P4 rep2 fallback: is the pi_hat-integration approach defensible, or should P4 rep2 be excluded from
    somatic classification?
Q7. Anything that would make a reviewer call the posterior circular, given that CalicoST CN and purity are
    inputs and CNV enrichment is later reported as supporting evidence?

---

# v1 amendments after review (2026-10-09). These supersede sections 3-6 where they conflict.

Review: REVIEW_astra_2026-10-08.md. User decisions: prototype on P4 rep1 + P6 rep2 (the two sections with near-complete
CalicoST purity); run as a SHADOW job alongside the current vote, compared on 2026-10-11; the matched-WES gate is
reported, not binding (8 evaluable P4 / 3 P6 positives).

A1. Counts are molecules, not reads. Use step-6 output under quality tag `baseQ0mapQ0_umi` (UMI-collapsed per-spot AD,
    being built by run_slurm/umi_count_2026-10-09/). Never fit on read-level AD.
A2. Information gate: only candidates with >= 3 covered spots that have finite purity AND >= 10 total molecules enter
    the model; everything else is `unresolved_low_information` and is counted, not dropped.
A3. Expected ALT fraction with a tumor/normal per-copy expression ratio r (prior centred at 1, fitted per section on
    1KGP heterozygous controls in copy-number-altered segments):
      D_s = 2(1-pi_s) + r pi_s C_s,  C_s = A_k + B_k
      germline het on haplotype h:  f = (1 - pi_s + r pi_s h_k) / D_s,   h in {A,B}, prior 1/2
      germline hom ALT:              f = 1 - error
      somatic:  f = r pi_s c m / D_s,  c on a grid in (0,1], m in feasible {1..h_k} on the retained haplotypes
    Shared parameters (h, c, m, e, variant-level allelic bias) are integrated OUTSIDE the product over spots.
A4. Every hypothesis gets an error/ambient floor epsilon (fitted on the step-4 sequence-error class), so one ALT molecule in
    a normal spot cannot annihilate a likelihood.
A5. Germline: a shared variant-level allelic-bias effect b ~ distribution learned from confident 1KGP heterozygous sites,
    plus global residual beta-binomial dispersion.
A6. Artifact and RNA editing are separate components. Editing: catalogue-informed prior (SComatic AllEditingSites as a
    PRIOR weight only, never as a hard mask; the catalogue stays an evaluation metric) with a site-level fraction that may
    differ between tumor-high and tumor-low spots. If editing and somatic are not distinguishable for a site, return
    unresolved.
A7. Deferred from the prototype: focal-patch hypothesis, EM over class priors, purity reconstruction. Class priors are
    fixed and conservative; tau = 0.95 provisionally.
A8. P4 rep2 and the missing-purity spots of P6 rep1 get no confident somatic calls in this version.
A9. CNV/LOH enrichment of calls is reported descriptively only, never as validation.

Evaluation (2026-10-11), each of P4 rep1 and P6 rep2, versus the frozen deterministic vote
(baseQ0mapQ0_umi_full_calicost and _ablation_no_calicost):
 G1 count audit (from the UMI task): >= 99% agreement on 200 spot-locus pairs.
 G2 blinded, coverage-matched germline (1KGP het) and artifact controls: one-sided 95% upper bound of the false-somatic
    rate <= 1% for each control type.
 G3 simulation on real coverage/purity, including shared ASE, structured editing, m > 1 and purity noise: somatic FDR <=
    5% at 5% prevalence; recall >= vote at matched false-positive rate.
 G4 stability: >= 90% of somatic calls keep P(S) >= 0.9 under purity noise SD 0.05, halved/doubled dispersion, and
    10x prior odds either way.
 R1 (reported, not binding): matched-WES alleles recovered, including abstentions, vs both baselines.
