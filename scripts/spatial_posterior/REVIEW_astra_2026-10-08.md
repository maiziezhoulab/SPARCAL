<!-- gpt-6-astra xhigh review of DESIGN.md v0, 2026-10-08. Prompt: see bottom. Verified by Claude: extractor at
     scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py:345 skips only unmapped reads (no dup/secondary/UMI/baseQ
     filter); finite CalicoST purity P4s1 700/700, P4s2 29/648 (all 0), P6s1 535/2552, P6s2 2260/2262. -->

**Recommendation: prototype in shadow mode; do not adopt this draft unchanged.** The direction is worthwhile, but its count assumptions and identifiability claims need correction. I made no changes.

First, update the baseline description: the [current code](/data/maiziezhou_lab/leiy4/snv_calling/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py:1449) requires both uniformity and prevalence for germline exclusion, uses VAF–purity correlation, and conditions several features on coverage. Compare against that deterministic implementation.

My audit of ALT-positive de novo candidates within CalicoST-retained barcodes found:

| Rep1 sample | Candidates | Median covered spots / total AD | Only one covered spot |
|---|---:|---:|---:|
| P4, 700 spots | 202,247 | 1 / 1 | 76.9% |
| P6, 2,552 spots | 549,316 | 1 / 1 | 83.0% |

These are **read-count statistics**, not verified molecule counts. Moreover, P6 rep1 has finite purity for only **535/2,552 spots**; 63.5% of its audited candidates have no coverage at a finite-purity spot.

**Section 3: the DNA formulas are conditionally correct; their application to RNA is incomplete.**

Writing \(C=A+B\), your formulas
\[
f_G=\frac{1-\pi+\pi h}{2(1-\pi)+\pi C},\qquad
f_S=\frac{\pi cm}{2(1-\pi)+\pi C}
\]
are correct for diploid normal cells, equal transcript yield per allele copy, and identical CN states in mutation-bearing and other tumor cells. Do **not** multiply the germline normal contribution by \(c\).

For RNA, introduce tumor-to-normal **per-copy expression ratio** \(r_{vs}\):
\[
D_{vs}=2(1-\pi_s)+r_{vs}\pi_sC_s,\quad
f_G=\frac{1-\pi_s+r_{vs}\pi_sh_s}{D_{vs}},\quad
f_S=\frac{r_{vs}\pi_scm}{D_{vs}}.
\]
These still assume no allele-specific expression. CalicoST estimates purity from aggregated LOH allelic counts and expression ratios; that does not establish \(r_{vs}=1\) at each candidate gene. [CalicoST methods](https://www.nature.com/articles/s41592-024-02438-9)

**Fixing \(m=1\) is inadequate.** In copy-neutral LOH \((0,2)\), a mutation preceding duplication can have \(m=2\): at \(\pi=0.6,c=1\), its expected fraction is 0.6, versus 0.3 under your model. Marginalize over feasible multiplicities on retained haplotypes, including single-copy and amplified mutations. Do not interpret \(c\) separately from \(m\) without additional evidence. If mutation-bearing cells have different CN, the denominator must also include their fraction-weighted CN.

Integrate shared \(h,c,m,e\) **outside the product over spots**. Add sequencing-error/ambient-RNA contributions under every hypothesis; otherwise one ALT in a nominally normal spot can annihilate the somatic likelihood. Include homozygous germline ALT as a germline component.

1. **Q1 — Separate editing from sequencing error.** A low-error-trained Beta prior cannot represent both reliably, and a broad constant-fraction prior cannot capture cell-type-dependent editing. Add a distinct editing/background component with region- or cell-type-varying fractions and catalogue-informed priors. Treat upstream sequence-error labels as noisy controls. When structured editing and somatic variation are observationally indistinguishable, return unresolved.

2. **Q2 — Keep the dictionary, defer it from the prototype.** Predefine patches using coordinates only. Allocate equal prior mass across scales, then uniformly across centres within scale; remove duplicate patches and the whole-tissue patch. Fix the total local-somatic prior mass independently of dictionary size. Average likelihoods with normalized weights; never select the maximum patch likelihood. A correctly normalized mixture already supplies a complexity penalty. Require covered observations outside a patch before claiming localization.

3. **Q3 — A global \(\rho\) is insufficient.** Independent beta-binomial noise does not capture persistent, same-direction allelic imbalance across spots. Introduce a shared variant-level allelic-bias effect, with its distribution learned from confidently heterozygous controls; partially pool gene effects only where supported. Start with global residual dispersion after that effect. Crucially, when \(n=1\), beta-binomial emissions are Bernoulli regardless of \(\rho\): increasing dispersion cannot fix most observations here.

4. **Q4 — Most candidates are not identifiable.** The singleton percentages above support that conclusion. Low information does **not** guarantee a posterior near the prior, nor abstention: one ALT can strongly favor germline over a low-error model, and EM can produce a class prior exceeding the threshold. Require an explicit information gate and Bayes-factor evidence. Use \(\tau=0.95\) provisionally, calibrated against held-out false-somatic rates. Reject unrestricted EM for the prototype: selected-candidate class proportions require selection-conditioned likelihoods. Germline/error controls measure class-specific errors; they cannot establish somatic posterior reliability.

5. **Q5 — The deduplication premise is false for the inspected AD path.** The [extractor](/data/maiziezhou_lab/leiy4/snv_calling/scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py:303) skips unmapped reads but neither checks duplicate flags nor collapses UB tags. At one audited P4 locus, AD \(=90,15\) included 77 REF and 8 ALT duplicate-flagged reads. Mpileup’s duplicate exclusion does not apply to this separate extractor. [Samtools documentation](https://www.htslib.org/doc/samtools-mpileup.html) Recount one consensus observation per barcode/UMI/locus, with explicit alignment/base-quality filtering. Overdispersion is not a substitute.

6. **Q6 — Exclude P4 rep2 from confident somatic classification initially.** Transferring a purity prior from other sections cannot recover this section’s missing spatial information; normal-candidate labels are also uncertain. Retain unresolved calls and report prior-sensitive scores only. A later LOH-based estimate is reasonable only with sufficient independent heterozygous evidence and held-out validation. Apply the same principle to missing-purity regions of P6 rep1.

7. **Q7 — CNV enrichment would be circular supporting evidence.** Using CalicoST inputs is legitimate conditional modelling; subsequently presenting agreement with those inputs as independent validation is not. Report enrichment descriptively. Evaluate with exact-allele matched-WES evidence and blinded negatives. Germline controls used to assess the model should ideally be excluded from CalicoST fitting, or evaluated through chromosome/block withholding. Replicate recurrence also rewards germline and recurrent artifacts.

**The single biggest risk is false confidence from counting repeated molecules as independent somatic evidence.** A magnitude-based likelihood can amplify this more severely than rank voting. Detect it on day one: recount a depth-stratified candidate/control panel, compare raw-AD versus molecule-based likelihood ratios, and inspect high-confidence somatic calls among blinded germline negatives. Confidence collapsing after deduplication is a stop signal.

**A minimal three-day prototype is feasible with targeted recounting.** The “step 7 only, existing counts suffice” promise is not.

- **October 9:** Freeze candidate alleles and deterministic full/no-CalicoST baselines. Audit/recount candidates with at least three covered finite-purity spots and ten total observations; reapply those gates using molecule counts. Keep other candidates unresolved. Reuse the [existing molecule-count benchmark](/data/maiziezhou_lab/leiy4/snv_calling/data/mechanistic_cnv_purity_vaf_2026-09-10/REPORT.md) and matched-normal controls.
- **October 10:** Implement global somatic, germline-with-shared-allelic-bias, and error/editing-background likelihoods; include multiplicity and an error floor. Freeze nuisance parameters and conservative priors using separate controls. Defer patches, EM, and purity reconstruction.
- **October 11:** Evaluate both samples, including every input candidate in coverage/abstention accounting. Keep WES labels out of tuning. Compare a diploid simplification alongside the CN-aware model.

**Adopt only if all these prespecified gates pass in each sample:**

- Independent recounts agree on at least 99% of 200 audited spot–locus count pairs; discrepancies are resolved before evaluation.
- For blinded, coverage-matched germline and artifact/editing controls separately, the one-sided 95% upper confidence bound on the false-somatic rate is ≤1%. Hide control annotations during scoring.
- In simulations preserving actual coverage and candidate selection, including shared ASE, structured editing, \(m>1\), and purity error, somatic FDR is ≤5% at 5% somatic prevalence; recall equals or exceeds voting at the same false-positive rate.
- At no greater call count than either frozen baseline, retain every WES allele recovered by either baseline and recover at least one additional WES allele per sample. Report recall against all WES alleles entering step 7, including abstentions.
- At least 90% of somatic calls retain \(P(S)\ge0.9\) under each prespecified perturbation: purity noise SD 0.05, halved/doubled residual dispersion, and tenfold higher/lower somatic prior odds.

Insufficient controls, insufficient observable positives, or any failed gate means **do not adopt; retain the shadow prototype**. The existing benchmark has only eight evaluable P4 rep1 and three P6 rep1 somatic positives, so even passing these operational gates would not establish calibrated real-world somatic probabilities.
---
Prompt:

You are a statistical-genetics peer reviewer. Review the design at
/data/maiziezhou_lab/leiy4/snv_calling/scripts/spatial_posterior/DESIGN.md
(a replacement for the rank-voting somatic classifier in
/data/maiziezhou_lab/leiy4/snv_calling/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py, classify_variants).
You may read code and data under /data/maiziezhou_lab/leiy4/snv_calling and /data/maiziezhou_lab/leiy4/CalicoST to check
facts (e.g. how sparse per-spot AD really is for de novo candidates in data/P4_tumor/1); do not modify anything.

Deliver:
1. Answers to Q1-Q7 in the doc, each with a concrete recommendation, not a survey.
2. Any modelling error in section 3 (check the expected-ALT-fraction formulas for H_G and H_S under tumor/normal
   mixing and allele-specific CN; check whether m=1 is adequate).
3. The single biggest risk that would make this worse than the current vote, and how to detect it early.
4. A minimal version that could be prototyped and evaluated by 2026-10-11 (3 days) on P4 rep1 and P6 rep1,
   with the exact acceptance criteria you would use to decide adopt / do not adopt.
If you disagree with a design choice, say so plainly and give your alternative. Be concise: at most ~1500 words.
