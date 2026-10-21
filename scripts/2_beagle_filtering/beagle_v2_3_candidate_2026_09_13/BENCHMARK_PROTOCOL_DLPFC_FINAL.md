# Frozen final DLPFC end-to-end no-harm protocol

Frozen before the P4/P6 configuration, replication, and posterior-ensemble
selection results were available. This is a functional replacement test for
the final selected WES-free Beagle module. It is not a variant-caller accuracy
benchmark and it does not use WES as input.

## Trigger and selected module

Run this protocol exactly once, after the checksum-verified P4/P6 selection has
chosen the final production module. Failed screen arms do not trigger DLPFC
clustering.

- If a single-run configuration is selected, run that exact `modelscale`,
  `niterations`, and `maxlr` setting with fixed seed `20260915` on every DLPFC
  section.
- If a posterior ensemble is selected, run that exact configuration with seeds
  `20260915`, `20260916`, and `20260917` on every section and apply the selected
  mean- or median-posterior aggregation rule without retuning its threshold.
- Bind the final-selection bundle, v2.3 code, Beagle jar, Java executable, raw
  inputs, and the frozen 88-artifact GRCh38 reference ledger by checksum.
- Refuse fallback, missing contigs, unequal exact-allele universes across
  ensemble members, or reuse of a non-empty output root.

The fixed DLPFC sections are `151507`, `151508`, `151509`, `151510`, `151669`,
`151670`, `151671`, `151672`, `151673`, `151674`, `151675`, and `151676`.
Each run starts from that section's frozen `baseQ0mapQ0` raw mpileup candidate
VCF. No DLPFC annotation, cluster label, WES call, or other truth information
is available to Beagle or the evidence router.

## Matrix endpoints

The primary replacement endpoint is the production compatibility view
`all_panel_present`: the union of non-reference-high-confidence,
reference-high-confidence, and uncertain panel-present alleles. Project exact
alleles to the historical position-only DLPFC matrix with the prespecified
`any` rule at multiallelic loci. This view is feature support only; it is not a
germline call set and must not be described as one.

Generate and checksum an anchor-only `P(nonref)>=0.90` matrix as a secondary
mechanistic diagnostic. It cannot decide replacement. A previous strict
anchor-only run had positive mean delta ARI but failed the success-rate gate
because sparse embeddings produced unstable fixed-EEE fits; that result is a
reason to keep anchors and compatibility support as different interfaces, not
a reason to relax the numerical contract.

Before clustering, report exact dataframe equality, feature counts, and
per-section set differences against the frozen legacy any-Beagle-record matrix.
These are deterministic representation checks. They do not waive the required
clustering run: run STAGATE followed by mclust even if all primary candidate
matrices are byte-identical or dataframe-identical to baseline.

## STAGATE followed by mclust

Use the frozen paired strict implementation and environment:

- candidate and baseline in the same process for each section and seed;
- STAGATE seeds 0 through 9, giving 120 runs per arm;
- log-normalization, radius cutoff 150, hidden dimensions 512 and 30, 1000
  epochs, learning rate 0.001, weight decay 0.0001, and gradient clipping 5;
- annotated-domain cluster count for both arms;
- mclust model `EEE`, with no model search or fallback; and
- the existing independently replayed numerical validity contract for the
  embedding, covariance, posterior, classifications, and saved artifacts.

The final module passes only if all frozen gates pass:

1. mean paired delta ARI is at least 0;
2. every donor mean delta ARI is greater than -0.01;
3. no section mean delta ARI is below -0.05;
4. mean paired delta NMI is at least -0.01;
5. every section has at least 9 of 10 paired successes;
6. there is no candidate failure paired with a successful baseline run;
7. total candidate successes are not lower than baseline successes; and
8. neither arm uses an mclust fallback.

If the primary compatibility view fails, the selected Beagle module does not
replace the current downstream interface, regardless of P4/P6 WES metrics. If
it passes, the clustering result supports functional non-inferiority only;
P4/P6 WES concordance, seed stability, and route conservation remain the
variant-level evidence.

This protocol was informed by earlier DLPFC diagnostics, so the final run is a
locked reproducibility/no-harm assessment rather than an independent
prospective validation cohort.
