# DLPFC genotype-aware STAGATE validation

**Completed:** 2026-08-24

**Cohort:** 12 spatialLIBD DLPFC sections, four serial sections from each of three donors
**Primary result:** Beagle-corrected pseudobulk genotype does not improve cortical-layer
clustering. The strongest SNV-derived representation is the relative REF+ALT depth profile at
panel loci, which is close to gene expression overall but has a donor-dependent effect. The
signal is real spatial structure, but the ablations attribute it to locus coverage/transcription
and allelic detection rather than locus-specific corrected genotype.

## Decision

STAGATE remains useful as a downstream **spatial decodability assay**: given any spot-by-feature
matrix, it asks whether the representation plus the fixed spatial graph reconstructs expert
cortical layers. It is not, by itself, a germline-genotype validator. The current results support
the limited manuscript statement that spatial SNV-derived matrices contain anatomical
information. They do not support the stronger statement that Beagle-corrected genotypes contain
or improve that information.

The prespecified success condition was that a corrected-GT representation should exceed:

1. an exact-locus spot-presence matrix with no GT weighting;
2. a depth-stratified, within-chromosome shuffled-GT control; and
3. gene expression, reproducibly across all three donors.

Neither corrected non-reference selection nor corrected dosage met this condition. Corrected
dosage averaged ARI 0.351, compared with 0.365 for exact-locus presence, 0.347 for shuffled-GT
dosage, and 0.412 for gene expression. The corrected-minus-shuffled difference was only +0.004
ARI and was positive in two of three donors. A sparse-data TF--IDF sensitivity analysis also
failed its prespecified scaling rule.

## Why the previous setting could approach gene expression

The original DLPFC matrix is not a matrix of independently inferred spot genotypes. Its rows are
spots and its columns are observed variant loci or genomic bins. A nonzero entry records that an
allele was detected at that spot. That value depends on transcription, gene/locus expression,
RNA coverage, allelic expression, mapping and sampling, as well as genotype. In addition,
STAGATE uses the Visium spatial-neighbor graph and was designed for spatial transcriptomic
features [1]. Thus good layer ARI demonstrates spatially aligned information, but it does not
identify genotype as the source of the information.

The inherited preprocessing applies per-spot `normalize_total(1e4)` followed by `log1p`.
Consequently, scalar total UMI depth is removed, but the **relative distribution of reads across
genomic bins remains**. Cortical layers have different transcriptional programs, so a relative
panel-locus coverage profile can resemble a compressed gene-expression profile. This explains
why the normalized depth representation can approach gene expression without requiring
layer-specific inherited genotypes.

The corrected Beagle call is also a pseudobulk, section-level value. Every spot in one section
inherits the same GT label for a locus. Multiplying spot observations by that shared value changes
locus selection or weighting; it does not generate spot-varying genotype. Anatomically distinct
spots from the same donor should not have different inherited germline genotypes in the first
place. The scientifically meaningful test is therefore whether donor-level GT improves the
spot-evidence representation, not whether STAGATE can find genetically distinct cortical layers.

## Production-flow audit

The audit found an important distinction between Beagle filtering and Beagle-corrected genotype.
The inspected Beagle command uses genotype likelihoods with `impute=false`, `gprobs=true`, and
`niterations=0`. Per-chromosome Beagle VCFs contain corrected GT values. However,
`create_filtered_vcf()` reads records from the original pseudobulk VCF and copies the original
lines into the combined in/out-panel VCFs according to whether the allele survived Beagle. The
production combined VCF and downstream spot matrix therefore use Beagle as an allele-membership
filter; corrected GT does not flow into the matrix.

This was not merely a file-format difference:

- 735,131 production matrix loci were audited across the 12 sections.
- 731,005 loci (99.44%) matched allele-exactly to both raw and Beagle VCFs.
- All 734,635 positions common to the unique raw/Beagle VCF positions had concordant REF/ALT
  alleles; there were zero allele mismatches.
- Beagle GT differed from the raw pseudobulk GT at 233,888 of 731,005 matched loci (32.00%).
- The corrected calls contained 124,690 homozygous-reference and 606,315 non-reference loci.

Pooled raw-to-corrected transitions were:

| Transition | Loci | Percent |
|---|---:|---:|
| 0/1 -> 0/0 | 20,157 | 2.76% |
| 0/1 -> 0/1 | 111,813 | 15.30% |
| 0/1 -> 1/1 | 5,261 | 0.72% |
| 1/1 -> 0/0 | 104,533 | 14.30% |
| 1/1 -> 0/1 | 103,937 | 14.22% |
| 1/1 -> 1/1 | 385,304 | 52.71% |

The new experiment consumes the per-chromosome corrected VCFs directly and matches them
allele-exactly to the production loci, so it actually tests corrected GT without changing the
underlying spot evidence.

## Cohort and fixed analysis

The sections were 151507--151510 (donor 1), 151669--151672 (donor 2), and
151673--151676 (donor 3). There were 47,329 labeled spots. Donors 1 and 3 had the six cortical
layers plus white matter (seven classes). Donor 2 had five observed label classes: 0, 3, 4, 5,
and 6. This matters because the existing manuscript Methods statement that every section was
clustered into seven groups is not true for the executable benchmark; the code uses the number of
label classes actually present, five for donor 2 and seven for donors 1 and 3.

All new cohort modalities used five STAGATE seeds per section. Existing production and gene-
expression estimates use their established ten seeds. The fixed model used a radius-150 spatial
graph, hidden dimensions `[512, 30]`, 1,000 epochs, learning rate 0.001, weight decay 0.0001,
gradient clipping at 5, and mclust EEE. There was no highly-variable-feature selection. Each
matrix used 250-kb genomic bins and the same scored spots. The biological replication unit is the
donor; section-level tests are retained as descriptive repeated-section summaries.

The primary representations were:

- **Production 1KGP:** the existing Beagle-filtered binary spot matrix.
- **Matched presence:** the same spot evidence restricted to loci matched to the corrected VCF,
  without using the corrected GT.
- **Corrected non-reference:** matched loci retained only when corrected GT is 0/1 or 1/1.
- **Corrected dosage:** spot presence weighted 1 for corrected 0/1 and 2 for corrected 1/1.
- **Shuffled dosage:** corrected GT shuffled within chromosome and pseudobulk-depth stratum before
  weighting, preserving genotype counts and their depth distribution.

## Primary genotype results

| Representation | Mean section ARI | SD across sections |
|---|---:|---:|
| Gene expression | 0.412 | 0.089 |
| Matched exact-locus presence | 0.365 | 0.102 |
| Production 1KGP, 250 kb | 0.363 | 0.112 |
| Beagle-corrected non-reference | 0.355 | 0.103 |
| Beagle-corrected dosage | 0.351 | 0.092 |
| Shuffled-GT dosage | 0.347 | 0.088 |

The donor means show that the apparent competitiveness with gene expression is concentrated in
donor 2 rather than reproduced across the cohort:

| Donor | Gene expression | Production | Matched presence | Corrected non-ref | Corrected dosage | Shuffled dosage |
|---|---:|---:|---:|---:|---:|---:|
| Donor 1 | 0.437 | 0.326 | 0.334 | 0.318 | 0.316 | 0.331 |
| Donor 2 | 0.409 | 0.475 | 0.452 | 0.452 | 0.440 | 0.427 |
| Donor 3 | 0.389 | 0.287 | 0.310 | 0.296 | 0.297 | 0.283 |

Paired results were consistent with the negative conclusion:

- Corrected non-reference minus matched presence: mean -0.010 ARI; all three donor means were
  negative; exact donor-level Wilcoxon P=0.25. Only 3/12 sections improved.
- Corrected dosage minus matched presence: mean -0.014; all three donor means were negative;
  donor P=0.25. Five of 12 sections improved.
- Corrected dosage minus shuffled dosage: mean +0.004; two of three donor means were positive;
  donor P=1.00. Eight of 12 sections improved.
- Corrected dosage minus gene expression: mean -0.061; one of three donor means was positive;
  donor P=0.50. Three of 12 sections improved.

With only three donors, the minimum possible two-sided exact Wilcoxon P value is 0.25. The point
is not that the negative result is established by a small P value; it is that the corrected
methods fail the directional, effect-size, and shuffled-control criteria that would be required
for a genotype claim.

## Feature ablation: what actually helps

Per-spot VCF `AD` was reconstructed allele-exactly at all matched candidate loci. Across the
cohort, 209,892,481 VCF records were scanned; 38,191,677 matched records carried AD and
12,702,480 had positive REF+ALT depth. The matrices covered 47,681 barcodes, 731,005 matched
loci, and 107,253 section-specific 250-kb bins.

Five ablations were fixed after the initial 151507 pilot and then run on all 12 sections:

| Representation | Mean section ARI | Interpretation |
|---|---:|---|
| Relative REF+ALT depth at panel loci | 0.425 | Strongest; relative coverage/transcription profile |
| Gene expression | 0.412 | Reference modality |
| Shuffled-nonref ALT count | 0.384 | GT-negative selection control |
| All-locus ALT count | 0.380 | Alternate-read profile without GT |
| Corrected-nonref ALT count | 0.380 | ALT profile selected by corrected GT |
| Matched exact-locus presence | 0.365 | Binary production-evidence control |
| Binary ALT detection | 0.354 | Loses count information |

Relative depth exceeded gene expression by +0.014 ARI overall, but the donor differences were
-0.064, +0.086, and +0.020. It was better in two of three donors (donor P=0.75) and 7/12
sections (descriptive section P=0.791). Thus it is reasonable to call depth **near gene-
expression performance**, not superior to gene expression.

Relative depth exceeded matched presence by +0.060 ARI, with a positive mean in all three donors
(donor P=0.25) and in 10/12 sections (descriptive section P=0.021). Counts and relative coverage
therefore add useful information beyond binary locus detection.

Corrected-nonref ALT exceeded matched presence by +0.015 overall and in all three donor means,
but it was indistinguishable from shuffled-nonref ALT: corrected minus shuffled was -0.004 ARI,
positive in only one of three donors, and positive in 6/12 sections. This distinguishes two
effects. ALT counts can be more informative than binary presence, but choosing those loci by the
specific corrected genotype adds no detectable information beyond a depth-matched random
selection.

The finer 151507 pilot reached ARI 0.420 for relative depth, 0.386 for corrected-nonref ALT,
0.366 for all-locus ALT, 0.358 for shuffled-nonref ALT, and 0.342 for binary presence. VAF sums
were lower (0.319 corrected-nonref and 0.317 all-locus); corrected-heterozygous ALT was 0.313 and
corrected-homozygous-alternate ALT was 0.290. The corrected-homozygous-reference ALT matrix was
degenerate in all five seeds. Splitting a sparse matrix into GT-class channels therefore loses
rather than reveals useful structure.

## Independent permutation controls

To test whether STAGATE's spatial graph could manufacture layer structure from an uninformative
matrix, section 151507 was subjected to five independently generated permutations at fixed model
seed 0. Spot permutation breaks spot-to-coordinate alignment. Within-spot genomic-bin permutation
preserves each row's values but destroys shared bin identity.

For matched presence, observed ARI was 0.386 at model seed 0 and 0.354 averaged over five model
seeds. Independent spot permutations averaged 0.006 (range 0.000--0.011); bin permutations
averaged 0.147 (0.131--0.156). None of the five permutations equaled the observed seed-0 result.
The plus-one empirical P value is 1/6=0.167, the finest possible with five permutations.

For relative depth, observed ARI was 0.424 at seed 0 and 0.420 over five model seeds. Spot
permutations averaged 0.003 (range -0.004--0.008). All five within-row bin permutations produced
embeddings with effective rank 2--4, below the seven fitted groups, and were retained as explicit
degenerate failures instead of forcing mclust.

These nulls show that the observed matrices contain genuine spot-aligned and genomic-bin-aligned
structure. In particular, the spatial graph alone did not recover layers after spot permutation.
They do not identify genotype as the source; the genotype-label shuffle is the relevant null for
that question, and corrected GT did not beat it.

## Preprocessing sensitivity

Because library normalization and log transformation were inherited from a gene-expression
workflow, a TF--IDF transformation tailored to sparse count/presence data was tested in 151507.
Five-seed mean ARIs were 0.279 for matched presence, 0.277 for corrected dosage, 0.272 for
corrected non-reference, and 0.253 for shuffled dosage. Gene expression in that section was
0.410, making the prespecified scale-up threshold 0.430. Corrected GT did not beat matched
presence and was far below the scale-up threshold, so TF--IDF was not expanded to the cohort.
This protects the analysis from turning a post hoc preprocessing search into a favorable claim.

## Implications for the paper

The current DLPFC benchmark can remain if its claim is tightened:

- **Supported:** the spatial matrix of panel-supported allele observations contains enough
  anatomical structure for STAGATE to recover cortical layers; 250-kb aggregation and relative
  locus coverage are useful representations.
- **Supported with qualification:** the best relative-depth representation is close to gene
  expression overall, but its advantage is donor-dependent and not replicated as superiority.
- **Not supported:** Beagle-corrected genotype improves layer clustering, cortical layers are
  separated by inherited genotype, or layer ARI validates individual germline calls.
- **Required Methods correction:** state that mclust used the number of observed label classes,
  five for donor 2 and seven for donors 1 and 3, rather than seven for every section.
- **Required terminology:** call the input an SNV-observation, allele-detection, or relative
  locus-coverage representation. Do not call the spot matrix a corrected-genotype matrix.
- **Recommended evidence:** retain donor means and a genotype-label shuffle. Section-level P
  values may describe serial sections but must not be presented as n=12 biological inference.

STAGATE is therefore still an option for the intended validation, provided the question is
phrased correctly: “Does this SNV-derived representation preserve cortical-layer information?”
If the intended question is “Are the individual germline genotypes correct?”, use an orthogonal
DNA truth set and allele/genotype concordance instead. If the question is “Does genotype itself
drive clustering?”, the present matched-locus and shuffled-GT ablations are the necessary
controls—and they answer no for this dataset.

## Limitations

- There are only three biological donors, so formal donor-level power is very limited.
- DLPFC sections are serial technical/anatomical replicates, not independent patients.
- Beagle GT is pseudobulk and does not resolve spot-level genotypes.
- RNA allelic counts are affected by expression, allele-specific expression, reference mapping
  bias and sampling [7,8].
- ARI against expert labels measures anatomical agreement, not variant-call accuracy.
- The depth representation uses candidate panel loci rather than all transcriptomic bases. It is
  a relative locus-depth control, not raw total UMI and not a pure gene-expression matrix.
- STAGATE stochasticity is summarized with five seeds for new cells, but the GPU implementation
  is not bit-reproducible.

## Reproducibility and source map

No production results were overwritten. All generated matrices and fits are under this dated
analysis directory. Matrix builders refuse replacement unless `--force` is explicitly supplied.

Core code:

- `clustering_benchmark/genotype_aware_2026_08_24/build_matrices.py`
- `clustering_benchmark/genotype_aware_2026_08_24/build_ad_ablation.py`
- `clustering_benchmark/genotype_aware_2026_08_24/build_permutation_replicates.py`
- `clustering_benchmark/genotype_aware_2026_08_24/summarize_results.py`
- `clustering_benchmark/genotype_aware_2026_08_24/plot_results.py`
- `clustering_benchmark/SPARCAL_clustering.py`

Authoritative tables under `analysis/`:

- `seed_level.csv`, `section_level.csv`, `donor_level.csv`, `method_summary.csv`
- `paired_tests.csv`, `label_audit.csv`, `matrix_audit_summary.csv`
- `ad_ablation_audit_summary.csv`, `allele_match_audit.csv`
- `permutation_null_summary.csv`, `tfidf_pilot_seed_level.csv`
- `figure_cohort_source.csv`, `figure_effect_source.csv`
- `figure_ad_cohort_source.csv`, `figure_permutation_source.csv`
- `figure_metadata.txt`

Final analysis figure and its manuscript-facing dossier:

- `/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v5/fig_germline_genotype_ablation.pdf`
- `/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v5/fig_germline_genotype_ablation.png`
- `/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figures/fig-dlpfc-beagle-genotype-ablation.md`

Recreate summaries and the figure from the `snv_calling` repository with:

```bash
python clustering_benchmark/genotype_aware_2026_08_24/audit_allele_matching.py
python clustering_benchmark/genotype_aware_2026_08_24/summarize_results.py
python clustering_benchmark/genotype_aware_2026_08_24/plot_results.py
```

## References

All entries below were checked against the manuscript's curated local BibTeX entries, including
their DOI fields.

1. Dong K, Zhang S. Deciphering spatial domains from spatially resolved transcriptomics with an
   adaptive graph attention auto-encoder. *Nature Communications*. 2022;13:1739.
   doi:10.1038/s41467-022-29439-6.
2. Maynard KR, et al. Transcriptome-scale spatial gene expression in the human dorsolateral
   prefrontal cortex. *Nature Neuroscience*. 2021;24:425-436.
   doi:10.1038/s41593-020-00787-0.
3. Pardo B, et al. spatialLIBD: an R/Bioconductor package to visualize spatially-resolved
   transcriptomics data. *BMC Genomics*. 2022;23:434. doi:10.1186/s12864-022-08601-w.
4. Browning BL, Zhou Y, Browning SR. A One-Penny Imputed Genome from Next-Generation Reference
   Panels. *American Journal of Human Genetics*. 2018;103:338-348.
   doi:10.1016/j.ajhg.2018.07.015.
5. 1000 Genomes Project Consortium. A global reference for human genetic variation. *Nature*.
   2015;526:68-74. doi:10.1038/nature15393.
6. Scrucca L, Fop M, Murphy TB, Raftery AE. mclust 5: Clustering, Classification and Density
   Estimation Using Gaussian Finite Mixture Models. *The R Journal*. 2016;8:289-317.
   doi:10.32614/RJ-2016-021.
7. Degner JF, et al. Effect of read-mapping biases on detecting allele-specific expression from
   RNA-sequencing data. *Bioinformatics*. 2009;25:3207-3212.
   doi:10.1093/bioinformatics/btp579.
8. Castel SE, et al. Tools and best practices for data processing in allelic expression analysis.
   *Genome Biology*. 2015;16:195. doi:10.1186/s13059-015-0762-6.
