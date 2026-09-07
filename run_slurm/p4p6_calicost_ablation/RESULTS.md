# P4/P6 corrected-hg19 CalicoST and SPARCAL results

## Execution record

- Corrected hg19 CalicoST outputs: `/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906`.
- Replicate-2 UMI dedup completed before the full SPARCAL pipeline: P4 115,744,423 to 95,064,281 reads (4,992 split BAMs; job 13868625), P6 86,325,218 to 48,428,945 reads (4,992 split BAMs; job 13868626).
- SPARCAL steps 1-6 completed for P4 rep2 (job 13868627) and P6 rep2 (job 13868628); repaired steps 7-8 completed in jobs 13871209 and 13869475.
- Full-versus-ablation spatial array 13874786 completed all four elements with exit code 0. Downstream analysis and eight Viewer studies completed in job 13874928 with exit code 0.

## CalicoST phylogeography

| sample | requested clones | realized clones | tree type | Newick |
| --- | ---: | ---: | --- | --- |
| P4 rep1 | 2 | 1 | singleton | `clone1;` |
| P4 rep2 | 2 | 2 | branching exact-perfect LOH tree | `(clone0:0,clone1:1);` |
| P6 rep1 | 3 | 1 | singleton | `clone1;` |
| P6 rep2 | 3 | 1 | singleton | `clone1;` |

P4 therefore has two section-level trees, one of which branches. P6 has two section-level singleton trees and zero branching trees. Startle/CPLEX was not used: three cases require no inference because they realize one clone, and P4 rep2 satisfies CalicoST exact perfect phylogeny directly.

## Spatial evidence-ablation results

Full mode uses informative tumor-purity correlation/proxy plus spatial clustering and clone-resolved CNV consistency. The ablation deliberately uses spatial clustering (zeta) only.

P4 rep2 has no informative CalicoST purity values, so full mode correctly uses spatial clustering plus CNV consistency for that section.

## Somatic callset comparison

| sample | classification | full_count | ablation_count | intersection | union | jaccard | full_only | ablation_only | full_retained_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P4_rep1 | somatic | 19522 | 19522 | 8813 | 30231 | 0.2915 | 10709 | 10709 | 45.14 |
| P4_rep2 | somatic | 8065 | 8065 | 6491 | 9639 | 0.6734 | 1574 | 1574 | 80.48 |
| P6_rep1 | somatic | 65649 | 65649 | 39481 | 91817 | 0.43 | 26168 | 26168 | 60.14 |
| P6_rep2 | somatic | 22326 | 22326 | 6832 | 37820 | 0.1806 | 15494 | 15494 | 30.6 |

## Per-spot burden comparison

| sample | spots | full_median_somatic_burden | ablation_median_somatic_burden | median_delta_full_minus_ablation | spearman_rho |
| --- | --- | --- | --- | --- | --- |
| P4_rep1 | 750 | 297 | 464 | -135 | 0.9666 |
| P4_rep2 | 691 | 220 | 227 | -7 | 0.9978 |
| P6_rep1 | 3719 | 213 | 227 | -14 | 0.9708 |
| P6_rep2 | 3319 | 58 | 73 | -13 | 0.8244 |

## CNV enrichment by call partition

| sample | partition | variants | cnv_covered | cnv_altered_fraction | loh_fraction | median_purity_correlation | median_cnv_consistency | median_spatial_clustering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P4_rep1 | ablation_only | 10709 | 9829 | 0.6585 | 0.6583 | 0 | 0.06 | 0.0619 |
| P4_rep1 | full_only | 10709 | 9370 | 0.1593 | 0.1593 | 0.0336 | 0.5 | 0.0238 |
| P4_rep1 | shared | 8813 | 7559 | 0.2068 | 0.2062 | 0.0158 | 0.5 | 0.0652 |
| P4_rep2 | ablation_only | 1574 | 1429 | 0.06228 | 0.06228 | NA | 0.5 | 0.0595 |
| P4_rep2 | full_only | 1574 | 1339 | 0 | 0 | NA | 0.5 | 0.0417 |
| P4_rep2 | shared | 6491 | 5511 | 0 | 0 | NA | 0.5 | 0.0833 |
| P6_rep1 | ablation_only | 26168 | 23523 | 0.1274 | 0.01229 | 0.0109 | 0.5 | 0.0269 |
| P6_rep1 | full_only | 26168 | 23770 | 0 | 0 | 0.0363 | 0.5 | 0 |
| P6_rep1 | shared | 39481 | 34944 | 0.0005437 | 0 | 0.0542 | 0.5 | 0.0362 |
| P6_rep2 | ablation_only | 15494 | 14015 | 0.5907 | 0.1933 | 0 | 0.03935 | 0.0556 |
| P6_rep2 | full_only | 15494 | 13883 | 0.09378 | 0.02291 | 0.0147 | 0.5 | 0 |
| P6_rep2 | shared | 6832 | 5958 | 0.1448 | 0.04062 | 0 | 0.5 | 0.04665 |

## Full-only versus ablation-only CNV/LOH contrasts

| sample | outcome | full_only_positive | full_only_negative | ablation_only_positive | ablation_only_negative | odds_ratio_full_vs_ablation_only | fisher_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P4_rep1 | cnv_altered | 1493 | 7877 | 6472 | 3357 | 0.09831 | 0 |
| P4_rep1 | loh | 1493 | 7877 | 6470 | 3359 | 0.0984 | 0 |
| P4_rep2 | cnv_altered | 0 | 1339 | 89 | 1340 | 0 | 8.927e-27 |
| P4_rep2 | loh | 0 | 1339 | 89 | 1340 | 0 | 8.927e-27 |
| P6_rep1 | cnv_altered | 0 | 23770 | 2998 | 20525 | 0 | 0 |
| P6_rep1 | loh | 0 | 23770 | 289 | 23234 | 0 | 9.048e-89 |
| P6_rep2 | cnv_altered | 1302 | 12581 | 8279 | 5736 | 0.0717 | 0 |
| P6_rep2 | loh | 318 | 13565 | 2709 | 11306 | 0.09784 | 0 |

Diagnostic: the current theta score fixes copy-neutral segments at 0.5, while altered segments begin at within-clone variant prevalence. Sparse variants can therefore score below the copy-neutral baseline. A full-only odds ratio below 1 is evidence of this scoring-direction bias, not evidence that CNV and SNVs are biologically incompatible.

Any full-only enrichment in altered/LOH segments is internal supporting evidence, not independent validation, because the full caller used CalicoST CNV.
