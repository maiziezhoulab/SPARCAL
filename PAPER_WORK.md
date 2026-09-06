# PAPER_WORK — prioritized work queue from the 2026-08-23 referee assessment

**Created 2026-08-23.** This is the *actionable* companion to
[REFEREE_REPORT_2026-08-23.md](REFEREE_REPORT_2026-08-23.md) (the full assessment) and
[PAPER_PLAN.md](PAPER_PLAN.md) §8 (its condensed form) and [PAPER_PLAN_DEPRECATED.md](PAPER_PLAN_DEPRECATED.md)
(parked results, several of which close items below without new compute).

**Scope of this file.** What to run, in what order, and where each result must land so that
runs done now are directly comparable to runs done later. It does not re-argue the paper's
claims: *the work below is worth doing whatever the final claim turns out to be.* Several
items produce numbers that could support the current framing, undercut it, or replace it —
that is the point of running them.

**Live job tracker stays [On_going.md](On_going.md).** Add a dated block there when you launch
anything from this queue; this file records the *plan*, On_going.md records the *runs*.

---

## 0. How to use this queue

Every item follows the same shape, so a result can be picked up months later by someone else:

| field | meaning |
|---|---|
| **Do** | the concrete action |
| **Why** | which referee finding it closes (IDs map to PAPER_PLAN §8) |
| **Inputs** | data already on disk — no item below requires data we do not have, unless marked ⛔ |
| **Output** | the exact directory + CSV that must be written. **Non-negotiable:** one dated directory per item, a `RESULTS.md` beside the CSVs, and the generating script under `scripts/postanalyze/` |
| **Compare** | the specific comparison the output enables |
| **Accept** | what result would count as closing the item — decided *before* the run |

Priority bands:

- **P0 — crucial.** A referee blocks acceptance on these at any venue. Do these first.
- **P1 — major.** Each needs real new work; each materially changes what the paper can claim.
- **P2 — moderate.** Fixable inside one revision cycle, mostly re-analysis or writing.
- **P3 — editorial.** Text and template. Cheap, but currently disqualifying in aggregate.

Status keys: `TODO` · `RUNNING` · `DONE` · `PARTIAL` (assets exist, see note) · `⛔BLOCKED` ·
`DEFER` (deliberately not now).

---

## 0.5 P0-PHASE — the phasing rework (opened 2026-08-23)

**Status: ACTIVE, highest priority. Side-car only — the shipped pipeline is not modified.**

### The discovery

`run_beagle.py` runs Beagle 4.1 with **`niterations=0`** and **`impute=false`**
(`BEAGLE_PARAMS` lines 96-99). `niterations=0` disables the Li-Stephens phasing iterations
entirely, so Beagle is used as a panel-based genotype *refiner*, never as a phaser. The
signature is visible in every output record: `AR2=0;DR2=0`. Confirmed empirically —
`chr10.vcf.gz` is 2,131/2,131 unphased, `all_filtered_in.vcf.gz` is 20,000/20,000 unphased,
and no `PS`/`HP`/`|` appears anywhere.

Two consequences compound it:
- `run_beagle.py` deletes Beagle's raw output (`os.remove(beagle_output)`), so no phased
  artifact survives even transiently. Zero `.temp.vcf.gz` remain on disk.
- `run_beagle_genotype_shifting.py` hardcodes `valid_genotypes = ['0/0','0/1','1/1']` and
  `target_genotypes = ['0/0','0/1','1/1']` — unphased strings only. Even if phase were
  produced upstream, step 3 would collapse it.

**NOT the cause (checked, and an earlier note in this file said otherwise):**
`merge_vcf_fields()` runs `bcftools annotate` with Beagle's output as the *base* file and
`GT` absent from its `-c` list, so the merge does **not** overwrite GT. The merge is innocent.

**Manuscript impact.** Methods state Beagle is used "to resolve **and phase** 1KGP-panel
genotypes", and Results state CalicoST runs "on the same phased genotypes". As configured,
no phased genotypes exist. This must be corrected regardless of whether the rework succeeds.

### Why phasing is the right axis

A haplotype *ratio* is conditioned on detection; a presence/absence call **is** a detection
event. That is precisely the confound behind C1 — spot-level variant presence is a function of
transcript capture, so recovered structure may be re-derived expression. Phased BAF is immune
to that confound by construction. Phasing also supplies SparcalNet's first **independent label
source** (read-backed phase), which is the fix for the label leakage that makes its current
99.8% accuracy meaningless.

### Non-destructive rule (user decision 2026-08-23)

- **Side-car only.** New code lives in a parallel tree and writes to parallel output
  directories. `scripts/1_calling/` … `scripts/7_spatial_filter/` are **not** edited.
- **Frozen baseline.** `figs/v5_2026-08-23/`, `data/sparcalnet_eval_2026-08-23/`, and
  `data/somatic_calibration_2026-08-23/` are the pre-phasing comparison set. Do not overwrite.
- **New outputs** go to new dated dirs and `figs/v6_*`.
- **Snapshot before anything.** Record a manifest + checksums of the `current` variant sets so
  any later drift is detectable.
- **FUTURE MERGE MARKER:** if phasing and/or the UMI-family threshold prove out, fold them into
  the shipped pipeline as a deliberate, separately reviewed change — patching `run_beagle.py`
  (`niterations`, retain raw output) and making genotype-shifting phase-aware. Until that
  review happens, the side-car is the only place these features exist.

### Work items

| ID | Item | Depends on |
|---|---|---|
| **P-0** | Side-car re-run of Beagle with `niterations>0`, retaining raw phased output in a parallel tree. First deliverable is a **one-chromosome probe** answering: does Beagle emit phase at all on this data, and at what rate? | — |
| **P-FEAS** | Feasibility count: het-pair distance distribution, UMI-linkage yield, and **what fraction of the somatic class has ≥2 ALT reads AND a phased het in linkage range**. Decides whether P-1/P-2 are a strategy or a footnote | P-0 |
| **P-1** | Read-backed phase concordance → switch-error rate → germline validation with no clustering and no external truth | P-0, P-FEAS |
| **P-2** | Haplotype-linkage somatic test: do ALT reads consistently carry one germline haplotype? Artifacts have no haplotype preference. A per-variant discriminator independent of the spatial cascade — the calibrated test C2 found missing | P-0, P-FEAS |
| **P-3** | Phased-BAF spatial representation. Encode as depth-shrunk signed deviation `d=(alt-ref)/(alt+ref+k)` so null is 0 and zero-fill is correct; disable count normalization; fall back to mirrored abs BAF if haplotype sign proves inconsistent across blocks. Re-run the STAGATE benchmark. Attacks C1 | P-0 |
| **P-4** | Beagle `GP`/`DS`/`AR2`/`DR2` calibration vs matched WES. These fields are computed and then dropped downstream — free features. Closes part of M2/M8 | P-0 |
| **P-5** | SparcalNet rebuild on phase-derived labels + phasing features | P-1, P-2 |
| **P-6** | Somatic + genotype integration | P-2, P-5 |

### Parallel track — independent of phasing, same priority tier

| ID | Item | Why |
|---|---|---|
| **X-1** | **Moran's I with a permutation null, replacing the ad-hoc ζ.** Produces a calibrated per-variant p-value instead of a rank, which yields a real FDR and **retires the top-10% quota**. Highest-leverage non-phasing change; fixes C2 at the root | Highest |
| **X-2** | **UMI-family consensus.** RT errors occur at first-strand synthesis and therefore live in exactly one UMI family by construction. Requiring ≥2 independent families is a principled filter aimed at the single-molecule majority (75–92% of calls), and gives a fair, evidence-based answer to the Monopogen depth-floor question (M9) | High |
| **X-3** | Mutational spectrum + RNA-editing mask as a **model component**, not a control (M3) | Medium |
| **X-4** | Distance from transcript 3′ end as a feature — terminal positions are artifact-enriched (M4) | Medium |
| **X-5** | Cross-section recurrence (DCIS1/DCIS2 are two sections; currently unused) | Medium |
| **X-6** | Ancestry-matched 1KGP super-population for germline resolution (M8) | Low |

### Sequencing

P-0 → P-FEAS → (X-1, X-2 in parallel, they need no phase) → P-2 → P-3 → P-5 → P-6.

---

## 1. P0 — crucial. Nothing ships before these.

Ordered by decisiveness, not by cost. **P0-1 and P0-2 are the two the referee said block
acceptance at any venue**, and P0-2 is the ablation explicitly asked for.

### P0-1 · DLPFC coverage and detection controls  ·  closes C1  ·  `TODO`

The single most important run in this queue. Every DLPFC claim rests on a comparison that has
never had a negative control, and the region-detection analysis already shows that a coverage
baseline wins when one is supplied.

- **Do** — run the identical STAGATE → mclust pipeline (same graph, same preprocessing, same
  10 runs/section, all 12 sections) on four new control matrices:
  1. `coverage_only` — one feature per spot, log total UMI, broadcast to the same 250-kb bin grid.
  2. `detection_only` — binarized *expression* of exactly the genes carrying the SPARCAL variants,
     binned identically. Genotype discarded; detection pattern preserved.
  3. `allele_permuted` — the SPARCAL matrix with genotypes shuffled within each bin, detection
     pattern held fixed.
  4. `smoothed_random` — Gaussian noise matrix of matched dimension and sparsity, to calibrate how
     much layer purity STAGATE's own spatial smoothing manufactures.
- **Why** — C1. Also calibrates m4 (run-to-run noise) and feeds C7.
- **Inputs** — `data/dlpfc/{section}/`; the same harness as `clustering_benchmark/`
  (env `snv_clustering`); bin grid from `data/dlpfc_binsize_multisection/`.
- **Output** — `data/dlpfc_negative_controls_2026-08-DD/`
  → `control_ari_knn_long.csv` (section × modality × run: ARI, kNN purity k=10 and k=30, seed),
  `control_summary.csv` (per modality: mean, sd, n, paired Wilcoxon vs SPARCAL),
  `RESULTS.md`. Script `clustering_benchmark/make_negative_controls.py`.
- **Compare** — SPARCAL spatially-augmented 250 kb (ARI 0.350, kNN 0.856) and SPARCAL 1KGP-only
  250 kb (ARI 0.363) against all four controls, on the *same* metric in the *same* embedding.
- **Accept** — SPARCAL must exceed `detection_only` by more than the run-to-run SD (0.041) on
  kNN purity, with donor-level statistics (P0-4). **If it does not, the modality claim in the
  title cannot stand** and the DLPFC section becomes a representation/limits result. Record the
  outcome either way — a null here is a publishable finding, not a failure.

### P0-2 · Pipeline component ablation  ·  closes C3  ·  `TODO`

Explicitly requested. Nobody can currently tell which part of SPARCAL does any work.

- **Do** — run all four tumor sections through five configurations, holding candidates,
  covered positions, and spot set constant:

  | config | error model | SparcalNet | Stage 1 | Stage 2 descriptors |
  |---|---|---|---|---|
  | `A_errmodel` | ✓ | — | — | — |
  | `B_sparcalnet` | ✓ | ✓ | — | — |
  | `C_stage1` | ✓ | ✓ | ✓ | — |
  | `D_stage2_nocnv` | ✓ | ✓ | ✓ | δ, ζ, ε |
  | `E_stage2_cnv` (shipped) | ✓ | ✓ | ✓ | δ, ζ, ε, θ |

- **Why** — C3, and it also supplies the descriptor-correlation matrix M7 needs.
- **Inputs** — the `current` post-dedup candidate sets (Decision D1); CalicoST outputs already
  on disk for all four sections.
- **Output** — `data/pipeline_ablation_2026-08-DD/`
  → `ablation_counts.csv` (config × section × class: n variants),
  `ablation_support.csv` (≥2 ALT reads %, ≥2 spots %, median depth),
  `ablation_cosmic.csv` (hit rate, xMHC-in/out, somatic-vs-unresolved ratio + Fisher P),
  `ablation_descriptor_corr.csv` (Spearman between δ, ζ, ε, θ, η per section),
  `RESULTS.md`.
- **Compare** — every downstream number in the paper, recomputed per configuration. Specifically:
  does θ (the CNV/LOH evidence, the paper's stated novelty) change anything relative to
  `D_stage2_nocnv`?
- **Accept** — a table showing each component's marginal contribution. **If θ moves nothing, the
  "CNV-guided calling" thesis (PAPER_PLAN §1) has no empirical support and must be restated.**
  Note this is the ablation that most directly tests the advisor's stated contribution.

### P0-3 · Somatic caller calibration and null model  ·  closes C2  ·  `TODO`

The classifier currently has no threshold with a meaning: Stage 2 votes on each descriptor's
top 20% and calls the top 10% somatic, so the callset size is an analyst choice
(P6 = 65,655 "somatic" calls from RNA).

- **Do** — two parts, both needed:
  1. **Quota sweep.** Re-run Stage 2 at somatic fractions {1, 2, 5, 10, 20, 30}% and recompute
     COSMIC hit rate, ALT-read support, spot support, and region-detection ARI at each. This costs
     no new calling — it is a re-ranking of an existing score.
  2. **Null.** Estimate a false-positive rate: run the identical pipeline on a section with no
     tumor (a DLPFC section, or the normal-annotated spots of P6) and count how many candidates
     the cascade would call somatic. Every such call is a false positive by construction.
- **Why** — C2. Also gives the FDR curve the paper currently lacks entirely.
- **Inputs** — existing Stage-2 score tables per section; DLPFC `current` matrices for the null.
- **Output** — `data/somatic_calibration_2026-08-DD/`
  → `quota_sweep.csv` (section × quota × metric), `null_fpr.csv`, `RESULTS.md`.
- **Compare** — how every headline somatic number moves as the quota moves; where (if anywhere)
  a defensible operating point sits.
- **Accept** — either a quota-independent result (strong), or an explicit statement of the
  operating point with its estimated FPR (acceptable). **Silence on this is not acceptable.**

### P0-4 · Donor-level DLPFC statistics  ·  closes C7  ·  `DONE 2026-08-23`

The 12 spatialLIBD sections are **3 donors**, 4 sections each, in pairs 10 µm apart. Every paper
p-value treats n=12. No new compute — this is a re-analysis of tables already on disk.

- **Do** — recompute every DLPFC test with donor as the unit of replication: (a) collapse to
  donor means and test at n=3; (b) linear mixed model with donor as random effect and section
  nested within donor. Affected: P=0.519, 0.0015, 4.9e-4, 0.021, 5e-4, 0.052, 0.68, 0.850.
- **Why** — C7.
- **Inputs** — `data/dlpfc/clustering_benchmark/ari_matrix_mean.csv`,
  `data/dlpfc_binsize_multisection/clustering_benchmark/binsize_multisection_long.csv`.
- **Output** — `data/dlpfc_donor_stats_2026-08-DD/` → `donor_level_tests.csv`
  (test, section-level P, donor-mean P, LMM P, effect size, 95% CI), `RESULTS.md`.
- **Compare** — section-level vs donor-level P for each claim in the paper.
- **Accept** — a table stating which comparisons survive. Any that do not must be rewritten or
  dropped. Donor structure must then appear in Methods and in every affected caption.

**Result.** The section-level tests were reproduced, but no comparison is significant by an
exact test on the three donor means (minimum attainable two-sided P=0.25). Local-purity effects
against binned GATK and Strelka2 are positive in all three donors; the gene-expression comparison
is mixed. See `data/dlpfc_donor_stats_2026-08-23/` and
`../SPARCAL_pnas_2026/figs/v5/fig_dlpfc_donor_reanalysis.{pdf,png}`.

### P0-5 · Stage-1 specification and UPV assessment into the SI  ·  closes C4  ·  `DONE 2026-08-23`

The α/β equations and the T_α/T_β thresholds are inside a live `\iffalse` block and appear
nowhere in the submission; the SI has three figures and no body text. UPV — one of four named
output classes — is therefore undefined and unassessed.

- **Do** — (a) move the Stage-1 equations and per-dataset thresholds into `SI_AppendixGuided.tex`
  with a justification for per-dataset tuning; (b) run the promised UPV BAF assessment: recompute
  BAF from I16 pileup counts, fit the two-component mixture, and state how many UPVs are
  artifact-like vs germline-like with an explicit upper bound on a somatic interpretation;
  (c) resolve "the ovarian section" — introduce it or remove it (see also the NCCE assay-scope note:
  probe-based FFPE cannot support SNV claims).
- **Why** — C4.
- **Output** — `data/upv_baf_assessment_2026-08-DD/` → `upv_baf_mixture.csv`, `upv_class_counts.csv`,
  `RESULTS.md`; plus SI text.
- **Accept** — a reader can reimplement Stage 1 from the SI alone, and UPV has a stated
  interpretation with evidence.

**Result.** The live SI now gives the equations and output-verified sample thresholds. The
assessment was expanded beyond the originally proposed mixture to include a 16-setting Stage-1
grid, a 27-setting GMM grid, depth-matched panel controls, complete-case and burden-adjusted
purity associations, exact editing/PON masks, same-build recurrence, direct CB-tagged read-level
$\Delta$BAF, and matched-normal WES support. UPV is retained as an origin-neutral class; the data
do not support an artifact/germline binary split or a somatic-rate estimate. See
`data/upv_baf_assessment_2026-08-23/`, `data/upv_delta_baf_2026-08-23/`, and
`UPV_P1_1_REPORT_2026-08-23.md`.

### P0-6 · Bibliography  ·  closes C5  ·  `DONE 2026-08-23` · *no compute, purely mechanical*

Four `\cite` commands exist in the manuscript; twelve references, eight never cited.

- **Do** — build `references.bib` to ~50–70 entries and cite at first mention. **Must be added:**
  COSMIC, Cancer Gene Census, CalicoST, mclust, samtools/bcftools mpileup, GATK4/Mutect2 (the 2011
  framework paper is not Mutect2), inferCNV, UMAP, 10x Visium, SpaceTracer, the RNA-editing catalog
  and single-cell PoN used in the masking control. **Must be cited (present but unused):** STAGATE,
  Beagle, 1000 Genomes, spatialLIBD, Maynard DLPFC, Ji cSCC, STMut, GraphST.
- **Accept** — zero uncited bib entries, zero uncited tools, no claim of citing something absent
  from the bibliography (currently true of SpaceTracer).

**Result.** `PNAS/references_sparcal.bib` contains 57 topic-relevant records, all cited in the
guided manuscript. DOI/official-record verification and paragraph placements are recorded in
`REFERENCE_AUDIT_2026-08-23.md`; there are zero missing and zero uncited bibliography keys.

### P0-7 · Manuscript completion  ·  closes C6  ·  `PARTIAL`

| item | action | status |
|---|---|---|
| **Fig. 1 pipeline diagram** | **LEAVE EMPTY.** Not finalized; keep the placeholder box and do not block anything else on it. Revisit once the ablation (P0-2) fixes what the diagram must show — drawing it before then would draw the wrong pipeline | `DEFER` — deliberate |
| Fig. 6 region detection | Rebuild on current results: add DCIS1 (28 foci) and Monopogen-somatic on DCIS. The shipped asset is `v2_2026-07-29`, predating both | `TODO` |
| DCIS1 has no figure | It is the only caller-independent annotation and the only whole-section one — it must be plotted, not text-only | `TODO` |
| Fig. 6 == SI Fig. S3 | Same file submitted twice; SI S3 is never cited from the main text. Pick one location | `TODO` |
| Fig. 5 panels (c)/(d) | Different extents — (d) is a crop, (c) whole section. Re-export at matched extent | `TODO` |
| Caption production notes | Delete "Nimbus Sans previews pending Arial" and "a five-trial box plot remains pending" from shipped captions | `TODO` |
| Length | 16 pp. compiled, 2,349-word Methods in main text | `TODO` |

---

## 2. P1 — major. Each changes what the paper can claim.

### P1-1 · Germline concordance against matched-normal WES  ·  closes M2  ·  `PARTIAL 2026-08-23` · ⭐ *highest value-per-day item in this file*

The one validation these data actually support, and currently a single subordinate clause in the
Discussion. P4 and P6 have matched normal exomes.

- **Do** — genotype concordance at expressed, adequately covered sites: sensitivity, precision,
  het/hom accuracy, stratified by RNA depth (1–3, 4–9, 10–29, 30+ reads) and by 1KGP panel
  membership (defined vs de novo).
- **Inputs** — `.../ST_datasets/STmut_Data/P{4,6}_Somatic_{Mutect2,GATK}/` and the matched normal
  BAMs; SPARCAL germline VCFs on the `current` sets.
- **Output** — `data/germline_concordance_2026-08-DD/` → `concordance_by_depth.csv`,
  `concordance_summary.csv`, `RESULTS.md`.
- **Compare** — SPARCAL germline vs WES truth; and the same metric for GATK and Strelka2 on the
  same sites, so the comparison is three-way rather than self-referential.
- **Accept** — if concordance is good at ≥10× RNA depth, **this becomes the paper's headline
  positive result** and the DLPFC section rests on a validated matrix. Note guardrail 6:
  the earlier audit gave 51–56% agreement — that number needs to be understood before it is
  reported, because as stated it does not support a strong claim.

**Result.** The matched-normal analysis is complete for SPARCAL, including all truth-site misses,
RNA-depth and WES-GT strata, exact 1KGP membership, raw GT, the small evaluable SparcalNet subset,
and reverse WES pileup of final calls. At RNA depth $\geq10$, exact-allele sensitivity is 0.718
(P4) and 0.666 (P6) overall, 0.844/0.796 for 1KGP alleles, and 0.059/0.016 for non-1KGP alleles.
Raw-GT agreement among detections is 0.916/0.875. See
`data/germline_concordance_2026-08-23/` and `UPV_P1_1_REPORT_2026-08-23.md`. Status remains
`PARTIAL` only because no current P4/P6 RNA-derived GATK or Strelka2 callset was found; the requested
same-input three-way comparator requires new harmonized caller runs and was not substituted with
tumor-WES or DLPFC calls.

### P1-2 · Mutational spectrum of every output class  ·  closes M3  ·  `TODO` · *cheap, high signal*

The first artifact check for any RNA-derived callset, entirely absent.

- **Do** — six-channel and 96-channel trinucleotide spectra for germline / UPV / retained somatic /
  unresolved, all four sections. Plus strand bias, position-in-read, and confirmation that ALT
  reads come from distinct UMIs rather than duplicates of one molecule.
- **Inputs** — existing class-stratified VCFs; reference FASTA per build.
- **Output** — `data/mutational_spectrum_2026-08-DD/` → `spectrum_6channel.csv`,
  `spectrum_96channel.csv`, `strand_bias.csv`, `RESULTS.md`.
- **Compare** — the somatic class against the germline class (which should look like a population
  spectrum) and against a COSMIC SBS reference.
- **Accept** — an explicit statement of the A>G/T>C fraction. **If the somatic class is A>G
  dominated it is an editing catalog**, and that must be reported. 2 of the 14 WES-corroborated
  calls are already A>G.

### P1-3 · Capture-geometry waterfall — decompose the 1%  ·  closes M4  ·  `PARTIAL` ⭐

Substantially already done and parked. `PAPER_PLAN_DEPRECATED.md` §1.3 axis 4 has the 3′-shift
measurement (SPARCAL 0.528 vs 0.499 gene-length-matched null; SpatialSNV 0.547) and the
expression bias (called genes ~60× median UMI) in `p6_3prime_bias_summary.csv` and
`p6_gene_expression_bias.csv`.

- **Do** — assemble those into a staged waterfall, per patient: of matched-WES somatic sites,
  what fraction lie in a gene expressed in the section → of those, within the 3′ capture window →
  of those, with ≥1 read → of those, showing the ALT allele. Each stage as a percentage.
- **Why** — M4. Converts "Visium covers ~1% of exome somatic positions" from a conflated number
  into a mechanism. This is the most transferable result in the whole project.
- **Output** — `data/capture_geometry_2026-08-DD/` → `waterfall_p4.csv`, `waterfall_p6.csv`,
  `RESULTS.md`.
- **Compare** — the ceiling this sets for 3′-capture chemistry generally, versus full-length or
  higher-resolution platforms.
- **Accept** — four stage percentages per patient that sum coherently to the observed ~1%.

### P1-4 · COSMIC same-basis comparison  ·  closes M5  ·  `DONE 2026-09-06`

- **Decision** — compare all classes and callers with the same COSMIC build and allele-exact
  matching definition; retain raw and xMHC-excluded summaries.
- **Writing rule** — describe the result as an external catalogue comparison/class separation,
  not as variant-level validation or evidence that individual calls are cancer-driving.
- **No additional analysis** — no separate COSMIC model or control panel is scheduled.

### P1-5 · Monopogen depth-floor matched ablation  ·  closes M9  ·  `TODO` · *one run, settles an assertion*

The paper asserts Monopogen's better support statistics are "a threshold difference rather than
better discrimination." That is currently an explanation, not a result.

- **Do** — apply Monopogen's floor (≥4 high-quality REF and ≥4 high-quality ALT bases) to SPARCAL's
  candidates and re-compare support statistics, COSMIC rates, and region-detection ARI on all four
  sections.
- **Output** — `data/monopogen_matched_floor_2026-08-DD/` → `matched_floor_support.csv`,
  `matched_floor_cosmic.csv`, `RESULTS.md`.
- **Accept** — if SPARCAL under a matched floor reproduces Monopogen's support profile, the
  assertion is proven and both directions of the comparison become fair.

### P1-6 · Leaked-allele confusion table  ·  closes M1  ·  `DONE 2026-08-23`

"SPARCAL assigned 64–82% to germline" is tautological (those alleles are in the 1KGP panel by
definition) and its complement is unreported.

- **Do** — for the leaked SpatialSNV alleles in each section, a full breakdown: not detected /
  germline / UPV / retained somatic / unresolved, with denominators.
- **Output** — `data/leaked_allele_confusion_2026-08-DD/` → `confusion_by_section.csv`, `RESULTS.md`.
- **Accept** — the somatic-misassignment rate on known common polymorphism. **This is a directly
  measured false-positive rate and is far more informative than the current number.**

**Result.** Across 29,882 allele-exact 1KGP matches, SPARCAL assigned zero to retained somatic,
zero to unresolved, and one DCIS2 allele to UPV; all remaining alleles were germline or not
detected. See `data/leaked_allele_confusion_2026-08-23/` and
`../SPARCAL_pnas_2026/figs/v5/fig_leaked_allele_confusion.{pdf,png}`.

### P1-7 · CalicoST circularity and descriptor independence  ·  closes M7  ·  `PARTIAL`

CalicoST infers purity from allele-specific expression in the same BAM that supplies the variants;
SPARCAL then ranks candidates by correlation with that purity. Predictor and evidence share a source.

- **Do** — (a) the descriptor correlation matrix, which P0-2 already produces; (b) substitute an
  orthogonal purity estimate — pathology tumor fraction, or expression deconvolution — and recompute
  the ranking. ⛔ A DNA-derived purity estimate is not available for these sections.
- **Output** — fold into `data/pipeline_ablation_2026-08-DD/` plus
  `orthogonal_purity_2026-08-DD/`.
- **Accept** — the somatic ranking is shown to be stable under a purity estimate that does not
  come from the same reads, or the circularity is stated as a limitation.

### P1-8 · Beagle validation and ancestry  ·  closes M8  ·  `TODO`

- **Do** — report imputation accuracy against WES truth (this falls out of P1-1), and state the
  ancestry composition of the donors and patients. Discuss panel-driven bias: variants private to
  under-represented ancestries fail to resolve as germline and flow into the de novo pool — and
  therefore potentially into the somatic class.
- **Accept** — a limitation paragraph grounded in a number, not a hedge.

### P1-9 · Independent section  ·  closes M10  ·  `⛔BLOCKED` / `DEFER`

DCIS2 — the section where SpatialSNV wins — is from the SpatialSNV publication's own dataset.
An independent cSCC or DCIS section from a public resource neither method was developed on would
fix this. **Requires new data; not achievable on the current timeline.** Record as a stated
limitation rather than pretending independence.

---

## 3. P2 — moderate

| ID | Item | Action | Cost |
|---|---|---|---|
| m1 | Title/claim mismatch | Settle after P0-1 lands. The title claims what eight pages decline to claim; the control decides which way it resolves | writing |
| m2 | Selected example sections | Fig. 2a–c shows three winners at their best-ARI runs. Add a median-ARI section, or show all 12 as small multiples | plotting |
| m3 | Viewer score discrepancy | Text quotes ARI 0.594 / 17-of-28 / mean J 0.377; the source note records `ours` = 0.604 / 13-of-28 / 0.310. A second profile `our_best` exists. **Name the profile reported and say how it was chosen** | 1 h |
| m4 | Run-to-run spread not shown | Median within-cell SD is 0.041; several tested differences are inside that band per run. Report spread beside every ARI/purity value (house style already requires this) | plotting |
| m5 | Mixed reference builds | GRCh37 for P4/P6, GRCh38 for DCIS. xMHC chr6:28–34 Mb is not the same interval in both; COSMIC releases differ. State the handling | writing |
| m6 | No dataset table | Add Table 1: spots, median UMIs/spot, median reads/spot, saturation, candidate positions, class counts, per section | 2 h |
| m7 | Front matter | W.Y. and P.B. have no contribution statement; X.M./X.M.Z. initials ambiguous; 6 keywords vs limit 5; no ethics statement; data availability has no accessions, DOI, version tag, or license | 2 h |
| m8 | Template misuse | Methods sit in a plain `\section*{Methods}` while `\matmethods{}` holds only data availability, so the compiled paper has a near-empty "Materials and Methods" after a 2,349-word Methods. Also: live `\PENDING` macro, dead `\iffalse` block, `LastPage` multiply-defined, 5 overfull boxes | 2 h |

---

## 4. P3 — editorial

- Section headings are process statements ("We tested…", "We compared…"). Seven of eight Results
  headings begin "We". Headings should state findings.
- "We observed" opens a large fraction of paragraphs.
- "Ubiquitous private variant (UPV)" is coined terminology used in the abstract without definition.
- Abstract 249/250 words, significance 119/120 — no headroom for anything this queue adds.
- The Discussion cites a read-quality evidence screen that is reported nowhere. Report or remove.
- SpaceTracer: "we cite but did not benchmark" with no citation given.
- Methods describe the same two-stage cascade three times.
- Strip all guided/TODO source comments — they currently enumerate every unfinished item.

---

## 5. Fix-now list — no compute, can be applied in one pass

These are unambiguous errors, not judgment calls:

1. **Discussion number error.** "250-kb binning improved mean ARI by approximately 0.14 for the
   1KGP-only, GATK, and Strelka2 matrices" → the true values are +0.097 / +0.142 / +0.064. The GATK
   value was applied to all three. Replace with "by 0.064–0.142".
2. **Viewer profile mismatch** (m3) — name the profile and use its numbers consistently.
3. **Broken SI forward references.** Methods promise threshold definitions and UPV BAF analyses "in
   the Supplementary Information"; the SI has no body text. Either add (P0-5) or remove the promise.
4. **Move the Stage-1 `\iffalse` block** into the SI source.
5. **Delete production notes from captions.**
6. **Remove the duplicate** of Fig. 6 / SI Fig. S3.
7. **Main-text bin-width account** is a partial version of its own SI (omits 500 kb ×2, 25 kb ×1,
   100 kb ×1, and the 100 kb ≈ 250 kb tie at cohort level).

**Verified clean — do not re-check:** every leakage percentage, WES-overlap fraction, and COSMIC
count in the manuscript recomputes exactly from its stated numerator and denominator.

---

## 6. Sequencing the queue

If everything runs, roughly:

| Wave | Items | Gate |
|---|---|---|
| **A — today, no compute** | P0-4, P0-6, §5 fix-now list, m3, m5, m6, m7, m8 | none |
| **B — launch immediately, they are the long poles** | P0-1, P0-2, P0-3 | cluster time |
| **C — cheap runs, high signal** | P1-2, P1-5, P1-6, P1-3 (mostly assembled already) | none |
| **D — after B lands** | P1-1, P1-4, P1-7, P0-5 | P0-2 output |
| **E — after A–D** | P0-7 figure rebuilds, m1, m2, m4, P3 | all results final |

**Fig. 1 is deliberately last and may stay empty** — the diagram should not be drawn until P0-2
establishes what the pipeline's components actually contribute.

---

## 7. Stale pointers to fix while you are in here

- **PAPER_PLAN §5 "MANUSCRIPT FILE ARRANGEMENT"** still names `PNAS/PaperDraft_v2.tex` as
  the live manuscript. The live file is now `PNAS/PaperDraftGuided.tex` with
  `PNAS/SI_AppendixGuided.tex` (see the manuscript repo's `CLAUDE.md`). Anyone following
  PAPER_PLAN will edit the wrong file.
- **PAPER_PLAN §6 guardrail 7** says region detection omits the coverage baseline "by editorial
  scope" (Decision L3). L3 is superseded — the baseline is plotted and discussed in the guided
  draft, and the SI note says so. Reconcile the guardrail with the decision.
