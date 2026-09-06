# On-going Tasks

Dynamically updated task list. When asked to "check ongoing tasks", Claude should:
1. Run `squeue -u chowx` (or `-j <job_ids>`) to check SLURM status
2. Check output files/logs for results
3. Update this file accordingly

> **Cleaned 2026-07-12.** Trimmed to what is *grounded and paper-bound* + *what we plan to
> do next* + *closed questions (do-not-rerun)*. The full blow-by-blow of the 50+ DLPFC
> representation experiments, the UMI-dedup ablation, OVAR_P5/SpaceTracer debugging, and all
> resolved job-tracking lives in git history (`On_going.md` before commit of 2026-07-12) and in
> the weekly report at
> `data/dlpfc_recovery_test/151507/clustering_benchmark/weekly_report/DLPFC_SNV_representation_report.md`.

**No SLURM jobs currently running** (checked 2026-07-12).

---

## ⭐ THE OPEN QUESTION (drives everything below): does SPARCAL's *model* add value a caller + coverage can't?

Neither benchmark we have actually demonstrates the novelty of SPARCAL's classifier (the
UPV/somatic denovo calls). Both are honest results, but both point *away* from the model:

1. **DLPFC germline clustering** — the winning SNV representation is the **1kG-defined set only**
   (`defined_bin250kb`). Adding SPARCAL's own UPV/somatic calls does **not** help on normal cortex
   (expected — no clonal structure to encode there). So this benchmark validates the *germline
   representation*, not the model.
2. **Tumor region detection (DCIS2)** — **coverage/UMI alone matches-or-beats every SNV-burden
   method** (`coverage_only` ARI 0.657 ≥ best SNV 0.629); after coverage-normalization the SNV
   signal collapses (~0.27–0.31). So region *detection* is a coverage story, not an SNV story.

**Consequence for the paper:** we cannot claim "our SNVs / our model find the tumor" from region
detection (coverage does it), nor claim model value from DLPFC (germline-only wins there). The
model's real, unique value — **sub-clonal / allele-specific / phylogenetic structure that coverage
cannot provide** — is **not yet demonstrated by any benchmark we have run.** Closing this is the
priority. See "Planned work" → clonal-structure benchmark.

---

## 🧭 PLAN (2026-07-12, FINALIZED) — validating the UPV & somatic sets

**Goal:** show that SPARCAL's *model output* (UPV + somatic; **1kG excluded — it's Beagle, not our
model, used only as a negative-control baseline**) carries real tumor/clonal signal that a coverage
baseline and standard germline callers cannot produce. Answers the ⭐ open question above.

**Resources found (2026-07-12):**
- **Weiman's `ST_CNV/` multi-tool tumor-region benchmark** (symlinked into repo as `copyKAT/` →
  `/data/maiziezhou_lab/Weiman/ST_CNV/`). **5 expression/CNV tools** run per-spot on P4(rep1/2),
  P6(rep1/2), DCIS1, DCIS2: **copyKAT, clonalScope, inferCNV, numbat, siCNV** — each emits
  malignant/normal spot labels (e.g. `siCNV/{s}/{s}_copykat_annotations.tsv`,
  `inferCNV/{s}/infercnv_annotations.txt`). Precision/recall/F1 vs pathology GT exist for the public
  datasets (Gao2021/Lee2020/…); **no single consensus/vote file exists yet** (only per-tool plots).
  **CalicoST is NOT in this set** and is judged unreliable → do not use it as GT.
- **DCIS2 pathology GT** (10 foci / 249 spots) in `SPARCAL_Benchmarking/analysis/region_method_benchmark/`;
  pankaj RCTD cell-type annotations for dcis2.
- **COSMIC hits** already computed (`/data/maiziezhou_lab/leiy4/COSMIC/`): P4 12.5% / P6 1.0% /
  DCIS1 1.36% / DCIS2 1.50% — but NOT yet split by 1kG vs UPV vs somatic.

**Proposed 3 validation axes (UPV & somatic, with 1kG as negative control):**

1. **COSMIC enrichment gradient (variant-level, same-basis external comparison).**
   Fraction of each set that are COSMIC cancer-mutation hits; hypothesis **somatic > UPV > 1kG**
   (1kG germline → should be at background). Enrichment test vs 1kG baseline (and vs random loci of
   matched trinucleotide/mappability). This isolates the classifier from Beagle and needs **no region
   GT at all.** *Co-headline (with Axis 2), per decision 2026-07-12.*
2. **Coverage-CONTROLLED spatial concordance (region-level, cross-modal).** Do UPV/somatic spots
   coincide with an **independent (expression-CNV) tumor-region label**, *after removing coverage*?
   Metric: logistic `tumor_label ~ somatic_burden + total_UMI` (is the somatic coefficient
   significant beyond coverage?) and/or within-coverage-stratum enrichment. **Coverage control is
   mandatory** — raw burden concordance is the confounded result we already have. GT tiers:
   **pathology (gold) where it exists (DCIS2)**; **calibrated CNV-tool consensus (silver)** elsewhere
   (P4/P6/DCIS1). Cross-modal (CNV vs SNV) ⇒ orthogonal, not circular.
3. **Sub-clonal agreement (stretch).** Compare SPARCAL somatic-derived spot clusters vs the CNV
   tools' clone assignments (numbat/clonalScope emit clones): does somatic resolve substructure
   coverage/expression can't?

**On "tool vote as ground truth" (user's question):** yes as a **silver standard**, but NOT a naive
equal-weight majority (the 5 tools disagree wildly per-sample, F1 0.0–0.99). **Calibrate the
consensus rule against pathology** where pathology exists (DCIS2 + public), freeze that rule, then
apply it where pathology is absent — and report it as cross-modal (expression-CNV) evidence, with the
coverage control doing the real rigor. Pathology stays primary where available.

**DECISIONS LOCKED (user, 2026-07-12):** (a) **co-headline** the COSMIC gradient AND the
coverage-controlled spatial concordance; (b) GT = **calibrated CNV-consensus + pathology** (pathology
primary where it exists); (c) scope = **P4rep1, P6rep1, DCIS1, DCIS2** now, **OVAR_P5 later** (user
will ask Weiman to run the 5 CNV tools on it — not yet available).

**Sequenced steps (Claude builds; user submits any sbatch):**

*Axis 1 — COSMIC gradient (start here; independent of Weiman, works on VCFs so P6 is fine):*
1. Per sample, take the categorized `germline_defined`(1kG) / `germline_denovo`(UPV) /
   `somatic_denovo`(somatic) VCFs; intersect each with COSMIC v103; compute per-set hit fraction.
   Test **somatic > UPV > 1kG** with a matched-random-loci null (trinucleotide/mappability-matched)
   for significance, using 1kG as the germline-background baseline.
2. Gene-level: are somatic COSMIC hits in known cancer genes (cSCC: KRT6B/SPINK5 already seen)?
   Fix the `COSMIC_validation.py` `GENE=.` bug first, or reuse the manual `isec`+`0003.vcf` path.

*Axis 2 — coverage-controlled spatial concordance (needs Weiman's per-tool labels + P6 matrix rebuild):*
3. Build the **calibrated CNV-consensus tumor-region label** from Weiman's 5 per-spot annotation
   TSVs (`ST_CNV/{siCNV,inferCNV,numbat,clonalscope,copyKAT}/{sample}/…`). Calibrate the consensus
   rule (tool subset / threshold) against **DCIS2 pathology** (+ the public-data F1 Weiman already
   scored), freeze it, apply to P4rep1/P6rep1/DCIS1/DCIS2.
4. Per-spot UPV/somatic burden (1kG as baseline) + per-spot total UMI (coverage) from SpaceRanger.
   **Rebuild the broken P6 SPARCAL matrix first** (or compute burden from `vcf_by_spot`).
5. Fit `tumor_label ~ set_burden + total_UMI` (and within-coverage-stratum enrichment): does
   UPV/somatic burden predict the tumor region **beyond coverage**? Compare somatic vs UPV vs 1kG
   coefficients; include the known-strong `coverage_only` baseline for reference. **Coverage control
   is mandatory** — raw concordance is the confounded result we already have.

*Axis 3 — sub-clonal agreement (stretch, after 1–2):* ARI/NMI of SPARCAL somatic-derived spot
clusters vs numbat/clonalScope clone labels.

**Coordination / prereqs:** confirm read access to Weiman's per-tool per-spot annotation TSVs for the
4 samples; ask Weiman to run the 5 CNV tools on OVAR_P5 (future); P6 SPARCAL matrix rebuild (Axis 2
only); all tumor data is **pre-UMI-dedup**.

---

## ✅ GROUNDED — DLPFC germline SNV clustering (positive control, paper-ready)

12 sections × 10 runs, `data/dlpfc/clustering_benchmark/ari_matrix_mean.csv` (fact-checked
2026-07-12, all numbers reproduce from the per-section `summary.csv`):

| Modality | 12-section mean ARI |
|---|---|
| gene_expr (upper reference) | 0.412 |
| **defined_bin250kb** (1kG-only, 250kb-binned) | **0.363** |
| gatk_bin250kb | 0.354 |
| strelka2_bin250kb | 0.257 |
| sparcal (raw, current pipeline = 1kG+UPV) | 0.217 |
| gatk (raw) | 0.211 |
| strelka2 (raw) | 0.193 |

Paired per-section Wilcoxon (see [[project_sparcal_vs_gatk_statistical_tie]]):
- **SPARCAL ≈ GATK — a statistical tie**, raw (p=0.52) *and* binned (p=0.68); mean diff +0.004–0.009,
  an order of magnitude below run-to-run std. **Do NOT claim SPARCAL > GATK from this benchmark.**
- **Both beat Strelka2**, but only *significantly* once binned (defined vs strelka2-bin p=0.001,
  gatk vs strelka2-bin p=0.002). GATK-beats-Strelka2 too → not SPARCAL-specific.
- **Binning (250kb) is the dominant, robust lever** (~+0.14 ARI for *every* caller; swept 25kb–1Mb,
  clean inverted-U peak at 250kb). Representation matters far more than caller identity.

**Framing for the paper:** DLPFC clustering is a **positive control** — SNV-based spatial clustering
is valid and SPARCAL's germline calls recover known cortical anatomy on par with GATK and above
Strelka2. It is **not** the demonstration of the model's novel calls.

**Defensible headline sentence:** *"SPARCAL is on par with GATK and both clearly outperform Strelka2
for spatial-domain recovery once genomically binned; the representation choice (binning) matters far
more than which caller produced the variant set."*

---

## ✅ GROUNDED — tumor cross-caller comparison (SPARCAL_Benchmarking, pre-UMI-dedup)

Repo: `/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/` (own `CLAUDE.md`). See
[[project_sparcal_benchmarking_ecosystem]] and [[project_paper_benchmark_strategy]].

- **Variant-set overlap (P4/DCIS, `6_compare_matrices.py`):** SPARCAL ∩ **Monopogen** Jaccard
  **0.587** (P4; two germline-leaning callers corroborate — good sanity result). SPARCAL ∩
  **SpatialSNV** Jaccard only **0.016–0.036** — SpatialSNV (Mutect2 somatic regime) genuinely calls
  *different loci*; ruled out as a key-format/coordinate artifact.
- **Moran's I (P4):** SpatialSNV's private calls are **not noise** — coverage-independent spatial
  structure survives de-confounding (residual I 0.351). But raw-burden Moran's I is confounded by a
  smooth per-spot coverage gradient (burden ~ UMI r=0.87–0.97) for *all* callers.
- **Region detection (DCIS2, GT = 10 pathologist foci / 249 spots,
  `analysis/region_method_benchmark/`):** `coverage_only` raw ARI **0.657** / F1 0.807 ≥ every
  SNV-burden method (best SNV merged 0.629 / 0.787). **Region detection is coverage-driven** —
  this is the caveat that blocks any "somatic burden reveals the tumor region" claim.
- **COSMIC somatic hit rates** (`/data/maiziezhou_lab/leiy4/COSMIC/`): P4 12.5%, P6 1.0%,
  DCIS1 1.36%, DCIS2 1.50%; recurrent skin-barrier gene hits (KRT6B, SPINK5) shared across the two
  cSCC patients (P4+P6).

**Known data caveat:** P6 SPARCAL `germline_denovo` matrix was stale/broken (1.25M cols vs true 291)
when these were built — rebuild before any P6 SPARCAL matrix claim. All tumor numbers are
**pre-UMI-dedup**.

---

## 🔜 PLANNED WORK (in priority order)

1. **Clonal-structure tumor benchmark — THE priority.** Design a benchmark where SNVs beat a
   coverage baseline because they resolve *sub-clonal / allele-specific / phylogenetic* structure,
   validated against CalicoST clone labels or CNV segmentation (P4/P6/DCIS). This is the experiment
   that would actually demonstrate the model's unique value (see ⭐ open question). **Not yet scoped
   into runs** — needs a ground-truth choice + a metric that is coverage-independent by
   construction. Discuss framing before building.
2. **BAF-GMM UPV sub-filter (step 7c)** — drafted
   (`scripts/6_spatial_filter/upv_baf_gmm_subfilter.py`), exploratory only: PURITY_CORR is
   uninformative within UPV → effectively BAF-only; dcis1's low-BAF mode is likely ASE-skewed
   germline het, not somatic. Planned upgrade: replace PURITY_CORR with a per-clone ΔBAF
   (tumor−normal) contrast. Not a reportable result yet.
3. **CHIP rule-out** on the final somatic set — drop variants in common CHIP genes (DNMT3A, TET2,
   ASXL1, JAK2, TP53, SF3B1, …) as a post-processing gene-list filter on the somatic VCF.
4. **Category code rename** (deferred): `germline_defined→germline`, `germline_denovo→upv`,
   `somatic_denovo→somatic` in `run_spatial_snv_filter_enhanced.py` + `final_snv_mat.py`.
5. **SpaceTracer — cite-only** (decision 2026-07-06): unaccepted preprint; one time-boxed
   single-sample attempt max, otherwise cite in Methods/Limitations and move on. Panel that stands
   without it: Strelka2, GATK, Monopogen, SpatialSNV.

---

## 🗄️ CLOSED — resolved questions, do NOT re-run

- **"Is SPARCAL better than GATK unbinned?"** → **No, statistical tie** (p=0.52). The old "clearly
  better" belief was the single-section 151507 pilot (0.304 vs 0.184) that did not generalize.
- **"Does adding SPARCAL's UPV/somatic calls beat 1kG-only on DLPFC?"** → **No.** Raw: germline
  (1kG+UPV) 0.177, defsom (1kG+somatic) 0.245, merged 0.160 — all < defined 0.277. Binned:
  germline 0.317, merged 0.287 < defined_bin250kb 0.351. **`somtop25_bin250kb` and
  `floor05_somtop25_bin250kb` promoted to all 12 sections 2026-07-12 → 0.350 / 0.351, below
  defined_bin250kb 0.363** (the single-section 0.364 was seed noise). UPV, not somatic, is the drag.
- **`defsom_bin250kb` "promising" thread** → closed; does not survive promotion (see above).
- **Every other representation trick** (raw or binned, 151507): TF-IDF 0.186, SVG-select 0.117–0.148,
  VAF/rawbin 0.129–0.137, prevalence-cap 0.111–0.155, exome-restriction 0.094–0.148, floor sweep
  ~neutral, `floor20∘somtop25` combo 0.317 — **all ≤ or < baseline. Only binning wins.** Treat
  further variant-selection tweaking on the DLPFC axis as low-yield.
- **"Does UMI dedup hurt normal-tissue clustering?"** → **No, ARI-neutral** (~+0.01, same-code
  matched). The old "pre>post 0.29 vs 0.20" claim was confounded by code drift — do NOT put "dedup
  hurts normal tissue" in the paper. See [[project_umi_dedup_ablation_finding]].
- **DLPFC ARI "regression" 0.28→0.21** → code drift (germline set grew ~27% from same BAMs), not
  dedup. See [[project_dlpfc_ari_regression_codedrift]].

---

## Reference — pipeline & benchmark run commands

Canonical pipeline run/resume, matrix generation, strelka2/GATK benchmark matrices, and the
clustering benchmark (`clustering_benchmark/`, env `snv_clustering`) are documented in
[CLAUDE.md](CLAUDE.md). Paper figure design + variant-category naming: [pipeline_intro.md](pipeline_intro.md).
