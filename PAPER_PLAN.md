# PAPER PLAN — SPARCAL: spatial SNVs as a new modality, CNV-guided

**Status:** ACTIVE plan as of **2026-07-28**, replacing the 2026-07-18 coverage-control-reanalysis
plan. Written from the advisor's direction (2026-07-28). Continues the **original manuscript**
`SPARCAL_pnas_2026/PNAS/PaperDraft.tex` (Fig. 1 pipeline / Fig. 2 DLPFC germline / Fig. 3 somatic /
Fig. 4 SPARCALViewer) rather than starting a new one.

**Findings that contradict or endanger these stories are NOT deleted** — they are parked in
[PAPER_PLAN_DEPRECATED.md](PAPER_PLAN_DEPRECATED.md), which is also the seed for the planned
follow-up benchmarking paper. Read that file before writing any sentence that a reviewer could
attack; it lists exactly which attacks are live.

> **⚠️ Read before working from this plan:** an external referee assessment (2026-08-23) is
> consolidated in [§8](#8-external-referee-assessment--2026-08-23). It scores the current draft
> **3.1/10, reject as submitted**, and challenges decisions L3, L4, D2 and D6. The prioritized
> work queue derived from it is **[PAPER_WORK.md](PAPER_WORK.md)** — start there, not here.

Live job tracker: [On_going.md](On_going.md). Manuscript repo: `/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026`.

---

## 1. Thesis (advisor, 2026-07-28)

> The contribution is **the modality**: applying SNV calling to spatial transcriptomics, using
> **CNV/clone context as calling guidance**, and showing that the variants SPARCAL calls **make more
> sense** than what other callers produce — plus **insight into which representation/setting works
> best** (germline vs UPV vs somatic; raw vs binned).

Explicitly **not** the claim: "SPARCAL wins on every metric." Explicitly **not** the endpoint:
clustering ARI alone — clustering is one metric among several, and the paper is expected to say
what is *behind* the numbers.

Deferred to the follow-up paper: the coverage/UMI-dominance benchmark of the whole spatial-SNV
field. **This paper does not contest coverage; it validates the variant sets.**

---

## 2. Paper structure

| Fig | Content | State |
|-----|---------|-------|
| **1** | SPARCAL pipeline (dual pathway, CNV/clone-guided cascade) | text written; diagram file still a placeholder |
| **2** | **Germline — unbinned.** SPARCAL vs Strelka2 vs GATK vs GE: ARI + spatial domains + **UMAP** | data partly there; **needs the `defined` 12-section promotion** (§4.1) |
| **3** | **Germline — binned (250 kb), separate section.** Ties GATK; `defsom` vs `defined`; bin-size sweep | data partly there; **needs `defsom_bin250kb` 12-section promotion** (§4.2) |
| **4** | **Somatic — why it is hard.** WES callability, exome rate, singleton support, Mutect2 germline leakage | fully grounded, no new compute needed (§4.3) |
| **5** | **Somatic — COSMIC cascade** germline > UPV > somatic > unresolved | grounded on `current` sets; **fragile, see §5** |
| **6** | **Somatic — tumor-region detection ARI** (SNV methods only, no coverage baseline) | SPARCAL-sets version exists; **cross-tool version is new work** (§4.5) |
| **7** | **Somatic — specific hits + biology** (per-sample CGC candidates, WES-confirmed variants, spatial maps) | annotation **DONE** 2026-07-28; old cSCC signature **falsified**, WES panel still to build (§4.6b) |
| **8** | **SPARCALViewer as standalone software** — tumor-region detection, region-scoped SNV export, profile scoring; **the tool that computes the tumor-section ARI** | code verified; needs a release + Fig. 4 panels rebuilt on current sets (§3 Story D) |

---

## 3. Story-by-story inventory: what we HAVE, what is MISSING

### Story A — germline, **unbinned**: "our SNVs beat Strelka2/GATK; our UMAP looks like gene expression"

**HAVE** (`data/dlpfc/clustering_benchmark/`, 12 sections × 10 STAGATE runs):

| modality (unbinned) | 12-section mean ARI | best-run mean |
|---|---|---|
| gene_expr | 0.4116 | 0.5174 |
| **sparcal** (current pipeline = 1kG + UPV) | **0.2167** | 0.2579 |
| gatk | 0.2114 | 0.2486 |
| strelka2 | 0.1933 | 0.2252 |

- Per-section: sparcal > gatk in **8/11**, sparcal > strelka2 in **8/11**.
- **151507 single-section sweep** (`data/dlpfc_recovery_test/151507/clustering_benchmark/`):
  **`defined1000G` (1kG-only, unbinned) = 0.277** vs gatk 0.173 vs strelka2 0.132 →
  **1.61× GATK, 2.10× Strelka2.** (`orig49k`, the old 49,602-column matrix, = 0.278.)
- UMAP + spatial PNG/PDF exist for **every section × modality × run** at
  `data/dlpfc/{section}/clustering/{modality}/run{0-9}/{umap,spatial}.{png,pdf}`, plus
  `combined_{section}.{png,pdf}` for all 12 (row 1 = spatial domains Hungarian-matched to GT,
  row 2 = UMAP colored by true layer). Paper-ready versions of 151507 and 151672 are already in
  `SPARCAL_pnas_2026/figs/fig2{a,b}_*`.

**MISSING / RISK:**

1. ✅ **RESOLVED 2026-07-28 — Story A has multi-section support, at a moderate effect size.**
   Job 12824499 completed all 12 sections × 10 runs of `defined1000G`:

   | modality (unbinned) | 12-section mean ARI |
   |---|---|
   | gene_expr | 0.4116 |
   | **defined1000G (1kG-only)** | **0.2660** |
   | sparcal (1kG+UPV, full pipeline) | 0.2167 |
   | gatk | 0.2114 |
   | strelka2 | 0.1933 |

   Paired per-section Wilcoxon: **vs GATK +0.043, 9/11 sections, p=0.032**; **vs Strelka2 +0.061,
   9/11, p=0.0068**; **vs `sparcal` +0.049, 9/12, p=0.042** (n=11 where 151671 lacks gatk/strelka2).
   ⇒ The advisor's claim is **supported but moderate**: 1.26× GATK across 12 sections, not the 1.61×
   of the 151507 pilot (0.276 vs 0.173 there). ⚠️ **Honesty note:** p=0.032 at n=11 is not
   overwhelming, and Bonferroni across the two pre-specified comparisons leaves the GATK one
   borderline (0.064). **The embedding metric is the stronger evidence (12/12, p=4.9e-4) and should
   carry the argument** — the drafted Results already leads with it.
   Second finding, on-thesis for "which setting is best": **1kG-only beats the shipped 1kG+UPV
   pipeline output significantly** — UPV degrades domain recovery on normal cortex, consistent with
   [[project_dlpfc_snv_representation_study]].
   **The `sparcal` (1kG+UPV) vs GATK tie (p=0.52) still stands** — guardrail §6.1 is unchanged; the
   win belongs specifically to the 1KGP-resolved subset.
2. **`sparcal` never beats gene expression on any unbinned section** (0/12, mean and best-run). The
   old draft's "151672: 0.701 vs 0.600" is from a superseded run and does not reproduce (current
   151672: sparcal 0.148 mean / 0.212 best, GE 0.540 / 0.740). **The "beats GE" panel must move to
   the binned story (§ Story B), where it is real.**
3. ✅ **The UMAP claim now HAS a number, and it holds — DONE 2026-07-28.**
   `clustering_benchmark/embedding_quality.py` → `data/dlpfc/clustering_benchmark/embedding_quality/`
   (12 sections × 10 runs, measured on the saved 30-d `embedding.npy`, never on a UMAP projection;
   null = 20 within-section label permutations on the fixed neighbour graph).

   **FINAL (rerun 2026-07-29 with `defined1000G` complete at 12/12):**

   | modality (unbinned) | kNN k=10 | excess | kNN k=30 | silhouette |
   |---|---|---|---|---|
   | gene_expr | 0.859 | 0.614 | 0.823 | +0.116 |
   | **defined1000G** | **0.644** | **0.399** | **0.614** | **+0.035** |
   | sparcal (1kG+UPV) | 0.573 | 0.327 | 0.511 | −0.007 |
   | gatk | 0.493 | 0.247 | 0.489 | −0.006 |
   | strelka2 | 0.451 | 0.205 | 0.449 | −0.035 |

   **`defined1000G` wins 12/12 sections on ALL THREE metrics at BOTH scales** (vs GATK: k10 +0.152,
   k30 +0.124, silhouette +0.040, each p=4.9e-4; vs Strelka2 +0.194/+0.165/+0.070, each p=4.9e-4;
   vs `sparcal` +0.072/+0.103/+0.042, each p=4.9e-4). ⇒ **This, not ARI, is Story A's headline
   evidence** — it is far more decisive than the ARI p=0.032.

   **Corrected interpretation (my earlier "the advantage is only local" read was about `sparcal`,
   and does NOT apply to `defined1000G`):** `sparcal` has a purely local advantage — big at k=10,
   attenuating at k=30, silhouette ≈ 0 — i.e. layer-consistent neighbourhoods without globally
   separated groups, which is exactly the regime where ARI registers a tie. **`defined1000G` recovers
   global structure too**: no attenuation at k=30, and it is the *only* unbinned modality with a
   **positive silhouette**. That is why its ARI moves while `sparcal`'s does not, and it makes the
   two benchmarks tell one coherent story rather than two conflicting ones.
   **Binned:** defined_bin250kb excess 0.608 at k=10 ≈ gene expression 0.614; gatk_bin 0.579,
   strelka2_bin 0.545. At k=30 gene expression pulls ahead (0.577 vs 0.498) and keeps a higher
   silhouette (0.116 vs 0.043).

   ⚠️ **Script limitation:** `embedding_quality.py` hardcodes only the `sparcal`-vs-{gatk,strelka2}
   Wilcoxon pairs, so the `defined1000G` comparisons above were computed separately from
   `embedding_quality_by_section.csv`. Generalize the comparison list before the next re-run
   (needed again when job 12836816's five new modalities land).
   ⚠️ UMAP trustworthiness was **not** computed: `SPARCAL_clustering.py` never saves the 2-D UMAP
   coordinates (only the PNG/PDF). Getting it would require adding an `np.save` there and re-running
   the clustering — out of scope, and not needed given the 30-d metrics above.
4. **Variant-universe caveat.** SPARCAL `defined` = 1KGP-panel *restricted*; the GATK matrix is
   1KGP-*filtered* (the opposite selection); Strelka2 is unfiltered PASS SNVs. This is a
   **representation** comparison, not a like-for-like caller-accuracy comparison — which is fine
   and actually on-thesis ("what setting is best"), but must be stated, or a reviewer states it
   for us.

### Story B — germline, **binned (250 kb)**: "on par with GATK; `defsom` sometimes beats `defined`"

**HAVE:**

| modality (250 kb bins) | 12-section mean ARI |
|---|---|
| gene_expr (reference) | 0.4116 |
| **defined_bin250kb** | **0.3626** |
| gatk_bin250kb | 0.3536 |
| somtop25_bin250kb | 0.3500 |
| floor05_somtop25_bin250kb | 0.3514 |
| strelka2_bin250kb | 0.2571 |

- defined vs gatk binned: **7/12 sections**, Wilcoxon **p = 0.68 → statistical tie** (this is exactly
  the advisor's framing, so it is on-message).
- defined vs strelka2 binned: **11/12 sections**, p = 0.001 (and gatk vs strelka2 p = 0.002 →
  the Strelka2 gap is not SPARCAL-specific; say so).
- **Beats gene expression** in 151669 / 151670 / 151671 / 151672 by mean; 151669 / 151670 / 151671 /
  151674 / 151676 by best run. **This is the "sometimes beats GE" panel.**
- Bin-size sweep 25 kb → 1 Mb, clean inverted-U peaking at 250 kb
  (`data/dlpfc_recovery_test/151507/clustering_benchmark/ari_binsize_sweep.*`).
- **151507 only:** `defsom_bin250kb` = **0.3636 > `defined_bin250kb` 0.3317** — the advisor's exact
  point that adding somatic calls helps. Also `germline_bin250kb` (1kG+UPV) 0.3166,
  `merged_bin250kb` 0.2869, `floor20_bin250kb` 0.3378, `somtop10_bin250kb` 0.3376.

**MISSING / RISK:**

5. ⚠️ **`defsom_bin250kb` exists on ONE section.** Its closest promoted relative,
   `somtop25_bin250kb`, **lost** on promotion (0.3500 vs defined 0.3626; the single-section 0.364
   was seed noise — On_going records this explicitly). So there is a real chance `defsom` also
   fails to replicate. **Promote it before writing the sentence** (§4.2). If it lands between
   "sometimes better" and "on par", the honest phrasing is *per-section win count*, not means.
6. `germline_bin250kb` and `merged_bin250kb` are also single-section — needed to make the
   "which setting is best" table complete (defined / defsom / germline / merged × raw / binned).

### Story C1 — somatic is intrinsically hard (the setup)

**HAVE — all grounded, no new compute required:**

- **WES callability:** RNA covers only **1.2 %** of P4 WES-somatic positions (40/3,451) and
  **1.0 %** of P6 (27/2,604) — expression restriction + dropout, not a SPARCAL failure.
- **Direct overlap:** SPARCAL-somatic ∩ WES-Mutect2-exome = **8 (P4) / 7 (P6)** positions.
- **Singleton support** (`data/artifact_evidence_benchmark/v2_cross_section_2026-07-16/`):
  61–82 % of somatic candidates are a **single alt UMI in a single spot**; ≥2 alt reads in only
  28.6 % (P4) / 18.4 % (P6) / 65.0 % (DCIS1); ≥2 spots in 19.2 % / 8.2 % / 22.6 %.
- **Tool-independence:** the published SpatialSNV callset is **77–87 % single-spot single-molecule**
  on the same tissue → this is a *platform* limit, not a SPARCAL artifact. (Safe to cite here; the
  wider version of this argument belongs to the follow-up paper.)

**MISSING:**

7. **A clean "exome rate" table for our own calls** — what fraction of `defined` / UPV / `somatic` /
   unresolved calls fall inside exome intervals, per sample. Exome-filtered matrices already exist
   (`filter_matrix_exome.py` → `*_exome_matrix.pkl`), but the per-class rate is not tabulated.
   ⚠️ mind the boundary-convention mismatch vs the DLPFC exome builder
   ([[project_exome_filtering]]).

### Story C2 — Mutect2 germline leakage in the WES "truth"

**HAVE:**

- ⚠️ **CORRECTED 2026-07-29 — the long-standing "28 %/31 %" figure is an ARTIFACT. Do not use it.**
  Allele-exact matching gives **P4 288/3,450 (8.3 %)** and **P6 247/2,585 (9.6 %)** — about a third
  of the previously circulated number. Cause, diagnosed and **independently re-verified with
  bcftools on chr1**: a naive region query against the per-chromosome 1000G panel returns records
  whose *span* overlaps the query point, not only records *at* it. On P4 chr1, the raw query returns
  190 panel records for 432 WES positions, but **133 of the 190 sit at a position that is not a WES
  position at all** (panel indels/SVs); only 57 are true position matches and 46 are allele-exact
  (10.6 % of chr1, bracketing the genome-wide 8.3 %). No script reproducing the original 976/811
  exists in the repo, so the artifact cannot be confirmed at source — but the corrected figure is
  cross-validated two independent ways and is the one to use.
  Leaked sites skew common (46 %/40 % at panel AF ≥ 1 %, vs 22 %/14 % for coordinate-but-not-allele
  matches) ⇒ **mechanism (a) common-SNP leakage is supported**; (b) LOH and (c) paralog/segdup are
  **not testable** — no allele-specific LOH segmentation and no segdup annotation exist for these
  samples. Say so rather than speculating.
- Allele-aware landing of the RNA-covered WES-somatic sites in SPARCAL's classes:
  P4 {defined 6, UPV 1, somatic 7, unresolved 11}; P6 {2, 1, 7, 7}.
- **The point for SPARCAL:** WES-somatic calls that land in our `defined` class are *by construction*
  in 1000G — i.e. SPARCAL correctly germline-flags the leakage in the bulk-DNA truth.
- Figure + tables: `COSMIC/wes_somatic_vs_sparcal_2026-07-13.{png,pdf}`,
  `data/somatic_validation_2026-07-13/data/{wes_1kg_contamination,wes_sparcal_breakdown}.csv`.

**MISSING:**

8. **Mechanism attribution** (advisor asked "what other causes might be possible?"). Currently we
   assert leakage without discriminating among: (a) **LOH** converting a germline het → hom in the
   tumor so the matched normal is under-called; (b) **low-coverage / tumor-contaminated matched
   normal**; (c) **common SNPs absent from the PON**; (d) **mapping artifacts at paralogs/segdups**.
   One cheap discriminating check: stratify the 976/811 leaked sites by 1KGP allele frequency
   (→ (c)) and by CalicoST LOH-segment membership (→ (a)). ~1 short script (§4.4).

### Story C3 — COSMIC cascade: germline > UPV > somatic > unresolved

**HAVE** (`current` sets, `COSMIC/somatic_vs_ambiguous_rates_2026-07-13.csv`):

| class | P4 | P6 | DCIS1 | DCIS2 |
|---|---|---|---|---|
| germline / 1kG (`defined`) | 12.58 % | 6.32 % | 9.05 % | 9.69 % |
| **UPV** | 3.06 % | 2.41 % | 1.82 % | 2.04 % |
| **somatic** | 1.245 % | 0.705 % | 1.014 % | 0.990 % |
| unresolved (discarded) | 0.938 % | 0.546 % | 0.874 % | 0.846 % |
| **somatic / unresolved** | **1.33×** | **1.29×** | **1.16×** | **1.17×** |
| Fisher p (1-sided) | 3.6e-5 | 3.1e-7 | 3.1e-2 | 1.2e-2 |

Monotone descending cascade in **all four samples**; kept-somatic beats discarded in all four.

**Artifact rule-out variants already tested** (advisor asked us to test and choose):

| variant | P4 | P6 | DCIS1 | DCIS2 | verdict |
|---|---|---|---|---|---|
| raw (no exclusion) | 1.33× | 1.29× | 1.16× | 1.17× | all significant |
| **HLA/xMHC excluded** (chr6:28–34 Mb) | 1.22× (p=4e-3) | 1.18× (p=1e-3) | 1.11× (p=0.10) | 0.99× (p=0.55) | **holds in cSCC only** |
| SComatic PON + RNA-editing mask | 1.17× | 1.13× | 1.32× | 1.00× | **do not adopt** — incoherent |

**Writing decision (user, 2026-09-06):** retain COSMIC as a same-basis external-catalogue
comparison across classes and callers. Report the raw and xMHC-excluded results consistently and
do not describe catalogue membership as variant-level validation or evidence of cancer-driving
calls. The `pre_dedup` sensitivity analysis remains outside the paper because `current` is the
locked pipeline version.

**MISSING:**

9. Lock **one** burden version for every number in the paper (see §5, Decision D1).
10. Keep the same-basis COSMIC framing synchronized across the manuscript, captions, and SI.
11. Paper label for the discarded class. Code token is `ambiguous`; suggest **"unresolved
    candidates"** in the manuscript (it is what the class actually is: below the rank cut, not
    proven artifactual).

### Story C4 — tumor-region detection ARI (no coverage baseline)

**HAVE** (`SPARCAL_Benchmarking/analysis/region_method_benchmark/`, DCIS2 pathology GT =
10 foci / 249 spots, 0.6×0.6 clip = 789 spots):

| SNV set | best raw ARI | F1 | precision | recall |
|---|---|---|---|---|
| merged | 0.629 | 0.787 | 0.812 | 0.763 |
| UPV+somatic | 0.610 | 0.774 | 0.799 | 0.751 |
| somatic | 0.606 | 0.776 | 0.788 | 0.763 |

Three detectors each (burden watershed baseline / marker-restricted / neighbour-pooled profile).
P6 pathology mask also available (`data/spatialsnv_reanalysis_2026-07-17/p6_gt/`, 1,504 tumor /
2,215 normal, whole section — no clip needed).

**✅ CROSS-TOOL REBUILD DONE 2026-07-29** — `SPARCAL_Benchmarking/analysis/region_method_benchmark/
current_2026-07-28/` (`benchmark_best_ari_all.csv` + per-sample CSVs). Rebuilt on the `current` sets
per D1, extended to SpatialSNV + Monopogen and to P6. Integrity checks passed: UPV derived as
`germline`−`1000G` matched the VCF `RACE=denovo` count exactly on **both** samples (DCIS2 36,323;
P6 1,744), and Monopogen's matrix orientation was verified (3,719/3,719 in-tissue overlap).

| sample | best → worst (raw burden, best over detector+params) |
|---|---|
| **DCIS2** | SpatialSNV **0.671** > coverage 0.657 > SPARCAL merged 0.612 / somatic 0.609 / upv+som 0.603 |
| **P6** | SPARCAL somatic **0.162** > coverage 0.157 > Monopogen 0.149 > SpatialSNV 0.117 > SPARCAL upv+som 0.091 / merged 0.087 |

**Reading (this is the honest one):** ⚠️ **the ranking flips between samples and nothing separates
from coverage.** SPARCAL is 3rd on DCIS2 and 1st on P6, but the P6 margin over coverage is +0.005 —
noise. Two further facts must be stated in the paper: (a) **DCIS2 is from SpatialSNV's own published
dataset**, so the one comparison they win is not independent; (b) **P6 absolute ARIs are 0.09–0.16
for every method** — the whole panel is near-useless there, versus 0.60–0.67 on DCIS2. The
determining variable is annotation geometry (compact foci vs diffuse leading edge), not caller.
⇒ **Do NOT present this as a caller ranking.** Drafted Results now frames it as a bound on what
burden-based region detection can do. Version note: the current-set rebuild reproduced the old
pre-dedup DCIS2 numbers within 0.007, so D1 did not move this result.

**MISSING:**

12. ~~Cross-tool comparison~~ — **DONE** (above).
13. ~~P6 extension~~ — **DONE** (above).
14. `coverage_only` = 0.657 is in the same CSV and **exceeds every SNV method**. Omitting it from
    the figure is the advisor's call and is defensible for *this* paper's scope — but do not delete
    the number, and expect the question at review (see §6 guardrails).

### Story C5 — specific somatic hits and their biology

**HAVE:**

- COSMIC ∩ **current** somatic-set gene lists: `cosmic_amb/{p4_tumor,p6_tumor,dcis1,dcis2}_somgenes.txt`
  (196 / 372 / 157 / 183 genes) plus `cosmic_amb/cgc_genes.txt` (Cancer Gene Census) and full
  `bcftools isec` output dirs per class per sample.
- **7 (P4) / 7 (P6) WES-confirmed, RNA-covered somatic positions inside the somatic set** — the
  gold-standard shortlist, DNA-validated.
- SPARCALViewer can project per-spot COSMIC-hit counts onto tissue (P4: 407 spots with ≥1 hit,
  max 7 at one spot).

**✅ RECOMPUTED ON THE CURRENT SETS, 2026-07-28** —
`scripts/postanalyze/cosmic_somatic_gene_annotation.py` →
`data/somatic_hits_2026-07-28/` (`SUMMARY.md` + 6 CSVs). Hit counts reproduce the locked table
exactly (P4 243 / P6 463 / DCIS1 188 / DCIS2 249). Verdict:

- ❌ **The draft's cSCC signature does NOT survive.** `KRT10`, `ITGB4`, `NOTCH1`, `CDKN2A` are
  **absent from every current somatic set**. `KRT6B` p.N227S appears in **P6 only** (not P4), so it
  is not shared. `SPINK5` is in both patients but at **different positions**, none protein-altering.
  ⇒ **Delete the "KRT6B p.N227S and SPINK5 p.G519= shared by both P4 and P6" sentence.** It was an
  artifact of the 8-variant callset.
- ⚠️ **What actually recurs across the two cSCC patients is HLA.** Of 24 shared genes, 4 are HLA
  (`HLA-B`, `HLA-DRB1`, `HLA-G`, `HLA-DQB2`) and they carry **24 of the 30 shared protein-altering
  hits (80 %)**. Across all samples HLA is 11–19 % of somatic COSMIC hits and **18–45 % of the
  protein-altering ones** (DCIS2 44.6 %). This independently corroborates the xMHC finding from a
  completely different direction — the "shared signature" was the HLA region all along.
- **Only 9 exact positions** are somatic in both P4 and P6; 8 are non-coding or synonymous, and the
  single missense is `NPEPL1` p.L465V (not a cancer gene). **Zero Cancer Gene Census genes are
  shared** between P4/P6 or between DCIS1/DCIS2.
- **Per-sample CGC hits are reportable but are singletons**, not a signature: P4 14 CGC genes/14
  hits (NOTCH2, MSH2, PTEN, ERBB3, SMARCA4, ARID1B, MAX …); P6 34 genes/49 hits (**HLA-A alone = 10**,
  plus TP53 p.P64T, MYC, ERBB4, RICTOR); DCIS1 7 genes/11 hits (PMS2 = 5, MAP3K1, SMARCA4);
  DCIS2 12 genes/13 hits (SPOP = 2, MDM4, YAP1, AXIN1, CTCF, ELF3, NFIB).
- **Consequence mix of the somatic hits:** 64–72 % non-coding/unannotated, 11–12 % synonymous,
  23–30 % missense, 0–5 nonsense per sample.

⇒ **Honest framing for Story C5:** report the per-sample CGC candidate hits as *individual
localizable candidates* (with n, protein change and spatial map), state plainly that no recurrent
cross-patient signature survives outside the HLA region, and let the **DNA-confirmed WES variants
carry the section**. Do not attempt a "shared cSCC driver signature" claim.

**MISSING:**

15. ~~CGC intersection~~ — **DONE** (above).
16. ✅ **DONE 2026-07-29 — and the result is thin. Reset expectations.**
    `scripts/postanalyze/somatic_evidence_package.py` → `data/somatic_evidence_2026-07-28/`
    (`wes_confirmed_somatic_annotated.csv` + per-variant barcode lists). The landing counts
    **reproduce the 2026-07-13 numbers exactly** (P4 {6,1,**7**,11}, P6 {2,1,**7**,7}), so the
    14-variant set is real. But as a biology panel it is weak:

    - **1 of 14** is in COSMIC at all — `GJB2` p.F141=, **synonymous**, and **not** a Cancer Gene
      Census gene. **0 of 14** are in CGC. **0** are protein-altering in a cancer gene.
    - **13 of 14 have no gene annotation at all** at the transcript level — `GJB2` is the *only*
      annotated locus in the set. (Corrected 2026-07-29: an earlier "11 of 14" here was a
      transcription slip; verified directly against `wes_confirmed_somatic_annotated.csv`.)
    - **2 of 14** are A>G and flagged as possible **RNA editing** (both P4 chr21:9,907,xxx, in a
      cluster that also looks editing-like).
    - Spot support is mostly thin: 5 of 14 are in ≤6 spots; the best are 46/3,650 (P6) and
      41/744 (P4).
    - Reassuring detail: **0 of 14 fall in xMHC** (explicitly checked, genuinely zero), so this set
      is at least not the HLA artifact.

    ⇒ **This cannot carry a "somatic hits with biological meaning" section.** It is honest evidence
    that a handful of SPARCAL somatic calls are DNA-confirmed and spatially localizable — worth a
    supplementary table and one Results sentence — but it is not a driver-discovery result and must
    not be written as one. **Decision needed (D7 below).**
17. ~~Recompute the stale KRT6B/SPINK5 claim~~ — **DONE**; the claim is falsified and must be cut.
18. Per-spot COSMIC-hit spatial maps exist for P4 only, and on the stale set → regenerate for
    P4/P6/DCIS1/DCIS2 (via SPARCALViewer, §3 Story D).
19. Decide whether the HLA finding is stated **in Story C5 as well** (recommended — it is the same
    fact the xMHC exclusion shows in §3 Story C3, arrived at independently, and pre-empts the
    obvious reviewer question about why the hits cluster on chr6).

### Story D — SPARCALViewer as a separate piece of software (advisor addition, 2026-07-28)

**Claim:** SPARCALViewer is a **standalone companion application**, not a figure-generating script:
it performs **tumor-region detection** from a study's spot×SNV matrix and **exports the SNVs of a
selected region** in a usable form — and it is **the software that produced the tumor-section ARI
numbers reported in this paper**.

**HAVE — verified in code, the claim is accurate as stated:**

- **Separate software.** `SPARCAL_Benchmarking/viewer/` is its own package (`sparcal_viewer/`, own
  `README.md` / `CHANGELOG.md` / `LICENSE` / `requirements.txt` / `tests/`, macOS build script
  `build_macos.sh`, shipped `dist/`), loadable per study via a `.config` file
  (`DCIS_2_SPARCAL/`, `OVAR_P5_SPARCAL/`). See [[project_sparcalviewer_project]].
- **Tumor detection is a real algorithm, documented.** `StudyData.auto_tumor_regions()`
  (`sparcal_viewer/data.py:480`) = per-spot burden intensity → hex-neighbour adjacency graph →
  smoothing → **hysteresis thresholds** (seed/grow percentiles) → **seeded priority-flood
  watershed** → **basin merging across shallow saddles** → region filtering/ordering. Written up
  step by step in `viewer/ALGORITHM.md`.
- **"Outputs the desired SNVs in a wise way."** Region-scoped SNV export
  (`main_window.py:_export_snvs` → `StudyData.export_snvs`, re-loadable back to its source group)
  plus profile-map export with/without background (`_export_profile_map`).
- **Profile comparison built in.** `_compare_profiles` scores any two spot-labelings by region-wise
  Jaccard, **ARI**, NMI, homogeneity, completeness and V-measure — this is how a caller's regions
  are scored against a pathologist annotation.
- **✅ It genuinely is how the tumor ARI is computed.**
  `SPARCAL_Benchmarking/analysis/region_method_benchmark/benchmark.py` imports the viewer package
  (`from sparcal_viewer.config import load_config`, `from sparcal_viewer.data import StudyData`)
  and calls `study.auto_tumor_regions(seed_pct=…, grow_pct=…)` at line 174 — the benchmark's
  `baseline` detector *is* the viewer's algorithm. So the sentence "region ARI in this paper was
  computed with SPARCALViewer" is literally true, not marketing.
- It also produced the DCIS2 ground-truth export schema (`profile,region_name,barcode`,
  `profile=="Ground Truth"`) and the P6 leading-edge mask used as pathology GT.

**MISSING / to decide:**

20. **Where the section goes.** Recommend a short Results subsection right before the region-detection
    figure (so the reader meets the tool before its ARI numbers), plus the existing Fig. 4 panels.
    The draft already has a Fig. 4 subsection — extend it rather than adding a second one.
21. **Availability.** A standalone-software claim needs a release: repo/DOI, version, install
    instructions, and the two study configs. Currently `dist/` is local only.
22. **Regenerate the Fig. 4 panels on the current sets** — the "407 spots in P4 carry ≥1 COSMIC-matched
    somatic variant, up to 7 at one spot" number is from the **stale 8-variant callset** and must be
    recomputed (same defect as §3 Story C5 item 17), and the P6/DCIS maps are still missing.
23. **Do not claim novelty for the detection algorithm itself** — it is a standard watershed on SNV
    burden. The claim is *usable software + reproducible region scoring*, which is defensible.

---

## 4. Experiment queue

Ordered by *story risk × cost*. Items 1–2 gate the germline half; items 5–6 gate the somatic half.

### 4.1 ⭐ Promote `defined1000G` (unbinned, 1kG-only) to 12 sections × 10 runs — **DONE 2026-07-28** (job 12824499)

✅ Ran: 12 sections × 10 runs, `defined1000G` mean ARI **0.2660** vs gatk 0.2114, strelka2 0.1933,
sparcal(1kG+UPV) 0.2167, gene_expr 0.4116. Paired Wilcoxon vs gatk +0.043 (9/11, p=0.032),
vs strelka2 +0.061 (9/11, p=0.0068), vs sparcal +0.049 (9/12, p=0.042). **Story A has its headline**,
but a modest one — see the guardrail in `RESULTS_V2_DRAFT.tex` (Bonferroni leaves the GATK comparison
borderline at 0.064; the embedding metric of §4.3, 12/12 sections at p=4.9e-4, is the stronger leg).

**⚠️ OPEN SUB-ITEM — section 151671 fails for the two unbinned baselines.** `gatk` and `strelka2`
crash in **all 10 runs** on 151671 with `Error in svd(shape.o, nu = 0) : infinite or missing values
in 'x'` — a degenerate mclust EEE fit on the unbinned embedding
(`data/dlpfc/151671/clustering/summary.csv`). The matrices exist; their **binned** counterparts
converge on that section, as do all 13 other modalities. Consequences, both handled 2026-08-07:

- The Results prose compared our n=12 mean against their n=11 mean. 151671 is `defined1000G`'s
  **best** section (0.3939 vs a 0.266 overall mean), so the unmatched form inflated us by +0.012 in
  a comparison whose whole effect is +0.043. **Fixed** — the head-to-head sentence now reports the
  matched 11 sections (0.254 / 0.211 / 0.193), which reproduces the paired differences and P-values
  exactly, since the Wilcoxon tests were always paired. Relative improvement is 1.2×, not 1.3×.
- §4.3's embedding metric is **unaffected** — STAGATE embeddings were saved even where mclust failed,
  so all 12 sections are present for every modality there. Verified 2026-08-07.

**⛔ DO NOT "close it" by forcing mclust to converge.** Investigating the fix on 2026-08-07 turned up
something much larger — see §4.1c. The 151671 rerun is now a footnote to that.

### 4.1c 🚨 THE BASELINE EMBEDDINGS ARE DEGENERATE IN EVERY SECTION (found 2026-08-07)

Effective rank of the 30-d STAGATE embedding (run0, relative-eigenvalue cut 1e-6), all 12 sections:

| modality | rank range |
|---|---|
| `sparcal` | **26–30** |
| `defined1000G` | 7–21 |
| `gene_expr` | 11–15 |
| **`gatk`** | **1–3** |
| **`strelka2`** | **1–3** |

The two external baselines' autoencoders collapse to 1–3 real dimensions **everywhere**, not just on
151671. 151671 is the same collapse reaching rank exactly 1, where mclust finally errors instead of
silently returning a partition of noise. Condition numbers there: gatk 2.3e18, strelka2 5.0e15.

**Cause — a density difference in the input matrices, not a bug.** Per-spot variant counts and
sparsity (151507 / 151671 / 151670):

| | density | variants/spot | columns in ≤1 spot |
|---|---|---|---|
| `sparcal` | 2.6% / 4.0% / 3.6% | 1,580 / 3,015 / 2,072 | 6% / 4% / 7% |
| `gatk` | 0.21% / 0.25% / 0.28% | 109 / 159 / 134 | **66% / 65% / 64%** |
| `strelka2` | 0.21% / 0.22% / 0.25% | 122 / 162 / 120 | **80% / 79% / 82%** |

SPARCAL's matrix is ~12× denser with ~10–20× more variants per spot, and two-thirds to four-fifths
of the baselines' columns are spot-private. A variant seen in one spot carries no spot-to-spot
covariance, so there is little for a graph autoencoder to learn from those matrices.

**⚠️ WHAT THIS MEANS FOR STORY A — this is a live, unresolved threat to the germline headline.**
Both legs of Story A compare against these degenerate embeddings:
- the ARI comparison (0.254 vs 0.211 / 0.193), and
- the embedding-purity metric (Fig. 2c, +0.152 vs gatk, 12/12 sections, p=4.9e-4) — which is
  currently phrased as the *stronger* leg.

"SPARCAL's embedding is more layer-pure than GATK's" is close to tautological when GATK's embedding
is rank 1–2. A reviewer who computes the rank sees this immediately. The honest reading is that we
may be measuring **callset density**, not variant quality — which is the same coverage/density
confound parked in [PAPER_PLAN_DEPRECATED.md](PAPER_PLAN_DEPRECATED.md) and deferred to paper #2,
except here it sits inside the current paper's main germline claim.

**Not yet decided — needs the user.** Options, roughly in increasing cost:
(a) State it as a limitation and reframe Story A as a comparison of *representations at the density
each caller delivers* (which is defensible, and close to what L1 already says);
(b) density-match the comparison (subsample SPARCAL's matrix to the baselines' variants/spot, or
restrict all three to non-private variants) and report whether the advantage survives;
(c) drop the embedding-purity leg and stand on ARI alone.
**Do not write another sentence of Story A until this is decided.**

**Code change made 2026-08-07:** `clustering_benchmark/SPARCAL_clustering.py:mclust_R` now (i) catches
the R-side `RRuntimeError` — the old guard only handled mclust *returning* NULL, which is why the
151671 cells were recorded as failures rather than falling back — and (ii) **refuses** to fit when
effective rank < G, instead of retightening the rank cut until something converges. Forcing a
7-component mixture into 1–2 dimensions yields a number, not a measurement. The degeneracy is now
surfaced rather than papered over.

### 4.1b Original scope note
Same harness as the existing sweep
(`clustering_benchmark/`, env `snv_clustering`, a6000 GPU array, ~5 min/section/modality; a6000
caps at 2 concurrent). Add `defined1000G` (and optionally `germline_raw`, `merged_raw`,
`defined_somatic` for the settings table) to `clustering_config.json` and rerun the 12-section array.
**Also compute a per-section paired Wilcoxon vs gatk and vs strelka2** — do not report means alone.

### 4.2 ⭐ Promote `defsom_bin250kb` to 12 sections × 10 runs — **DONE 2026-07-29**
✅ 12 sections × 10 runs. `defsom_bin250kb` 0.3520 vs `defined_bin250kb` 0.3626 → **−0.0106, wins
6/12, Wilcoxon p=0.57**, i.e. adding the somatic class to the binned representation changes nothing.
Story B's "somatic calls also make sense" sentence is **not** supported by this; the single-section
151507 result (defsom 0.375 > defined 0.332) does not generalize. `germline_bin250kb` and
`merged_bin250kb` also completed — the settings table is full (15 modalities in
`data/dlpfc/clustering_benchmark/ari_table.csv`).

### 4.3 Embedding-quality metric for the UMAP panel — **DONE 2026-07-28**
✅ `data/dlpfc/clustering_benchmark/embedding_quality/` — kNN layer purity (k=10, k=30) + UMAP
trustworthiness + silhouette, against a within-section label-permutation null, **all 12 sections for
all modalities** (unaffected by the 151671 mclust failure in §4.1, since the embeddings are saved
before the clustering step). SPARCAL more layer-pure than GATK in **12/12** sections, mean difference
+0.152, paired Wilcoxon p=4.9e-4. This is Story A's **strongest** leg and Fig. 2c; do not lean on
ARI alone.

### 4.4 WES leakage mechanism split
Stratify the P4 976 / P6 811 leaked WES-somatic sites by (a) 1KGP allele frequency and (b) CalicoST
LOH-segment membership. One short script; answers the advisor's "what other causes?".

### 4.5 ⭐ Cross-tool region detection (DCIS2 + P6) — **DONE 2026-07-28**
✅ `SPARCAL_Benchmarking/analysis/region_method_benchmark/current_2026-07-28/` — all four arms
(**SPARCAL, SpatialSNV, Monopogen, coverage-only**) on **both** DCIS2 and P6, via
`benchmark_current.py`. This is the source for SI Fig. S2. Outcome is **not** a caller ranking and
must never be written as one: the ordering **reverses** between the two samples, SpatialSNV beats us
on DCIS2 (0.671 vs 0.612), no method separates meaningfully from the coverage baseline, and DCIS2 is
drawn from the SpatialSNV publication's own dataset, so that comparison is not independent. Coverage
baseline is omitted from the figure per L3-as-amended but plotted dashed in the SI version.

### 4.5b Original scope note
Run `SPARCAL_Benchmarking/analysis/region_method_benchmark/benchmark.py`'s three detectors on the
SpatialSNV and Monopogen matrices as additional "SNV sets", plus extend the harness to the P6
pathology mask. Report ARI/F1/precision/recall **per tool**, coverage baseline omitted from the
figure per the advisor (but kept in the CSV). Matrices: `SpatialSNV/results/{dcis2,p6}/matrix/…`,
`SPARCAL_Benchmarking/monopogen/P6_rep1/…`.

### 4.6 Somatic-hit biology package — **(a) and (c) DONE 2026-07-28**
(a) ✅ CGC intersection + per-class gene annotation — `scripts/postanalyze/cosmic_somatic_gene_annotation.py`
→ `data/somatic_hits_2026-07-28/`. (c) ✅ shared-cSCC signature recomputed on the current sets — the
draft's KRT6B/SPINK5 claim is **falsified**; what recurs is HLA (§3 Story C5).
**(b) still to do — now the section's headline:** annotate the 7 (P4) + 7 (P6) WES-confirmed somatic
variants (gene / protein change / COSMIC ID / per-spot spatial map).
**(d) still to do:** regenerate per-spot COSMIC-hit maps for all four samples on the current sets,
via SPARCALViewer (also fixes the stale "407 spots / max 7 hits" Fig. 4 number).

### 4.7 Exome-rate table per variant class
Per sample × class fraction of calls inside exome intervals. Cheap.

### 4.8 Additional COSMIC modeling — **NOT scheduled** (Decision D2)
The paper uses the same catalogue definition for all classes and callers, with the xMHC-excluded
comparison reported separately. No additional COSMIC model is required for the paper.

### 4.9 ⭐ Rebuild the region-detection inputs on `current` matrices (Decision D1) — **DONE 2026-07-28**
✅ Rebuilt as `region_method_benchmark/current_2026-07-28/` (driver: `benchmark_current.py`). The
region figure and the COSMIC figure now describe the same variant sets. **Trap left in place:** the
ORIGINAL `benchmark.py:33` still hardcodes `dcis2_pre_umidedup_2026-06-25/final_matrices/baseQ0mapQ0`
— running that file directly reproduces the pre-dedup numbers and silently violates D1. Use
`current_2026-07-28/benchmark_current.py`, or repoint line 33 before anyone runs the old entry point.

### 4.9b Original scope note
`region_method_benchmark/benchmark.py` hardcodes
`snv_calling/data/dcis2_pre_umidedup_2026-06-25/final_matrices/baseQ0mapQ0`. With `current` locked as
the paper's only version, repoint it at the current DCIS2 tree and re-run before §4.5, otherwise the
region figure and the COSMIC figure describe different variant sets (~6× size difference) — the exact
inconsistency that sank the previous plan. Do this **first**, then build §4.5 on top of it.

---

## 5. Decisions — locked and open

**Locked (advisor, 2026-07-28):**
- **L1.** Paper = modality + CNV-guided calling + "which setting is best" insight. Not a
  win-on-all-metrics claim.
- **L2.** Germline gets **two separate sections**: unbinned (variant sets make sense) and binned
  (spatial representation makes sense).
- **L3.** Somatic region detection is reported **without** the coverage/UMI baseline in the figure.
- **L4.** The UMI-dominance / coverage-confound benchmark is **deferred to a second paper**.
- **L5.** Continue `PaperDraft.tex`; do not splice in `RESULTS_PIVOT_DRAFT.tex`.

**Locked (user, 2026-07-28):**

- **D1 — variant-set version: `current` everywhere.** The shipped post-dedup pipeline is the paper's
  single version; state it explicitly in Methods. **Consequence:** no `pre_dedup` number may appear
  anywhere, and the region-detection benchmark must be **rebuilt on current matrices** — it presently
  reads `data/dcis2_pre_umidedup_2026-06-25/final_matrices/` (§4.9). The `pre_dedup` COSMIC null is
  a version-sensitivity note in [PAPER_PLAN_DEPRECATED.md](PAPER_PLAN_DEPRECATED.md) §3.2, not a
  paper number.
- **D2 — COSMIC framed as a same-basis class/caller comparison, no cancer-relevance claim.** Keep the cascade figure and
  the monotone ordering; write it as *"the classifier separates the variant classes, and the ordering
  is consistent across four samples"* — **never** *"COSMIC validates our somatic calls as
  cancer-relevant."* Use the same COSMIC build and allele-exact definition throughout; §4.8 does
  not schedule an additional model.
- **D3 — Story A headline set: `defined` (1kG-only), promoted to 12 sections** (§4.1). Report
  `sparcal` (1kG+UPV) alongside as the full-pipeline default, so the paper shows both the best
  representation and what the pipeline actually emits. The 1KGP-restricted-vs-1KGP-filtered caveat
  (§3 Story A, risk 4) must be stated in the same paragraph.
- **D4 — region detection is cross-tool:** SPARCAL vs **SpatialSNV** vs **Monopogen**, on **DCIS2 and
  P6** (both pathology GTs), coverage baseline omitted from the figure but retained in the CSV (§4.5).

**Still open:**

- **D7 — RESOLVED 2026-08-20: option (b).** Fig. 7 is reframed **entirely around the HLA/MHC
  recurrence caution**; the next task is to establish what that recurrence means biologically, and
  to find a defensible alternative framing if it does not hold up. Original statement of the
  decision follows.
  **What Story C5 (Fig. 7) actually claims, now that both of its legs are weak.** The COSMIC
  signature is falsified (HLA-driven), and the WES-confirmed set is 14 variants with 0 cancer-gene
  hits and **13** unannotated (§3 Story C5 item 16 — this line previously said 11, contradicting
  the 2026-07-29 correction two hundred lines above it; 13 is the value verified against
  `wes_confirmed_somatic_annotated.csv`, where `GJB2` is the only annotated locus).
  Options: **(a)** merge Fig. 7 into the platform-limit
  section as "what individual retained calls look like at this depth" — honest, removes a weak
  standalone figure, and the material still appears; **(b)** keep Fig. 7 but reframe it entirely
  around the **HLA/MHC recurrence caution**, which is our most novel somatic-side finding and
  generalizes to any spatial-RNA callset; **(c)** keep it as a per-sample CGC candidate list with
  heavy caveats. **Recommend (b)** — it converts the weakest section into the one field-level
  contribution the somatic half actually has, and (a) can fold in underneath it.
- **D8 — whether the somatic half still justifies Figs. 5–7 as three separate figures.** With COSMIC
  demoted to class separation, region detection not favouring us (§3 Story C4), and C5 thin, three
  somatic figures may oversell. Consider consolidating to two.

- **D5 — which sections for the Fig. 2 UMAP panel.** 151507 and 151672 are already rendered, but
  151672 is one of the three sections where unbinned SPARCAL *loses* to GATK. Recommend replacing
  151672 with **151509** (largest unbinned SPARCAL margin: 0.325 vs gatk 0.303 vs strelka2 0.227) or
  **151676** (0.232 / 0.179 / 0.177). Revisit once §4.1 gives 12-section `defined` numbers — the
  right pair should be chosen on `defined`, not on `sparcal`.
- **D6 — HLA exclusion presentation.** Recommend reporting **both** (raw + xMHC-excluded) and framing
  the cascade as **tumor-type-specific**: robust in high-TMB cSCC (P4/P6), HLA-driven/absent in
  low-TMB DCIS. Do **not** adopt the SComatic mask (it moves the ratio incoherently across samples).
  Compatible with D2 either way.

### ⚠️ MANUSCRIPT FILE ARRANGEMENT — CHANGED 2026-08-07. READ BEFORE EDITING ANY .tex

The manuscript was split into a frozen "before" and a live "after" so both could be
presented at the 2026-08-07 advisor meeting. They get merged once the advisors approve.

| file | role | edit it? |
|---|---|---|
| `PNAS/PaperDraft_v2.tex` | **THE LIVE MANUSCRIPT.** Self-contained: front matter, Introduction, Methods, Results, Discussion, Conclusions all in one file, no `\input`. | **YES — this one** |
| `PNAS/PaperDraft.tex` | Frozen at the committed GitHub state. The version the advisors have already seen. | No |
| `PNAS/RESULTS_V2_DRAFT.tex` | **NO LONGER MAINTAINED.** Reset to HEAD 2026-08-07 and frozen with `PaperDraft.tex`. | No |

**`RESULTS_V2_DRAFT.tex` is no longer the single source of truth**, despite what its own
header still says (that header could not be corrected without dirtying the frozen pair
before the meeting — fix it at merge time). Both frozen files are clean at HEAD; keep them
that way until the merge, or the "before" side of the comparison stops being the GitHub
version.

**Consequences to watch:**
- `RESULTS_V2_PREVIEW.tex` still `\input`s `RESULTS_V2_DRAFT.tex`, so **the preview wrapper
  now previews the OLD Results.** Do not use it to check new text — compile
  `PaperDraft_v2.tex` directly. (`_probe.tex` has the same problem; it is a scratch file.)
- Nothing feeds `PaperDraft_v2.tex` any more. It is edited directly.
- The generator that originally produced it (`make_paperdraft_v2.py`) was **deleted**
  2026-08-07: it rebuilt v2 *from* `PaperDraft.tex`, so running it after the freeze would
  have silently overwritten every post-review change with the old text.
- Pre-revert copies of both files were saved to the session scratchpad
  (`pre_revert_2026-08-07/`). Session-scoped — move them into the repo if they need to last.

**At merge time, decide:** does the merged manuscript go back to the `\input` split (restore
one source of truth, keep the preview wrapper working) or stay monolithic (simpler to send,
but `RESULTS_V2_PREVIEW.tex` and `_probe.tex` should then be deleted rather than left
pointing at a dead file)?

### Amendments (2026-08-04 advisor review → 2026-08-06 work)

Locked decisions above are left as written; this block records what has since
changed and why. Where the two conflict, **this block wins**.

- **L3 and the baseline clause of D4 are SUPERSEDED.** Both said to omit the
  coverage/UMI baseline from the region-detection figure. That figure is no longer
  in the main text — it moved to SI Fig. S2 — and in SI the baseline is plotted
  (dashed). Rationale: the number sits in the released CSV
  (`benchmark_norm_bestARI.csv`), so hiding it is an editorial choice a reviewer
  can catch, and coverage-only *beats* every SNV set on DCIS2 (0.657 vs 0.629).
  Showing it pre-empts the question instead of inviting it. The rest of D4
  (cross-tool, DCIS2 + P6) stands.
- **Figure numbering changed.** Bin-width sweep (former main Fig. 3c) → **SI
  Fig. S1**; region detection (former main Fig. 6) → **SI Fig. S2**; downstream
  main figures renumbered **7→6, 8→7**. Figure *filenames* were deliberately not
  renamed, because the generating scripts reference them — so main Fig. 6 is
  `fig7_hla_wes_evidence.pdf`. The mapping is recorded in the header of
  `RESULTS_V2_DRAFT.tex`; Fig. S1 is the one exception and does have a matching
  name (`figS1_binsize_sweep.pdf`, redrawn 2026-08-06).
- **The inverted-U claim is withdrawn.** The bin sweep is non-monotonic (500 kb
  below both 250 kb and 1 Mb) and 250/125/100 kb are not separable at n=5
  (shared range 0.260–0.353). 250 kb is now described as a pragmatic,
  dataset-specific choice. Never restore the "clean inverted-U" sentence.
- **The introduction's SpatialSNV claim was wrong and is fixed.** It said no
  population reference informs their germline/somatic distinction. One does —
  Mutect2's probabilistic population-AF filter. The corrected claim is that this
  is not panel-based genotype resolution and phasing, and that being tumor-only
  they emit no germline class. Verified: they drop *all* `germline`-flagged
  records (0 of ~760k retained calls across four samples).
- **NEW RESULT — germline leakage is a real, reportable difference.** Allele-exact
  against 1KGP, genome-wide: 2.17% (P4), 2.71% (P6), 4.67% (DCIS1), 4.39% (DCIS2)
  of SpatialSNV's retained callset are common panel variants, and SPARCAL routes
  64–82% of those same positions to germline in the same tissue.
  `data/spatialsnv_callset_quality_2026-08-06/`. Caveat for Methods: SPARCAL is
  not exactly 0% — DCIS2 has 1 leak in 25,154 (0.004%).
- **Cross-method concordance is descriptive, not a confidence axis.** Report overlap/Jaccard on
  the shared call universe, but do not infer variant quality from callset intersection alone.
- **The two callers barely agree** (Jaccard 0.027–0.050; 14–42% of SPARCAL's calls,
  2.8–7.2% of SpatialSNV's). This is model-free and reportable on its own.
- **NEW NEGATIVE RESULT — the CNV/LOH method comparison does not work; recommend
  dropping it from this paper.** Two design faults, both now documented:
  (i) the intended control set is not heterozygous — SPARCAL's 1KGP-defined class
  is 59–71% homozygous-alt and only 14–22% het, so it cannot detect *loss of*
  heterozygosity; (ii) copyKAT "loss" segments are low-*expression* segments, so
  region and depth are confounded — the depth drop appears in diploid spots too
  (6.9 vs 9.1, DCIS1), which carry no copy loss. Rebuilt with heterozygosity
  defined empirically in diploid spots and read out in aneuploid spots, with
  depth required in both groups and then stratified. Outcome: the method
  comparison is null inside validated LOH (14.1% vs 12.9%, p=0.71) **and violates
  its own negative control** — in copy-balanced regions, where no difference
  should exist, SPARCAL reads 42.7% vs SpatialSNV 14.5% (p=8.7e-8). The metric
  tracks callset VAF composition, not consistency with copy state. Fixing it
  needs allele-specific copy number and a VAF-matched design, which is new work.
  `data/loh_allelic_test_2026-08-06/RESULT.md`.
- **NEW POSITIVE RESULT — DCIS1's clone-loss segments are genuine LOH.** At sites
  heterozygous in normal spots, median |BAF−0.5| in the tumour clone is 0.250 in
  loss segments vs 0.077 in balanced (pooled p=6.5e-12), and it holds in **every**
  depth band (p=0.044 / 0.016 / 2.3e-6 / 0.0059), so it is not the depth artifact
  above. Only 35.0% of normal-heterozygous sites stay heterozygous in the clone
  (vs 73.4% balanced). **DCIS2 shows no such signature** (p=0.18, null in every
  band) — do not treat copyKAT segments as LOH by default. This is usable as
  modest evidence that the copy-number backbone SPARCAL conditions on is real in
  at least one sample, and it is non-circular: θ uses within-clone prevalence and
  never sees BAF.
- **D7 RESOLVED 2026-08-07 → option (b).** The Fig. 7 subsection now leads with the
  MHC/HLA recurrence caution and folds the Cancer Gene Census candidate list and the
  14 WES-confirmed variants underneath it (option (a)'s material is retained, not
  deleted). New subsection title: *"Apparent cross-patient recurrence in RNA-derived
  somatic calls is dominated by the MHC."* Rationale: the candidate list is almost all
  single observations and recurs in neither cSCC patient, so leading with it invites the
  driver-discovery reading the data cannot support; the MHC result is the one field-level
  contribution the somatic half has, and `fig7_hla_wes_evidence.pdf`'s caption was already
  written around it. Note this is main **Fig. 7**, not Fig. 6 — the earlier note here said
  Fig. 6, which predates the 2026-08-06 insertion of the SpatialSNV head-to-head at Fig. 5.
- **BUG FIXED with it:** the CGC candidate sentence cited **Fig. 7c**, but panel (c) is the
  14 WES-confirmed variants (`fig7_hla_wes_evidence.py:12`; `ax_c` title *"WES-confirmed
  somatic variants (n=14; 0 Cancer Gene Census)"*). There is no panel for the CGC
  candidates. The citation moved to the WES paragraph, and the recurrence paragraph now
  cites Fig. 7a,b.
- **D8 RESOLVED 2026-08-07 → NO CONSOLIDATION.** D8 (consider merging the somatic figures
  down to two) predates the advisor's 2026-08-04 review and is superseded by it: the review
  explicitly asks to keep the COSMIC cascade and the somatic-hits figure in main (comment 4),
  to move region detection to SI (comment 5, done → Fig. S2), and to add the SpatialSNV
  comparison the paper lacked (comment 1, done → Fig. 5). Every one of the four somatic-half
  main figures is therefore individually requested. D8's real concern — overselling a thin
  somatic half — is handled by reframing, not cutting: COSMIC is class separation only (D2),
  region detection is in SI, and Fig. 7 leads with the caution (D7 above). If a length limit
  later forces a cut, the cheapest merge remains Fig. 5c into Fig. 4.
- **Methods keying inconsistency RESOLVED 2026-08-07 (disclosure, not re-analysis).** The
  older matrix-level overlap analysis cannot be re-keyed —
  `scripts/postanalyze/ssnv_crossmethod_jaccard_p6.py:4` records that the bundle-matrix
  column key carries no allele. The Methods now state that the spot-level projection is
  position-only and why, and point to the allele-exact subsection for the variant-level
  comparisons. Never present a position-only overlap number and an allele-exact one as
  comparable.

---

## 6. Framing guardrails (things a reviewer will attack)

1. **Never write "SPARCAL outperforms GATK" from the unbinned 12-section ARI as currently computed**
   — it is a tie (p=0.52). The claim is only available if §4.1 delivers `defined` at ~0.28 across 12
   sections, and even then it must be phrased as a **representation** comparison (§3 Story A, risk 4).
2. **Never repeat the draft's 151507 = 0.356 / 151672 = 0.701 numbers.** They are from a superseded
   run and do not reproduce (current: 0.178 / 0.148 mean).
3. **The Strelka2 gap is not SPARCAL-specific** — GATK beats Strelka2 too, at similar significance.
4. **UMAP is not evidence on its own.** Ship the embedding metric (§4.3) with the picture.
5. **Do not call the discarded class "artifacts."** COSMIC is a DNA cancer catalogue, not an
   RNA-artifact catalogue; we have no direct artifact evidence for those loci.
6. **Do not claim genotype correction.** The audit against matched-normal WES gave 51–56 % agreement
   and the pipeline does not rewrite `GT` — it is a filtering/confidence layer.
7. **Region detection omits the coverage baseline by editorial scope, not because it is absent.**
   Have the answer ready; keep `coverage_only` = 0.657 in the supplementary CSV.
8. **DLPFC is a positive control, not a demonstration of the somatic model** — normal cortex has no
   clonal structure to encode.

---

## 6.5 Manuscript drafting status (2026-07-28)

**Restructured Results + Discussion + Conclusions DRAFTED** at
`SPARCAL_pnas_2026/PNAS/RESULTS_V2_DRAFT.tex`. Eight Results subsections in the Fig. 1–8 order of
§2, a rewritten Discussion and Conclusions, plus draft replacement **abstract and significance
statement** (in a `\begin{comment}` block at the end of the file). The whole splice-in set is now in
this one file; the **Introduction needs no change**.

Discussion positions the *representation* finding (binning ≫ caller identity) as the transferable
result, states the platform limit as a field-level norm, reports the COSMIC result as class
separation only, foregrounds the **MHC/HLA recurrence caveat**, and contains one explicit paragraph
conceding that region-level burden is entangled with coverage and deferring the coverage-controlled
benchmark to the separate second paper — this is the sentence that keeps this paper consistent with
[PAPER_PLAN_DEPRECATED.md](PAPER_PLAN_DEPRECATED.md) instead of contradicting it.

- **It supersedes `RESULTS_PIVOT_DRAFT.tex`** (the coverage-reanalysis draft, now paper #2's).
  Do not splice both.
- Every number carries a source-path comment and was verified against the CSV on disk; pending
  numbers use a `\PENDING{}` macro that renders visibly, so an unresolved value cannot ship silently.
- **Do not splice into `PaperDraft.tex` until Discussion + Conclusions are rewritten too** — the
  existing abstract makes three claims this draft does not support: "consistently more concordant
  than either caller" (binned is a tie, p=0.68), "strong concordance with matched WES" (false — ~1%
  callability), and COSMIC "supports biological relevance" (catalogue membership is comparative,
  not driver validation). The draft
  abstract in the comment block fixes all three.
- The existing **Introduction needs no rewrite** — it already argues modality + CNV-as-calling-evidence,
  which is the new thesis.

**Open `\PENDING{}` items and what resolves each:** unbinned 12-section ARI + Wilcoxon (job 12824499);
the 2×2 settings table incl. `defsom` (job 12824500 → the defsom clustering array); kNN embedding
purity (`embedding_quality.py`); cross-tool region-detection table
(`region_method_benchmark/current_2026-07-28/`); WES leakage mechanism split and the WES-confirmed
somatic variant table (`data/somatic_evidence_2026-07-28/`).

## 7. Data & script index

- DLPFC clustering benchmark: `data/dlpfc/clustering_benchmark/{ari_matrix_mean,ari_matrix_best,ari_table}.csv`;
  per-section `data/dlpfc/{s}/clustering/{modality}/run{0-9}/`; harness `clustering_benchmark/`
  (env `snv_clustering`).
- 151507 full modality sweep (54 modalities): `data/dlpfc_recovery_test/151507/clustering_benchmark/ari_full_experiment_table.csv`
  + `ari_binsize_sweep_table.csv` + `weekly_report/DLPFC_SNV_representation_report.md`.
- COSMIC: `/data/maiziezhou_lab/leiy4/COSMIC/` (v103 GRCh37+GRCh38, CGC TSVs, per-sample analyses);
  current-set intersections + gene lists in `cosmic_amb/`; rates in
  `COSMIC/somatic_vs_ambiguous_rates_2026-07-13.csv`.
- Somatic validation tables/figures: `data/somatic_validation_2026-07-13/`.
- Region detection: `SPARCAL_Benchmarking/analysis/region_method_benchmark/`.
- Support/singleton evidence: `data/artifact_evidence_benchmark/v2_cross_section_2026-07-16/`.
- Pathology GT: DCIS2 `SPARCAL_Benchmarking/viewer/DCIS_2_SPARCAL/tumor_groups.csv`;
  P6 `data/spatialsnv_reanalysis_2026-07-17/p6_gt/`.
- WES: `.../ST_datasets/STmut_Data/P{4,6}_Somatic_{Mutect2,GATK}/` (hg19, chr-prefixed).
- Re-scoring tool: `scripts/postanalyze/sparcal_set_benchmark.py` (+ baseline
  `data/set_benchmark/baseline_pre_dedup/` — note it is the **pre_dedup** anchor, see D1).
- Related memories: [[project_sparcal_vs_gatk_statistical_tie]],
  [[project_dlpfc_snv_representation_study]], [[project_dlpfc_ari_regression_codedrift]],
  [[project_exome_filtering]], [[project_sparcal_benchmarking_ecosystem]],
  [[project_paper_benchmark_strategy]].

---

## 8. External referee assessment — 2026-08-23

Full simulated PNAS referee report against `PNAS/PaperDraftGuided.tex` (95,788 B) and
`PNAS/SI_AppendixGuided.tex`, read as a reviewer with field knowledge. **Full report: [REFEREE_REPORT_2026-08-23.md](REFEREE_REPORT_2026-08-23.md).** The actionable queue
derived from it is **[PAPER_WORK.md](PAPER_WORK.md)** — that file, not this section, is what to
work from.

**Verdict: reject as submitted. Overall readiness 3.1/10 for PNAS.**

| criterion | score |
|---|---|
| Conceptual advance | 4/10 |
| Evidence for central claims | 2/10 |
| Controls | 2/10 |
| Statistical treatment | 3/10 |
| Methods & reproducibility | 3/10 |
| Manuscript readiness | 1/10 |
| Transparency & self-criticism | **9/10** |
| Fit for PNAS | 2/10 |

### 8.1 The core structural problem

The manuscript disclaims almost every result it reports — correctly — and after the disclaimers,
one positive claim is left standing: that the spatially augmented 250-kb representation matches
gene expression on 10-NN cortical-layer purity (0.856 vs 0.859). **That claim has never been run
against a coverage or detection control**, while the region-detection section demonstrates that a
no-SNV coverage baseline wins when one is supplied. The paper therefore proves it knows the
control is decisive and does not apply it to its own headline.

A referee's question becomes: *what am I being asked to accept as new and true?* On the current
text there is one sentence, and it is uncontrolled.

### 8.2 Critical findings (block acceptance at any venue)

| ID | Finding |
|---|---|
| **C1** | **No coverage/detection control on the DLPFC headline.** A germline-SNV presence matrix is a thresholded function of transcript capture, so layer structure may be gene expression re-derived through a lossy filter. Compounded by STAGATE smoothing on the spatial graph, which partly manufactures kNN purity regardless of input features → PAPER_WORK **P0-1** |
| **C2** | **Somatic caller uncalibrated.** Top-20% descriptor votes, top-10% somatic quota, per-dataset hand-tuned T_α/T_β, no FDR, no null. Callset size is an analyst choice: 65,655 "somatic" calls in P6 from RNA, 74.8–91.6% on one ALT UMI in one spot → **P0-3** |
| **C3** | **SparcalNet never evaluated; no component ablated.** No training set, label source, class balance, split, cross-validation, or performance number appears anywhere. Nobody can tell which part of SPARCAL does any work, including whether θ (the CNV/LOH evidence, i.e. the stated thesis) contributes anything. *Note (corrected 2026-08-23):* the paper's "64 and 32 neurons" **is** right — it describes `run_supplimentary_models.py`, the canonical step 5; `run_sparcal_net.py` (100,50) is the unused, buggy one. The live issue is that Methods name neither, so a reader cannot tell which classifier produced the results → **P0-2** |
| **C4** | **Stage 1 is absent from the submission.** α/β equations and thresholds sit in a live `\iffalse` block; the SI has three figures and no body text. Methods promise UPV BAF analyses that do not exist. UPV — a named output class — is undefined and unassessed. Also the only mention of "the ovarian section", a dataset never introduced → **P0-5** |
| **C5** | **Four `\cite` commands; 12 references; 8 never cited.** Uncited but used: STAGATE, Beagle, 1000G, spatialLIBD, Ji cSCC, STMut, GraphST. Absent entirely: COSMIC, CGC, CalicoST, mclust, mpileup, Mutect2, inferCNV, UMAP, 10x Visium, SpaceTracer (which the Discussion claims to cite) → **P0-6** |
| **C6** | **Manuscript unfinished.** Fig. 1 is a placeholder box; Fig. 6's caption says a box plot "remains pending"; Fig. 4's says "Nimbus Sans previews pending Arial"; Fig. 6 is a `v2_2026-07-29` asset predating both the Monopogen DCIS run and the DCIS1 annotation it is captioned with; DCIS1 has no figure at all despite being the only caller-independent, whole-section annotation; Fig. 6 and SI Fig. S3 are the same file; 16 pp. → **P0-7** |
| **C7** | **Pseudoreplication.** The 12 spatialLIBD sections are 3 donors × 4 sections, adjacent pairs 10 µm apart. Every headline p-value treats n=12. The word "donor" does not appear in the manuscript → **P0-4** |

### 8.3 Major findings

| ID | Finding | → |
|---|---|---|
| **M1** | Germline classification is tautological ("by construction, because they are in the 1KGP panel") and its complement — what happened to the other 18–36% — is unreported | P1-6 |
| **M2** | The one validation these data support is missing: germline concordance vs matched-normal WES. Currently one subordinate clause in the Discussion | **P1-1** ⭐ |
| **M3** | No mutational spectrum. The first artifact check for RNA-derived variants. 2 of 14 WES-corroborated calls are already A>G | P1-2 |
| **M4** | "Visium covers ~1% of whole-exome somatic positions" conflates 3′ capture geometry, expression, depth, and allelic dropout. Most exonic mutations are absent from the library at any depth | P1-3 |
| **M5** | **Resolved by author decision:** retain COSMIC as a same-basis external benchmark across classes/callers, report xMHC exclusion separately, and avoid variant-level validation or driver claims | P1-4 |
| **M6** | xMHC correctly diagnosed then not acted on — exclusion should be primary, inclusion the sensitivity check. No HLA-aware realignment; somatic HLA LOH vs mismapping never distinguished | P1-4 |
| **M7** | CalicoST circularity: purity is inferred from allele-specific expression in the same BAM that supplies the variants, then used to rank those variants. ε = ζ·δ makes the voting scheme's voters correlated | P1-7 |
| **M8** | Beagle/1KGP imputation on spatial RNA unvalidated (ASE, NMD, reference bias violate its model); donor/patient ancestry never reported | P1-8 |
| **M9** | Monopogen's depth-floor explanation is asserted, not demonstrated. One matched-floor run settles it | P1-5 |
| **M10** | No independent cohort. Four sections, ≤3 tumors; DCIS2 — where SpatialSNV wins — is SpatialSNV's own dataset | P1-9 ⛔ |

### 8.4 Internal inconsistencies found

1. **Discussion says binning gained "approximately 0.14" for all three matrices**; Results give
   +0.097 / +0.142 / +0.064. The GATK value was applied to all three. **Wrong as written.**
2. **Viewer scores**: text reports ARI 0.594 / NMI 0.602 / hom 0.625 / comp 0.581 / V 0.602,
   17-of-28, mean J 0.377; the in-file verification note records `GT vs ours` = 0.604 / 0.575 /
   0.520 / 0.642 / 0.575, 13-of-28, 0.310. Five of six disagree; a second profile `our_best`
   exists and is not named.
3. SPARCAL-vs-GATK reported as P=0.850 in main text and P=0.68 in SI Fig. S2 — different
   representations, but nothing tells the reader that.
4. Main-text bin-width account omits 500 kb ×2, 25 kb ×1, 100 kb ×1 and the 100 kb ≈ 250 kb tie.
5. Methods → SI forward references point at content that does not exist.
6. Discussion cites a read-quality evidence screen reported nowhere.
7. **Clean:** every leakage %, WES-overlap fraction, and COSMIC count recomputes exactly.

### 8.5 Decisions this assessment challenges

Recorded so the conflict is explicit rather than discovered later. **These are advisor/user
decisions and are not overridden by this section** — but each now has a referee-side argument
against it that should be answered.

- **L3 / D4 (coverage baseline omitted from the region figure)** — already superseded in practice;
  the guided draft plots and discusses it. §6 guardrail 7 is stale. The referee's point is that
  the paper carries the liability of the coverage finding without the credit for it.
- **L4 (coverage benchmark deferred to paper #2)** — the referee reads
  `PAPER_PLAN_DEPRECATED.md` §1 as the stronger half of the study: a coverage-conditioned test
  *with a working positive control* (TSK 0.807, epithelial 0.946 beat coverage; no SNV burden from
  four tools does), STMut at 14,565 gold WES loci giving AUC **0.432** — below chance, with tumor
  spots carrying *lower* burden — cross-method Jaccard 0.049, and subclone structure at η²(UMI)
  0.71–0.74. Splitting has produced a paper with the caveats and a paper with the evidence.
- **D2 (COSMIC as a same-basis comparison; no additional model scheduled)** — retained by author
  decision; use identical catalogue matching across classes/callers and keep claims comparative.
- **D6 (report both raw and xMHC-excluded)** — referee wants xMHC-excluded as *primary*.

### 8.6 Venue

- **PNAS, current draft** — reject; likely without review, given Fig. 1 and four citations.
- **PNAS, fully revised** — poor fit regardless. The result is significant to the spatial-variant
  community, not to a general readership.
- **Nature Methods (Analysis)** — plausible only if merged with the parked coverage benchmark and
  given ground truth (simulation with spiked variants at known VAF/depth/3′ distance), broader
  tool coverage, and a generalizable detectability model. ~25–35%.
- **Genome Biology** — strong fit for the honest bounded version. ~65–75%.
- **Cell Genomics** — underrated middle option. ~45%.
- A **bioRxiv preprint** establishes priority against SpaceTracer and others and forecloses nothing.
