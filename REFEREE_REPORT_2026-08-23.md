# Referee report — SPARCAL PNAS submission, 2026-08-23

Full simulated PNAS referee assessment of `SPARCAL_pnas_2026/PNAS/PaperDraftGuided.tex`
(95,788 B, modified 06:29) and `PNAS/SI_AppendixGuided.tex`, read as a reviewer with field
knowledge. This is the **complete** report; [PAPER_PLAN.md](PAPER_PLAN.md) §8 is the condensed
version and [PAPER_WORK.md](PAPER_WORK.md) is the actionable queue derived from it.

Also published as an artifact: <https://claude.ai/code/artifact/c2963a06-c309-4dec-9c51-56922b3708fd>

Reviewed: main text, SI, all figure assets, and the in-source verification notes. Scope:
scientific claims, controls, statistics, reproducibility, journal readiness.
Manuscript state at review: 16 pp. compiled · 249-word abstract · 119-word significance
statement · ~8,000-word body · 6 main figures · 3 SI figures · 12 references.

---

## Recommendation: **Reject** — overall readiness **3.1 / 10** for PNAS

Not submittable as it stands, and the central positive claim is not yet controlled. The
underlying work is salvageable, but not on this timeline and probably not at this journal.

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

---

## Summary assessment

This manuscript is unusually honest and unusually incomplete. Almost every result it reports,
it also retracts — correctly. The problem is that after the retractions, nothing load-bearing
is left standing, and the one claim that is left standing has never been tested against the
control the paper itself proves elsewhere is decisive.

The authors describe SPARCAL, a variant caller that starts from an aligned Visium BAM, resolves
genotypes against 1000 Genomes, and uses per-spot tumor purity and clone-resolved copy number to
sort candidates into germline, "ubiquitous private", retained somatic, and unresolved classes.
Four analyses follow: cortical-layer recovery in 12 DLPFC sections, an assay-limit
characterization in four tumor sections, a COSMIC-based class-separation test, and a
tumor-region-detection benchmark. SPARCALViewer is presented as a fifth contribution.

Read as a referee, the paper walks itself back at every step. The DLPFC binning result is
disclaimed as "not a clustering win." The spatial augmentation is shown to *reduce* global
agreement (0.350 vs 0.363, P = 0.021). The COSMIC enrichment is concentrated in the extended
MHC, absent in DCIS2 outside it, and null after depth adjustment (OR 1.04, P = 0.56). The
region-detection benchmark is disclaimed as "not a caller ranking" and shows a no-SNV coverage
baseline finishing first on two of three annotations. The focus-separation analysis is null
against its own baseline. The 14 WES-corroborated somatic calls are explicitly "not a
driver-discovery result."

This candour is a genuine virtue, and I would rather review this manuscript than the version
that claimed all of these as wins. But a paper cannot consist entirely of results the authors
decline to claim. The referee's question becomes: **what, specifically, am I being asked to
accept as new and true?** On the current text the answer is one sentence — that a spatially
augmented 250-kb SNV representation matches gene expression on 10-nearest-neighbour
cortical-layer purity (0.856 vs 0.859) — and that sentence rests on an uncontrolled comparison.

### What genuinely works

- **The assay-limit characterization.** That ~1% of matched-WES somatic alleles are re-observed
  in Visium RNA, and that 75–92% of somatic candidates from three independent callers rest on
  one molecule in one spot, is a real, citable, field-relevant bound. Demonstrating it across
  SPARCAL, SpatialSNV *and* Monopogen is the right design.
- **Germline leakage in the reference standards.** 8.3–9.6% of matched tumour/normal Mutect2
  calls, and 2.2–4.7% of released SpatialSNV calls, being allele-exact 1000G matches is useful
  and slightly alarming. The in-source note documenting that the previously circulated 28%/31%
  figures were position-span artifacts is exactly the self-correction that should be preserved.
- **The coverage baseline in region detection.** Running a no-SNV baseline and reporting that it
  wins is the most valuable single result in the paper.
- **The MHC/HLA diagnosis.** Identifying the extended MHC as the dominant source of apparent
  cross-sample recurrence in RNA-derived callsets is a caution the field needs.
- **Numerical bookkeeping.** Every leakage percentage, WES-overlap fraction, and COSMIC count I
  recomputed from the stated numerators and denominators checks out exactly.

---

## Critical findings — each blocks acceptance on its own

### C1 · The headline result has no coverage control, and the paper elsewhere proves that control is decisive

The only surviving positive claim is that a spatially augmented 250-kb SNV representation reaches
10-NN cortical-layer purity of 0.856 against gene expression's 0.859. But a germline-SNV
spot×variant presence matrix is a thresholded function of transcript capture: a variant is
"present" in a spot only where its transcript was captured and sequenced deeply enough. Layer
structure in that matrix may therefore be gene expression, re-derived through a lossy detection
filter, with the genotype contributing nothing.

The manuscript demonstrates that it understands this confounder — it runs a no-SNV coverage
baseline for tumour-region detection, where that baseline beats every caller on two of three
annotations, and it states that per-spot burden is collinear with UMI count at Spearman
0.83–0.97. It then does not run the equivalent control for the DLPFC analysis, which is the
analysis carrying the paper's title claim. Every occurrence of "coverage" as a baseline is in the
tumour sections; there is none in the DLPFC benchmark.

Compounding: 10-NN purity is measured in a STAGATE embedding, and STAGATE explicitly smooths over
the spatial neighbour graph. High local neighbourhood purity is therefore partly manufactured by
the method, independent of the input features. Measuring in the 30-D embedding rather than the
UMAP is more honest than a UMAP, but does not address the smoothing.

**Required:** identical pipeline on (i) a coverage-only feature matrix; (ii) a detection-matrix
control — binarized expression of exactly the genes carrying the variants, same bins, genotype
discarded; (iii) an allele-permuted control preserving detection pattern; (iv) a smoothed-random
matrix to calibrate how much purity STAGATE manufactures. If the SNV matrix does not separate
from (ii), the modality claim in the title cannot stand.

### C2 · The somatic caller is uncalibrated; its output size is set by quota, not by evidence

Stage 2 gives each descriptor a vote on its own top 20% of survivors, then calls the top 10% of
Stage-2 survivors somatic. Stage-1 thresholds are hand-tuned per dataset (defaults T_alpha = 0.5,
T_beta = 0.4; T_beta = 0.2 for DCIS, T_alpha = 0.3 for P4).

Three consequences, all disqualifying as stated. First, the number of somatic calls is a fixed
fraction of the candidate pool, so it tracks depth and pool size rather than mutational burden.
Second, there is no null model, no FDR, no calibration curve, and no score threshold with an
interpretable meaning — a variant is somatic because it ranked in an arbitrary decile. Third, the
thresholds are tuned on the same sections used for evaluation, with no held-out data.

The output is biologically implausible on its face: 19,523 somatic calls in P4 and **65,655** in
P6, from RNA. A hypermutated cSCC exome carries on the order of 10³ coding mutations, and the
paper's own analysis says only ~1% are observable in Visium RNA at all. Combined with 74.8–91.6%
of these calls resting on a single ALT UMI in a single spot, the honest description of this class
is that it is artifact-dominated.

### C3 · SparcalNet is never evaluated, and no component is ablated

"Machine learning" is a listed keyword. The entire description is: an MLP with hidden layers of 64
and 32 neurons, features including BAF, read depth, mapping-quality metrics, base-quality metrics,
and segregation statistics. Absent: training set, label source, class balance, split,
cross-validation, any performance number, hyperparameters, optimizer, seeds, and an ablation
against the error model alone.

No referee can accept a classifier whose accuracy is never reported. The same gap applies to the
pipeline: no ablation shows what CalicoST purity and CNV/LOH evidence contributes, what Stage 1
contributes over Stage 2, or what SparcalNet contributes over the error model.

> **Verified in code 2026-08-23 (corrected).** There are **two** parallel classifier
> implementations, and an earlier draft of this report cited the wrong one:
> `scripts/4_classifier/run_supplimentary_models.py` uses
> `MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)` and **is**
> the canonical pipeline step 5 (root `CLAUDE.md` lines 113 and 192: "features → classified
> VCFs (true variants vs artifacts)"). **The manuscript's "64 and 32 neurons" is therefore
> correct.** `scripts/4_classifier/run_sparcal_net.py` uses `(100, 50)` and is explicitly
> flagged in `CLAUDE.md` as *not* the wired-in classifier — "that has the `no_variance`
> label-encoder bug."
>
> Two reproducibility points survive and should be addressed: (i) the Methods name neither
> script, so a reader with the repo cannot tell which of two classifiers — one of them known
> buggy — produced the published results; (ii) the `no_variance` class never occurs in either
> input VCF for any of the four tumor samples (confirmed against saved `label_encoder.pkl`
> artifacts: `classes_ = ['heterozygous', 'homozygous']`), so a nominally 3-class problem is
> empirically 2-class.
>
> **The substance of C3 is unaffected:** the classifier is still never evaluated anywhere in
> the manuscript — no training set description, label source, class balance, split,
> cross-validation, or performance number.

### C4 · The Stage-1 algorithm is not present anywhere in the submission

The equations for spatial uniformity (α), global prevalence (β), and the T_alpha/T_beta
thresholds sit inside an `\iffalse … \fi` block in the main source, with a note saying they are
"retained below but suppressed … until they are moved into the SI source." They were not moved.
The SI contains three figures and no body text.

The Methods therefore promise two things that do not exist: "The threshold definitions and the
BAF analyses used to assess whether UPVs are artifact-like or germline-like are reported in the
Supplementary Information." Neither is there. UPV — one of four output classes named in the
abstract and significance statement — is undefined, unthresholded, and unassessed. The suppressed
block is also the only place "the ovarian section" appears, a dataset never introduced.

### C5 · Four in-text citations; eight of twelve references never cited

The manuscript contains `\cite` for exactly four works: Strelka2, GATK, Monopogen, SpatialSNV.
Never cited though used or discussed: STAGATE, Beagle, 1000 Genomes, spatialLIBD, Maynard DLPFC,
Ji cSCC, STMut, GraphST.

Missing from the bibliography entirely: COSMIC, Cancer Gene Census, CalicoST, mclust,
samtools/bcftools mpileup, GATK4/Mutect2 (only the 2011 framework paper is present), inferCNV,
UMAP, the single-cell PoN and RNA-editing catalog used in the masking control, 10x Visium, and
SpaceTracer — which the Discussion explicitly says "we cite" while providing neither citation nor
entry.

A PNAS paper in this area carries 50–80 references. At twelve with four cited, this would not
survive editorial screening, and it means the work has not been positioned against the
literature.

### C6 · The manuscript is not finished; the figures contradict the text

- **Figure 1 is a placeholder box** reading "[GUIDED PLACEHOLDER — upload the revised pipeline figure]".
- **Figure 6's caption states "a five-trial box plot remains pending."**
- **Figure 4's caption ends "All panels are Nimbus Sans previews pending Arial."**
- **Figure 6 is stale relative to its own text.** The asset dates from 2026-07-29 and predates both
  the Monopogen DCIS run and the DCIS1 annotation. The text reports Monopogen somatic at ARI 0.660
  on DCIS2 and a full four-way DCIS1 comparison; the figure plots neither.
- **DCIS section 1 has no figure at all** — the only annotation independent of every caller
  compared, the only one covering a whole section so true negatives can be scored, and the one
  carrying the strongest version of the paper's central negative result.
- **Figure 6 and SI Figure S3 are the same file**, and SI Fig. S3 is never cited from the main text.
- **Figure 5 panels (c) and (d) are at different extents** — (d) a crop, (c) the whole section —
  so the side-by-side the caption invites is not valid. Flagged in a source comment, not fixed.
- **Length:** 16 compiled pages with a 2,349-word Methods section in the main text.

### C7 · Pseudoreplication: the 12 DLPFC sections are 3 donors, adjacent sections 10 µm apart

The spatialLIBD DLPFC resource comprises 12 sections from **three** donors, four sections each,
arranged as two pairs of directly adjacent serial sections per donor. Adjacent sections in a pair
are separated by 10 µm — for clustering purposes, near-duplicates of the same tissue.

Every headline P-value treats these as twelve independent units in a paired Wilcoxon test:
P = 0.519 (SPARCAL vs expression, purity), 0.0015 (vs GATK), 4.9×10⁻⁴ (vs Strelka2), 0.021
(augmented vs 1KGP-only), 5×10⁻⁴ (binning), 0.052 (ARI vs expression). Effective sample size is
three donors, or at most six position-blocks. The word "donor" does not appear in the manuscript.

---

## Major findings

| ID | Finding |
|---|---|
| **M1** | **Germline classification is tautological.** The manuscript states it itself: "SPARCAL assigns them to its germline class *by construction, because they are in the 1KGP panel*." The abstract's "64–82%" is not a classification result but a statement about detection coverage. The complement is unexplained — what happened to the other 18–36%? If a nontrivial share landed in the somatic class, that is a measured false-positive rate on known common polymorphism, and it is not reported. |
| **M2** | **The one validation these data support is missing:** germline concordance against matched-normal WES. P4 and P6 have matched normal exomes. Sensitivity, precision, and genotype accuracy at expressed, adequately covered sites, stratified by depth, appear nowhere. It surfaces only as a buried clause: "an audit against matched-normal exome data at high-confidence sites did not support a genotype-correction claim." |
| **M3** | **No mutational spectrum.** For RNA-derived variants the substitution spectrum is diagnostic: A>G/T>C excess ⇒ ADAR editing; C>T/G>A at low VAF ⇒ deamination; strand asymmetry ⇒ library artifact. Nothing is shown for the 19,523–65,655 somatic candidates. The only spectrum information is that 2 of 14 WES-corroborated calls are A>G — a 14% rate in the best-supported subset. |
| **M4** | **"Visium covers approximately 1% of whole-exome somatic positions" misdiagnoses the paper's own best result.** Visium 3′ chemistry sequences roughly the terminal few hundred bases of each transcript. Most exonic mutations are not undersampled — they are *absent from the library* at any depth. That is a capture-geometry limit, not a coverage limit. Decomposed properly (expressed gene → within 3′ window → ≥1 read → ALT observed) this becomes the most durable contribution in the manuscript. |
| **M5** | **The COSMIC result is null after adjustment for its own confounder, yet occupies a main figure.** Outside the xMHC: 0.99-fold (P = 0.55) in DCIS2, 1.11-fold (P = 0.099) in DCIS1. After depth adjustment: OR 1.04 (P = 0.56) and 1.00 (P = 0.95). Absent in two of four sections, non-surviving in the other two. Also: Fisher's exact on non-independent variants overstates significance; no multiple-testing correction; one-sided tests need pre-specification. |
| **M6** | **The MHC finding is correctly identified and then not acted on.** The extended MHC is the most polymorphic region of the genome, exceptionally highly expressed in tumour and immune-infiltrated tissue, and mismaps against a linear reference routinely — precisely the region that would dominate a depth-driven catalog-hit rate for artifactual reasons. Exclusion should be the default with inclusion as sensitivity, at which point the headline becomes "present in hypermutated cSCC, absent in DCIS." No HLA-aware realignment; somatic HLA LOH versus mismapping never distinguished. |
| **M7** | **CalicoST circularity is not addressed.** CalicoST infers purity, clones, and allele-specific CN from allele-specific expression in the *same BAM*. SPARCAL then ranks candidates by correlation with that purity (δ) and consistency with those CNV/LOH segments (θ). Predictor and evidence derive from the same allelic signal in the same reads. ε = ζ·δ compounds it — a variant scoring well on both effectively receives three of four votes from two underlying quantities. CalicoST is also uncited. |
| **M8** | **Beagle/1000G imputation on spatial RNA is unvalidated.** Beagle assumes DNA genotypes at roughly uniform, unbiased depth. RNA violates this through allele-specific expression, imprinting, NMD, and reference bias at heterozygous sites. Separately, panel-based resolution inherits panel ancestry composition: variants private to under-represented ancestries fail to resolve as germline and flow into the de novo pool, and potentially into the somatic class. Donor and patient ancestry are not reported. |
| **M9** | **The Monopogen comparison lacks the one ablation that would settle it.** The paper correctly identifies Monopogen's ≥4 ref / ≥4 alt floor and concludes it "selects the sites already above the floor." That is asserted, not demonstrated. Apply the same floor to SPARCAL and re-compare — one run. |
| **M10** | **No independent validation cohort.** Four sections from at most three tumours; whether DCIS1 and DCIS2 are the same patient is never stated. No held-out data anywhere. DCIS section 2 — where SpatialSNV finishes first at ARI 0.671 — is drawn from the dataset accompanying the SpatialSNV publication. Flagging a non-independent benchmark does not repair it. |

---

## Moderate findings

| ID | Finding |
|---|---|
| **m1** | **Title claims what the body spends eight pages declining to claim.** Title: SPARCAL "establishes spatially resolved SNVs as an analytical modality." Body: not a clustering win; not a caller ranking; not evidence of a detection signal independent of coverage; not variant-level validation; not driver discovery; region burden is "a descriptive readout rather than an independent tumor-detection signal." |
| **m2** | **Displayed examples are selected on the quantity being displayed, at their best runs.** Fig. 2a–c shows 151672 (highest mean ARI), 151510 and 151676 (largest positive margin within prespecified blocks), each at its best-ARI run of ten. Disclosed, which is to the authors' credit, but three winners at their best runs invites exactly the inference the caption then asks the reader not to draw. |
| **m3** | **SPARCALViewer scores appear quoted from the better of two profiles.** Text: ARI 0.594, NMI 0.602, hom 0.625, comp 0.581, V 0.602, 17/28 at J ≥ 0.25, mean J 0.377. The verification note in the same file records `DCIS1 GT vs ours`: ARI 0.604, NMI 0.575, hom 0.520, comp 0.642, V 0.575, mean J 0.310, 13/28. Five of six disagree; homogeneity/completeness are inverted. A second profile `our_best` exists in the figure comment. |
| **m4** | **Run-to-run noise is the size of several reported effects.** Median within-cell SD across five STAGATE runs is 0.041 (STAGATE is seeded but not bit-reproducible: `cudnn.benchmark` enabled alongside deterministic mode, non-deterministic scatter atomics). Comparisons at 0.350 vs 0.363, 0.350 vs 0.354, 0.856 vs 0.859 sit inside that band per run. Every value in the main text is currently a bare point estimate. |
| **m5** | **Reference builds mixed with no liftover statement.** GRCh37 for P4/P6, GRCh38 for DCIS. `chr6:28–34 Mb` is not the same interval in both; COSMIC v103 GRCh37 and GRCh38 releases differ in content. Ranges quoted as single findings (1.16–1.33-fold, 6.9–18.5% xMHC share, 18–45% HLA) each span both builds. |
| **m6** | **No dataset table.** Per section: spots, median UMIs/spot, median reads/spot, saturation, tissue area, candidate positions, class counts — none reported. Every depth-dependent claim is uninterpretable without them. |
| **m7** | **Front matter.** W.Y. and P.B. have no contribution statement. "X.M." and "X.M.Z." are ambiguous initials for two authors. Six keywords against a limit of five. No ethics statement for human tumour tissue. Data availability names a GitHub URL only — no accessions, DOI, version tag, license, or environment spec. |
| **m8** | **Template misuse.** Methods sit in a plain `\section*{Methods}` while `\matmethods{}` holds only data availability, so `\showmatmethods` renders a near-empty "Materials and Methods" after a 2,349-word Methods section. Plus a live `\providecommand{\PENDING}` macro, a dead `\iffalse` block, `LastPage` multiply-defined, five overfull boxes, and a float overflowing by 0.11 pt at line 224. |

---

## Minor and editorial

- Section headings are process statements — seven of eight Results headings begin "We tested…",
  "We compared…", "We observed…". PNAS headings should state findings.
- "We observed" opens a large fraction of paragraphs.
- "Ubiquitous private variant (UPV)" is coined terminology used in the abstract without definition
  or literature grounding.
- Abstract 249/250 words; significance statement 119/120. No headroom for required additions.
- The Discussion cites a read-quality evidence screen reported nowhere in the manuscript or SI.
- SpaceTracer: "we cite but did not benchmark, as we were unable to complete an end-to-end run on
  our data" — and no citation is given.
- Methods describe the same two-stage cascade three times.
- Guided/TODO source comments enumerate every unfinished item and must be stripped.

---

## Internal inconsistencies

| Where | Manuscript says | Source says | Status |
|---|---|---|---|
| Discussion ¶2 | "improved mean ARI by **approximately 0.14** for the 1KGP-only, GATK, and Strelka2 matrices" | +0.097 / +0.142 / +0.064 | **Wrong** — GATK value applied to all three. Use 0.064–0.142 |
| Viewer results | ARI 0.594 / NMI 0.602 / hom 0.625 / comp 0.581 / V 0.602; 17 of 28; mean J 0.377 | `GT vs ours`: 0.604 / 0.575 / 0.520 / 0.642 / 0.575; 13 of 28; 0.310 | **Wrong or mislabelled** — five of six disagree; `our_best` unnamed |
| SPARCAL vs GATK | main text "indistinguishable (P = 0.850)" | SI Fig. S2 "indistinguishable (7 of 12, P = 0.68)" | Different representations; reader cannot tell. Label both |
| Bin-width sweep | "125 kb for five sections and 250 kb for three" | 125×5, 250×3, 500×2, 25×1, 100×1; cohort 100 kb 0.361 vs 250 kb 0.362 | Main text is a partial account of its own SI; the tie is omitted |
| Region detection | three annotations incl. DCIS1; Monopogen somatic on DCIS2 | Fig. 6 asset dated 2026-07-29, DCIS2 + P6 only | **Figure predates its caption**; DCIS1 unplotted |
| Methods → SI | "reported in the Supplementary Information" | SI has three figures, no body text | **Broken forward reference** |
| Discussion | "a read-quality evidence screen … added nothing" | No such experiment reported | Unsupported — report or remove |
| Leakage & COSMIC arithmetic | 2.168% (2,264/104,439); 8.3% (288/3,450); 1.245% (243/19,523) … | all recompute exactly | **Verified correct** |
| SparcalNet architecture | "hidden layers of 64 and 32 neurons" | canonical `run_supplimentary_models.py`: `(64, 32)`; unused `run_sparcal_net.py`: `(100, 50)` | **Correct as written.** But the Methods name neither script, and two classifiers ship — one flagged buggy |

---

## What a resubmission needs, in priority order

1. **Settle C1** — coverage, detection, and permutation controls for DLPFC. Decides whether there
   is a paper. Run before writing anything else.
2. **Promote the germline concordance analysis (M2)** to a main result.
3. **Calibrate the somatic caller (C2)** and show its spectrum (M3).
4. **Report SparcalNet properly and ablate the pipeline (C3).**
5. **Re-run every DLPFC statistic with donor as the unit (C7).**
6. **Decompose the 1% (M4)** into capture geometry, expression, depth, allele observation.
7. **Make xMHC exclusion the primary analysis (M6).**
8. **Finish the manuscript (C4, C5, C6).**

## Venue

- **PNAS, current draft** — reject, likely without review.
- **PNAS, fully revised** — poor fit regardless. Significant to the spatial-variant community,
  not to a general readership.
- **Nature Methods (Analysis)** — plausible only if merged with the parked coverage benchmark
  ([PAPER_PLAN_DEPRECATED.md](PAPER_PLAN_DEPRECATED.md) §1) and given ground truth (simulation
  with spiked variants at known VAF/depth/3′ distance), broader tool coverage, and a
  generalizable detectability model. ~25–35%.
- **Genome Biology** — strong fit for the honest bounded version. ~65–75%.
- **Cell Genomics** — underrated middle option. ~45%.
- A **bioRxiv preprint** establishes priority against SpaceTracer and others, and forecloses
  nothing.

The negative results here are better than the positive ones. A paper built around the
capture-geometry decomposition, the three-caller single-molecule floor, the coverage-baseline
result, the MHC diagnosis, and the germline leakage in published reference standards — with
SPARCAL as the instrument rather than the headline — is a strong, citable paper. The manuscript
is already about 70% written in that direction.
