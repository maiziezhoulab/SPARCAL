# Pipeline Intro — Figure Design + Paper Wording

This doc has two parts:
- **§1–§7 — Figure-design discussion:** how to draw the main pipeline schematic.
- **§8 — Paper wording:** ready-to-adapt text for the abstract, methods, and
  results that refer to the pipeline detail and the variant categories.

Goal of the figure: a single panel a reader can follow top-to-bottom and
understand *what each stage decides*. The old 7-panel figure is a good skeleton
but (a) over-details the classifier and the genotype tables, and (b) under-
explains the novel step — **how a de-novo SNV is partitioned in the spatial
filter**, which the old figure draws as an abstract score scatter that no longer
even matches the code.

## 0. Current pipeline as run (updated 2026-10-09) — read this first

§1–§8 below date from June 2026. Where they disagree with this section, **this section wins**.
Source of truth for wording is the Methods of `SPARCAL_pnas_2026/PNAS/PaperDraftGuided.tex`.
Stale items below: §4.1 "BAF-GMM sub-filter (NEW TODO)" is no longer a pipeline step (it became the
SI UPV assessment); §7 CHIP rule-out is an optional user filter, not a default step; §8.3 per-spot
profiling describes presence as "a record exists", which is the bug fixed in step S6+ below.

### 0.1 Steps (code names as in the job chain and the run console)

| Step | Name | What it decides | Notes for the figure |
|---|---|---|---|
| S1 | Candidate calling | pooled mpileup over all spots → candidate SNVs; pooled 0/0 genotype gate | input = aligned Visium BAM only; no Mutect2 |
| S2 | Beagle routing | Beagle + 1KGP refine genotype posteriors (`gl=`, `impute=false`; unphased GT+GP) for exact panel alleles → **in-panel germline** route | DLPFC germline benchmark uses this route only (no somtop25) |
| S3 | Genotype shifting | corrected genotypes for panel sites → training labels | |
| S4 | Sequence-error model | 12 genotype-transition patterns, pattern-specific BAF/depth thresholds | |
| S5 | SparcalNet | MLP (64, 32) classifies non-panel candidates: non-variant / het / hom | |
| S6 | Per-spot profiles | per-spot BAM + VCF at every retained candidate | AD counted per read, no UMI collapse |
| **S6+** | **Molecule recount (new, 2026-10-09)** | per spot and locus: primary reads only, collapsed by UB, majority base per molecule → `ref_mol`, `alt_mol` | `scripts/presence_fix/recount_spot_alleles.py` |
| S7 | Spatial filter | CalicoST purity / clones / CNV-LOH; Stage 1 uniformity + prevalence → UPV; Stage 2 votes (δ, ε, ζ, θ; top 20% each; top 10% of survivors) → somatic, rest ambiguous | ambiguous never shown |
| D1 | Matrices | spot × locus matrices per class; unbinned (UMAP) and 250 kb bins (ARI) | |
| D2 | Downstream | clustering (ARI, KNN purity, UMAP), region detection, COSMIC, SPARCALViewer | |

### 0.2 The presence rule (open decision, 2026-10-09)

Before 2026-10-09 every production matrix set a spot to 1 wherever S6 wrote a record, i.e. wherever
any read covered the locus. Only 15–30% of P4/P6 and DLPFC 1s, and 8.4% of DCIS2 somatic 1s, carry an
ALT read. Arms under test (adopt a new rule only if it performs better):

- **G0 / record** — current behaviour (coverage = presence).
- **G1** — keep loci with ≥1 ALT molecule in at least one spot; values unchanged.
- **G2 / alt_mol** — spot = 1 only if it has ≥1 ALT molecule at the locus.

DLPFC (G0–G2), P4/P6 step 7 (record / alt_mol ± CalicoST) and the Fig. 5 DCIS/P6 matrices
(G0–G2) are being run. **The figure must show whichever rule is adopted**; until then draw S6+ as a
step but leave its label to the decision.

### 0.3 What Fig. 1 (P1) should show — draft list for the author to edit

1. **Single sample input:** aligned 10x Visium BAM (+ spot coordinates, reference, 1KGP panel). Mark
   "no Mutect2 / no external callset".
2. **Primary pathway, left to right:** S1 candidates → S2 Beagle/1KGP routing (in-panel germline,
   blue) → S4–S5 sequence-error model + SparcalNet on non-panel candidates → S6/S6+ per-spot
   molecule-level allele counts.
3. **Secondary pathway (parallel):** CalicoST → per-spot purity, clone labels, allele-specific CNV/LOH;
   thick labelled arrow into Stage 2 only.
4. **Spatial filter (hero panel):** Stage 1 (purity-independent) → **UPV** (purple, uninterpreted
   label; interpretation deferred to SI); Stage 2 vote → **somatic** (red); ambiguous not drawn.
   Two small spot-map icons: ubiquitous (UPV) vs focal (somatic).
5. **Outputs:** spot × SNV matrices (unbinned / 250 kb), class VCFs, viewer overlays.
6. **Downstream icons:** layer clustering (DLPFC), tumor-region detection (DCIS/P6), COSMIC context,
   SPARCALViewer.
7. Colours fixed: germline blue, UPV purple, somatic red (see `CLAUDE.md` house palette in the
   manuscript repo for the exact hex values). Do not put the ambiguous bucket or any accuracy number
   in the schematic.

### 0.4 Drawing requirements (for Codex)

- Editable vector master: **SVG** (text as text, not paths) plus **PDF** export with embedded
  TrueType text (`pdf.fonttype 42` if drawn in Matplotlib); PNG at 300 dpi for preview only.
- Arial for all non-gene text (see the Arial gate in the manuscript `CLAUDE.md`).
- No figure title in the artwork; panel letters only. Caption lives in the `.tex`.
- Write to `SPARCAL_pnas_2026/figs/<new version dir>/fig1_pipeline_overview.{svg,pdf,png}` and update
  `figures/fig1-pipeline.md` and `FIGURE_PROVENANCE.md`.

---

## Decisions locked (2026-06-01)

- **Naming = neutral / behavioral.** Categories are named by *spatial behavior*,
  not by an unproven germline/somatic identity, because category (b) is genuinely
  a mixture. Identity is argued in the Discussion, not asserted in the label.

  | Figure label | Color | Code category (current) | Meaning |
  |--------------|-------|-------------------------|---------|
  | **(a) germline** | **blue** | `germline_defined` | 1000G-known germline (free pass) |
  | **(b) Ubiquitous Private Variants (UPV)** | **purple** | `germline_denovo` | not in 1000G + spatially ubiquitous; *mix* of novel germline + permissive/early(or truncal) somatic |
  | **(c) somatic** *(= denovo_focal)* | **red** | `somatic_denovo` | focal, clustered, tumor/clone/CNV-linked; late "dot-mutation" somatic |

  > **Paper term for (b): "Ubiquitous Private Variants (UPV)".** Chosen because it
  > encodes the *exact* filter logic — *Private* = absent from 1000G; *Ubiquitous* =
  > passed the spatial broad/uniform rule-out — without claiming a mechanism.


- **Fixed 3 categories everywhere.** The pipeline always emits the same three,
  regardless of normal vs tumor tissue. The fact that (b) collapses to ~pure
  germline in *normal* tissue (no somatic to contaminate it) is explained in the
  **text only** — it does not change the outputs or the figure structure.

- **Colors:** germline = blue, somatic = red, everywhere (results plots + figure).
  (b) UPV = purple, signaling "in between / mixed".

- **Ambiguous bucket:** not shown in the figure.

---

## 1. What the old figure gets right (keep)

- Three-column top row reads as a story: *call → genotype-correct → filter*.
- The Beagle/1000G banner across the top as an "anchor reference" motif.
- Bottom row (CNV/clones → spatial filter → downstream) is the right flow.
- The germline/somatic split shown as the *output* of the spatial module.

## 2. What changes

| Panel | Problem | Fix |
|-------|---------|-----|
| **3. SparcalNet** (`n×27 → Dense → MaxPool …`) | Layer-size numbers are too detailed for an overview. | **Keep the architecture cartoon's structure** (it usefully fills the space) but **strip the dimension numbers** — show the shape of the net, not its hyperparameters. Detailed sizes go to a Supp. Fig. |
| **2. Genotype Correction** + **4. Undefined SNVs Filtering** | Two big panels, many tiny `Chr/Pos/GT` tables, Pool A/B / "corrected GT" / "all sequencing errors". Training path vs inference path is unreadable. | **Merge** into one panel. Show the *decision*: in-1000G → training labels; outside-1000G → SparcalNet inference → keep/drop. Replace the literal GT tables with at most one iconic mini-table. |
| **6. Spatial Filtering** | Drawn as an abstract "Germline Score vs Somatic Score" scatter — and that scatter describes the **old v1 scoring algorithm**, not the current code. | **Redesign** around the real **rule-out cascade** and the spatial signature behind each call — see §4. |
| **5. CNV & Cancer Clones** | Eagle2→Calicost→tables chain is visually heavy and its link into the somatic decision is a thin, easy-to-miss arrow. | Keep, **shrink internal tables**, and **thicken + label** the arrow into the somatic stage ("clone labels + CNV/LOH segments"). |

**Net effect:** 7 panels → ~5–6. Freed space goes to the spatial-filter panel.

---

## 3. Proposed layout (mockup)

```
┌────────────── Beagle genotype-posterior refinement (1000 Genomes panel) ──────────────┐
▼                                                                                       ▼
┌─────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────┐
│ 1. Preprocessing│ → │ 2. Genotype correction &      │   │  reference split:         │
│  Visium BAM     │   │    classification             │ ← │   in-1000G  → train labels│
│  → per-spot     │   │  ┌─────────────────────────┐  │   │   out-1000G → infer       │
│    SNV profile  │   │  │ SparcalNet (3-class:    │  │   │             → keep / drop │
│                 │   │  │ non-var / het / hom)    │  │   └───────────────────────────┘
│                 │   │  │ [arch shape, no numbers]│  │
│                 │   │  └─────────────────────────┘  │
└─────────────────┘   └───────────────┬───────────────┘
                                       │ kept SNV pool
                                       ▼
┌─────────────────┐   ┌───────────────────────────────────────────┐   ┌────────────────┐
│ 3. CNV & clones │ ═►│ 4. SPATIAL FILTER (rule-out cascade, §4)   │ → │ 5. Downstream  │
│  Calicost:      │   │   (b) UPV      ◄ ruled out first (+BAF-GMM) │   │  tumor detect  │
│  tumor purity,  │   │   (c) somatic  ◄ what survives (−CHIP)      │   │  clustering    │
│  clone labels,  │   │   (a) germline ◄ 1000G free pass           │   │  feature anal. │
│  CNV/LOH segs   │   │   outputs: germline / UPV / somatic        │   │                │
└─────────────────┘   └───────────────────────────────────────────┘   └────────────────┘
    (purity, clone labels, CNV/LOH) ═══════════► feed ONLY the somatic stage
```

Outputs leaving panel 4 must be the three locked categories, colored
**blue / purple / red**.

---

## 4. The spatial-filter panel (the hero panel)

Must answer: **given a de-novo SNV (not in 1000G), how is it partitioned?**
The current algorithm (`run_spatial_snv_filter_enhanced.py`) is a **rule-out
cascade** — *not* the v1 score-and-compare. The figure must mirror the cascade.

> **v1 (old, do not draw):** compute germline-score and somatic-score per SNV,
> assign to whichever is higher → the score scatter.
> **v2 (current):** first **rule out** the broad/uniform variants as
> **UPV**, then rank *what remains* to pull out the focal `somatic`.

### 4.1 The real decision logic (ground truth from code)

```
                       de-novo SNV (outside 1000G, passed classifier)
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        │ STAGE 1 — RULE OUT broad variants  →  (b) UPV                  │
        │   α = spatial uniformity   (evenly spread across tissue?)      │
        │   β = global prevalence    (in a large fraction of spots?)     │
        │   IF α > 0.5 AND β > 0.2  →  UPV  (removed from cascade here)  │
        │   ┌───────────────────────────────────────────────────────┐   │
        │   │ NEW (TODO): BAF-GMM sub-filter inside the UPV set      │   │
        │   │   germline het clusters at BAF≈0.5 / hom≈1.0;          │   │
        │   │   a lower-BAF component = candidate (early/mosaic)     │   │
        │   │   somatic. 2-component GMM splits UPV →                │   │
        │   │     UPV-germline-like   vs   UPV-somatic-candidate.    │   │
        │   │   (seq-error model already drops BAF<0.05.)            │   │
        │   └───────────────────────────────────────────────────────┘   │
        └───────────────────────────────┬───────────────────────────────┘
                   (b) UPV  ◄────────────┤ yes
                                         │ no  → survives to Stage 2
                                         ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ STAGE 2 — RANK survivors, take the focal ones → (c) somatic    │
        │   each feature votes for its top-20%:                          │
        │     δ  purity correlation   (presence tracks tumor purity)     │
        │     ζ  spatial clustering   (neighbors share it)               │
        │     ε  clone-specific = ζ·δ                                    │
        │     θ  CNV/LOH consistency  (matches Calicost clone CNV)       │
        │   top-10% by votes  →  (c) somatic ;  rest → ambiguous (drop)  │
        │   ┌───────────────────────────────────────────────────────┐   │
        │   │ NEW (TODO): CHIP rule-out (post-filter on (c))         │   │
        │   │   drop variants in common CHIP genes (DNMT3A, TET2,    │   │
        │   │   ASXL1, JAK2, TP53, …) — likely clonal-hematopoiesis  │   │
        │   │   blood contamination, not tissue somatic.            │   │
        │   └───────────────────────────────────────────────────────┘   │
        └───────────────────────────────────────────────────────────────┘

   Free pass:  in-1000G  →  (a) germline   (never enters the cascade)
```

### 4.2 The key visual idea — two spot maps, not a scatter

Replace the score scatter with **two side-by-side tissue spot maps** that make
the rule-out self-evident: (b) is ruled out because it's *everywhere*; (c)
survives because it's *focal and tumor-linked*.

```
   STAGE 1 rules this OUT first:                STAGE 2 keeps this:
        (b) UPV  [purple]                            (c) somatic  [red]
   uniform · prevalent (α,β high)            focal · purity/clone/CNV-linked

   ┌───────────────────────┐                ┌───────────────────────┐
   │ ● ● ● ● ● ● ● ● ● ● ● │                │ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ │
   │ ● ● ● ● ● ● ● ● ● ● ● │                │ ○ ○ ● ● ● ○ ○ ○ ○ ○ ○ │
   │ ● ● ● ● ● ● ● ● ● ● ● │                │ ○ ● ● ● ● ● ○ ○ ○ ○ ○ │ ◄ tumor
   │ ● ● ● ● ● ● ● ● ● ● ● │                │ ○ ○ ● ● ● ● ○ ○ ○ ○ ○ │   clone
   │ ● ● ● ● ● ● ● ● ● ● ● │                │ ○ ○ ○ ● ● ○ ○ ○ ○ ○ ○ │   (+CNV/LOH)
   └───────────────────────┘                └───────────────────────┘
   present across ALL tissue                concentrated in one clone,
   (incl. normal regions)                   overlapping a CNV/LOH segment
   → high α (uniform), high β (prevalent)   → high δ,ζ,θ
```

Compact "evidence bars" under each map (the features that actually fire):

```
 (b) UPV:      α uniform ▓▓▓▓▓░   β prevalent ▓▓▓▓░░   (+BAF-GMM sub-split)
 (c) somatic:  δ purity-corr ▓▓▓▓░  ζ cluster ▓▓▓▓▓  θ CNV/LOH ▓▓▓▓░  (−CHIP)
```

A one-line caption carries the identity hedge the naming was chosen for:
*"UPV (Ubiquitous Private Variants) = not in 1000G + spatially ubiquitous; a mix
of novel germline + permissive/early somatic. In normal tissue it is ≈ germline."*

### 4.3 Show the Calicost hand-off explicitly

Tumor purity / clone labels / CNV-LOH segments enter **only at Stage 2** (the
somatic ranking). Draw the labeled arrow from the CNV-&-clones panel landing on
the **somatic** map (purity→δ, clone labels→ε/enrichment, CNV/LOH→θ). Stage 1 is
deliberately purity-*independent* — worth annotating, since "broad regardless of
tumor purity" is exactly why (b) is ruled out before any tumor evidence is used.

### 4.4 Optional inset

A small α–β plane (uniformity × prevalence) with the Stage-1 threshold box
(α>0.5, β>0.2) drawn as the "rule-out region" can be a secondary inset to show
the cut quantitatively. Keep it subordinate to the two spot maps; cut it if the
panel crowds.

---

## 5. Concrete to-do for v2 of the figure

1. Panel 3: keep the SparcalNet architecture **shape**, delete the dimension
   numbers; relabel as a 3-class classifier. Full sizes → Supp. Fig.
2. Merge panels 2 & 4 into one "genotype correction & classification" panel;
   remove the literal `Chr/Pos/GT` tables (keep ≤1 iconic mini-table).
3. Redesign the spatial panel around the **rule-out cascade** (§4.1) + the
   **two-spot-map** contrast (§4.2). Drop the v1 score scatter.
4. Make the Calicost → Stage-2(somatic) arrow explicit and labeled; annotate
   Stage 1 as purity-independent (§4.3).
5. Output labels = **germline (blue) / UPV (purple) / somatic (red)**.
6. Keep the 1000G/Beagle top banner; keep panels 1 and downstream as-is.
7. Add the one-line identity-hedge caption under the spatial panel (§4.2).
8. Draw the two new sub-steps (BAF-GMM inside UPV, CHIP rule-out after somatic)
   as small nested boxes — see §7.

---

## 6. Resolved

- **Naming:** neutral/behavioral — germline / **UPV (Ubiquitous Private
  Variants)** / somatic (see locked table at top). ✔
- **Normal vs tumor:** fixed 3 categories everywhere; normal-tissue reading of
  (b) explained in text only. ✔
- **Ambiguous bucket:** omitted from the figure. ✔
- **Colors:** germline blue, somatic red, UPV purple, everywhere. ✔
- **Model branding:** keep **"SparcalNet"** (the network) and **"SPARCAL"** (the
  overall model/method) for the paper. ✔
- **Code rename** (`germline_defined`/`germline_denovo`/`somatic_denovo` →
  `germline`/`upv`/`somatic`): **marked, deferred** to a later pass. ✔

---

## 7. Planned method additions (TODO — not yet implemented)

These came out of the (b)-naming discussion; both refine the spatial-filter
outputs and should appear as small nested boxes in panel 4.

1. **BAF-GMM sub-filter inside the UPV set.** UPV is a mix of novel germline and
   permissive/early(or truncal) somatic. Fit a 2-component Gaussian mixture on the
   per-variant BAF/VAF: germline het variants cluster at BAF≈0.5 (hom≈1.0), while
   a lower-BAF component flags **UPV-somatic-candidates**. Splits UPV →
   `UPV-germline-like` vs `UPV-somatic-candidate`. Note the existing sequencing-
   error model already removes very-low BAF (<0.05), so the GMM operates on
   what's left. *Lightweight, fits naturally inside Stage 1.*

2. **CHIP rule-out on the final somatic set.** Clonal Hematopoiesis of
   Indeterminate Potential (CHIP) introduces blood-derived clonal mutations that
   masquerade as tissue somatic calls. Drop variants falling in a common CHIP
   gene list (DNMT3A, TET2, ASXL1, JAK2, TP53, SF3B1, …) from category (c).
   **Recommendation: implement as a post-processing filter** on the somatic VCF
   (a gene-list intersection annotation), not inside the core cascade — it's a
   simple, swappable exclusion that doesn't need the spatial machinery. Keep it a
   separate step so the gene list can be updated independently.

*(Tracked as live tasks in `On_going.md` → "Planned method additions".)*

---

## 8. Paper wording

Ready-to-adapt phrasing for the manuscript. **SPARCAL** = the overall method;
**SparcalNet** = the neural-network classifier inside it. Terminology is fixed to
the locked categories: **germline / UPV / somatic** (blue / purple / red).

### 8.1 Terminology box (define once, reuse everywhere)

> Throughout, we partition called variants into three classes by their spatial
> behavior rather than by an assumed origin. **Germline** variants are those
> present in the 1000 Genomes Project (1KGP) reference panel. **Ubiquitous
> Private Variants (UPVs)** are absent from 1KGP yet distributed ubiquitously and
> uniformly across the tissue section; lacking trio data, we do not attempt to
> separate their rare-germline, de-novo, and early/truncal-somatic constituents,
> and we note that in normal (non-tumor) tissue this class is expected to be
> predominantly germline. **Somatic** variants are absent from 1KGP and spatially
> focal, enriched in tumor clones and consistent with their copy-number profile.

### 8.2 Abstract (one-paragraph version)

> Spatial transcriptomics resolves gene expression across tissue, but calling
> single-nucleotide variants (SNVs) per spot is confounded by sparse, error-prone
> per-barcode coverage and by the difficulty of separating germline from somatic
> variation in situ. We present **SPARCAL**, a pipeline that calls germline and
> somatic SNVs from 10x Visium per-barcode alignments. SPARCAL refines genotypes
> against the 1000 Genomes panel with Beagle, classifies each candidate as a true
> variant or a sequencing artifact with a neural network (**SparcalNet**), and
> then applies a spatially-aware rule-out filter that integrates tumor purity,
> clonal structure, and allele-specific copy number (from CalicoST) to partition
> variants into three spatially-defined classes — germline, Ubiquitous Private
> Variants (UPVs), and focal somatic mutations. Applied to [DCIS / DLPFC / breast
> tumor] sections, SPARCAL yields spot×SNV matrices that support tumor detection,
> spatial-domain clustering, and downstream variant analysis.

### 8.3 Methods

**Variant calling and per-spot profiling.** Per-barcode alignments (split from
the SpaceRanger `possorted_genome_bam.bam` by `CB` tag) are pileup-called with
samtools/bcftools in multi-sample mode, restricted to in-tissue spots, producing
an initial per-spot SNV profile. The base- and mapping-quality thresholds
(e.g. `baseQ0mapQ0`) propagate through all downstream steps.

**Genotype-posterior refinement and classification.** At candidates represented
by an exact allele in the 1000 Genomes panel, Beagle refines genotype posteriors
from pooled genotype likelihoods. The production call uses `gl=` and
`impute=false`, so it returns unphased GT plus GP at target markers rather than
phased haplotypes or newly imputed reference-only markers. Variants present in 1KGP supply
labeled training examples (consistent genotypes, corrected genotypes, and
non-variant sequencing-error sites); variants outside 1KGP are scored by a
per-site sequencing-error model and classified by **SparcalNet**, a neural
network that assigns each candidate to one of three states (non-variant,
heterozygous, homozygous). Predicted non-variants are discarded; retained
variants form the kept SNV pool.

**Clonal and copy-number context.** In parallel, CalicoST results from the same
section provide per-spot tumor purity, spot-to-clone assignments, and
allele-specific copy-number (including LOH) segments per clone.

**Spatial rule-out filter.** De-novo (non-1KGP) variants are partitioned by a
two-stage cascade (1KGP-known variants bypass the cascade as germline):

> *Stage 1 — germline/UPV exclusion.* A variant is assigned to the **UPV** class
> if it is both spatially uniform (low coefficient of variation of inter-spot
> distances, α > 0.5) and globally prevalent (present in a large fraction of
> spots, β > 0.2). This stage is deliberately independent of tumor purity, so
> that broadly-distributed variants are removed before any tumor-specific
> evidence is considered.
>
> *Stage 2 — somatic selection.* Remaining variants are ranked by tumor-purity
> correlation, local spatial clustering, a clone-specific term, and CNV/LOH
> consistency with the CalicoST segments. Each feature votes for its top-ranked
> variants, and the most-voted fraction is called **somatic**; the rest are left
> ambiguous and excluded.

Color convention in all figures: germline = blue, UPV = purple, somatic = red.

**Final matrices.** The three classes are merged into a spot×SNV matrix and a
per-spot variant table for downstream analysis (`final_snv_mat.py`).

### 8.4 Methods — planned refinements (describe as future/optional)

> *(Include only if implemented before submission — see §7.)*
> Within the UPV class, a two-component Gaussian mixture on B-allele frequency
> can be used to flag a lower-BAF sub-population as candidate somatic variants,
> separating them from the germline-like component centered at BAF ≈ 0.5 (the
> upstream sequencing-error model already removes sites with BAF < 0.05). To
> reduce blood-derived contamination of the somatic class, variants falling in
> canonical clonal-hematopoiesis (CHIP) genes (e.g. *DNMT3A*, *TET2*, *ASXL1*,
> *JAK2*, *TP53*) are removed in a post-processing step.

### 8.5 Results — phrasing snippets

- "SPARCAL recovered N germline, N UPV, and N focal somatic SNVs per section
  (median, baseQ0mapQ0)."
- "As expected for normal cortical tissue, the UPV class in DLPFC was dominated
  by germline-consistent variants, whereas tumor sections (DCIS, breast) showed a
  distinct focal-somatic class concentrated in CalicoST tumor clones."
- "Spot×SNV matrices from SPARCAL clustered into spatial domains concordant with
  [annotated layers / histology], outperforming/comparable to germline-only
  callers (Strelka2, GATK) benchmarked on the same spots."

### 8.6 Naming caveats to keep in the text (not the figure)

- UPV is a **mixture**; do not call it "germline" without the 1KGP-absence +
  ubiquity qualifier. The normal-tissue ≈ germline statement belongs in the text.
- "Somatic" here means **spatially-focal, tumor-clone/CNV-consistent**; state the
  operational definition so it is not read as orthogonally-validated somatic
  calling.
