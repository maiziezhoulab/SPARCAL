# SPARCAL — progress report, 2026-08-07

Covers everything done since the **2026-08-04 advisor review**. Prepared for the 2026-08-07 meeting.

Authoritative detail lives in [PAPER_PLAN.md](PAPER_PLAN.md) (§5 Amendments, §4.1–4.1c). Manuscript:
`SPARCAL_pnas_2026/PNAS/`. Nothing is committed to git yet — all manuscript changes are staged in the
working tree for review.

---

## 1. Status against the seven review comments

| # | Comment | Status | Outcome |
|---|---|---|---|
| 1 | No Results section shows SpatialSNV's disadvantage | **Done** | New **main Fig. 5**, head-to-head on all four tumor sections |
| 2 | Bin sweep → SI; 250 kb may not generalize | **Done** | Sweep → **SI Fig. S1**; inverted-U claim withdrawn |
| 3 | How do the Fig. 4 findings benefit the community? | **Done — negative** | Concordance is *not* a confidence axis; stays a limitation |
| 4 | Old Figs. 5 and 7 belong in main | **Done** | Both retained in main; Fig. 7 reframed (see §3) |
| 5 | Region detection → SI | **Done** | Now **SI Fig. S2** |
| 6 | Fig. 8a colouring unclear | Author handling | — |
| 7 | SPARCALViewer should have its own repo | Accepted | Not yet split out |

---

## 2. New results

### 2.1 Germline leakage — the head-to-head the paper was missing (positive)

Matching alleles exactly against the 1000 Genomes panel, a reference used by neither method during
calling, genome-wide across autosomes:

| | P4 | P6 | DCIS1 | DCIS2 |
|---|---|---|---|---|
| SpatialSNV calls that are common panel variants | **2.17%** | **2.71%** | **4.67%** | **4.39%** |
| Of those, the share SPARCAL routed to germline in the same tissue | 64% | 82% | 75% | 72% |

The second row is the load-bearing one: both pipelines saw these positions and disposed of them
differently. **We are deliberately fair to them in the text** — SpatialSNV *does* discard every
record Mutect2 flags as `germline` (0 of ~759,000 retained calls carry that flag). What we report is
what survives Mutect2's *probabilistic* population-allele-frequency filter in tumor-only mode.
SPARCAL's own near-zero rate is a design consequence, not a contest won, and is not exactly zero
(DCIS2 retains 1 in 25,154).

Also model-free and reportable: **the two callers agree on only 2.7–5.0% of their union**
(Jaccard 0.029 / 0.050 / 0.027 / 0.039).

### 2.2 Concordance × COSMIC — descriptive comparison

Caller agreement and COSMIC membership are reported on the same allele-exact matching basis used
for every callset. Callset intersection is descriptive and is not treated as a standalone measure
of variant quality.

### 2.3 CNV/LOH comparison — recommend dropping

The originally proposed design had two faults, both found by building it:

1. **The intended control set was not heterozygous.** SPARCAL's 1KGP-defined class is 59–71%
   homozygous-alt and only 14–22% het. A set that homozygous cannot detect *loss of* heterozygosity.
2. **copyKAT "loss" segments are low-*expression* segments**, so region and depth are the same axis.
   The clincher: the depth drop also appears in *diploid* spots (6.9 vs 9.1, DCIS1), which carry no
   copy loss.

Rebuilt properly — heterozygosity defined empirically in diploid spots, read out in aneuploid spots,
depth required in both groups then stratified — the method comparison is null inside validated LOH
(14.1% vs 12.9%, p=0.71) **and violates its own negative control**: in copy-*balanced* regions, where
no difference should exist, SPARCAL reads 42.7% vs SpatialSNV 14.5% (p=8.7e-8). The metric tracks
callset VAF composition, not consistency with copy state. Fixing it needs allele-specific copy number
and a VAF-matched design — new work, not a re-run.

**One salvage worth keeping:** DCIS1's clone-loss segments are **genuine LOH**. At sites
heterozygous in normal spots, median |BAF−0.5| in the tumour clone is 0.250 in loss segments vs 0.077
in balanced (pooled p=6.5e-12), holding in *every* depth band; only 35.0% of normal-het sites stay
het in the clone vs 73.4% balanced. **DCIS2 shows nothing** (p=0.18) — copyKAT segments should not be
treated as LOH by default. This is non-circular (θ uses within-clone prevalence and never sees BAF).
*Not currently in the manuscript — a scope decision (§5).*

---

## 3. Figure arrangement

Resolved both open figure decisions in line with the review:

- **Fig. 7 reframed** to lead with the **MHC/HLA recurrence caution** — our one field-level somatic
  contribution — folding the Cancer Gene Census candidate list and the 14 WES-confirmed variants
  underneath it. The candidate list is almost all single observations and recurs in neither cSCC
  patient, so leading with it invited a driver-discovery reading the data cannot support. Nothing
  was deleted, only reordered.
- **No consolidation of the somatic figures.** All four are individually requested by the review
  (Fig. 4 setup, Fig. 5 = comment 1, Figs. 6–7 = comment 4). The earlier worry about overselling a
  thin somatic half is handled by reframing rather than cutting.

Current main sequence: 1 pipeline · 2 unbinned germline · 3 binned germline · 4 platform limit ·
5 SpatialSNV head-to-head · 6 COSMIC cascade · 7 MHC recurrence · 8 SPARCALViewer.
SI: S1 bin-width sweep · S2 region detection.

---

## 4. Two errors found in our own numbers, both fixed

1. **Unmatched sample sizes in the germline headline.** The text compared our mean ARI over **12**
   sections against the baselines' over **11**. The excluded section (151671) is our *best*
   (0.394 vs a 0.266 overall mean), so the mismatch inflated us by +0.012 in a comparison whose whole
   effect is +0.043. Corrected to the matched 11 sections: **0.254 vs 0.211 (GATK) vs 0.193
   (Strelka2)** — which reproduces the paired differences and p-values exactly, because the
   Wilcoxon tests were always paired. Relative improvement is **1.2×, not 1.3×**.
2. **Wrong figure panel cited.** The CGC candidate sentence cited Fig. 7c, but panel (c) is the 14
   WES-confirmed variants; there is no panel for the CGC candidates. Citation moved.

---

## 5. Open issue we are setting aside — for the advisor's view

**The two external baselines' embeddings are degenerate, and not only on one section.**

Effective rank of the 30-dimensional STAGATE embedding, all 12 DLPFC sections:

| modality | rank |
|---|---|
| `sparcal` | **26–30** |
| `defined1000G` | 7–21 |
| `gene_expr` | 11–15 |
| **`gatk`** | **1–3** |
| **`strelka2`** | **1–3** |

This is **not a bug in our pipeline, and not specific to section 151671** — 151671 is simply where
the rank reached exactly 1 and the clustering step errored out instead of silently returning a
partition of noise. The cause is a density difference in the input matrices (section 151507):

| | density | variants/spot | columns present in ≤1 spot |
|---|---|---|---|
| `sparcal` | 2.6% | 1,580 | 6% |
| `gatk` | 0.21% | 109 | **66%** |
| `strelka2` | 0.21% | 122 | **80%** |

Our matrix is ~12× denser with 10–20× more variants per spot, and two-thirds to four-fifths of the
baselines' columns are spot-private. A variant seen in a single spot carries no spot-to-spot
covariance, so a graph autoencoder has little to learn from those matrices.

**Why it matters.** Both legs of the germline story compare against these embeddings — the ARI
comparison, and the embedding-purity metric (Fig. 2c, +0.152 vs GATK, 12/12 sections, p=4.9e-4),
which the draft currently calls the *stronger* leg. "Our embedding is more layer-pure than GATK's" is
close to tautological when GATK's embedding is rank 1–2. The honest reading is that we may partly be
measuring **callset density rather than variant quality** — the same coverage/density confound
already deferred to the second paper, except here it sits inside this paper's main germline claim.

**We did not paper over it.** The obvious fix — forcing the clustering to converge by discarding
dimensions until it does — would have produced a number rather than a measurement: a 7-component
mixture fitted in one dimension. The code now refuses that case explicitly and surfaces the
degeneracy instead.

**Three ways forward, for discussion:**

| | Approach | Cost |
|---|---|---|
| (a) | State it as a limitation; frame the comparison as *representations at the density each caller delivers* | text only |
| (b) | **Density-match** — subsample our matrix to the baselines' variants/spot, or restrict all three to non-private variants — and report whether the advantage survives | ~1 day |
| (c) | Drop the embedding-purity leg; stand on ARI alone | text only |

(b) is the only one that answers the question rather than manages it. **Set aside pending the
advisor's view; no further Story A text will be written until it is decided.**

---

## 6. Two new datasets now in the project

Both were run through the SPARCAL pipeline to the matrix step and are documented in
[CLAUDE.md](CLAUDE.md).

- **OVAR_P5** — ovarian cancer, **one patient, one section** (`P5_sr13`, 2,108 spot BAMs), GRCh38.
  Standard **poly-A capture** Visium. Behaves normally; currently underused (see §6.2).
- **NCCE** — Japanese **gastric-cancer** cohort: **5 patients × 3 serial sections = 15 sections**,
  5,202 in-tissue spots. Patients 6/8/10/11 non-responders, **patient 14 responder**; A/B/C are
  serial sampling timepoints. **Probe-based (RNA-templated ligation) chemistry — see §6.1. We are
  withdrawing it as a results dataset.**

### 6.1 🚨 NCCE is probe-based, and probe chemistry cannot carry SNVs (found 2026-08-07)

NCCE's SpaceRanger `outs/` contain **`probe_set.csv` — Visium Human Transcriptome Probe Set v2.0**
(RNA-templated ligation, 54,581 probes), and its metrics report *"Reads Mapped to Probe Set: 0.98"*
rather than *"Reads Mapped to Genome"*. In RTL chemistry **the sequenced molecule is a synthetic
probe oligo carrying reference sequence**, not the patient's transcript. Verified three ways, with
**OVAR_P5 as a matched poly-A positive control on the same pipeline, reference and code path**:

| Test (chr1:1–50 Mb unless noted) | NCCE_14A (probe) | OVAR_P5 (poly-A) |
|---|---|---|
| Covered-interval width — median / p90 / max (dp≥5) | **50 / 50 / 99 bp** | 102 / 328 / 2,736 bp |
| Callable bases (dp≥5) | 84,511 | 1,398,486 |
| Median depth *inside* the callable footprint | **96** | 23 |
| Common 1000G SNPs per kb of footprint | 0.77 | 1.68 |
| **Common-SNP detection, genome-wide, dp≥10** | **12/1,588 = 0.8%** | **16,715/65,930 = 25.4%** |

The 50 bp median interval is exactly the ligated probe-pair footprint. The detection gap is **not a
coverage artifact — it widens with depth**, which is the opposite of what a coverage effect does:

| depth band | NCCE detection | OVAR detection |
|---|---|---|
| 10–20 | 1.4% | 20.5% |
| 20–50 | 1.6% | 26.5% |
| 50–200 | 0.2% | 36.3% |
| **≥200** | **0.0% (0 / 254)** | 37.9% |

At depth ≥200 — where detection should be easiest — NCCE finds **zero** of 254 common SNPs sitting
inside its own callable footprint. Two compounding mechanisms explain it: 10x designs probes to
**avoid** common polymorphisms (hence 0.77 vs 1.68 panel sites per kb), and a mismatch under the
probe **blocks ligation**, so the variant molecule drops out instead of being sequenced. The assay
does not merely miss variants — it *selects against* them.

Third, independent check on the **final callset**: the `1000G`/`defined` class is ~100%
panel-matching *by construction*. Across all 15 NCCE sections it is **29.8%** (range 20.7–45.7%)
versus **88.3%** for OVAR_P5. Not a coordinate-offset bug — only 3–4% of non-members lie within
±5 bp of a panel site.

**Consequence.** The 133 NCCE somatic-pattern SNVs are not demonstrably genotype, so the
patient-14 responder narrative (19 → 10 → 8 loci; 29.4% → 18.0% → 10.0% of spots) must be
withdrawn as a mutation result. It has a mundane competing explanation: probe coverage tracks
target-gene expression, which is itself spatial, so "hotspots" can be expression domains. Section
14A's sequencing saturation is 0.98, so the rising-UMI/falling-burden trend is also consistent with
better consensus suppressing error-driven calls. `data/ncce/biological_analysis/report_v1/` should
stop recommending a manuscript subsection; its existing cautions (copyKAT circularity, UMI range
178–55,844) are real but do not reach this, because they assume the underlying calls carry
mutation signal.

**Nothing about SPARCAL failed** — it ran all 15 sections without error, which is precisely the
hazard. The constraint applies identically to SpatialSNV, Monopogen, SpaceTracer and any other
RNA-based caller. It is a **field-level assay constraint, not a SPARCAL weakness**.

**✅ All previously published datasets were checked and are clean.** No `probe_set.csv` and no
probe metrics keys anywhere in P4/P6/DCIS1/DCIS2; the assay-agnostic interval test confirms it:

| dataset | median / p90 / max interval | chemistry |
|---|---|---|
| DLPFC 151507 | 83 / 356 / 2,146 | poly-A |
| P4 rep1 | 98 / 312 / 1,885 | poly-A |
| P6 rep1 | 111 / 305 / 1,312 | poly-A |
| DCIS1 | 105 / 384 / 2,895 | poly-A |
| DCIS2 | 120 / 367 / 2,702 | poly-A |
| OVAR_P5 | 102 / 328 / 2,736 | poly-A |
| **NCCE (all 15)** | **50 / 50 / 99** | **probe (RTL)** |

**No existing paper claim is affected.** Going forward the onboarding rule is *check for
`probe_set.csv`* — **not** "avoid FFPE". The discriminator is the probe set, not the fixation:
CytAssist runs on fresh-frozen and is still probe-based.

**Recommendation — drop NCCE as a results dataset, keep one paragraph of scope.** Dropping the
biological story is not in question. But the scope statement is worth a paragraph plus an optional
SI panel, for three reasons: (i) it is the only result in the current plan with **no coverage
confound** — NCCE has *higher* in-footprint depth and still loses, so it is immune to the density
critique in §5; (ii) it pre-empts the obvious "does this work on FFPE?" review question with data
rather than silence; (iii) framed as a field constraint it costs SPARCAL nothing. The honest
counter-cost is that it invites "so your method only works on poly-A" — true, but true of the whole
field, and better answered on our terms. **Decision D-NCCE below.**

### 6.2 OVAR_P5 is the more useful of the two, and is currently underused

Its only present role is a SPARCALViewer demo profile. It is a clean poly-A tumor section —
2,108 spots, 188,742 unique variants, 61,850 1000G-defined at **88.3% panel purity**, full
four-class matrices plus exome-restricted variants. Two candidate uses, neither yet scoped:

- **Story A density rebuttal (§5).** An independent poly-A tumor section that is neither DLPFC nor
  from SpatialSNV's own published data — it widens the base beyond the 12 DLPFC sections if we take
  option (b).
- **Story C4 third sample.** Region-detection ARI currently rests on DCIS2 and P6, where the
  ranking flips and nothing separates from coverage; a third sample is worth more than another
  metric on the same two. **Blocker:** no tumor annotation was supplied (`MANIFEST.json`). CalicoST
  clone calls exist for this patient — worth asking Pankaj.

---

## 7. Decisions needed

1. **Baseline embedding degeneracy (§5)** — option (a), (b) or (c)?
2. **DCIS1 LOH validation (§2.3)** — does it earn a place in the paper on its own, given it holds in
   one of two samples?
3. **HLA exclusion presentation** — recommendation is to report both raw and xMHC-excluded, framing
   the cascade as tumor-type-specific (robust in high-TMB cSCC, HLA-driven/absent in low-TMB DCIS).
4. **SPARCALViewer repository split** — confirm, then it can be done.
5. **D-NCCE (§6.1)** — NCCE is withdrawn as a results dataset either way. Does the probe-chemistry
   scope statement go in the paper (one paragraph, optional SI panel), or do we drop NCCE silently?
6. **OVAR_P5 scope (§6.2)** — leave as a viewer demo, or pull it into Story A and/or C4? The C4 use
   needs a tumor annotation we do not currently have.

Also outstanding on our side, not blocked on anyone: the Methods section still describes the
pre-2026-07-28 direction and needs rewriting; Fig. 1 is a placeholder pending the author-drafted
pipeline diagram; the title is undecided.
