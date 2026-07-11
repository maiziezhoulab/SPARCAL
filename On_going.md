# On-going Tasks

Dynamically updated task list. When asked to "check ongoing tasks", Claude should:
1. Run `squeue -j <job_ids>` to check SLURM status
2. Check output files/logs for results
3. Update this file accordingly

---

## 🔶 NEW (2026-07-10 PM) — Follow-ups from the 48-experiment recap: somtop+bin sweep, floor20 correction, GATK job tracking, combined-figure update, weekly report

**1. Somatic-threshold sweep, NOW BINNED (user's request) — built + wired, ready to submit.**
The raw somtop02/05/10/25 sweep (defined ∪ top-F% most-spatially-clustered somatic) never beat
0.277 raw. Never tested binned. Built via `build_binned_matrices.py` (local, no sbatch) on the
already-existing `DLPFC_151507_SPARCAL_defined_somtop{F}_matrix.pkl` sources:

| modality | bins | mean SNV/bin | occupancy |
|---|---|---|---|
| somtop25_bin250kb | 8,897 | 6.8 | 3.2% |
| somtop10_bin250kb | 8,871 | 6.6 | 3.1% |
| somtop05_bin250kb | 8,855 | 6.5 | 3.1% |
| somtop02_bin250kb | 8,852 | 6.5 | 3.1% |

Wired into `clustering_config_recovery_test.json` (now 50 modalities). **USER SUBMITS:**
```bash
sbatch clustering_benchmark/run_clustering_recovery_test.slurm
```
Then read `data/dlpfc_recovery_test/151507/clustering/summary.csv`, compare each `somtopF_bin250kb`
mean vs. `defsom_bin250kb` **0.364** (the untightened full-somatic-set binned result) and
`defined_bin250kb` **0.351** (recovery-tree, same-seed baseline). Tests whether tightening the
somatic filter before binning pushes past 0.364, or whether the untightened set was already right.

**2. Correction — floor20 is NOT an improvement over `defined_bin250kb`, just a smaller matrix at
equal accuracy.** User initially read `defined_bin250kb` (main tree, 10-run, **0.332**) vs.
`floor20_bin250kb` (recovery tree, 5-run, **0.338**) as a small win. Those are from two *different*
seed counts/trees. Same-tree, same-5-seeds comparison (both recovery tree): `defined_bin250kb`
**0.351 ± 0.052** vs. `floor20_bin250kb` **0.338 ± 0.016** — floor20 is actually **0.013 lower**,
i.e. statistically indistinguishable-to-slightly-worse, not better. **The real, reportable finding**:
floor20 reaches statistically the same accuracy with **3,847 bins vs. 8,845 (2.3× smaller matrix)** —
an efficiency result, not an accuracy gain. Already correctly stated this way in the "✅ DONE" entry
below; this note exists because the discrepancy came up in conversation and is worth flagging clearly.

**3. GATK bin250kb 12-section resubmit — job `12480568` submitted 2026-07-10, tracking.** Correct
`--export` syntax this time (env var set before `sbatch`, not packed into `--export`, so no comma
reaches sbatch's own parser):
```bash
MODALITIES=gatk_bin250kb N_RUNS=10 sbatch --array=0-11 --export=ALL clustering_benchmark/run_clustering.slurm
```
At submit: queued (`PENDING`, reason `QOSGrpGRES` — a6000 2-concurrent cap). Will populate
`gatk_bin250kb` in the main `data/dlpfc/{s}/clustering/` tree for all 12 sections (currently only
151507 has a 5-seed measurement, from the recovery tree). `ari_matrix_mean.csv` / `ari_boxplot*`
auto-regenerate on completion (flock-guarded tail step in `run_clustering.slurm`).

**4. `make_combined_figure.py` updated (user's request) — now plots ONLY the binned callers + GE.**
Was `['sparcal', 'strelka2', 'gatk', 'gene_expr']` (raw matrices). Changed to
`['defined_bin250kb', 'strelka2_bin250kb', 'gatk_bin250kb', 'gene_expr']`, `OURS = 'defined_bin250kb'`.
`py_compile` clean. **Regenerate all 12 sections' combined figures after job `12480568` finishes**
(needs `gatk_bin250kb` populated for every section first, else those columns are silently skipped):
```bash
python clustering_benchmark/make_combined_figure.py --all
```
(writes `data/dlpfc/{s}/clustering/combined_{s}.png/.pdf` for all 12, incl. the user-flagged
`combined_151671.pdf`). Single-section form still works: `--section_id 151671`.

**5. Weekly-report draft written + saved locally with figures.** All 48 section-151507 experiments,
glossary of every modality token, headline findings table (bin=necessary, exome=harmful, UPV=no
help, defsom=promising-but-unconfirmed, floor20=efficiency-not-accuracy), full per-category results,
4 embedded figures (full-experiment boxplot, bin-size sweep, 12-section cross-modality, spatial
domain comparison). Saved as a self-contained folder (report + copies of its 4 source PNGs, ~3.8MB):
`data/dlpfc_recovery_test/151507/clustering_benchmark/weekly_report/DLPFC_SNV_representation_report.md`.
Also published (text-only, no images — artifacts need self-contained/base64 assets, not done yet)
as an Artifact: https://claude.ai/code/artifact/4bcc2d3b-e17c-4639-be5d-eab85ad2fabc — ask if a
graphics-embedded refresh of the artifact is wanted.

---

## ✅ DONE (2026-07-10) — Does adding SPARCAL's OWN variants (UPV/somatic), THEN binning, beat `defined_bin250kb`? — job `12475342` COMPLETED, ANSWERED

**User's question:** every prior test of adding SPARCAL's novel calls (UPV, somatic) to the defined
set was done on the RAW (unbinned) matrix and all hurt. Nobody had tested adding them **and then
binning** — i.e. does the aggregation mechanism that rescued `defined_bin250kb` (0.351) also rescue
UPV/somatic once they're aggregated into the same 250kb bins, instead of sitting as raw dilutive
columns? This was a real gap, not yet run.

**Status of the previously-proposed "4 attempts" (Q4 representation study, A/B/C/D) — all 4 fully
resolved, nothing outstanding:**

| Exp | Modality | Result | Verdict |
|---|---|---|---|
| A | `defined_tfidf` | 0.186 | hurts |
| B | `defined_svg05k/10k/20k` | 0.11–0.148 | hurts |
| C | `defined_bin*` | **0.351 @ 250kb** | **winner** (already promoted, see above) |
| D | `defined_vaf` / `defined_rawbin` | 0.13–0.16 | hurts (rank-collapsed embedding) |

Separately, every raw (unbinned) variant-addition test also already completed and confirmed hurts:
`defined_somatic` 0.245, `defined_upvrule` 0.263, `defined_floor{02,03,05,10,20,50}` 0.223–0.286 (all
≤ baseline except floor05's 0.286, barely above), `defined_somtop{02,05,10,25}` 0.181–0.198,
`defined_cap*` 0.11–0.16 (crashes — Q2 showed dropping dense variants is actively harmful, density
IS the signal). **So: no raw-matrix variant-selection trick beats `defined1000G` (0.277) / and only
binning beats it (0.351). But "adding UPV/somatic, then binning" was never actually tested — until now.**

**Correction (user, 2026-07-10): "somatic" must mean 1kG + somatic, not somatic alone** — by analogy
with germline = 1kG + UPV. Checked: the pipeline's own `somatic` per-barcode class is ALREADY only
the top-10%-by-vote focal denovo (DLPFC has no purity/clone/CNV features to refine it further — this
is a property of step-7's classification itself, not an extra subsetting choice), so union(1kG,
somatic) turns out to be **byte-identical** to the already-existing, already-tested `defined_somatic`
matrix (verified: same shape 4226×70,264, same values after alignment). **So the raw "1kG+somatic"
number was already known: 0.245 (hurts)** — no new raw run needed there. What's genuinely new is
binning it, and two more items the user requested (floor20+bin, exome-filtering).

**Built (local/direct, no sbatch, ~5 min total):**
1. Generated the true canonical `somatic` (13,419 SNVs) and `merged` (=1kG+UPV+somatic, everything
   SPARCAL calls, 74,223 SNVs) class matrices for 151507 via `generate_sparcal_matrices.py --dataset
   DLPFC --section_id 151507 --classes somatic merged`. `germline` (=`normal`, 1kG+UPV, 60,804 SNVs)
   already existed. New script `build_defined_plus_somatic_matrix.py` unions 1000G ∪ somatic →
   `defsom` matrix (70,264 cols, = defined_somatic, see correction above).
2. Binned four sets at 250kb via the generalized `build_binned_matrices.py`: `germline_bin250kb`
   (8,868 bins, 4.1% occ), `merged_bin250kb` (9,192 bins, 4.1% occ), `defsom_bin250kb` (9,178 bins,
   3.2% occ, from the defsom/1kG+somatic matrix), and **`floor20_bin250kb`** (3,847 bins, 6.4% occ —
   bins the ALREADY-BUILT `defined_floor20` matrix, 19,998 cols; queued since the bin-size-sweep DONE
   entry above, now actually built). User's observation that motivated this: `defined_floor20`
   (drops the 36,847 lowest-prevalence/rarest defined variants, keeps only 19,998) scores 0.277 ≈
   baseline unbinned — losing 65% of columns costs ~nothing raw. Question: does removing that dead
   weight BEFORE binning concentrate the aggregate signal into fewer, denser, more informative bins
   (beat 0.351), or does it just starve some bins entirely (hurt)?
3. **New: exome-capture filtering** (user's idea, not yet tried in this study). New script
   `build_exome_filtered_matrices.py` intersects SNV columns against
   `/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/regions/Twist_Exome_Core_Covered_Targets_hg38.bed`
   (same Twist exome kit already used elsewhere in this repo for DLPFC exome QC, chr-prefix stripped
   to match our bare-chromosome column keys, overlapping BED intervals merged for an exact
   bisect-based point-in-region test). Built for three variant sets: `1000G_exome` (56,845→2,428
   cols, 4.3% in exome), `normal_exome` (=germline_exome, 60,804→2,534, 4.2%), `defsom_exome`
   (70,264→2,877, 4.1%). Rationale: exome-captured regions are the best-characterized, most
   artifact-resistant part of the genome — orthogonal, annotation-based (not data-driven) filter,
   worth testing whether narrowing to just these SNVs helps vs. the data-driven cap/floor filters
   (which already showed density, not annotation quality, is what matters — this tests whether that
   still holds when restricted to exons).
4. Wired 10 new modalities into `clustering_config_recovery_test.json` (now 46 total, all pkl paths
   verified to exist): `germline_raw`, `germline_bin250kb`, `merged_raw`, `merged_bin250kb`,
   `defsom_bin250kb`, `floor20_bin250kb`, `defined_exome`, `germline_exome`, `defsom_exome`,
   `merged_exome`. (Dropped the pure-somatic-only `somatic_raw`/`somatic_bin250kb`/`somatic_exome`
   modalities — superseded by the corrected `defsom` variant set per the correction above; the raw,
   unfiltered `DLPFC_151507_SPARCAL_somatic_exome_matrix.pkl` file still exists under
   `data/dlpfc/151507/matrix/` from the P4/P6/DCIS/OVAR_P5 exome-filtering pass, just not wired here.)
   Existing 36 modalities stay cached — only these 10 compute (46 new runs, 5 seeds each).

**SUBMITTED 2026-07-10 as job `12475342` (`sbatch clustering_benchmark/run_clustering_recovery_test.slurm`)
— COMPLETED in 36m13s, exit 0.** All 10 new modalities × 5 seeds = 50 runs present in
`data/dlpfc_recovery_test/151507/clustering/summary.csv`, all `status=ok`, correct 4221-spot shape.

**RESULT — variant set (SPARCAL, additive), raw vs. bin250kb ARI** (5-seed means; std in parens):

| variant set (SPARCAL, additive) | raw ARI | bin250kb ARI |
|---|---|---|
| 1000G (defined only) | 0.277 | **0.351** (prior winner) |
| germline (1kG + UPV) | 0.177 (±0.015) | 0.317 (±0.056) |
| defsom (1kG + somatic) | 0.245 (±known) | **0.364** (±0.062) |
| merged (1kG + UPV + somatic, everything) | 0.160 (±0.007) | 0.287 (±0.081) |
| floor20 (1kG, rarest 65% dropped) | 0.277 (±known) | 0.338 (±0.016) |

**RESULT — exome-restricted (annotation filter, raw only):**

| exome-restricted | cols kept | ARI (±std) |
|---|---|---|
| `defined_exome` (1kG ∩ exome) | 2,428 | 0.134 (±0.027) |
| `germline_exome` (1kG+UPV ∩ exome) | 2,534 | 0.142 (±0.009) |
| `defsom_exome` (1kG+somatic ∩ exome) | 2,877 | 0.094 (±0.027) |
| `merged_exome` (1kG+UPV+somatic ∩ exome) | 2,961 | 0.148 (±0.013) |

**Answers to the questions posed:**
1. **`germline_bin250kb` (0.317) does NOT beat `defined_bin250kb` (0.351)** — binning does NOT rescue
   UPV enough to help beyond pure 1kG. It's still a huge recovery over raw germline (0.177→0.317,
   +0.140), so binning fixes most of UPV's raw-matrix dilution damage, but not all of it. **This means
   the "SPARCAL is doing nothing" concern is NOT resolved** — the pipeline's actual UPV output (as
   currently used in the germline/normal matrix) doesn't add value over defined-only, whether raw or
   binned.
2. **Surprise: `defsom_bin250kb` (0.364) slightly BEATS `defined_bin250kb` (0.351)** — this contradicts
   the going-in hypothesis (somatic is spatially focal/clone-consistent, predicted to underperform
   after binning). However it's noisy — 5-seed range 0.247–0.430, std 0.062, roughly double
   `defined_bin250kb`'s own std (0.059 from the bin-size sweep) — so this is a *weak, uncertain* win,
   not a robust one. Worth a higher-seed-count rerun before treating it as a real result.
3. **`merged_bin250kb` (0.287) is worse than both `defined_bin250kb` and `defsom_bin250kb`** — adding
   UPV on top of the somatic-augmented set makes things worse, consistent with UPV being dead weight
   (see point 1). Confirms: of the three additive combinations, defsom is the best, germline is
   second, merged (everything) is the worst — UPV, not somatic, is the drag.
4. **`floor20_bin250kb` (0.338) does NOT beat `defined_bin250kb` (0.351)** — slightly below, small
   gap relative to its own std (±0.016), so essentially neutral/marginally negative. Dropping the
   rarest 65% of defined variants before binning neither helps nor meaningfully hurts; the few-dense-
   bins hypothesis didn't pan out as an improvement.
5. **Exome-restriction hurts across the board** (0.094–0.148, all well below the unrestricted raw
   1000G baseline 0.277), confirming the prior finding that aggressive column-count cuts on this data
   hurt regardless of whether the cut is prevalence-based (`cap02`/`floor50`, which crashed similarly)
   or annotation-based (exome). Restricting to exons is not a useful filter for this clustering task.

**Bottom line:** `defined_bin250kb` (0.277 raw → 0.351 defined-only-binned) remains the strongest
robust SNV-only representation. The one modality that edges it out (`defsom_bin250kb` 0.364) does so
narrowly and noisily — not yet a confident claim. No combination that includes UPV beats binned-1kG.

---

## 🔶 NEW (2026-07-10) — Strelka2/GATK 250kb binning PROMOTED to all 12 DLPFC sections — HALF DONE (strelka2 only), gatk still needs resubmit

Follow-up to the recovery-test finding (job `12470601`) that binning is a generic sparse-matrix win
(Strelka2/GATK gained as much as SPARCAL from 250kb binning on 151507 alone). This promotes that to
all 12 sections for a true 12-section apples-to-apples comparison against the already-promoted
`defined_bin250kb` (12-section mean 0.363).

**Built (local/direct, no sbatch, ~3 min total):**
1. Built `DLPFC_{s}_strelka2_bin250kb_matrix.pkl` and `DLPFC_{s}_gatk_bin250kb_matrix.pkl` for the 11
   sections that didn't have them yet (151507 already did, from the recovery test) via
   `build_binned_matrices.py --caller {strelka2,gatk} --filter germline --grouping 6 --out_prefix ""
   --bin_sizes 250000`. All 12 sections × both callers verified present.
2. Added `strelka2_bin250kb` and `gatk_bin250kb` as modalities in the **main**
   `clustering_config.json` (now 7 modalities total): `{caller: strelka2/gatk, filter: bin250kb,
   grouping: ""}` — matches the `DLPFC_{s}_{caller}_bin250kb_matrix.pkl` naming exactly. Verified
   every (section × modality) pkl path resolves and exists.

**SUBMITTED 2026-07-10 as job `12475080` (`--array=0-11 --export=ALL,MODALITIES=strelka2_bin250kb,gatk_bin250kb,N_RUNS=10 clustering_benchmark/run_clustering.slurm`) — COMPLETED, but only HALF ran: `gatk_bin250kb` never computed.**

**Bug found:** `sbatch --export=...` splits its whole argument on commas to separate `KEY=VALUE`
pairs — a comma *inside* a value breaks it. `MODALITIES=strelka2_bin250kb,gatk_bin250kb` was parsed
as `MODALITIES=strelka2_bin250kb` plus a stray bare token `gatk_bin250kb` (silently dropped), so
`gatk_bin250kb` was **never passed to `--modalities`**. Confirmed by grepping the `Modalities:` banner
line in every section's log (`slurm_output/clustering/section_{0..11}.out`) — all 12 say
`Modalities: strelka2_bin250kb` only, no `gatk_bin250kb` anywhere. `N_RUNS=10` DID come through fine
(single-valued, no comma), so the 10 runs/section that did happen are trustworthy.

**RESULT — `strelka2_bin250kb` 12-section mean ARI** (from
`data/dlpfc/clustering_benchmark/ari_matrix_mean.csv`, new column added alongside the existing 5):

| Modality | 12-section mean ARI |
|---|---|
| gene_expr (baseline) | 0.412 |
| **defined_bin250kb** | **0.363** |
| strelka2_bin250kb | **0.257** |
| sparcal | 0.217 |
| gatk | 0.211 |
| strelka2 (raw) | 0.193 |

Strelka2 gains +0.064 absolute from 250kb binning at the full 12-section scale (0.193→0.257) — close
to the +0.089 seen on 151507 alone in the earlier single-section test, confirming binning is a generic
sparse-matrix win, not SPARCAL-specific, at scale. `defined_bin250kb` (0.363) still leads decisively
over binned strelka2 by +0.106.

**`gatk_bin250kb` still needs to be run — resubmit with the export bug avoided** (set the var in the
shell environment instead of packing it into `--export`, so the comma never reaches sbatch's own
parser):
```bash
MODALITIES=gatk_bin250kb N_RUNS=10 sbatch --array=0-11 --export=ALL clustering_benchmark/run_clustering.slurm
```

---

## ✅ DONE — `defined_bin250kb` PROMOTED to all 12 DLPFC sections — job `12470671` COMPLETED

**Done 2026-07-10 (this session), all local/direct commands (no sbatch — reused already-completed
step-7 output, ~30 min wall-clock):**
1. Built the `1000G` (defined-only) matrix for the 11 sections that previously only had `normal`
   (=germline, 1000G+UPV) — `generate_sparcal_matrices.py --dataset DLPFC --section_id {s}
   --classes 1000G`. All 12 sections now have `data/dlpfc/{s}/matrix/
   DLPFC_{s}_SPARCAL_1000G_matrix.pkl` (151507: 56,845 cols; range across sections 50,110–89,906).
2. Ran `clustering_benchmark/build_binned_matrices.py --section_id {s}` (default bin sizes
   1mb+250kb) for all 12 sections → every section now also has
   `DLPFC_{s}_SPARCAL_defined_bin250kb_matrix.pkl` (+ `defined_bin1mb`, unused downstream but
   free to keep).
3. Added `defined_bin250kb` as a 5th modality in the **main** `clustering_benchmark/
   clustering_config.json` (the all-12-section config, not the recovery-test one):
   `{"caller": "SPARCAL", "filter": "defined_bin250kb", "grouping": ""}` — matches the
   `DLPFC_{s}_SPARCAL_defined_bin250kb_matrix.pkl` naming exactly, no `pkl_path` override needed.

**SUBMITTED 2026-07-10 — job `12470146` (`--array=0-11 --export=ALL,MODALITIES=defined_bin250kb,N_RUNS=10
clustering_benchmark/run_clustering.slurm`).** At submit: task 0 (151507) RUNNING on gpu0208, tasks
1–11 PENDING on `QOSGrpGRES` (a6000 2-concurrent cap; will serialize through as slots free). Existing
4 modalities (sparcal/strelka2/gatk/gene_expr) already have 10 cached runs/section in
`data/dlpfc/{s}/clustering/summary.csv` and are skipped — only `defined_bin250kb` computes.

**Job `12470146` was accidentally cancelled by user 2026-07-10 mid-run. Checked `sacct` +
per-section output dirs — actual state (task idx → section, per `SECTIONS=` array in
`run_clustering.slurm`):**
- 151507, 151508, 151509, 151510 (idx 0–3): **fully done**, all 10 runs cached.
- 151669 (idx 4): **fully done** — all 10 runs + `summary.csv` rows present; job was killed
  right at the tail end, after compute finished.
- 151670 (idx 5): **partial** — run0/run1 cached; run2 killed mid-computation (empty dir, no
  `ari.txt`); runs 3–9 never started.
- 151671–151676 (idx 6–11): **never started** (still PENDING when cancelled), no output.

**Resubmit is safe and will NOT recompute finished work** — `run_modality()` in
`SPARCAL_clustering.py:352-353` checks each run's `ari.txt` before recomputing, so 151507–151510
+151669 skip instantly, 151670 only redoes run2 (+3–9), and 151671–151676 run fresh.

**RESUBMITTED 2026-07-10 as job `12470671`** (same command as above), submitted concurrently with
the recovery-test job `12470601` (see next section) — no conflict, disjoint output trees
(`data/dlpfc/*/clustering/defined_bin250kb/` vs. `data/dlpfc_recovery_test/151507/clustering/`).
Only 1 of the 12 array tasks will actually run at a time alongside `12470601` (account
`maiziezhou_lab_acc` a6000 cap = 2 concurrent GPUs); the rest queue on `QOSGrpGRES` and serialize
automatically as slots free up.

**RESULT (job `12470671`, completed 2026-07-10) — 12-section mean ARI** (from
`data/dlpfc/clustering_benchmark/ari_matrix_mean.csv`, all 12 sections × 10 runs):

| Modality | 12-section mean ARI |
|---|---|
| gene_expr (baseline) | 0.412 |
| **defined_bin250kb** | **0.363** |
| sparcal (current pipeline, incl. UPV) | 0.217 |
| gatk | 0.211 |
| strelka2 | 0.193 |

`defined_bin250kb` nearly closes the gap to the gene-expression baseline and is a **~67% relative
gain** over the current `sparcal` pipeline matrix (0.217→0.363), holding consistently per-section
(range 0.24–0.58, above 0.217 in every section). This is the strongest result of the representation
study so far — candidate for the paper's main SNV-clustering figure.

**Known data gap (pre-existing, not from this job):** section 151671's `strelka2` modality has all
10 runs failing with an R `mclust` SVD error (`infinite or missing values in 'x'`) —
silently NaN'd out of the aggregate table. Not blocking, but fix before reporting a clean
strelka2 column.

**Open question raised by user (2026-07-10) — is binning a SPARCAL-specific win or a generic
sparse-binary-matrix win? — GENERALIZATION DONE, ready to test on 151507.** If aggregating into 250kb
bins helps because it recovers regional signal from per-site dropout, that logic isn't specific to
SPARCAL's variant set — Strelka2 (58,979 SNVs) and GATK (51,553 SNVs) are also binary spot×SNV
matrices from the same low-per-spot-coverage Visium data. Without binning them too, a
`defined_bin250kb` win over baseline `strelka2`/`gatk` could be confounded by "binning helps any
sparse caller" rather than "SPARCAL's variant set aggregates better."

**Done this session (2026-07-10, all local/direct — no sbatch):**
1. Generalized `clustering_benchmark/build_binned_matrices.py` — was hardcoded to read
   `SPARCAL_1000G` and write `defined_bin{TAG}`. Now takes `--caller/--filter/--grouping` (or explicit
   `--input`) to bin ANY `DLPFC_{s}_{caller}_{filter}[_{grouping}]_matrix.pkl`, with `--out_caller`
   and `--out_prefix` controlling the output name. **Default invocation is byte-identical to the old
   behavior** (verified: regenerated `SPARCAL_defined_bin250kb` equals the pre-existing pkl exactly).
2. Built the 151507 binned matrices for the other two callers (1mb + 250kb each):
   ```bash
   python clustering_benchmark/build_binned_matrices.py --section_id 151507 --caller strelka2 --filter germline --grouping 6 --out_prefix ""
   python clustering_benchmark/build_binned_matrices.py --section_id 151507 --caller gatk     --filter germline --grouping 6 --out_prefix ""
   ```
   → `DLPFC_151507_{strelka2,gatk}_bin250kb_matrix.pkl` (strelka2: 8,478 bins, occ 1.0%; gatk: 8,412
   bins, occ 1.1%). Note: both are markedly sparser post-bin than SPARCAL's defined_bin250kb (8,845
   bins, occ **3.0%**) — first hint that SPARCAL's set aggregates denser regional signal. All three
   share the same 4,226 in-tissue barcodes.
3. Wired 4 modalities into `clustering_config_recovery_test.json` (isolated `data/dlpfc_recovery_test`
   tree, default `lognorm`, 5 seeds): `strelka2_raw`, `strelka2_bin250kb`, `gatk_raw`,
   `gatk_bin250kb`. Raw + binned in the SAME tree/seeds → clean raw-vs-bin per caller, alongside the
   already-present `defined1000G` 0.277 / `defined_bin250kb` 0.351. (Existing modalities are cached
   and skipped — only these 4 compute.)

**SUBMITTED 2026-07-10 as job `12470601`, COMPLETED.** Config had 36 modalities total; 32 already
cached, the 4 new ones (`strelka2_raw`, `strelka2_bin250kb`, `gatk_raw`, `gatk_bin250kb`) computed
5 runs each.

**RESULT — ANSWERED: binning is a generic sparse-binary-matrix win, not SPARCAL-specific.**
From `data/dlpfc_recovery_test/151507/clustering/summary.csv` (5-run means):

| Caller | raw ARI | bin250kb ARI | Δ (absolute) |
|---|---|---|---|
| SPARCAL (defined1000G) | 0.277 | 0.351 | +0.074 |
| GATK | 0.190 | 0.266 | +0.076 |
| Strelka2 | 0.136 | 0.225 | +0.089 |

GATK and Strelka2 gain **as much or slightly more** in absolute ARI than SPARCAL from 250kb binning
— so the aggregation benefit is generic to sparse spot×SNV matrices, not a SPARCAL-specific effect.
However, the fair post-bin comparison still favors SPARCAL: its lead over GATK/Strelka2 is
essentially unchanged after binning (gap to GATK 0.087→0.085; gap to Strelka2 0.142→0.126) — i.e.
SPARCAL's variant set isn't disproportionately helped by binning, it was already ahead and stays
ahead.

**Possible next step (not yet done): promote strelka2/gatk 250kb binning to all 12 sections** —
run the generalized `build_binned_matrices.py` per section for both callers and add
`{caller: strelka2, filter: bin250kb, grouping: ""}` (+ gatk) to the **main**
`clustering_config.json`, to get a 12-section apples-to-apples binned comparison matching the
`defined_bin250kb` 12-section run above. Would need ~22 more matrix-build calls (local, no sbatch)
+ a new clustering array job.

**How the 250kb bin length was chosen — swept 7 sizes (1Mb→25kb) on section 151507, 5 runs each.**
Results are stored at `data/dlpfc_recovery_test/151507/clustering/summary.csv` (modalities
`defined_bin{1mb,500kb,250kb,125kb,100kb,50kb,25kb}`), plotted in
`data/dlpfc_recovery_test/151507/clustering_benchmark/ari_binsize_sweep.{png,pdf}` (+ table
`ari_binsize_sweep_table.csv`), generated by `clustering_benchmark/make_binsize_sweep_plot.py`:

| Bin size | n bins | mean ARI (5 runs) |
|---|---|---|
| 1Mb | 2,695 | 0.277 |
| 500kb | 5,061 | 0.246 |
| **250kb** | **8,845** | **0.351** |
| 125kb | 14,076 | 0.310 |
| 100kb | 16,027 | 0.303 |
| 50kb | 22,252 | 0.263 |
| 25kb | 27,993 | 0.244 |

250kb is a clean, symmetric peak — an inverted-U over bin size, not a monotonic trend: coarser bins
(1Mb/500kb) merge spatially-uncorrelated regions and wash out signal back toward the raw unbinned
`defined1000G` ARI (0.277); finer bins (100kb→25kb) re-sparsify and decay back down toward the
`defined_rawbin`/no-aggregation floor (0.129). 250kb sits at the top of the curve (+0.074 over raw),
which is the empirical justification for picking it as the default bin length (not a preset choice —
it was swept and selected post hoc). The same plot also shows the strelka2/gatk 250kb-binned
reference bars (blue/green) for scale, confirming 250kb still beats both other callers even at their
own optimum bin size (untested whether 250kb is *their* optimum too — only 250kb was swept for
strelka2/gatk so far).

---

## ⏳ WATCH NOW — `defined_vaf`/`defined_rawbin` mclust NaN — ROOT-CAUSED + FIXED, 9 runs left to compute

**Not yet submitted (user submits — see command below).** Step 2 of the 2026-07-10 plan (bin sweep
done → this → promote winner to 12 sections).

**Root cause found (CPU diagnostics, no sbatch needed):** STAGATE training itself is clean — zero
NaN through 60 epochs, loss converges smoothly. The failure is downstream: on these two matrices
(`preprocess="none"`, ultra-sparse raw alt-evidence, mean nonzero rate 0.28%) the 30-d STAGATE
embedding doesn't have enough independent signal to fill 30 dimensions, so most collapse toward
near-zero variance — **rank 6/30 at epoch 60, and confirmed by a full production-path rerun to keep
collapsing further to rank 2/30 by epoch 1000** (covariance condition number ~1.7e7 already at epoch
60). Fitting a full 30-d Gaussian-mixture covariance (`mclust`'s `"EEE"` model + its fallback model
search, which includes unconstrained per-cluster `VVV`) to that is both statistically meaningless
along the dead dimensions and numerically fatal — R's internal SVD eventually hits an exactly
singular matrix: `Error in svd(shape.o, nu=0): infinite or missing values in 'x'`.

**Fix applied:** `clustering_benchmark/SPARCAL_clustering.py:mclust_R` now projects the embedding
onto its informative PCA subspace (eigenvalue > 1e-6 × the largest) before calling `Mclust`. Verified
three ways: (1) synthetic pathological embedding → fix succeeds where a hand-rolled repro of the old
path was inconclusive (didn't itself crash — real per-cluster singularity needs real spatially
imbalanced data, not a synthetic unimodal blob); (2) **zero regression** — replayed the fix against
the cached `defined_bin250kb` run0 embedding, got byte-identical cluster labels and ARI (0.384924);
(3) **real end-to-end confirmation** — reran `defined_vaf` run0 through the actual production code
path (CPU, full 1000 epochs): printed `embedding rank 2/30 (dropping 28 near-zero-variance dims)`
and completed with **ARI = 0.1588** where it previously crashed on all 5/5 seeds. Low ARI is expected
(severely rank-collapsed representation), not a concern — the point was completion, not quality.

**Remaining work:** only `defined_vaf` run0 is cached (`ari.txt` exists on disk); the other 9 runs
(`defined_vaf` runs 1–4, `defined_rawbin` runs 0–4) have no `ari.txt` yet, so caching will recompute
them automatically — now through the fixed `mclust_R`, on GPU (fast, unlike my CPU validation).

**USER SUBMITS (Claude does not sbatch) — from repo root:**
```bash
sbatch clustering_benchmark/run_clustering_recovery_test.slurm
```
Then read `data/dlpfc_recovery_test/151507/clustering/summary.csv`: all 10 `defined_vaf`/
`defined_rawbin` rows should now have real ARI values (no more `status` error text). Compare their
5-seed means vs `defined1000G` **0.277** and `defined_bin250kb` **0.351** — expect both to underperform
given the rank collapse (repo prior: "expect vaf≈rawbin" since VAF is nearly binary already). **Then:**
promote `defined_bin250kb` to all 12 DLPFC sections (the last item from the 2026-07-10 plan).

---

## ✅ DONE — DLPFC bin-size sweep — CONFIRMED: `defined_bin250kb` is the peak

**Completed 2026-07-10 07:06 (job exit 0).** Plan agreed with user 2026-07-10: (1) bin-size sweep
first, (2) root-cause the `defined_vaf`/`defined_rawbin` mclust NaN failure, (3) promote the winner
to all 12 DLPFC sections. Step 1 is done; **step 2 (NaN root-cause) is now next.**

5 new bin sizes (500kb/125kb/100kb/50kb/25kb) clustered cleanly (5 seeds each, all `status=ok`,
correct 4221-spot shape). Result, vs `defined1000G` baseline **0.277**:

| bin size | bins | mean SNV/bin | ARI (5-seed mean ± std) |
|---|---|---|---|
| 1mb | 2,695 | 21.1 | 0.277 ± 0.057 |
| 500kb | 5,061 | 11.2 | 0.246 ± 0.044 |
| **250kb** | **8,845** | **6.4** | **0.351 ± 0.059 ← peak** |
| 125kb | 14,076 | 4.0 | 0.310 ± 0.064 |
| 100kb | 16,027 | 3.5 | 0.303 ± 0.039 |
| 50kb | 22,252 | 2.6 | 0.263 ± 0.026 |
| 25kb | 27,993 | 2.0 | 0.244 ± 0.037 |

**Conclusion:** 250kb is a genuine local peak, not a fluke. Fine side is a clean monotone decay
(125kb→25kb) as bins shrink toward single-SNV resolution and lose the aggregation benefit. Coarse
side dips at 500kb (0.246, below even 1mb) — likely seed noise (one of 5 seeds landed at 0.174; the
other four average ~0.264) rather than real non-monotonicity, but both coarse points sit well below
250kb regardless. **No further bin-size sweeping needed — `defined_bin250kb` stands as the winner.**

**Next: root-cause `defined_vaf`/`defined_rawbin`** (see DONE entry below for the `Mclust`/`svd` NaN
error — not yet instrumented/confirmed; hypothesis is a NaN STAGATE embedding under
`preprocess="none"` on these two very-sparse raw-alt-evidence matrices). **Then:** promote
`defined_bin250kb` to all 12 DLPFC sections (151507 alone is not yet paper-ready). Also queued from
discussion: a controlled `floor20 ∘ bin250kb` modality (does floor-filtering before binning sharpen
or starve the aggregate signal?) — run as one modality against `defined_bin250kb` once priorities
allow, not folded into other sweeps.

---

## ✅ DONE — DLPFC SNV-selection/representation clustering — job `12454721` — WINNER FOUND: `defined_bin250kb` 0.351

**Completed** 2026-07-10T00:04:29 (started 22:32:09, elapsed 1h32m — well under the 2–4h estimate,
8h limit). Exit code 0. Recovery-test harness on **section 151507 only**, all **27 modalities × 5
seeds = 135 runs** present in `data/dlpfc_recovery_test/151507/clustering/summary.csv`; 125 succeeded,
**2 modalities (`defined_vaf`, `defined_rawbin`) failed all 5/5 seeds** with an R error from mclust:
`Error in svd(shape.o, nu = 0) : infinite or missing values in 'x'` (visible in `run.err`/`run.out`,
rows written as `ari=NaN`) — likely a degenerate/near-constant embedding for these two representations;
not yet root-caused, no other modality hit it.

**Final means (5-seed avg), sorted, vs. baselines defined1000G 0.277 / gene_expr 0.412:**

| modality | ARI | vs. baseline |
|---|---|---|
| **`defined_bin250kb`** | **0.351** | **BEATS baseline by +0.074** — best of the run |
| `defined_floor05` | 0.286 | slightly above |
| `orig49k` (cached) | 0.278 | ≈ baseline |
| `defined1000G` (cached, baseline) | 0.277 | — |
| `defined_bin1mb` | 0.277 | ≈ baseline (predicted neutral — confirmed) |
| `defined_floor20` | 0.277 | ≈ baseline |
| `defined_upvrule` | 0.263 | below (confirms prior: UPV rule-based subset hurts) |
| `defined_floor03` | 0.259 | below |
| `defined_floor02` | 0.253 | below |
| `defined_somatic` (cached) | 0.245 | below (confirms: adding somatic hurts) |
| `defined_floor50` | 0.240 | below |
| `defined_floor10` | 0.223 | below |
| `defined_somtop02/05/10/25` | 0.198 / 0.190 / 0.186 / 0.181 | below, monotone with tightening (predicted) |
| `defined_tfidf` | 0.186 | below (predicted ↓ — confirmed) |
| `defined_cap30` | 0.156 | below |
| `defined_svg20k` | 0.148 | below |
| `defined_cap20/02/10`, `defined_svg10k/05k` | 0.11–0.12 | below (predicted ↓ — confirmed) |
| `defined_vaf`, `defined_rawbin` | **FAILED (NaN)** | mclust svd error, inconclusive |

**Takeaways:** (1) `defined_bin250kb` (250kb genomic binning of the defined/1000G SNV set, 8,845
features) is the standout winner — meaningfully closes the gap to the gene_expr upper reference
(0.412) without leaving the SNV-only regime. `defined_bin1mb` (2,695 features) is only neutral, so
250kb is the better bin size, not "coarser is better." (2) Every other prior was confirmed: capping,
SVG-selection, tf-idf, somatic/UPV augmentation, and top-%-somatic all hurt; floor-filtering is
roughly neutral-to-slightly-positive at floor02–floor20 and degrades by floor50.

**Next steps (not yet done):** (1) root-cause the `defined_vaf`/`defined_rawbin` mclust svd failure
(check for constant/degenerate features feeding STAGATE) and rerun those two. (2) Per the original
plan, promote `defined_bin250kb` to all 12 DLPFC sections before it goes in the paper — this single
151507 result is not yet a paper-ready claim. (3) No auto-aggregate/boxplot exists for the
recovery-test tree (`make_ari_boxplot.py` targets `data/dlpfc` — do not point it here, it would
clobber the post-dedup benchmark); any figure needs a new path. Full context: memory
[[project_dlpfc_snv_representation_study]] + [[project_dlpfc_ari_regression_codedrift]].

---

## 2026-07-09 (PM) — Q1 ANSWERED (adding somatic HURTS) + Q2 "clean the defined set" SET UP

**Q1 — "does adding the pipeline's focal/spatially-clustered SOMATIC denovo variants to the
defined set improve clustering?" → NO, it HURTS.** Read `data/dlpfc_recovery_test/151507/
clustering/summary.csv` (job `12453071`, 5 seeds each):

| modality | matrix | mean ARI (5 seeds) |
|----------|--------|--------------------|
| orig49k | real 0.28 matrix, 49,602 | **0.278** |
| defined1000G | defined-only, 56,845 | **0.277** |
| **defined_somatic** | defined ∪ ~13k focal denovo, 70,264 | **0.245** ↓ |

Adding the ~13k "somatic" denovo columns drops ARI 0.277 → 0.245. Interpretation: DLPFC
"somatic" is only top-10% denovo by `spatial_clustering` (no CalicoST clone/CNV features on
normal tissue), so in normal cortex these are just moderately-clustered denovo noise with no
real clonal structure → they add columns without layer signal. **Confirms: for the DLPFC
positive control, the best matrix is defined/1kG-only; do NOT fold denovo (somatic OR UPV) in.**

**Q2 — "does DROPPING the uniformly/densely distributed variants from the DEFINED set beat
0.277?" → BUILT + WIRED, awaiting cluster run.** Tests the handoff hypothesis that it's the
DENSITY/UNIFORMITY axis (not denovo-ness) that hurts. `clustering_benchmark/
build_defined_clean_matrices.py` (ran directly, fast) loads `DLPFC_151507_SPARCAL_1000G_matrix.pkl`
(4226×56,845), computes per-column prevalence + the pipeline's exact spatial-uniformity alpha,
and writes cleaned copies (same rows/values, fewer columns) to `data/dlpfc/151507/matrix/`:

| modality | rule | cols kept (dropped) |
|----------|------|---------------------|
| defined_cap30 | drop prev > 30% | 56,628 (−217) |
| defined_cap20 | drop prev > 20% | 56,458 (−387) |
| defined_cap10 | drop prev > 10% | 55,920 (−925) |
| defined_cap05 | drop prev > 5% | 54,672 (−2,173) |
| defined_cap02 | drop prev > 2% | 51,148 (−5,697) |
| defined_upvrule | pipeline UPV rule: alpha>0.5 & beta>0.2 | 54,927 (−1,918) |

Basis caveat: prevalence/alpha computed on the matrix itself (post-smoothing), the exact data
STAGATE sees — slightly higher prev than the pipeline's raw-`vcf_by_spot` basis; noted. Defined
median prev is 0.26% (mean 1.08%) with a heavy tail up to 94% — the caps bite only that tail.
Per-column scores saved: `data/dlpfc/151507/matrix/DLPFC_151507_SPARCAL_defined_clean_scores.tsv`.
The 6 pkls are added as modalities in `clustering_config_recovery_test.json` (clustering is
cached — the 3 existing modalities won't recompute).

**USER SUBMITS (Claude does not sbatch) — from repo root:**
```bash
sbatch clustering_benchmark/run_clustering_recovery_test.slurm    # computes only the 6 new modalities × 5 seeds
```
Then read `data/dlpfc_recovery_test/151507/clustering/summary.csv`: compare each `defined_cap*`/
`defined_upvrule` mean vs **defined1000G 0.277**. **Hypothesis: defined_clean > 0.277 ⇒ density
is the harmful axis ⇒ a general GT-free "drop dense/uniform SNVs" rule.** If even the aggressive
cap02 (−5,697) doesn't beat 0.277, density within the defined set is NOT the lever.

**Q2 RESULT — DONE (job 12453703), HYPOTHESIS REFUTED. Density is SIGNAL, not noise.** Dropping dense
defined variants does the OPPOSITE of help — it CRASHES ARI (matrices confirmed correct size, 4221
spots, 30-d emb): defined1000G **0.277** | cap30 (−217, prev>30%) **0.156** | cap20 (−387) 0.124 |
cap10 (−925) 0.111 | cap05 (−2,173) 0.117 | cap02 (−5,697) 0.123 | upvrule (−1,918, α>0.5&β>0.2) 0.263.
Removing just the **217 highest-prevalence** defined variants halves ARI ⇒ the dense/high-prevalence
defined variants are **load-bearing**: they carry the aggregate per-spot-coverage signal (cortical
layers differ in cellularity → coverage → detection), matching the Moran's-I finding that defined
variants cluster via AGGREGATE coverage. **The "drop dense/uniform SNVs" rule is DEAD for the defined
set.** Reconciles with UPV/somatic: harm there is *adding* flat denovo columns (dilution), not density
per se. **defined1000G 0.277 stands as the SNV-binary ceiling.**

**⇒ Revised priors for Q4 representation study:** TF-IDF *down-weights* common variants → now predicted
to HURT (suppresses the signal); binning (aggregates) → help/neutral; SVG-select (keeps high-Moran's-I,
discards the aggregate) → hurt.

**Q2b — LOW-prevalence FLOOR sweep (user's mirror idea) — BUILT + WIRED.** Since capping HIGH prevalence
crashed ARI, test the opposite: drop ultra-rare variants (the standard scanpy/ATAC `min_cells` filter —
a variant in 1 spot can't carry reproducible spatial signal). `clustering_benchmark/
build_defined_floor_matrices.py` keeps variants present in ≥N spots:

| modality | keep | dropped (present in <N spots) |
|----------|------|-------------------------------|
| defined_floor02 | 53,026 | 3,819 singletons (0.1% of detections) |
| defined_floor03 | 48,664 | 8,181 |
| defined_floor05 | 41,636 | 15,209 |
| defined_floor10 | 30,427 | 26,418 |
| defined_floor20 | 19,998 | 36,847 |
| defined_floor50 | 9,518 | 47,327 |

Added to `clustering_config_recovery_test.json` (now **27 modalities**). Prediction: floor02/03 (drop
singletons/doubletons, negligible aggregate mass) neutral-to-slightly-helpful (denoise + shrink);
aggressive floors (10/20/50) likely HURT if the aggregate-coverage signal is spread across many rare
variants (same mechanism as the cap crash). The floor02–floor50 curve maps exactly how much of the
signal lives in the rare tail.

**Q3 — somatic step-sweep (user's idea) — BUILT + WIRED.** Q1 added the FULL somatic set (top-10%
denovo by votes) and it hurt (0.245). But on DLPFC ~half that set has `f_spatial_clustering`=0 (not
actually clustered — purity/clone/cnv vote features are NA on normal tissue). So TIGHTEN the somatic
filter: keep only the top-X% MOST spatially-clustered somatic, union with defined, and see if a
smaller focal set recovers toward/above 0.277. `clustering_benchmark/build_somatic_sweep_matrices.py`
ranks the 13,419 somatic by `f_spatial_clustering` (med 0.0, max 0.167) and writes defined ∪ top-f:

| modality | somatic kept (top-f) | total cols |
|----------|----------------------|------------|
| defined_somtop25 | 3,355 | 60,200 |
| defined_somtop10 | 1,342 | 58,187 |
| defined_somtop05 | 671 | 57,516 |
| defined_somtop02 | 268 | 57,113 |

Added as modalities in `clustering_config_recovery_test.json` (now 13 total).

**SUBMIT ORDER (important — avoid a summary.csv write race):** the defined-clean job `12453703` is
STILL RUNNING (read the config at launch → won't pick up the 4 somtop modalities). Submit the
somatic sweep **AFTER 12453703 finishes**; caching then computes ONLY the 4 new somtop modalities
and merges cleanly:
```bash
sbatch clustering_benchmark/run_clustering_recovery_test.slurm   # after 12453703 done → computes the 4 somtop × 5 seeds
```
Read `data/dlpfc_recovery_test/151507/clustering/summary.csv`: compare each `defined_somtop*` mean
vs defined1000G **0.277** and defined_somatic **0.245**. Expectation: monotone recovery toward 0.277
as f shrinks; if even somtop02 stays < 0.277, no somatic subset helps the DLPFC positive control
(consistent with normal tissue having no real clonal/somatic structure).

**Q4 — SNV representation/feature-selection study (lit-review-driven, user asked, all GT-free) — BUILT
+ WIRED.** Insight from the review: our matrix is binary + ultra-sparse = the **scATAC-seq** problem,
and this is the **spatially-variable-feature selection** problem. Two literatures apply directly:
(1) binary-data representation → **TF-IDF/LSI** (scATAC standard; up-weight rare, down-weight common —
our `normalize_total+log1p` is a mismatched count model); (2) SVG benchmark (Genome Biology 2023/2025):
**SVG selection beats HVG for clustering, Moran's I is best & sparse-robust** — but must DECOUPLE
structure from prevalence (raw Moran's I rewards ubiquitous variants; that was the earlier "dead end").
Four experiments, each a modality on the defined matrix (build scripts in `clustering_benchmark/`):

| exp | modality(ies) | what | build script |
|-----|---------------|------|--------------|
| A | `defined_tfidf` | TF-IDF preprocess (reuses 1000G pkl, `preprocess:"tfidf"`) | code: SPARCAL_clustering.py |
| B | `defined_svg05k/10k/20k` | prevalence-cap 10% → top-K by Moran's I (5k/10k/20k) | build_svg_matrices.py |
| C | `defined_bin1mb` (2,695 bins) / `defined_bin250kb` (8,845) | genomic binning, count/bin/spot | build_binned_matrices.py |
| D | `defined_vaf` vs `defined_rawbin` | graded VAF vs matched binary, `preprocess:"none"` | build_vaf_matrix.py |

Code: added an **opt-in `preprocess` field** to `SPARCAL_clustering.py:load_snv_section`
(`lognorm` default = unchanged; `tfidf` = Signac RunTFIDF scale 1e4; `none` = pass-through). All
existing modalities are byte-identical (default) so caching holds. TF-IDF verified correct (rare cols
up-weighted 4.77 > 3.14 common). B: Moran's I on capped defined has median≈0, only 239 vars I>0.05
(defined variants are individually unstructured → cluster via AGGREGATE coverage; svg20k min-I≈0 is
already noise — tests the aggregate hypothesis). **D caveat:** mean VAF of detected entries = 0.974
(Visium per-spot depth → almost all detected SNVs have 0 ref reads → VAF≈1), so `defined_vaf` is
nearly binary; expect vaf≈rawbin. Config `clustering_config_recovery_test.json` now has **21
modalities**; py_compile clean, all pkls present, no stray keys in the loop.

**SUBMIT (after 12453703 finishes — same write-race reason; caching computes only the uncomputed):**
```bash
sbatch clustering_benchmark/run_clustering_recovery_test.slurm   # computes 4 somtop + 8 A/B/C/D = 12 new × 5 seeds = 60 runs
```
60 runs × ~1–3 min on a6000 ≈ 1–3 h (4h slurm limit; resumable — if it times out just resubmit, cached
runs skip). Then read `summary.csv` and compare every new modality vs **defined1000G 0.277** (and
gene_expr **0.412** as the upper reference). Priors: A (tfidf) is the most likely to beat 0.277; C (bin)
tests the aggregate-coverage hypothesis; B (svg) likely ≤0.277 if aggregate coverage is the mechanism;
D likely ties rawbin. **NOTE:** these all cluster on 151507 only — promote any winner to all 12 sections
before it goes in the paper.

---

## 2026-07-09 — DLPFC ARI regression RESOLVED (= UPV inclusion, not dedup) + denovo A/B in progress

**Full analysis + evidence in memory:** `project_dlpfc_ari_regression_codedrift.md` and
(corrected) `project_umi_dedup_ablation_finding.md`. Summary for the next session:

**RESOLVED — the 0.28→0.21 DLPFC clustering drop is CODE DRIFT, not UMI dedup.** The July
`generate_sparcal_matrices.py` builds the "normal"/germline matrix as defined+denovo, folding
in **UPV** (germline-denovo, ~4k dense/uniform columns). UPV are *worse than useless* for layer
clustering. Proof (recovery A/B, job `12450060`, isolated tree `data/dlpfc_recovery_test/151507`,
5 runs seeds 0-4):
- `orig49k` (real 0.28 matrix `temp/snv_matrix.pkl`, 49,602) → **0.278** (reproduces archived 0.288)
- `defined1000G` (current post-dedup, **UPV dropped**, 56,845) → **0.277** (full recovery)
- full germline WITH upv (60,804) → **0.177** (the regressed value)
- Dedup itself is ARI-neutral (same-code post 0.214 vs pre 0.204; 97.4% identical columns).
- **Paper fix: build the DLPFC clustering matrix as `--classes 1000G` (defined-only), no UPV.**

**A/B harness (reusable) — `clustering_benchmark/`:**
- `SPARCAL_clustering.py` now honors a `pkl_path` override in a modality config (cluster any pkl).
- `clustering_config_recovery_test.json` (modalities: orig49k, defined1000G, defined_somatic) +
  `run_clustering_recovery_test.slurm` (a6000 GPU, output isolated → `data/dlpfc_recovery_test/`,
  NEVER touches the protected `data/dlpfc/{s}/clustering{,_pre_dedup}` baselines). Runs are cached,
  so resubmitting only computes new modalities.

**Moran's I selection = DEAD END** (`denovo_spatial_analysis.py`, job `12452133`, outputs
`data/dlpfc/151507/denovo_spatial/baseQ0mapQ0/`): per-variant spatial autocorrelation INVERTS —
defined (helps) median I≈0, UPV (hurts) highest (+0.031), because `classify_variants` Stage-1
*defines* UPV as denovo with alpha(spatial_uniformity)>0.5 AND beta(global_prevalence)>0.2. So don't
select denovo by Moran's I. DLPFC caveat: Stage-2 somatic-vs-ambiguous votes on purity/clone/cnv
features that are **NA without CalicoST**, so DLPFC `somatic` = top-10% denovo by spatial_clustering.

**`defined + somatic` A/B — FINISHED (job `12453071`), NOT YET READ.** Matrix built by
`clustering_benchmark/build_defined_somatic_matrix.py` → `data/dlpfc/151507/matrix/
DLPFC_151507_SPARCAL_defined_somatic_matrix.pkl` (defined1000G ∪ ~13k focal denovo). Clustered as
modality `defined_somatic`. **NEXT SESSION: read `data/dlpfc_recovery_test/151507/clustering/summary.csv`**,
compare `defined_somatic` mean vs `defined1000G` **0.277** (does focal denovo add signal?) and vs
full-germline **0.177**. (Ambiguous, 120k cols, dropped — too large / ill-defined for DLPFC.)

**NEXT APPROACH (user's idea, agreed) — prevalence-cap "clean the defined set" experiment (GT-FREE):**
Run the DEFINED set through the UPV rule and remove defined variants that satisfy it, to test whether
it's DENSITY/UNIFORMITY (not denovo-ness) that hurts. Both scores are GT-free
(`calculate_spatial_uniformity_score` = CV of carrying-spot pairwise distances; `calculate_global_prevalence_score`
= spots_with/total). Plan: score every defined variant's alpha/beta from raw spot detection
(`output_VCFs/spotprofiles/{qf}/vcf_by_spot/`), build `defined_clean` matrices at the hard UPV
threshold (alpha>0.5 & beta>0.2) AND a **prevalence-cap sweep** (drop defined with prev >0.3/0.2/0.1/0.05
— needed because defined median prev is only 0.26%, so the hard threshold may flag too few to move ARI),
add as recovery-test modalities, cluster vs 0.277. Hypothesis: `defined_clean` **> 0.277** ⇒ density is
the harmful axis ⇒ a general, transferable "drop dense/uniform SNVs" rule.

**GT design justification (settled this session):** GT (cortical-layer labels) is an EVALUATION
yardstick ONLY (ARI at the very end) — never a pipeline input or SNV-selection criterion. Clustering
is unsupervised; DLPFC is a POSITIVE CONTROL that validates the method for the no-GT tumor datasets.
**Bright line: every SNV filter must be GT-free** (prevalence, spatial uniformity, Moran's I, quality)
so it transfers. Selecting SNVs by correlation with layers (Fisher-vs-layer, cross-layer variance) is
CIRCULAR/forbidden as a method (oracle-only). The prevalence-cap experiment is correctly GT-free.

---

## 2026-07-08 — SpaceTracer gnomAD/dbSNP reference fix + DCIS1 time-boxed submission

**FIXED:** gnomAD v2.1 / dbSNP 138 (hg19) reference allele mismatch in SpaceTracer's `prior` rule. Job 12351949 (P6) was stagnated for 3+ days repeating the error; root cause was non-fatal coordinate/build incompatibility. Changed `logging.error` → `logging.debug` in `/data/maiziezhou_lab/leiy4/SpaceTracer/module/read_file.py:416`.

**SUBMITTED:** SpaceTracer DCIS1 end-to-end test via `/data/maiziezhou_lab/leiy4/SpaceTracer/run_sample.sh` — **job 12417865** (2026-07-08 10:15 AM). Single time-boxed attempt per 2026-07-06 decision (if successful → benchmark; if fails → cite preprint only). Monitoring log: `tail -f /data/maiziezhou_lab/leiy4/SpaceTracer/runs/slurm_output/spacetracer_12417865.err`.

---

## 2026-07-08 — DLPFC PRE-DEDUP regeneration for a fair 10-run clustering comparison (SET UP, ready to submit)

**Why:** the post-dedup clustering benchmark (10-run) finished (sparcal mean ARI **0.217**), but
the pre-dedup baseline was only **5-run** (0.281) AND its SPARCAL matrices were overwritten in place
by the 2026-07-07 dedup rerun (data/dlpfc/{s}/matrix now = deduped 60,804-col; the 49,602-col
pre-dedup version is gone; spatial_filter_purity also clobbered). To compare fairly (10 vs 10) we must
**regenerate the pre-dedup SPARCAL matrices** then run 10× sparcal clustering. User decision (2026-07-08):
**regenerate + 10× rerun, and DO NOT overwrite the post-dedup results.**

**Non-destructive design — env flag `DLPFC_PREDEDUP=1`.** Added a backward-compatible flag to all 8
pipeline scripts (steps 1–8) that flips ONLY the DLPFC paths: BAMs ← the read-only ORIGINAL non-dedup
source `/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{section}/bam_bycell/*.bam`
(confirmed present, 4992/section × 12), output → **`data/dlpfc_prededup/{section}`**. Default (unset) =
unchanged → post-dedup `data/dlpfc` is never touched. Verified: all 8 compile; path resolution correct
both ways; dataset name stays `DLPFC` so every format branch still fires (no OVAR_P5-style gap).
Files: steps 1/6 override bam_base_path+bam_pattern+output_dir; steps 2–5 output_dir; step 7 output_base;
step 8 DATASET_OUTPUT_BASE. New runner **`run_slurm/dlpfc/run_pipeline_DLPFC_prededup.sh`** (array 0-11,
exports the flag, resumable START_STEP).

**Clustering phase:** new `clustering_benchmark/clustering_config_prededup.json` (sparcal only, both
paths → data/dlpfc_prededup) + runner `clustering_benchmark/run_clustering_prededup.slurm` (10 runs,
sparcal only, writes to data/dlpfc_prededup/{s}/clustering; deliberately SKIPS make_ari_boxplot.py /
make_combined_figure.py because those default to data/dlpfc and would clobber the post-dedup aggregate).

**SUBMIT (from repo root; user submits — Claude does not sbatch):**
1. **`bash run_slurm/dlpfc/run_pipeline_DLPFC_prededup.sh` — SUBMITTED 2026-07-08, job `12413536`
   (array 0-11).** At submit: tasks 0–3 (151507–151510) RUNNING on cn1801/cn1804/cn1812, tasks 4–11
   PENDING on `QOSGrpCpuLimit` (CPU-quota queue; start as running ones free). Steps 1–8 → data/dlpfc_prededup
   (long: mpileup ~hours/section). **VERIFY per section when done:** matrix
   `data/dlpfc_prededup/{s}/matrix/DLPFC_{s}_SPARCAL_normal_matrix.pkl` exists with col count ≈ the
   pre-dedup 49,602 scale (NOT the deduped 60,804), and data/dlpfc/{s} mtimes are UNCHANGED (Jul-7).
2. after (1): `sbatch --array=0-11 clustering_benchmark/run_clustering_prededup.slurm`   # 10× sparcal
3. after (2): aggregate `data/dlpfc_prededup/*/clustering/summary.csv` vs post-dedup
   `data/dlpfc/*/clustering/summary.csv` for the 10-vs-10 sparcal comparison.

**Post-dedup 10-run baseline to compare against (means):** gene_expr 0.412 · **sparcal 0.217** ·
gatk 0.211 · strelka2 0.193 (gatk/strelka2 over 11 sections — 151671 mclust svd failure).

---

## 2026-07-09 — pre-dedup regen CHECKED: 11/12 done, matrix col-count anomaly, user picked "5+5" mix

**Pipeline `12413536` status:** 11/12 sections COMPLETE (151507–151675). Task 11 (151676) still on
Stage 1 (mpileup), started ~03:25, other sections took 5–12h for the full 8-stage chain — check back later.

**Anomaly found:** regenerated matrices do NOT match the expected ≈49,602-col (old pre-dedup) scale —
they're bigger and inconsistent across sections, e.g. 151507 61,837 · 151671 72,847 · 151674 110,127
(row/spot counts DO match the old archive exactly, e.g. 4226 for 151507 — only the SNV/column axis
differs). Likely cause: **unseeded `np.random.choice` training subsample** in
`scripts/4_classifier/run_supplimentary_models.py:772` — reruns on identical BAMs are not
deterministic, so "regenerate the pre-dedup matrix" does not reproduce the original 49,602-col set.

**User decision (2026-07-09): combine anyway — "5+5" mix, NOT a clean 10-run.** Keep the existing
archived **run0–4** at `data/dlpfc/{s}/clustering_pre_dedup/sparcal/run{0-4}/` (computed on the
now-deleted original 49,602-col matrix) as-is, and add **5 NEW runs (run5–9)** computed on the
freshly regenerated `data/dlpfc_prededup/{s}/matrix/...` matrix. The resulting "10-run" pre-dedup
box entry will therefore mix ARI values from two different underlying variant sets, not just
mclust-seed randomness — flagged as a caveat for the paper if this number is reported.

**Command for the user to submit (once section 151676's matrix is ready — currently task 11 of
`12413536` is still running):**
```bash
sbatch --array=0-11 --export=ALL,RUNS=5,6,7,8,9 clustering_benchmark/run_clustering_prededup.slurm
```
Writes `data/dlpfc_prededup/{s}/clustering/sparcal/run{5..9}/` + merges into
`data/dlpfc_prededup/{s}/clustering/summary.csv` (resumable — safe to resubmit, only recomputes
missing runs). If 151676 isn't ready yet, sections 0–10 can be submitted now
(`--array=0-10`) and 11 appended later (`--array=11 ...`).

**After that finishes:** aggregate old run0–4 (`data/dlpfc/{s}/clustering_pre_dedup/sparcal/run{0-4}/ari.txt`)
+ new run5–9 (`data/dlpfc_prededup/{s}/clustering/sparcal/run{5-9}/ari.txt`) into one combined ARI
box plot, written to a **new** location (not overwriting `data/dlpfc/clustering_benchmark_pre_dedup/`
or the post-dedup `data/dlpfc/clustering_benchmark/`) — planned output dir:
`data/dlpfc_prededup/clustering_benchmark_combined/`.

---

## 2026-07-07 — OVAR_P5: step-3 fix VERIFIED healthy; NEW `1000G=0` matrix anomaly (runner --exclude_vcf)

**Chain `12390112..117` COMPLETED — the step-3 `extract_metrics` fix worked, all previously-zero
numbers are now real:** step 3 `Total beagle variants` 5552/4955/… (was 0), transitions **16,744
shifted / 45,212 stable** (was 0/0); step 5 classifier results non-empty (het 33,648 / hom 93,908 /
predictions 344,363); step 7 `Total SNV instances 4,038,488` / 126,764 unique (was 0), classification
**germline 71,540 (defined 61,850 + denovo 9,690), somatic 11,720 (denovo), ambiguous 105,482**;
per-barcode txts 2,109 germline / 2,099 somatic.

**4-class SPARCAL matrices built** (`generate_sparcal_matrices.py`, direct): rows 2,108 —
germline **9,562**, somatic **11,720**, merged **21,282**, but **`1000G` = 0**.

**⚠️ NEW ANOMALY — `1000G` matrix is empty; germline matrix is denovo-only.** All 55,225 rows in the
germline per-barcode txts are `race==denovo`, **0 `defined`** → the 1000G class (germline ∩ defined)
is empty and the germline matrix (9,562) omits the 61,850 defined/1KGP variants. **Root cause: the
OVAR_P5 step-7 runner `run_slurm/ovar_p5/7_spatial_filter_n_matrix.sh` passes
`--exclude_vcf ${BEAGLE_VCF}` AND `--kept_variants ${BEAGLE_VCF}` where `BEAGLE_VCF=all_filtered_in.vcf.gz`
(61,989 = the beagle-KEPT 1KGP-concordant "defined" variants).** `--exclude_vcf` removes those from
every spot's set (log: `Exclusion pool: 61978` → `SNVs after pool filtering: … -> 0` for the defined
tier), so the per-spot germline files keep only denovo. **P6/DCIS pass NEITHER `--exclude_vcf` nor
`--kept_variants`** (P6 `SPATIAL_ARGS` has only tumor_purity/clone/cnv/min_expression), so their
defined variants survive → P6 `1000G` matrix = 125,269, germline per-barcode files hold both defined
(8,626) + denovo (7,343). Excluding `all_filtered_in` (the germline you want) looks like a copy-paste
bug — if anything, only `all_filtered_out` (beagle-rejected junk) would be a sensible exclude.
- **FIX APPLIED 2026-07-07 (user approved):** removed `--exclude_vcf ${BEAGLE_VCF}` and
  `--kept_variants ${BEAGLE_VCF}` from `run_slurm/ovar_p5/7_spatial_filter_n_matrix.sh` (now matches
  P6/DCIS, which pass neither). `bash -n` clean; only comments mention `BEAGLE_VCF` now.
- **STEP 7 RE-RUN — job `12399551` COMPLETED ✅ 2026-07-07T14:40 (34m46s, MaxRSS ~2.3 GB).** With the
  fixed runner (no `--exclude_vcf`/`--kept_variants`). Step 6 `vcf_by_spot` reused. Summary now healthy:
  germline **71,447 (defined 61,850 + denovo 9,597)**, somatic **11,729 (denovo)**, ambiguous 105,566;
  per-barcode txts **2,109 germline / 2,100 somatic**, and germline txts carry BOTH `defined` (2.5M
  rows) and `denovo` (2.9M rows) — the fix worked (was denovo-only).
- **MATRICES REBUILT ✅ 2026-07-08** (`generate_sparcal_matrices.py`, direct). Note: the 13:52 matrices
  were STALE (built before the 14:40 step-7 finish → 1000G=(2108,0), germline=9562 denovo-only). Rebuilt
  from the corrected step-7 output — **anomaly RESOLVED**, all cross-checks hold:
  **1000G (2108, 61,850)** (was 0) · germline (2108, **71,447**) = 1000G+UPV · somatic (2108, **11,729**)
  · merged (2108, **83,176**) = germline+somatic. OVAR_P5 pipeline is now COMPLETE end-to-end.

---

## 2026-07-05 — OVAR_P5: pipeline started (step 1 mpileup submitted)

**OVAR_P5 (section `P5_sr13`) was stuck at step 0 only** — 2,108 per-barcode BAMs
in `data/ovar_p5/P5_sr13/split_BAM/` (job 12299132, done 2026-07-02), no downstream
outputs (no VCFs/matrices).

- **`12368216` mpileup_ (step 1) — COMPLETED ✓ 2026-07-05 20:15** (elapsed 3h55m,
  MaxRSS 21.7 GB, node cn1817). 22 region VCFs
  `output_VCFs/mpileup_multi_bam/baseQ0mapQ0/region_chr{1..22}.vcf.gz` (autosomes
  only) + `merged_sorted_gt.vcf.gz` = **458,126 records**, index valid. (Log's
  "Total SNPs found: 3669582" = pre-merge raw per-region count.) NB output files are
  `region_chr*.vcf.gz`, not `chr*.vcf.gz`.
- **Steps 2→6 — COMPLETED** (afterok chain `submit_chain_2_7.sh`, 2026-07-05):
  `12372470` beagle (7m) → `12372471` genotype-shift (34s) → `12372472` seq-error (3s)
  → `12372473` nn_classifier (10s) → `12372474` single-BAM filter (30m). Step 6 verified:
  **2108/2108 spots with SNVs, 2108 per-spot VCFs** in
  `output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot/`.
- **Step 7 (`12372475`) — FAILED in 8s, now FIXED (2026-07-06).** `run_spatial_snv_filter_enhanced.py`
  argparse `choices` list omitted `ovar_p5` (`error: argument --dataset: invalid choice: 'ovar_p5'`),
  even though its DATASET_CONFIGS + spatial-file dispatch already handle OVAR_P5. Added
  `'ovar_p5'` to the choices list (one line). Verified clone_labels.tsv exists (43 KB) and
  `run_generate_matrix.py` already accepts ovar_p5 (choices derived from its config) → no
  other blocker.
- **Step 7 RE-RAN 2026-07-06 16:08 but produced DEGENERATE output — BROKEN, needs debug.**
  `spatial_filter_summary.txt`: `Total SNV instances: 0`, `Unique variants: 0`,
  `Spots with germline variants: 0`, `Spots with somatic variants: 0`. Classification:
  **germline 61,978 (all `defined`, 0 denovo), 0 somatic, 0 ambiguous**. The
  `germline/{defined,denovo}` dirs hold only the combined variant lists — **NO per-barcode
  `{barcode}.txt`** were written (the spatial filter matched 0 variants to the 2108 spots).
  Consequence: the only matrix is `OVAR_P5_P5_sr13_bcftools_normal_6_matrix.pkl` with shape
  **(1, 61978)** — 1 row, not 2108 — i.e. garbage. **No 4-class SPARCAL matrices.**
  - **ROOT CAUSE FOUND + FIXED 2026-07-07 — step 3 (`run_beagle_genotype_shifting.py`)
    `extract_metrics` had no `OVAR_P5` branch.** Its dataset dispatch was
    `if dataset in [DLPFC,…,DCIS]: … elif [P4,P6]: … else: raise ValueError("Unknown dataset
    format")`. For OVAR_P5 every beagle variant raised, and that exception is **silently
    swallowed** by the `try/except (ValueError, IndexError): continue` in
    `_load_beagle_variants` → `beagle_variants` stayed **empty for all 22 chr** (`Total beagle
    variants: 0` in the log) → the pre/post genotype join matched nothing → step 3 wrote
    `metrics_by_transition={}`, `all_transitions={}` (0 shifted, 0 stable) though it counted
    458,126×22 = 10,078,772 variants. Empty transition pkls → step-5 classifier collected
    **0 training examples** → crashed at feature extraction (`No valid features extracted`),
    leaving `Classifier/…/results/` empty → step 6/7 saw only the 61,978 Beagle `defined`
    (1KGP) variants, 0 denovo → degenerate matrix.
    **FIX:** added `"OVAR_P5"` to the DLPFC/DCIS branch of `extract_metrics`
    (`run_beagle_genotype_shifting.py:192`). Verified inline: now returns `(BAF=1.0, DP=4)`,
    no raise. (The identical branch in step 4 `run_sequence_error_model.py:293` already had
    OVAR_P5 — step 4 was fixed a prior session but step 3 was missed; both are the same bug.)
  - **Full dataset-dispatch sweep of steps 1–8 (2026-07-07):** the ONLY OVAR_P5 gap was
    step 3 above. Confirmed OVAR_P5 present/handled in: step 1 mpileup (`:583`), step 2 beagle
    (config-driven), step 4 (`:293`), step 5 classifier (config + chr auto-detect, no format
    branch), step 6 filter_bams (falls into the identical `else` branch — already ran, 2108
    per-spot VCFs), step 7 spatial_filter (`:276,:537`), and both matrix builders
    (`run_generate_matrix.py:41`, `generate_sparcal_matrices.py:51`).
  - **Downstream sweep (steps 4–8) confirmed clean 2026-07-07.** The step-7 "Total SNV
    instances: 0" was NOT an independent step-7 bug: its per-spot load worked perfectly
    (log: `Loaded 2467808 total SNV instances across 2108 spots`, chr-keys matched, RACE
    parsed). The collapse `2467808 -> 0` was pool logic — with 0 denovo, step-6's SNV pool =
    only the 61,978 `defined` variants, so `vcf_by_spot` held only those, and step-7's
    exclusion pool (= those same 61,978) removed everything. Pure downstream of the empty
    classifier. All downstream steps handle OVAR_P5 (4 `:293`, 5 config+chr-autodetect,
    6 ran OK, 7 `:276/:537`, matrix builders `run_generate_matrix.py:41` /
    `generate_sparcal_matrices.py:51`). Step 3 was the ONLY gap.
  - **CHAIN RE-SUBMITTED 2026-07-07 (`submit_chain_2_7.sh`, with the step-3 fix):**
    `12390112` beagle → `12390113` genotype-shift → `12390114` seq-error → `12390115`
    nn_classifier → `12390116` single-BAM filter → `12390117` spatial_filter (afterok chain;
    all PENDING at submit, `12390112` held on `QOSGrpCpuLimit`). Re-running step 6 is
    essential (regenerates `vcf_by_spot` with the denovo/somatic variants). **VERIFY when
    done:** step-3 log `Total beagle variants: >0` + non-zero shifted/stable; step 5 writes
    non-empty `Classifier/…/results/neural_network_*.vcf.gz`; step-7 summary `Total SNV
    instances > 0` with non-zero denovo/somatic + per-barcode `{barcode}.txt` under
    `spatial_filter_purity/…/{germline,somatic}/`. Then run
    `python scripts/6_spatial_filter/generate_sparcal_matrices.py --dataset OVAR_P5 --section_id P5_sr13`
    for the 4-class SPARCAL matrices. **Do NOT trust the current (1, 61978) matrix.**
- Config: GRCh38, `chr`-prefix, single section, 2108 in-tissue spots (see memory
  `project_ovar_p5_dataset`).

---

## 2026-07-02 (later) — DLPFC: SPARCAL_normal rename + UMI dedup (split OOM-failed; fix applied 2026-07-05)

**DLPFC matrix renamed** to `DLPFC_{section}_SPARCAL_normal_matrix.pkl` (step 8
`generate_sparcal_matrices.py --classes normal`; `normal` = all germline). All 12
sections regenerated (151507 = 4226 × 49,602, cols match old, rows now align with
strelka2/gatk 4226). Old `bcftools_normal_6` deleted (all 12). Clustering benchmark
updated: `clustering_benchmark/clustering_config.json` sparcal modality →
`{caller: SPARCAL, filter: normal, grouping: ""}`; `SPARCAL_clustering.py` +
`st_loading_utils.py` drop the grouping token when empty (strelka2/gatk unchanged).

**DLPFC UMI dedup — SUBMITTED 2026-07-02 19:36.** DLPFC has no possorted BAM —
only read-only per-cell `bam_bycell/*.bam` (CB+UB tags). `run_slurm/dlpfc/0_umidedup_split_DLPFC.sh`
(array 0-11): merge → `umi_tools dedup --per-cell` → split by CB →
`data/dlpfc/{s}/bam_bycell_dedup/`. Pipeline reads deduped BAMs via a new optional
`bam_base_path` config field in `mpileup_pipeline.py` + `run_filter_bams_by_snv_pools.py`
(repoints only the BAM glob; `base_path` still serves read-only spatial).

- **`12303350_[0-11]` umidedup_DLPFC — RAN but the SPLIT step FAILED on all 12.**
  merge (step 1) + `umi_tools dedup` (step 2) SUCCEEDED — every section has an intact
  `data/dlpfc/{s}/dedup_tmp/merged.dedup.bam` (151507: 227.5M reads, 4984 barcodes).
  BUT step 3 `samtools split -@ 16 -d CB -M 6000` was **OOM-`Killed`** on every section
  (`.err`: "line 80: Killed … samtools split"). The dedup BAM is coordinate-sorted, so
  all ~5000 per-cell writers stayed open the whole pass; `-@ 16` gave each its own
  thread-pool block queue → >128 GB RSS → killed ~3 min in. The script had **no error
  check on the split**, so it "indexed" the truncated files and printed a false
  `[done] … BAMs = N`. Result in `bam_bycell_dedup/`: only 172–494 files/section (of
  4984), all **byte-identical + truncated** (each exactly 1,983,855 B, no EOF marker,
  unreadable, 0 `.bai`). The deduped pipeline never validly ran.
- **`12303351` dlpfc_pipeline — mis-submitted then CANCELLED** (unchanged; irrelevant now).

- **`12367865_[0-11]` (retry 2026-07-05, `-@1` single-thread) — STILL OOM-Killed**, but
  the new guards caught it (every task aborted at `ERROR: samtools split failed/killed`,
  no corrupt/false-success output, dedup correctly skipped). Single-threading did NOT
  help → threads were not the cause.

**ROOT CAUSE (found 2026-07-06):** the OOM is the **giant header**, not threads/FD limits.
`merged.dedup.bam` carries a **~1.1M-line header** (194 @SQ but **19,968 @RG + 34,944 @PG**,
one set per input per-cell BAM accumulated through merge+dedup). `samtools split` holds
the parsed header in memory and attaches it to every one of ~5000 open output streams →
>128 GB. Measured on a 4,870-cell slice: **fat header = 13.8 GB RSS (stalls); slim
@HD+@SQ header = 0.86 GB RSS, all 4,870 files produced + quickcheck-OK.**

**FIX applied 2026-07-06 to `0_umidedup_split_DLPFC.sh` step 3:** reheader the dedup BAM to
a minimal `@HD`+`@SQ` header on the fly and pipe straight into split
(`samtools reheader slim.hdr DEDUP | samtools split -@1 -M 6000 -d CB … -`), no 13 GB temp.
Dropped @RG/@PG are unused downstream (bcftools mpileup ignores them; per-cell BAMs get the
same lean header the non-dedup source BAMs already have). Kept: quickcheck gate on the
split output, `|| exit 1` guards, and merge+dedup skip when a valid `merged.dedup.bam`
exists. Mechanism verified end-to-end on a slice (streaming `reheader|split -` → 4,870
files, all quickcheck-OK).

**`12379642_[0-11]` (2026-07-06, with the reheader fix) — SUCCEEDED on all 12 sections.**
Each: `[done] … BAMs = 4992`, 4992 `.bam` + 4992 `.bai`, 0 errors (vs 173–501 corrupt
before). DLPFC UMI dedup + per-cell split is **COMPLETE**.

**DEDUPED PIPELINE COMPLETE 2026-07-07 — all 12 sections ✅** (array `12389935_[0-11]`, steps
1–8). All 12 `DLPFC_{section}_SPARCAL_normal_matrix.pkl` freshly overwritten with the deduped
call set (mtimes Jul 7 02:57–13:53; 151674/`_9` finished last — its Stage-7 spatial filter ran
~4h15m, not stuck). DLPFC now also emits real somatic denovo variants (e.g. 151674: germline
denovo 31,832, somatic 38,774).

**POST-DEDUP CLUSTERING BENCHMARK — pre-dedup results archived, 10× run pending (2026-07-07).**
The benchmark now lives in `clustering_benchmark/` (was `SPATIAL_SNV/`) and is resumable (reuses
`clustering/{modality}/run{i}/ari.txt`). The old results were **pre-dedup 5-run** (e.g. sparcal
n_snvs 49,602), so to avoid mixing pre/post-dedup runs they were **renamed aside**:
`data/dlpfc/clustering_benchmark` → `…_pre_dedup`, and each `data/dlpfc/{s}/clustering` →
`clustering_pre_dedup` (all 12). All benchmark inputs verified present: sparcal (Jul-7 post-dedup),
strelka2/gatk (unchanged), gene_expr (h5 live). **NEXT (user to run):** the germline clustering
benchmark, 10 runs, all 4 modalities × 12 sections:
`sbatch --array=0-11 --export=ALL,N_RUNS=10 clustering_benchmark/run_clustering.slurm`
(a6000, cap 2 concurrent → serializes ~2 sections at a time; the slurm script auto-runs
`make_combined_figure.py` + flock-guarded `make_ari_boxplot.py` → fresh `data/dlpfc/clustering_benchmark/`).
Compare post-dedup ARI vs pre-dedup baseline (sparcal 0.281; gene_expr 0.404 > sparcal > gatk 0.205
> strelka2 0.180).

---

## 2026-07-02 — UMI-dedup pipeline FINISHED (all 4 samples) + matrix/pipeline refactor ✅

**All four dedup samples (P4, P6, dcis1, dcis2) are complete through matrices.**
No pipeline jobs running (`squeue` clean). Steps 1–7 were done for all four on
2026-07-01 (P4 needed the step-5 refix rerun `12276786..788`; DCIS `12270329..335`).
Today the matrix step was rebuilt for all four with the new generator.

### New canonical matrix generator — `scripts/6_spatial_filter/generate_sparcal_matrices.py`
Replaces the flaky `run_generate_matrix.py`/`final_snv_mat.py` path juggling (stale
`--filter-subdir`, case-sensitive `data/P6_tumor`, UPV-matrix corruption). Reads step-7
per-barcode `spatial_filter_purity/{qf}/{germline,somatic}/{bc}.txt` (has `race` col
defined/denovo) → **4 binary int8 matrices sharing one row index**, named
**`{STUDY}_{section}_SPARCAL_{class}_matrix.pkl`** (model always `SPARCAL`, no `_6`):

| sample | rows | 1000G | germline (=1000G+UPV) | somatic | merged |
|--------|------|-------|-----------------------|---------|--------|
| P6_TUMOR 1 | 3650 | 125,269 | 127,013 | 65,655 | 192,668 |
| P4_TUMOR 1 | 744  | 42,375  | 51,866  | 19,523 | 71,389  |
| DCIS dcis1 | 1454 | 140,773 | 193,893 | 18,536 | 212,429 |
| DCIS dcis2 | 1807 | 129,764 | 166,087 | 25,154 | 191,241 |

Cross-checks hold (germline=1000G+UPV, merged=germline+somatic). Old-named matrices
(`*_bcftools_normal_6`, `*_bcftools_somatic_6`) **deleted** — superseded & reproducible.
Each sample's `matrix/` now holds exactly the 4 SPARCAL pkls.

### Unified run_pipeline scripts (steps 1–8, resumable `START_STEP` arg)
`run_slurm/{P4_tumor,P6_tumor}/run_pipeline_{P4,P6}.sh`, `run_slurm/DCIS/run_pipeline_DCIS.sh`
(`--array=1-2`), updated `run_slurm/dlpfc/run_pipeline_DLPFC.sh`. One job runs all steps +
matrix, aborts on first failure. `sbatch …/run_pipeline_DCIS.sh 8` = matrix-only rerun.
Step 5 = `run_supplimentary_models.py` (not the buggy `run_sparcal_net.py`).

### DCIS output cleansed to `data/dcis1` / `data/dcis2` only
Removed stale April artifacts `data/dcis{section_id}` (597M), `data/dcis/dcis{1,2}` (29M),
`data/DCIS` (80M). DCIS dual section-id documented: numeric (1/2) for steps 1–6, prefixed
(dcis1/dcis2) for steps 7–8, both → `data/dcis{1,2}`. See CLAUDE.md "Standard run pipeline".

**Remaining (optional):** DLPFC still uses the old `bcftools_normal_6` naming (SPATIAL_SNV
loader keyed on it) — its run_pipeline keeps both that and the new step-8 SPARCAL matrices.

---

## 2026-07-01 — Results check + P6 matrix fix + P4/DCIS dedup submission

### DLPFC benchmarking (clustering, all 12 sections) — DONE ✅
Array `12156700_0..11` all COMPLETED (2026-06-25). Aggregate outputs in
`data/dlpfc/clustering_benchmark/` (`ari_boxplot{,_by_section}.{png,pdf}`,
`ari_table.csv`, `ari_matrix_{mean,best}.csv`). **Mean ARI across 12 sections:**
gene_expr **0.404** > sparcal **0.281** > gatk **0.205** > strelka2 **0.180**.
SPARCAL is the best SNV caller in **8/12** sections; gene_expr best overall 11/12.
Confirms the 151507 pilot ordering (sparcal > gatk > strelka2). Caveat still:
only GATK has the 1000G filter.

### UMI dedup — step 0 done for 4 samples ✅
`umi_tools dedup --per-cell` + `samtools split -M 6000`, all COMPLETED:
| Sample | pre → post reads | dup removed | split BAMs |
|--------|------------------|-------------|------------|
| P6 rep1 (12133315) | 84.3M → 51.5M | ~39% | 4,992 |
| P4 rep1 (12149523) | 104.5M → 84.7M | ~19% | 4,992 |
| DCIS1 (12149524_1) | 508.4M → 329.7M | ~35% | 4,992 |
| DCIS2 (12149524_2) | 488.9M → 311.6M | ~36% | 4,992 |
Deduped `split_BAM/` in place for all; old non-deduped split_BAM backed up aside.

### P6 dedup pipeline (steps 1–7) — filtering DONE; matrix FIXED 2026-07-01 ✅
Steps 1–7 completed (job chain up to 12156808). Categorized VCF counts (deduped):
germline_defined 125,269 · UPV 1,744 · somatic_denovo 65,655 · ambiguous_denovo 590,897.
**Matrix step in job 12156808 FAILED** (`run_generate_matrix.py --filter-subdir
filtered_snvs` → `spatial_analysis/.../filtered_snvs` not found — the known stale-path bug).
**Regenerated manually 2026-07-01** from the real per-barcode dirs →
`data/P6_tumor/1/matrix/`:
- `P6_TUMOR_1_bcftools_normal_6_matrix.pkl` — **3651 × 127,013** (= germline_defined
  125,269 + UPV 1,744; **NOT** the ~1.25M raw-merged set → the old UPV-matrix corruption
  bug is GONE ✅).
- `P6_TUMOR_1_bcftools_somatic_6_matrix.pkl` — 3649 × 65,655.

**Fixes applied 2026-07-01 (so future runs don't repeat the failure):**
- `scripts/6_spatial_filter/run_generate_matrix.py:save_snv_matrix` now writes to the
  dataset's real `output_subdir` (case-sensitive FS: was writing `data/p6_tumor` ≠
  `data/P6_tumor`).
- `run_slurm/{P4_tumor,P6_tumor}/7_spatial_filter_n_matrix.sh` matrix call switched from
  `--filter-subdir filtered_snvs` to `--input-dir .../spatial_filter_purity/${QF}/germline`.

### P4 / DCIS dedup pipeline (steps 1–7) — SUBMITTED 2026-07-01

**DCIS chain — RUNNING ✅** (dependency-chained, dedup ablation, dcis1+dcis2, baseQ0mapQ0):
`12270329` mpileup (array 1-2) → `330` beagle → `331` genotype_shift → `332` seq_err →
`333` nn → `334` filter_bams → `335` spatial_filter. Step 1 confirmed doing real work
(>13 min). DCIS scripts were already correct (env `snv_caller`, `scripts/1_calling/...`).
DCIS matrices come from `8_final_snv_mat_dcis{1,2}.sh` (its step 7 doesn't call
run_generate_matrix) — run those after step 7 finishes.

**P4 chain — FIRST ATTEMPT `12270322..328` FAILED SILENTLY (exit 0, ran in seconds).**
Root cause: stale P4 scripts. Step 1 used a non-existent env `snv_caller_new` and the old
path `scripts/calling/mpileup_pipeline.py` (real: `scripts/1_calling/…`); no error-check →
exit 0 → the `afterok` chain cascaded through, steps 6/7 likely touched stale/old P4 data.
**Fixed 2026-07-01** (env `snv_caller`, correct paths):
- step 1: `snv_caller_new`→`snv_caller`, `scripts/calling/`→`scripts/1_calling/`, comment `module load`
- step 2: dropped stray `--caller Monopogen` (run_beagle.py has no `--caller`), added `--quality-filter baseQ0mapQ0`
- step 3, 4: `snv_caller_new`→`snv_caller`
- (steps 5/6/7 were already OK; ablation edits from earlier still in place)
**P4 chain — 2nd attempt `12270741..747` COMPLETED, but step-5 classifier CRASHED → 0 denovo.**
Steps 1–4 are GOOD (mpileup 2h31m fresh on deduped BAMs; beagle `all_filtered_in` 42,523 /
`all_filtered_out` 529,582; seq-err ran). **Step 5 used `run_sparcal_net.py`, which crashed:
`ValueError: y contains previously unseen labels: 'no_variance'`** (trains on het/hom only,
then hard-looks-up `no_variance` in `apply_to_vcf` → `neural_network_*.vcf.gz` all EMPTY).
Consequence: step 6 saw only the 42,523 defined (1KGP) variants; step 7 classified **100%
germline_defined, 0 UPV, 0 somatic** (pre-dedup P4 had UPV 7,215 / somatic 4,924). The P4
`normal` matrix (745 × 42,523) is therefore **germline-only / incomplete — do not use yet.**
- **Fix 2026-07-01:** P4 `5_neural_network.sh` switched from `run_sparcal_net.py` to
  `run_supplimentary_models.py --model-type neural_network --max-training-samples 90000`
  (exactly what P6/DCIS use and which works). `run_sparcal_net.py` has a latent
  label-encoder bug (no `no_variance` class at train time) — avoid until fixed.
- **ACTION: re-run P4 steps 5 → 6 → 7 only** (steps 1–4 outputs are valid, no need to redo).
  Then verify `germline_denovo`/`somatic_denovo` VCFs are non-empty and the `normal` matrix
  col count ≈ germline_defined + UPV.

**DCIS chain `12270329..335` — HEALTHY ✅.** Uses `run_supplimentary_models.py` (not
sparcal_net); step 5 produced full output (dcis1 hom 143,770 / het 96,767 / no_var 449,510;
dcis2 similar), both "completed successfully". As of this check: steps 1–5 done, step 6
`filter_bams` (12270334) RUNNING, step 7 pending. Then run `8_final_snv_mat_dcis{1,2}.sh`.

**Lesson:** these per-step SLURM scripts swallow errors and exit 0. When chaining with
`afterok`, watch for jobs that "COMPLETE" in seconds (`sacct -j <id> --format=JobID,State,Elapsed`)
— that means the step no-op'd, not that it succeeded.

---

## 2026-06-24 — UMI dedup ablation (P6 rep1) + SpatialSNV/SPARCAL comparison

### Why
SPARCAL's mpileup calling does NOT do **UMI deduplication**, nor GATK
**SplitNCigarReads / BQSR**. UMI dedup is the field standard for 10x variant calling
(cellsnp-lite, vartrix, SpatialSNV all do it) → highest reviewer risk. Plan: a dedup
**ablation** on one pilot section (P6 rep1) to show whether the final variant set /
spatial categories move (turns "you skipped dedup" into a figure). SplitNCigar/BQSR
are lower-risk — defend with a splice-junction-distance figure + the "SparcalNet IS a
learned error model (subsumes BQSR)" argument.

### Terminology (post-dedup)
Each retained alignment = one **molecule**, not a read. In deduped outputs rename:
depth → **UMI count / molecular depth**; alt/ref → **alt/ref UMI count**; VAF →
**molecular VAF** (more accurate — not amplification-inflated).

### Done (P6 rep1)
- **BACKUP:** `data/P6_tumor/1` → `data/P6_tumor/1_pre_umidedup_2026-06-24` (rename).
- **NEW pre-calling step:** `run_slurm/P6_tumor/0_umidedup_split_P6.sh`
  - `umi_tools dedup --per-cell --cell-tag=CB --umi-tag=UB --method=directional`
    (umi_tools 1.1.6, SpaceTracer env) on possorted → `possorted.dedup.bam`
  - `samtools split -d CB` (spatialsnv env v1.23.1) → regenerates `split_BAM/{bc}.bam`.
    `split_BAM/` had been cleaned up; rebuilding it **deduped** means every downstream
    step that reads `split_BAM/` inherits dedup — no other code change. Split naming
    verified to match `get_bam_list_for_tumor` (`{barcode}-1.bam`).
- **Step 0 (dedup+split) DONE** (job 12133315, 2026-06-24/25):
  - **Dedup result: 84,253,988 → 51,485,809 reads = ~39% duplicates removed** (real PCR
    inflation; citable ablation number). `possorted_genome_bam.dedup.bam` = 4.1 G.
  - **🐞 split bug found+fixed:** `samtools split -d` defaults to **`-M 100`** → only 100 of
    ~4,986 cells got a BAM; the rest (3.6 G) went to `_nobarcode.bam`. Fixed with **`-M 6000`**
    (ulimit -n is 524288, no FD concern). Re-split the *existing* dedup.bam (no re-dedup) →
    **4,992 per-spot BAMs** in `…/outs/split_BAM/` (+ indexes). Script patched.
- **Submission order** (from repo root):
  1. ~~`sbatch run_slurm/P6_tumor/0_umidedup_split_P6.sh`~~ **DONE** (split_BAM ready, deduped)
  2. ~~`sbatch run_slurm/P6_tumor/1_mpileup_pipeline.sh`~~ **DONE 2026-06-25 (job 12145890)** —
     COMPLETED, exit 0:0, **5h10m**, MaxRSS ~39 GB. Loaded 3,650 in-tissue / 1,343 out-of-tissue
     BAMs OK (the `.gz` fix worked); 22/22 regions; `merged_sorted_gt.vcf.gz` = 28 MB, 1 sample.
     **Ablation data point (calling stage):** deduped **1,540,060** merged sites vs pre-dedup
     **1,512,275** (`1_pre_umidedup_2026-06-24/`) = **+27,785 (+1.8%)** — i.e. the merged call set
     is essentially unchanged despite ~39% read-duplication removal; the real dedup effect to watch
     is downstream (categories/matrices, steps 7–8). Live log: `slurm_output/P6_tumor/b13m20/…out1`
     (script's duplicate `--output` → SLURM writes `b13m20/…out1`/`…err1`, not `b0m0/`).
  3. ~~`sbatch run_slurm/P6_tumor/2_beagle_pipeline.sh`~~ **DONE 2026-06-25 (job 12152678)** —
     COMPLETED, exit 0:0, **7.5 min**, MaxRSS ~56 GB; 22/22 chromosomes, 0 failures.
     Output: `output_VCFs/beagle/baseQ0mapQ0/{chr*.vcf.gz ×22, all_filtered_in.vcf.gz,
     all_filtered_out.vcf.gz}`. **Ablation (beagle stage):** `filtered_in` (kept) **126,034**
     dedup vs **126,870** pre-dedup = **−836 (−0.7%)**; `filtered_out` 1,414,026 vs 1,385,405.
     → consistent with step 1: dedup barely moves the call set; watch for the effect downstream.
     Live log (script's last `--output`): `slurm_output/P6_tumor/baseQ13mapQ20_section2_beagle_pipeline_P6_tumor.out`.
  4. ~~`sbatch run_slurm/P6_tumor/3_beagle_genotype_shifting.sh`~~ **DONE 2026-06-25 (job 12152771)**
     — COMPLETED, exit 0:0, **2m40s**. (Note: this step does NOT rewrite `all_filtered_{in,out}.vcf.gz`
     — those are step-2/run_beagle outputs; it produces the **genotype-shift analysis** under
     `data/P6_tumor/1/metrics/beagle/baseQ0mapQ0/`: `P6_TUMOR_1_shifted_results.pkl`,
     `_stable_results.pkl`, `_shifted_detailed_counts.csv`, + plots — feeds step 4.)
  5. ~~`sbatch run_slurm/P6_tumor/4_seq_err_model.sh`~~ **DONE 2026-06-25 (job 12152848)** —
     COMPLETED, exit 0:0, **20 s**. (Edited the script: commented out the active `--section_id 2`
     line — sec2 N/A for the ablation.) **Ablation (seq-err stage):** of 1,414,026 variants,
     **188,760 (13.35%) flagged sequence errors**, 1,225,266 (86.65%) kept. Output:
     `output_VCFs/SeqErrModel/baseQ0mapQ0/{sequence_error,sequence_no_error}.vcf.gz` + summary.
  6. ~~`sbatch --array=1 run_slurm/P6_tumor/5_neural_network.sh`~~ **DONE 2026-06-25 (job 12152936_1)**
     — COMPLETED, exit 0:0, **4 min**, MaxRSS ~3.4 GB. Validation variant-only **F1 0.687**; applied
     to 1,225,266 variants. **Filtered (thr 0.5):** hom 1/1 **551,373**, het 0/1 **112,518**,
     no-variance 0/0 **478,398** → **663,891 true variants**. Output:
     `output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_{predictions,homozygous,heterozygous,no_variance}.vcf.gz` + `neural_network_model.pkl`.
  7. ~~`sbatch run_slurm/P6_tumor/6_single_bam_snp_filter.sh`~~ **DONE 2026-06-25 (job 12153130)** —
     COMPLETED, exit 0:0, **2h00m**, MaxRSS ~23 GB. **The previously-FAILED step now WORKS on the
     deduped `split_BAM/`.** SNV pool combined = **789,925** (defined 126,034 + denovo 663,891).
     Processed 4993 BAMs, 4993 filtered / 0 failed; **4991/4993 (99.96%) have detected SNVs**;
     783,565 unique detected variants. **NOTE the real output path is
     `output_VCFs/spotprofiles/baseQ0mapQ0/{bam_by_spot/ (4993), vcf_by_spot/ (4991)}`** +
     `all_detected_variants_summary.vcf.gz` / `all_variants.vcf.gz` — NOT the `BAM_filtered/…/snv_positions/`
     path in CLAUDE.md (stale for this script version). Harmless: per-chrom "fetch called on bamfile
     without index" errors are only for `_nobarcode.bam` (the catch-all bin, correctly skipped).
  8. ~~`sbatch run_slurm/P6_tumor/7_spatial_filter_n_matrix.sh`~~ **SUBMITTED 2026-06-25 (job 12156808)**
     — `run_spatial_snv_filter_enhanced.py` (clone+CNV integration via CalicoST P6_sec1: tumor-purity
     / clone_labels / cnv_seglevel — all verified present) → viz → `run_generate_matrix.py`
     (`--filter-subdir filtered_snvs --output-name normal`) → score-scatter plots. **Edited the script:
     loop `for SECTION_ID in 1` (was `1 2`)** — sec2 N/A. Outputs:
     `data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/{germline,denovo,somatic,...}` + matrices.
     Log `slurm_output/P6_TUMOR/baseQ0mapQ0/spatial_filter_n_matrix_P6.out`. Then `8_final_snv_mat.sh`.
     ⚠️ **VERIFY the P6 UPV matrix bug is gone** (new `germline_denovo`/UPV matrix cols ≈ its VCF
     record count, NOT ~1.25M; a `somatic` matrix should also be produced).

### ✅ Pre-existing step-1 blocker (P6 rep1) — RESOLVED 2026-06-25
Config `barcode_file = …GSM4565825_barcodes.tsv.gz` but disk had only the uncompressed
`GSM4565825_barcodes.tsv` → step 1 built an empty BAM list. **Fixed:** created
`…/Meta_Data/GSM4565825_barcodes.tsv.gz` via `gzip -c … > ….gz` (env `gzip` has no `-k`;
original kept). Verified 3,650 barcodes in both. Step 1 then launched (job 12145890).

### Runtime
63.2M mapped reads. dedup (single-threaded) is the bottleneck ~4–10 h; split+index ~1 h
→ ~5–12 h in the 48 h window. Watch rate in `…/outs/umidedup.log`; the post-dedup read
count printed in the `.out` = duplication rate (an ablation number itself). Fallback if
too slow: chr-sharded dedup (22 parallel, merge).

### ⚠️ P6 `germline_denovo` (UPV) MATRIX is BROKEN — verify on re-run (found 2026-06-25)
While calibrating SpatialSNV-vs-SPARCAL magnitudes, found the P6 UPV **matrix** is corrupt:
- `P6_tumor/1_pre_umidedup_2026-06-24/final_matrices/baseQ0mapQ0/germline_denovo_spot_snv_matrix.pkl`
  has **1,253,838 columns**, but the categorized UPV **VCF**
  (`spatial_filter_purity/baseQ0mapQ0/germline/denovo/germline_denovo.vcf.gz`) has only **291**
  variants (matches step-7c "P6 → 0/291"). The matrix ≈ the raw merged set (~1.3M) → built from
  the wrong input at step 7b/8.
- P6 `germline_defined` matrix (106,884) is correct; P6 `somatic` matrix is **missing**.
- P4 matrices all match their VCFs (gd 37,985 / UPV 7,215 / som 4,924) — **P4 is fine, only P6 is bad**.
- True P6 SPARCAL total ≈ **120,702** (106,884 + 291 + 13,527), comparable to SpatialSNV ~92k —
  the earlier "P6 SPARCAL = 1.36M, 15× larger" was an artifact of this broken matrix.
- **ACTION:** when the P6 dedup pipeline re-runs steps **7b/8** (matrix generation), **verify the
  new P6 `germline_denovo` matrix has ~291 columns, NOT ~1.25M**, and that a `somatic` matrix is
  produced. Do NOT use the stale matrix for any benchmark in the meantime.

### TODO — Monopogen spot×SNV matrix (missing piece of the 3-way benchmark)
**Status: CONVERTER BUILT 2026-06-25, not yet run at scale.**
- **Script:** `SPARCAL_Benchmarking/monopogen_to_spot_matrix.py` (+ runner
  `SPARCAL_Benchmarking/4_monopogen_to_pkl.slurm`, array 0-3 for the 4 samples).
- **Design:** Monopogen `out/germline/merged.vcf.gz` is **pseudobulk single-sample**
  (one col `possorted_genome_bam`, all PASS, hg19 chr-prefixed) → no per-spot info, so
  we use the **strelka2 precedent**: pysam pileup of the CB-tagged possorted BAM at
  Monopogen PASS-SNV positions, **allele-aware** (≥`--min-alt-reads` ALT-base reads ⇒
  present, default 1). Writes the `.pkl` directly in the SpatialSNV/SPARCAL contract
  (uint8, rows=in-tissue `AAAC…-1`, cols 4-part no-`chr` `{chrom}_{pos}_{ref}_{alt}`).
- **Inputs verified:** all 4 Monopogen VCFs + the 4 hg19 possorted BAMs (CB tags carry
  `-1`) + `tissue_positions.csv` (NEW spaceranger format — **has a header**, unlike
  DLPFC's headerless `tissue_positions_list.csv`; loader skips it). P4_rep1: 51,567 SNV
  alleles (1,143 indels skipped), 750 in-tissue spots. **chr22 smoke test passed**
  (617 SNVs × 750 spots, cols `22_17074104_G_A`).
- **Row-set note:** rows = ALL 750 in-tissue spots; SPARCAL P4 matrix has 744 → intersect
  to the shared spot set before the 3-way compare (same as strelka2/GATK row caveat).
- **NEXT:** `cd SPARCAL_Benchmarking && sbatch 4_monopogen_to_pkl.slurm` (~13 min/sample
  like the original SNP-profile scan), then it joins SpatialSNV + SPARCAL in P4/P6.
- We have Monopogen **VCFs** but the matrix below is the contract to fill.
- **Monopogen VCFs** (collaborator CanLuo): `/lfs/…/CanLuo/ST_SNV/Monopogen/{P4,P6}_rep{1,2}/out/
  germline/merged.vcf.gz`. Counts: P4_rep1 52,710 / P4_rep2 49,957 / P6_rep1 103,013 /
  P6_rep2 111,121. **No DCIS** (Monopogen never run on DCIS → 3-way compare is P4/P6 only).
- Only VCF-level overlaps exist (`run_slurm/overlap/overlap_Monopogen_SPARCAL/`, venn PNGs) — not matrices.

**How we converted before / the standard:**
- For **SpatialSNV** we used `SPARCAL_Benchmarking/callback_to_pkl.py` — but that consumed
  SpatialSNV's **CallBack MTX**. Monopogen has no CallBack; it's a plain VCF.
- The **standard for turning a VCF-based caller into the benchmark matrix** is the **strelka2
  precedent**: scan the merged BAM by `CB` tag at the caller's PASS-SNV positions (allele-aware:
  ≥1 read carrying the ALT base ⇒ "present" in that spot) → per-spot `<barcode>.txt` →
  `run_generate_matrix.py` → spots×SNV `.pkl`.
- **Reference scripts:** `scripts/tools/strelka2_to_spot_snvs.py` (the BAM-scan-by-CB step) +
  `scripts/6_spatial_filter/run_generate_matrix.py` (the `.pkl` builder). See the strelka2
  spot-matrix array in this file ("Strelka2 spot×SNV matrix" section) for the exact pattern.
- **Contract to match:** uint8 binary, rows=barcodes (`AAAC…-1`), cols=`{chrom}_{pos}_{ref}_{alt}`
  (no `chr`), spots×SNVs — same as `…/final_matrices/…/*_spot_snv_matrix.pkl` and the SpatialSNV
  pkls. (Reuse `callback_to_pkl.py`'s `chr`-strip + 4-part-key conventions for consistency.)
- Build per section from the Monopogen VCF + the (deduped, once available) merged/per-spot BAM,
  then it joins SpatialSNV + SPARCAL in the P4/P6 comparison. **(Conversion to be done in a
  separate session.)**

### SpatialSNV vs SPARCAL (the comparison)
| | **SpatialSNV** | **SPARCAL (ours)** |
|---|---|---|
| Type | somatic **caller** | spatial variant **interpretation** framework |
| Calling | GATK **Mutect2** (reassembly, PON, somatic LL, FilterMutectCalls) | mpileup/bcftools + **SparcalNet** (learned 3-class genotype/error NN) |
| Preprocessing | UMI dedup + SplitNCigar + BQSR | none (this ablation tests adding dedup) |
| Germline/somatic | gnomAD/PON priors only; one pseudobulk set | **spatial partition** germline / UPV / somatic |
| Clone/CNV | none | **CalicoST** CNV/LOH + clone labels |
| Phasing | none | Beagle/Eagle vs 1000G |
| Output | pseudobulk VCF + spot alt/ref/depth MTX | 3 spatial per-spot variant sets + matrices |

**Missing vs them (front-end):** somatic-grade caller, UMI dedup, SplitNCigar/BQSR.
**Our novelty:** SparcalNet (learned error model), spatial germline/UPV/somatic
categorization, CalicoST clone/CNV linkage, phasing/imputation, spatial-domain ARI
validation. Framing: *their "model" = off-the-shelf Mutect2 stats; ours = SparcalNet +
spatial model.* SpatialSNV benchmark tracker: `SpatialSNV/On_going.md`.

---

## Planned method additions (TODO — not yet implemented)

Came out of the variant-category naming discussion (2026-06-02). Both refine the
spatial-filter outputs. Design notes + paper wording: [pipeline_intro.md](pipeline_intro.md) §7–§8.
Category naming locked: **germline / UPV (Ubiquitous Private Variants) / somatic**
(was `germline_defined` / `germline_denovo` / `somatic_denovo`; code rename deferred).

- [~] **BAF-GMM sub-filter inside the UPV set — step 7c DRAFTED (2026-06-02).**
  Script: `scripts/6_spatial_filter/upv_baf_gmm_subfilter.py`. Runners:
  `run_slurm/{DCIS/7c_upv_baf_gmm.sh (dcis1+dcis2), P4_tumor/7c_upv_baf_gmm.sh,
  P6_tumor/7c_upv_baf_gmm.sh}`. Also wired into the enhanced filter via
  `run_spatial_snv_filter_enhanced.py --run_baf_gmm_subfilter` (subprocess hook,
  non-fatal on failure). Non-destructive: reads `germline_denovo.vcf.gz`
  (PURITY_CORR) + `merged_sorted_gt.vcf.gz`, fits a 2-D GMM on `[BAF, PURITY_CORR]`,
  writes `germline/denovo/gmm_subfilter/{upv_germline_like,upv_somatic_candidate}.vcf.gz`
  + TSV + `upv_baf_gmm.png` + summary. Ran on all 4 (corrected BAF):
  dcis1 → 1467 somatic-candidate / 2618; dcis2 → 0 / 3584; P4 → 0 / 7215; P6 → 0 / 291.
  - **BUG FOUND + FIXED — BAF field was 0 for all high-depth variants.**
    `merged_sorted_gt.vcf.gz` FORMAT `BAF` is 0 for every site whose I16 contains
    scientific notation (high depth) — root cause: `parse_i16` in
    `scripts/1_calling/mpileup_pipeline.py` used `int(x)` over all 16 I16 values;
    `int("1.38392e+07")` throws → silent `[0]*16` fallback → BAF=0 → falsely flagged
    `DiscordantBAF` (but kept). dcis1: exactly 1344/1,058,191 merged variants, 100%
    coincident with sci-notation I16; in the UPV set 340/4085 (mostly hom-ALT germline,
    true alt-frac ≈1.0). **Fix:** `parse_i16` now `int(float(x))`
    (fixes future step-1 runs). GT unaffected (from PL); variant set unaffected.
    Same latent bug in `mpileup_pipeline_old.py` + `all_caller_pipeline.py` (not fixed).
    **7c works around it now** by recomputing BAF = alt/(ref+alt) from I16 in
    `lookup_baf` — so 7c is correct without re-running step 1.
    **OPEN:** existing merged VCFs (all datasets) still carry the wrong BAF; the
    seq-error model consumed it. Decide whether a step-1 re-run is worth it (only
    0.1% of variants, the high-depth ones; they're kept regardless).
  - **PURITY_CORR is uninformative within UPV (unchanged finding).** Clipped to ≥0
    and flat (UPV are ubiquitous by definition → presence-vs-purity saturated). 2-D
    GMM degenerates to BAF-only. P4/P6/dcis2 have no low-BAF mode → 0 (conservative).
  - **HARD BAF ceiling added (`--somatic-baf-max`, default 0.35).** The soft GMM
    component (mean 0.319) has wide variance and spilled past 0.5, putting germline-
    het (BAF≈0.5) variants in the somatic set. The hard per-variant gate
    `somatic ⇔ (GMM somatic posterior > 0.5) AND (BAF < 0.35)` excludes the het/hom
    modes. dcis1: 1467 → **739** somatic-candidate, BAF range now 0.108–0.349
    (median 0.25); dcis2/P4/P6 still 0. Plot draws the cap line.
  - **ΔBAF experiment DONE (2026-06-02) — NEGATIVE result.**
    `scripts/6_spatial_filter/upv_delta_baf_experiment.py`. Per-spot read counts are
    NOT in any VCF: `vcf_by_spot` records **coverage-presence only** (step 6
    `filter_bam_one_chrom` marks a position "detected" on read overlap, never checks
    the base; sample=`./.`, INFO copied from merged pseudobulk → identical I16 across
    spots). So ΔBAF requires the reads: scanned the CB-tagged `possorted_genome_bam.bam`,
    pooled ref/alt per UPV position into tumor vs normal spot groups (tumor_proportion
    tertiles). dcis1, 1753 variants w/ ≥5 reads each side: ΔBAF symmetric around 0
    (q5/q95 = −0.33/+0.34, median 0); current BAF<0.35 somatic-candidates average
    ΔBAF = **−0.032** (NOT tumor-enriched). ⇒ **No tumor-associated somatic population
    inside UPV**; the low-BAF UPV variants are tumor-independent = ASE-skewed germline.
    Outputs: `gmm_subfilter/upv_delta_baf.{tsv,png}`.
    Caveats on the negative: tumor/normal spots are mixtures (tumor≥0.66, normal≤0.39
    purity → diluted contrast), per-spot reads sparse, CalicoST purity inferred — but
    the signal is flat enough that a strong somatic population would still show.
  - **CALL:** treat UPV as a germline-dominated mixed bucket; do NOT force a somatic
    sub-call. 7c (BAF-GMM) stays exploratory/supplementary, not a reportable somatic set.
  - **Step 6 per-spot `AD` — DRAFTED + unit-validated (2026-06-02).**
    `scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py` now tallies real per-spot
    ref/alt read counts while scanning each barcode BAM and writes
    `FORMAT=GT:AD:DP` (was `./.`) into `vcf_by_spot/{barcode}.vcf.gz`.
    `filter_bam_one_chrom(... pos_to_info)` returns `allele_counts`; aggregated in
    `filter_bam_by_positions` → written in `save_detected_snvs`. AD verified equal to an
    independent pysam pileup at every tested position. **Two behavior changes to know
    before re-running step 6:** (1) detection no longer `break`s at the leftmost SNV a
    read covers — every covered position is now recorded (fixes under-detection in dense
    regions, so per-spot variant counts may rise); (2) sample is real `GT:AD:DP`, with
    `./.` only where a read spans the locus but has no aligned base (deletion/clip).
    Perf: adds one `get_reference_positions` per kept read (slower; still parallel/30-worker).
    **Not yet re-run** — re-running regenerates all `vcf_by_spot` VCFs (and any matrix
    built from them). Payoff: ΔBAF + an allele-aware spot×SNV matrix (use ALT-count≥1 as
    presence instead of coverage) become computable straight from the VCFs, no BAM re-scan.
    `final_snv_mat.py`/`run_generate_matrix.py` can be updated to consume `AD` when desired.
  - **Candidate effective features (the principle: anything that does NOT depend on
    spatial presence-distribution, which UPV saturates).** Ranked:
    (1) **gnomAD/dbSNP annotation** — cheapest, highest value; peels off the
    rare-germline-not-in-1000G contaminant directly (UPV = "not in 1KGP", but gnomAD
    is far larger). (2) **per-clone ΔBAF** (read-level VAF, tumor vs normal spots).
    (3) **CNV/LOH allelic-imbalance consistency** vs CalicoST per-clone allele-specific
    CN. (4) **phasing consistency** (germline co-segregates with a Beagle/Eagle
    haplotype; somatic breaks phase). (5) mutational trinucleotide context / SBS
    signature (weak per-variant). BAF alone is necessary-not-sufficient (ASE confound).
  - Eventually fold the chosen version into Stage 1 of `run_spatial_snv_filter_enhanced.py`.
- [ ] **CHIP rule-out on the final somatic set.** Drop variants in common CHIP
  genes (DNMT3A, TET2, ASXL1, JAK2, TP53, SF3B1, …) from category (c) — likely
  clonal-hematopoiesis blood contamination. Implement as a **post-processing**
  filter on the somatic VCF (gene-list intersection), not in the core cascade.
- [ ] **Code rename** (deferred): `germline_defined→germline`,
  `germline_denovo→upv`, `somatic_denovo→somatic` across
  `run_spatial_snv_filter_enhanced.py` + `final_snv_mat.py` (keep 3-way output).

---

## SPATIAL_SNV benchmark clustering (post-processing — added 2026-06-09)

**Post-processing step: spatial-clustering benchmark of all callers.** Cluster each
caller's spot×SNV matrix (+ a gene-expression baseline) with STAGATE→mclust and score
spatial domains against the DLPFC layer annotations (ARI). This is the downstream
comparison that consumes the per-caller matrices.

Files in `SPATIAL_SNV/`:
- `clustering_config.json` — config (sections, modality→caller/filter/grouping, paths, model params)
- `SPARCAL_clustering.py` — main script: load matrix → STAGATE → mclust → ARI, saves outputs (PNG **+ editable PDF**)
- `run_clustering.slurm` — array job; `--array=0-0` (151507) / `--array=0-11` (all 12)
- `make_ari_boxplot.py` — aggregate every section's `summary.csv` → 4-modality ARI box plot (PNG+PDF) + `ari_table.csv`/`ari_matrix.csv`
- `make_combined_figure.py` — per-section concatenated figure: row1 = spatial domains (GT + each modality, colors Hungarian-matched to GT layers), row2 = per-modality UMAP colored by true layer (PNG+PDF)

**Four modalities** (config `modalities`): `sparcal` (bcftools/normal), `strelka2` (germline),
`gatk` (germline), `gene_expr` (Visium h5 baseline). gene_expr loads the SpaceRanger
`{section}_filtered_feature_bc_matrix.h5` directly (no pkl) via `load_gene_expr_section` —
same `normalize_total+log1p`, no HVG, so all 4 share identical STAGATE preprocessing.

**Outputs:**
- `data/dlpfc/{section}/clustering/{modality}/` — embedding.npy, cluster_labels.csv, ari.txt, umap.{png,pdf}, spatial.{png,pdf}
- `data/dlpfc/{section}/clustering/summary.csv` — per-section ARI table
- `data/dlpfc/{section}/clustering/combined_{section}.{png,pdf}` — concatenated figure
- `data/dlpfc/clustering_benchmark/ari_boxplot.{png,pdf}` + `ari_table.csv` + `ari_matrix.csv`

**Run (from repo root; then activate snv_clustering for the figure scripts):**
```bash
sbatch --array=0-11 SPATIAL_SNV/run_clustering.slurm     # 12 sections × 4 modalities
source activate snv_clustering
python SPATIAL_SNV/make_ari_boxplot.py
python SPATIAL_SNV/make_combined_figure.py --section_id 151507
```

**GPU (2026-06-09):** `run_clustering.slurm` runs on `batch_gpu` / `maiziezhou_lab_acc`
with `--gres=gpu:nvidia_rtx_a6000:1`. `train_STAGATE` auto-uses cuda:0 (no code change).
A6000 ≈ 14 epochs/s vs 8 s/epoch on CPU (~110×); ~5 min/section for all 4 modalities.
**Constraint:** our account is authorized for 2080_ti/titan_x/quadro/a6000, but only the
**a6000 still has live nodes** (others decommissioned) and the a6000 cap is **2 concurrent
GPUs** — so a 12-section array runs ~2 at a time. (Stale Slurm quota: 51 on the dead
2080_ti, 2 on the live a6000.) Possible ask to ACCRE/PI: rebalance the quota toward a6000,
or grant access to the idle l40s/a100 nodes.

**Bugs fixed during setup (2026-06-09):**
- `STAGATE_pyG/utils.py` `Transfer_pytorch_Data`: `torch.Tensor(lbl)` crashed on string labels — LabelEncode first.
- `STAGATE_pyG/gat_conv.py`: `NoneType` removed from `torch_geometric.typing` (PyG 2.x) — added `NoneType = type(None)` locally.
- `SPARCAL_clustering.py` numpy shim: pkl serialized with NumPy 2.x (snv_caller); env has 1.24.4 — register `numpy._core` alias for `numpy.core` before unpickling.
- **`mclust_R` rpy2 3.6.x regression (fixed 2026-06-09).** Job "completed" but ARI=NaN for
  all modalities. Two stacked breaks vs the older cluster: (1) `numpy2ri.activate()` is
  removed in rpy2 3.6 (now raises "activate and deactivate are deprecated"); (2) passing the
  embedding positionally into `Mclust` throws `length of 'dimnames' [2] not equal to array
  extent` under rpy2 3.6.4 / mclust 6.1.2. **Fix:** push the embedding into R's global env
  under a `localconverter` context, then call `Mclust(emb_mat, G=k, modelNames="EEE")` from an
  R string. Verified. Not a logic bug — an env/version regression (will bite other scripts
  that still call mclust the old way: the notebooks, `classification_codepart.py`, `st_loading_utils.py`).

**Environment:** `snv_clustering` (Python 3.10, torch 2.2.2+cu121, PyG 2.5.2, scanpy 1.11.4,
rpy2 3.6.4, R 4.5.1 mclust 6.1.2). `STAGATE_pyG` via PYTHONPATH (setup.py refs missing README.rst).

**Status (2026-06-09):** pipeline working end-to-end on **151507** (job 11731374, ~5 min on
a6000) — ARI **sparcal 0.304 > gatk 0.184 > strelka2 0.128**. gene_expr not yet clustered.
**NEXT:** resubmit `--array=0-11` (now 4 modalities) → run the two figure scripts for the
full 12-section boxplot + combined figures. Caveat: filtering differs per caller (only GATK
has the 1000G filter) — note in methods.

---

## Active SLURM Jobs

### Step 6 re-run with per-spot AD fix — submitted 2026-06-02

Re-running `6_single_bam_snp_filter.sh` to regenerate `vcf_by_spot/` with real
`GT:AD:DP` (the AD correction + leftmost-`break` under-detection fix in
`run_filter_bams_by_snv_pools.py`).

| Dataset | Section | Job | Status |
|---------|---------|-----|--------|
| DCIS | 1 + 2 | 11548544 (`filter_bams_DCIS`) | RUNNING (cn1817) |
| P4_TUMOR | 1 | 11548546 (`filter_bams_P4_TUMOR`) | **FAILED** |
| P6_TUMOR | 1 | (`filter_bams_P6_TUMOR`) | **FAILED** |

**P4/P6 failure = missing input, NOT the AD code.** `run_filter_bams_by_snv_pools.py`
→ `FileNotFoundError: No BAM files found at .../P{4,6}_Tumor_output/outs/split_BAM/*.bam`
(the bash loop still prints "All replicates processed", so it looks done in seconds).
Diagnosis: the `outs/` dir EXISTS and still has `possorted_genome_bam.bam` (+.bai), but
the **`split_BAM/` subdir is gone** — the per-barcode BAMs were cleaned up after the
original Feb-2026 run. **To unblock P4/P6:** re-run **step 0** (`0_split_bam.sh`,
splits possorted BAM by CB tag + barcode list) to regenerate `split_BAM/`, THEN re-run
step 6. (Alternative, bigger: rework step 6 to scan the possorted BAM by CB tag directly
— like the ΔBAF experiment — and skip storing thousands of per-barcode BAMs.) DCIS
unaffected (its `split_BAM/` exists).

> 🔔 **REMIND THE USER next session:** P4 sec1 and P6 sec1 have **no `split_BAM/`**
> (only the possorted BAM survives). If this is still not resolved when you next read
> this, tell the user to **ask their colleagues** (who own/produced the P4/P6 Visium
> data) to restore or regenerate the per-barcode split BAMs at
> `/lfs/.../STmut_Data/P{4,6}_Visium/spaceranger_align_rep1_hg19/P{4,6}_Tumor_output/outs/split_BAM/`.
> Until then P4/P6 step 6 (and the whole AD/UPV re-run chain) stays blocked.

> ⚠️ **DOWNSTREAM RE-RUN REQUIRED after step 6 finishes (read this before using any
> result for the affected sections).** The step-6 re-run changes `vcf_by_spot/` (real
> AD + more detected positions), so everything built from it is now **stale** and must
> be re-run in order:
> 1. **Step 7** spatial filter (`7_spatial_filter_n_matrix.sh` → `run_spatial_snv_filter_enhanced.py`)
>    — regenerates the UPV (`germline_denovo`) / germline / somatic per-spot sets.
> 2. **Step 8** final matrices (`8_final_snv_mat_*.sh` → `final_snv_mat.py`).
> 3. **7c** UPV BAF-GMM (`7c_upv_baf_gmm.sh`) — depends on the new step-7 UPV set.
> Also: any benchmark/comparison matrix and the ΔBAF experiment derive from these.
> Currently only **DCIS** will have fresh step-6 output; P4/P6 are blocked (see above).



### Strelka2 germline array — RESOLVED 2026-06-01 (bioconda 2.9.10 rebuild)

Script: `strelka2/run_slurm/strelka2_germline_dlpfc.sh`
**Root cause (finally):** the prebuilt `strelka-2.9.2.centos6_x86_64` binary is broken against
the current OS (fails even on the login node with a 5 KB demo file — NOT compute-specific, NOT
a path issue). All the realpath/panfs/`/tmp`/`/dev/shm` work was a phantom. Fixed by rebuilding
via bioconda: `strelka=2.9.10=hdfd78af_2` (see `strelka2/scripts/install_strelka_conda.sh`,
which also re-applies the Bug 1 SMTP-timeout patch). Runner now resolves the configure script
via `$STRELKA_CONFIG`/PATH. Full details: `strelka2/DEBUGGING.md` (2026-06-01 entry).

**ALL 12 SECTIONS — DONE** (job 11528358_0 + array 11528559_1..11, all COMPLETED exit 0,
~11 min each). Validated 2026-06-01 via `strelka2/scripts/validate_strelka2_outputs.sh`:
all 12 `variants.vcf.gz` integrity-OK, PASS-SNV 58k–88k/section, all 24 GRCh38 contigs.
Output: `data/dlpfc/{section}/strelka2/results/variants/{genome,variants}.vcf.gz`.

---

### Strelka2 spot×SNV matrix (cross-tool comparison) — job array 11534438 (2026-06-01) — IN PROGRESS

Script: `scripts/tools/strelka2_spot_matrix_dlpfc.sh` (array 0-11). Two steps/section:
1. `scripts/tools/strelka2_to_spot_snvs.py` — scan merged BAM by CB tag at strelka2 PASS-SNV
   positions (allele-aware: ≥1 read carrying the ALT base = present), PASS SNVs only →
   one `<barcode>.txt` per in-tissue spot in `data/dlpfc/{section}/strelka2/spot_snvs/`.
2. `scripts/6_spatial_filter/run_generate_matrix.py --caller strelka2 --output-name germline` →
   `data/dlpfc/{section}/matrix/DLPFC_{section}_strelka2_germline_6_matrix.pkl`.
Resources: 22 CPU / 64 GB / 4 h. Actual: ~34–49 min, ~6–7 GB RSS per section.
Rationale/design: see [[project-strelka2-matrix-comparison]] + this session's notes. The merged
BAM carries CB:Z: tags so the single-BAM scan works (no per-cell BAM iteration).

| Task | Section | Status | Matrix (spots × SNVs) |
|------|---------|--------|------------------------|
| 0 | 151507 | DONE | 4226 × 58,979 |
| 1 | 151508 | DONE | 4384 × 53,032 |
| 2 | 151509 | DONE | 4789 × 60,951 |
| 3 | 151510 | DONE | 4634 × 57,929 |
| 4 | 151669 | DONE | 3661 × 58,319 |
| 5 | 151670 | RUNNING (cn1815) | — |
| 6 | 151671 | RUNNING (cn1816) | — |
| 7 | 151672 | RUNNING (cn1817) | — |
| 8 | 151673 | RUNNING (cn1815) | — |
| 9 | 151674 | RUNNING (cn1816) | — |
| 10 | 151675 | PENDING (QOSGrpCpuLimit) | — |
| 11 | 151676 | PENDING (QOSGrpCpuLimit) | — |

Matrices validated (tasks 0-4): binary int8 {0,1}, rows=barcodes, cols=`chrom_pos` (same key
format as pipeline matrices → directly comparable), spots = in-tissue counts, ~82% of PASS SNVs
observed in ≥1 spot, median 84–149 SNVs/spot. Logs: `strelka2/slurm_output/spot_matrix_{0..11}-*.{out,err}`.

**Next (optional):** head-to-head vs pipeline's `DLPFC_{section}_bcftools_normal_6_matrix.pkl`
(shared-spot count, SNV-column Jaccard).

---

### DLPFC Stage 7+7b rerun — job array 11333398 (submitted 2026-05-26) — COMPLETED WITH PARTIAL FAILURE

Script: `run_slurm/dlpfc/rerun_stage7_DLPFC.sh`
Reason: Stage 7 in job 11288684 used wrong script (`run_spatial_snv_filter.py` reading from empty `BAM_filtered/snv_vcf/`). Now uses `run_spatial_snv_filter_enhanced.py` which reads from `spotprofiles/baseQ0mapQ0/vcf_by_spot/` (same as P4/DCIS).
Dataset: DLPFC · all 12 sections · `baseQ0mapQ0` · 4 h / 4 CPU / 64 GB per task
`--tumor_purity_file` made optional in enhanced script (normal tissue → all spots default to purity 0.0).

| Task | Section | Stage 7 | Stage 7b | Notes |
|------|---------|---------|----------|-------|
| 0 | 151507 | DONE | FAILED | `run_generate_matrix.py` → `FileNotFoundError: spatial_analysis/baseQ0mapQ0/filtered_snvs` missing |
| 1 | 151508 | DONE | FAILED | same as 151507 |
| 2 | 151509 | DONE | FAILED | same as 151507 |
| 3 | 151510 | DONE | DONE | |
| 4 | 151669 | DONE | FAILED | same as 151507 |
| 5 | 151670 | DONE | FAILED | same as 151507 |
| 6 | 151671 | DONE | DONE | |
| 7 | 151672 | DONE | DONE | |
| 8 | 151673 | DONE | DONE | |
| 9 | 151674 | DONE | DONE | |
| 10 | 151675 | DONE | DONE | |
| 11 | 151676 | DONE | DONE | |

**Stage 7b failure root cause:** `run_generate_matrix.py` reads from `spatial_analysis/baseQ0mapQ0/filtered_snvs` (old path), empty for these 5 sections. Stage 7 (enhanced) already wrote per-barcode `.txt` to `spatial_filter_purity/baseQ0mapQ0/germline/` ✓.

**Fix applied (2026-05-27):**
- `run_generate_matrix.py`: added `--input-dir` flag (bypasses old path construction); added header-line skip (`pos.isdigit()` guard).
- New rerun script: `run_slurm/dlpfc/rerun_stage7b_DLPFC.sh` — **array 0-11 for all 12 sections**, reads from `spatial_filter_purity/{QUALITY_FILTER}/germline/`. Covers all 12 for consistency (7 already had old-pipeline matrices; those will be overwritten with enhanced-pipeline output).
- **Ready to submit:** `cd /panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling && bash run_slurm/dlpfc/rerun_stage7b_DLPFC.sh`

**Logs:** `slurm_output/DLPFC/baseQ0mapQ0/stage7_{0..11}.{out,err}`

---

### Strelka2 germline array — all prior attempts FAILED (see strelka2/DEBUGGING.md)

Jobs 11272103, 11287046, 11287211, 11287522, etc. — all 12 sections, all failed.
Three bugs identified and fixed (SMTP hang, panfs path resolution, shutil.copy2 truncation).
Superseded by jobs 11390244 + 11390247 above.

## Completed Tasks

| Job ID | Script | Config | Completed | Notes |
|--------|--------|--------|-----------|-------|
| 11360329 | `rerun_stage7b_DLPFC.sh` | DLPFC all 12 sections · `baseQ0mapQ0` | 2026-05-27 | Stage 7b ✓ all 12 sections. Matrices in `data/dlpfc/{section}/matrix/DLPFC_{section}_bcftools_normal_6_matrix.pkl`. Input: `spatial_filter_purity/baseQ0mapQ0/germline/` (enhanced pipeline output). |
|--------|--------|--------|-----------|-------|
| 11288684 | `run_pipeline_DLPFC.sh` | DLPFC all 12 sections · `baseQ0mapQ0` | 2026-05-25/26 | Stages 1–6 ✓ all sections. Stage 7 used wrong script (bug). Stage 7b: 5 sections failed (151507/08/09/151669/70 — no pre-existing filtered_snvs), 7 succeeded via old Jul-2025 data. Rerun as job 11333398. |
| 11065771 | `vcf_visualizer.sh` | DCIS sec1 · germline_denovo · exome BED | 2026-05-18 | exit 0 · 1h09m |
| 11065872 | `vcf_visualizer.sh` | DCIS sec1 · germline_denovo · exome BED | 2026-05-18 | exit 0 · 1h05m |
| 11066319_1 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec2 · germline denovo | 2026-05-18 | exit 0 · 25s · 176 unique SNVs (exome) |
| 11066319_2 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec2 · somatic denovo | 2026-05-18 | exit 0 · 33s · 837 unique SNVs (exome) |
| 11066319_3 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec2 · germline defined | 2026-05-18 | exit 0 · 36s · 4,716 unique SNVs (exome) |
| 11066713_1 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec1 · germline denovo | 2026-05-18 | exit 0 · 10s · 242 unique SNVs |
| 11066713_2 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec1 · somatic denovo | 2026-05-18 | exit 0 · 3s · 639 unique SNVs |
| 11066713_3 | `vcf_visualizer_per_barcode_presented.sh` | DCIS sec1 · germline defined | 2026-05-18 | exit 0 · 12s · 5,182 unique SNVs |
| 11066715_1 | `vcf_visualizer_per_barcode_presented.sh` | P4 sec1 · germline denovo | 2026-05-18 | exit 0 · 7s · 479 unique SNVs |
| 11066715_2 | `vcf_visualizer_per_barcode_presented.sh` | P4 sec1 · somatic denovo | 2026-05-18 | exit 0 · 7s · 545 unique SNVs |
| 11066715_3 | `vcf_visualizer_per_barcode_presented.sh` | P4 sec1 · germline defined | 2026-05-18 | exit 0 · 9s |
| 11066716_1 | `vcf_visualizer_per_barcode_presented.sh` | P6 sec1 · germline denovo | 2026-05-18 | exit 0 · 10s · 106 unique SNVs |
| 11066716_2 | `vcf_visualizer_per_barcode_presented.sh` | P6 sec1 · somatic denovo | 2026-05-18 | exit 0 · 10s · 536 unique SNVs |
| 11066716_3 | `vcf_visualizer_per_barcode_presented.sh` | P6 sec1 · germline defined | 2026-05-18 | exit 0 · 7s |

### Output files — per_barcode_visualizer (all datasets, sec1)

**DCIS sec1** (`data/dcis1/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 242 SNVs, median 12/spot, 1434/1454 spots covered
- `germline/per_barcode_visualizer/germline_defined_*` — 5,182 SNVs, median 82/spot
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 639 SNVs, median 2/spot, 1040/1454 spots covered

**P4 tumor sec1** (`data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 479 SNVs, median 228/spot (743/744 spots)
- `germline/per_barcode_visualizer/germline_defined_*` — 3,377 SNVs, median 132/spot (741/744 spots)
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 545 SNVs, median 3/spot (574/744 spots)

**P6 tumor sec1** (`data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 106 SNVs, median 16/spot (3605/3650 spots)
- `germline/per_barcode_visualizer/germline_defined_*` — 3,251 SNVs, median 15/spot (3637/3650 spots)
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 536 SNVs, median 0/spot (1609/3650 spots)

**DCIS sec2** (`data/dcis2/spatial_filter_purity/baseQ0mapQ0/`)
- `germline/per_barcode_visualizer/germline_denovo_*` — 176 SNVs, median 18/spot (1803/1807 spots)
- `germline/per_barcode_visualizer/germline_defined_*` — 4,716 SNVs, median 77/spot (1803/1807 spots)
- `somatic/per_barcode_visualizer/somatic_denovo_*` — 837 SNVs, median 2/spot (1371/1807 spots)
