# SPARCAL model freeze v1 (2026-10-20)

One rerun of every tumor section through the frozen model, written to a new root
`data/freeze_v1/<dataset>/<section>/`. Legacy trees (`data/P4_tumor`, `data/P6_tumor`,
`data/dcis{1,2}`, `data/ovar_p5`) are not modified. Paper figures are regenerated from this root.

## Why a full rerun

The production spatial filter requires a step-6 spot-profile bundle with `MANIFEST.json` +
`RUN_COMPLETE` certifying the frozen Beagle router (`p_nonref >= 0.90`). No existing tumor bundle
has one (e.g. P6 rep1 spot profiles date from 2026-06-25, before the 2026-09-16 Beagle freeze), so
job 14715033 (P6 CNV gate) failed at validation. Manifests were deliberately not back-filled.

## What is frozen

| Component | Setting | Source |
|---|---|---|
| Beagle | Beagle 4.1, `gl=`, modelscale 2 / niterations 0 / maxlr 5000, seed 20260915, hard anchor `GP(0/1)+GP(1/1) >= 0.90` | `data/releases/beagle_v2_3_default_20260916_v1/DECISION.md` |
| Classifier | SparcalNet = `run_supplimentary_models.py --model-type neural_network` (MLP 64,32) | step 5 |
| Presence | **G1**: a spot is 1 where step 6 wrote a record, at loci with >= 1 ALT **molecule** (UB-collapsed, S6+) somewhere in the section | `apply_presence_rule` in `run_spatial_snv_filter_enhanced.py` (user decision 2026-10-19) |
| CalicoST gate | CNV default-on, auto-off unless a diploid reference clone and a >=5%-altered tumor clone survive; purity opt-in | `scripts/6_spatial_filter/calicost_gate.py`; verdicts in `data/calicost_gate_2026-10-19/` |
| CNV feature (theta) | depth-stratified Mantel–Haenszel log OR, tumor vs reference clone | `--cnv_feature clone_enrichment` |
| UPV annotation | v1 tiers (editing → confident → low_depth) | `scripts/6_spatial_filter/upv_v1_annotate.py` |

Gate verdicts (2026-10-19):

| Section | CNV | Purity | Note |
|---|---|---|---|
| P4 rep1/rep2 | off | on | tumor clones <= 2.6% altered |
| P6 rep1/rep2 | **on** (ref 0, tumor 2) | off | clone 1 dropped (size + UMI AUC 0.95/0.98); purity 3% finite |
| DCIS1/DCIS2 | off | on | no diploid reference clone (all spots in altered clones) |
| OVAR_P5 | **on** (ref 0, tumor 1, 6.2% altered) | off | no purity file |

Model changes relative to earlier runs: DCIS step 7 no longer passes the legacy
`--exclude_vcf/--kept_variants all_filtered_in.vcf.gz` (P4/P6 dropped these 2026-09-06; the frozen
step-6 bundle carries the Beagle routes). NCCE is excluded (probe-based FFPE). DLPFC is not rerun:
its frozen-Beagle G1 matrices and clustering already exist in
`data/experiments/dlpfc_presence_fix_20261009_v1/`.

## Run

```bash
sbatch run_slurm/freeze_v1/run_freeze_v1.slurm                     # 7 elements, 3 at a time
START_STEP=7 sbatch --array=<i> run_slurm/freeze_v1/run_freeze_v1.slurm   # resume one element
```

Array: 0 P4 rep1 · 1 P4 rep2 · 2 P6 rep1 · 3 P6 rep2 · 4 DCIS1 · 5 DCIS2 · 6 OVAR_P5 (starts at
step 0: UMI dedup into `data/ovar_p5/P5_sr13/split_BAM_dedup/`).

P6 elements also build the CalicoST-free arm A (`spatial_filter_purity/baseQ0mapQ0_spatial_only`,
matrices `..._SPARCAL_spatial_only_*`) from the same bundle; the production step 7 is arm B.

## Provenance caveat

The code is in a working tree with uncommitted changes. Record `git rev-parse HEAD` plus
`git diff --stat` at submission and commit the freeze code before the paper is submitted.
