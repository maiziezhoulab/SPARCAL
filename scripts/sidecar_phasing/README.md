# sidecar_phasing

This is a **NON-DESTRUCTIVE side-car**. It does not modify, run, or import
`scripts/2_beagle_filtering/run_beagle.py` or any other file under `scripts/1_calling/`,
`scripts/2_beagle_filtering/`, `scripts/3_classifier_prep/`, `scripts/4_classifier/`,
`scripts/5_refilter_bam/`, or `scripts/6_spatial_filter/`, and it never writes into any
`data/<sample>/output_VCFs/` tree. It is a **candidate for future merge into `run_beagle.py`
only after review** — nothing here should be treated as replacing the shipped pipeline until
that review happens.

## What this is

`run_beagle_phased.py` reruns Beagle 4.1 on a single chromosome of a single sample with
`niterations` and `impute` configurable (the shipped pipeline hardcodes `niterations=0`, which
disables phasing iterations, and deletes Beagle's raw output before anyone can inspect it). This
side-car does the opposite on both counts: `niterations` defaults to 5 (Beagle's own factory
default), and the raw Beagle output is always retained, never deleted.

Reference-panel resolution and the "does this chromosome have any marker overlap with the 1000
Genomes panel" guard are copied (re-typed, not imported) from `BeaglePipeline` in
`run_beagle.py` as of 2026-08-24, so behavior matches the shipped pipeline on those two points.
Everything else (output location, retention, configurable phasing parameters, the stats
function) is new.

## Output location

Everything is written under `data_sidecar_phased/<dataset_output_dir>/<chrom>/<quality_filter>/<run_tag>/`,
e.g. `data_sidecar_phased/P4_tumor/1/chr10/baseQ0mapQ0/niter5_imputeF/` — a tree parallel to
(never inside) `data/<sample>/output_VCFs/`. `run_tag` encodes `niterations`/`impute` (and, for
the diagnostic `--input-field` option, the input field) so multiple configurations for the same
chromosome coexist without collision. Each run directory holds:

| file | contents |
|---|---|
| `<chrom>.beagle_raw.vcf.gz(.tbi)` | Beagle's own output, verbatim. **Retained, never deleted** (the shipped pipeline's `run_beagle_command` deletes the equivalent `.temp.vcf.gz`). |
| `<chrom>.beagle_raw.log` | Beagle's stdout/stderr — exact command line + run stats. |
| `<chrom>.merged.vcf.gz(.tbi)` | Original INFO/FORMAT fields re-annotated onto the raw calls, same `bcftools annotate` recipe as `merge_vcf_fields` in `run_beagle.py`. Parity artifact only — not needed to answer the phasing question, since GT/AR2/DR2 are not touched by the merge. |
| `<chrom>.stats.json` | Output of `compute_phasing_stats()`: record/het counts, % phased, AR2/DR2 summary, whether a `PS` FORMAT tag is declared. |
| `run_meta.json` | Exact params, timings (marker-check / Beagle / merge / total wall seconds), timestamp. |

Re-running with the same dataset/section/chrom/quality-filter/niterations/impute/input-field is
**idempotent** — it skips the Beagle call and reuses the existing `run_meta.json`/`*.stats.json`
unless `--force` is passed.

## Usage

```bash
conda activate snv_caller

python scripts/sidecar_phasing/run_beagle_phased.py \
    --dataset P4_TUMOR --section_id 1 --chrom chr10 --quality-filter baseQ0mapQ0 \
    --niterations 5 --threads 16

# analyze an arbitrary VCF (e.g. the shipped niterations=0 output) with the exact same
# stats function, no Beagle run:
python scripts/sidecar_phasing/run_beagle_phased.py --analyze-only \
    --vcf-path data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/chr10.vcf.gz
```

Key flags: `--niterations` (default 5), `--impute` (default off, matches shipped), `--no-gprobs`
(gprobs defaults on, per spec), `--input-field {gl,gt,gtgl}` (default `gl`, matches shipped —
**but see Stage 2 below: `gt=` is the one that actually phases**), `--force` to rerun over an
existing run_tag, `--custom-input-vcf PATH` + `--run-label LABEL` (Stage 2: feed Beagle a prior
pass's retained output instead of the resolved original mpileup VCF, for two-pass
call-then-phase runs; the true original mpileup VCF is still used as the merge-parity step's
annotation source regardless).

## Stage-1 probe result (2026-08-24)

The first use of this script was a probe on P4_tumor/1, chr10 only, comparing
`niterations=0` (shipped, read from existing output, not rerun) against `niterations=5` and
`niterations=10` side-car runs, plus two supplementary diagnostics (`impute=true`,
`--input-field gtgl`). **Verdict: phase was not recovered at any setting tested — 0.0% of
genotypes phased in all 5 configurations**, even though the Beagle log confirms real phasing
iterations executed. **This verdict was correct as far as it tested (`gl=`/`gtgl=` really don't
phase, at any `niterations`) but incomplete — see Stage 2 immediately below, which found the
actual fix.** Full comparison table and exact commands: `data/sidecar_phasing_probe_2026-08-23/RESULTS.md`.

## Stage-2 probe result (2026-08-24, same day) — supersedes the Stage-1 headline

Stage 2 tested the one axis Stage 1 hadn't: Beagle's `gt=` input parameter (called genotypes)
instead of `gl=`/`gtgl=` (genotype likelihoods / GT-preferred-else-likelihoods). **Verdict:
`gt=` phases 100% of chr10's heterozygous sites** — both fed directly with the original
mpileup VCF and in a two-pass (`gl=` calls genotypes, `gt=` phases them) pattern, and **at
`niterations=0` just as well as at `niterations=5`** — so `niterations` was never the lever;
the load-bearing change is `gl=` → `gt=`. `impute=true` is legal under `gt=` (unlike under
`gl=`, where it silently no-ops) but **crashes** with a `beagle.27Jul16.86a.jar`-internal
`ArrayIndexOutOfBoundsException` on this data — phasing itself (`impute=false`) is unaffected.
Modern alternative phasers (Beagle 5.4, SHAPEIT5, SHAPEIT2) are available on this HPC system via
`module load`, zero install needed, and were confirmed to run (not exercised on real data here).
Full write-up, exact commands, the crash traceback, the alternative-phaser search, and the
panel-AF distribution of the phased het sites: `data/sidecar_phasing_probe_2026-08-23/RESULTS.md`
(Stage-2 section, appended after Stage 1 — read both; Stage 1's record is unchanged and still
accurate for what it tested).
