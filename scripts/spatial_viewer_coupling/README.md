# Additive spatial filter ↔ SPARCALviewer coupling

Read [DESIGN.md](DESIGN.md), its pre-implementation [self-review](DESIGN_REVIEW.md),
and [implementation notes](DESIGN_IMPLEMENTATION_NOTES.md). No production changes are needed.
Python requires NumPy, SciPy, pandas and PyYAML; the viewer kernels are imported read-only.
The registered cluster environment is `snv_clustering`.

```bash
# From /data/maiziezhou_lab/leiy4/snv_calling
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.spatial_viewer_coupling.test_coupling
python -m scripts.spatial_viewer_coupling.test_evaluator
sbatch scripts/spatial_viewer_coupling/run_registered.slurm
sbatch --dependency=afterok:JOB_ID scripts/spatial_viewer_coupling/evaluate_registered.slurm
sbatch --dependency=afterok:JOB_ID scripts/spatial_viewer_coupling/export_registered.slurm
```

All outputs are under `data/spatial_viewer_coupling_2026-09-16/`. Existing paths fail closed;
use a new `--run-id` / `--evaluation-id` for a new run. Upstream version is represented by
path+SHA256, not inferred from filenames or Beagle settings. `--prepared` accepts an existing
validated additive preparation; changing upstream data requires a fresh preparation.

For a serialized contract handoff, use `audit.load_pair(A, B)` and `exchange.advance(A,B,out)`:

```bash
python -m scripts.spatial_viewer_coupling.exchange \
  --variant-bundle RUN/SECTION/coverage_Coff/iteration_00/A \
  --prior-bundle RUN/SECTION/coverage_Coff/iteration_00/B \
  --output NEW_EXCHANGE_DIRECTORY
```

This independently reloads/validates both contracts, rejects mismatched section, build,
barcodes, graph, provenance or quotas, and computes the next registered update. At most
iteration 3 is admitted. Raw VCF files are never rewritten. The real P4 round-trip check is
recorded in `data/spatial_viewer_coupling_2026-09-16/exchange_smoke/final_result.json`.

The `viewer_studies/` export is a new viewer profile for each section. Open its `.config`;
`CoupledFrozen` contains the iteration-3 regions. The viewer's existing **Phase-aware tumor
posterior** method can display the frozen generic posterior via its existing sidecar schema.
The model name and provenance identify the actual prior source; coverage exports contain
no phase or purity evidence. The matrix is an ALT>0 compatibility view. Exact REF/ALT,
five-state probabilities, weights and QC remain the primary A bundle, linked by the export
manifest. The old viewer does not yet expose every A field in its GUI; Python consumers
can read them through `contracts.read_a` or the stricter cross-contract audit.

The evaluators (`evaluate.py` and `evaluate_monotone.py`) alone read pathology/WES/DCIS labels. Numerical label-swap replays prohibit
all file I/O. Inference and evaluation completion markers are distinct: an inference bundle
with `label_swap_status=pending_post_freeze_evaluator` is usable for replay, but no evaluation
claim is complete until the evaluator's `COMPLETE.sha256` exists and its swap table passes.

To reproduce `CoupledFrozen` in the viewer Auto dialog, set seed=90, grow=80,
min-size=5 and split=0.4. The profile stores these regions directly; other dialog
settings legitimately produce a different segmentation of the same frozen posterior.

## Corrected default evaluation (monotone coverage nulls)

`evaluate_registered.slurm` now always calls `evaluate_monotone.py`. Read the
[post hoc correction protocol](MONOTONE_NULL_PROTOCOL.md). Legacy `evaluate.py`
stays immutable for old score-cache provenance and is not the reporting entry.
The correction restores post-residual smoothing for coverage, exports every
transform chain, and compares every arm against both rank and monotone-PCHIP
coverage nulls in raw and residual space. All iterations use the complete grid.

```bash
# Reuse the existing sealed cache, without rerunning inference or filter:
sbatch scripts/spatial_viewer_coupling/evaluate_registered.slurm \
  --source-evaluation-id evaluation_v1 --evaluation-id evaluation_v2_monotone
# A future Beagle/input run automatically gets both nulls; cache generated if absent:
sbatch scripts/spatial_viewer_coupling/evaluate_registered.slurm \
  --run-id NEW_RUN --evaluation-id NEW_EVALUATION
python -m scripts.spatial_viewer_coupling.test_monotone
```

Existing output directories fail closed. New evaluations record source-cache,
input-bundle, graph, code and protocol hashes. The old report is archived in
`data/spatial_viewer_coupling_2026-09-16/report_v1_archive/`; evaluation_v1 and
registered_v1 remain untouched. Nulls are evaluator controls, not new viewer
profiles or inputs to the inference loop.
