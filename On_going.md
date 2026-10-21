# On-going Tasks

**Live only.** Jobs currently on the cluster, and results that just landed but are
not yet analysed. **Clear an entry the moment its analysis is finished** — the
finished content moves into that week's files in `report/`. This file is not an
archive; keep it readable at a glance.

History before 2026-09-11 is archived at
[`report/On_going_archive_pre_2026-09-11.md`](report/On_going_archive_pre_2026-09-11.md).
Past results and discussion now live in `report/MM_DD_record.md` (see CLAUDE.md
§ Weekly report / record convention).

When asked to "check ongoing tasks": run `squeue -u chowx`, inspect output files,
update this file in place, and delete anything already analysed.

## Running

| Job | Name | Submitted | What / next step |
|---|---|---|---|
| 14740437_[0-6%3] | `sparcal_freeze_v1` | 2026-10-20 (Claude, on request) | **Model freeze v1** — `sbatch run_slurm/freeze_v1/run_freeze_v1.slurm` (array 0-6%3: P4×2, P6×2, DCIS×2, OVAR). New root `data/freeze_v1/`; legacy untouched. P6 elements also build arm A (CalicoST-free). README: `run_slurm/freeze_v1/README.md`. Codex is unavailable (quota); Claude now does all work. Next while it runs: DCIS localization evaluator, spectrum, SparcalNet held-out. |

---

## Landed, not yet analysed

- 2026-10-20: old 09-13 job table cleared (SpaceTracer TIMEOUT 09-14; 13874185/13874186/13874220-1 cancelled 10-13; 13963391 cancelled 10-19). P6 CNV-gate job 14715033 failed at the step-6 manifest check and was folded into the freeze.
- None otherwise. The accepted DLPFC compatibility regression has already moved to
  `report/09_11_{record,report}.md`; it is not a live or pending result.

## Next up (detail in `report/09_11_record.md` §5)

- Build the UPV filter on the confident set (DP>=50 / DP>=100).
- CHECK 2: verify gnomAD REF matches ours at allele-exact hits
  (`data/dnafree_2026-09-08/validation/gnomad_ref_check.py`, written, not yet run to completion).
- Construct the cross-individual recurrence negative set (P4<->P6, DCIS1<->DCIS2).
- **After the current model promotion is complete (do not start now):** run the
  prespecified P6 spatial-filter paper validation (spatial block bootstrap/permutation,
  fixed detector-neighborhood sensitivity, and P6-vs-UMI-dominant-context interaction).
  Existing positive results and the no-overclaim boundaries are frozen in
  `report/09_18_record.md` §7.5.
- **After promotion, phase-aware viewer validation:** materialize separately named
  P4/P6 replicate profiles for purity-only, phase-CNV-only, combined, and
  molecular/spatial additions; freeze first, then evaluate. The implemented contract,
  P6 real-data smoke result, and phase limitations are recorded in `report/09_18_record.md` §7.6.
