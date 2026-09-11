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

---

## Running on the cluster (2026-09-11)

| Job(s) | Name | State | Note |
|---|---|---|---|
| 13869433, 13874366, 13874368 | `spacetra*` | R (3–4 d) | SpaceTracer runs |
| 13874220, 13874221, 13869441 | `spacetra*` | PD (Dependency) | queued behind the above |
| 13874185 | `somatic_*` | PD | **DependencyNeverSatisfied — will never start, needs requeue or cancel** |
| 13869442 | `spacetra*` | PD | **DependencyNeverSatisfied — same** |
| 13874186 | `region_p*` | PD (Dependency) | region-detection pipeline |

> Two jobs are in `DependencyNeverSatisfied` and are dead in the queue. Decide
> whether to requeue or `scancel` them.

## Landed, not yet analysed

_(none — everything from the 2026-09-11 cycle has been written up in
[`report/09_11_report.md`](report/09_11_report.md))_

## Next up (detail in `report/09_11_record.md` §5)

- Build the UPV filter on the confident set (DP>=50 / DP>=100).
- CHECK 2: verify gnomAD REF matches ours at allele-exact hits
  (`data/dnafree_2026-09-08/validation/gnomad_ref_check.py`, written, not yet run to completion).
- Construct the cross-individual recurrence negative set (P4<->P6, DCIS1<->DCIS2).
