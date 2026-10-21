# P4/P6 spatial filter to SPARCALViewer

This entry point runs the production spatial filter, canonical matrix builder,
and self-contained viewer-study packaging for P4/P6 replicates in one SLURM
array.  Every output is namespaced by a required run label; canonical outputs
and previous viewer studies are not overwritten.

```bash
# Audit-only mechanistic VAF feature (recommended first run)
sbatch run_slurm/p4p6_spatial_to_viewer/run_all.slurm ad_vaf_shadow_v1 shadow full

# Explicit impact experiment: include the VAF feature in somatic voting
sbatch run_slurm/p4p6_spatial_to_viewer/run_all.slurm ad_vaf_vote_v1 vote full

# Matched ablation under the corrected AD/coverage semantics
sbatch run_slurm/p4p6_spatial_to_viewer/run_all.slurm ad_coverage_no_calicost_v1 off spatial_only_no_calicost
```

The four viewer studies are written below:

```text
/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/dist/studies/<RUN_LABEL>/
  P4_rep1_SPARCAL_somatic_<RUN_LABEL>/
  P4_rep2_SPARCAL_somatic_<RUN_LABEL>/
  P6_rep1_SPARCAL_somatic_<RUN_LABEL>/
  P6_rep2_SPARCAL_somatic_<RUN_LABEL>/
```

Each package contains the viewer `.config`, spatial assets, binary SNV matrix,
coverage table, QC/manifest files, and the spatial-filter evidence tables.  To
resume only matrix/viewer stages, run `run_spatial_to_viewer.py --start-at
matrix` (or `viewer`) with the same label.  Reusing an existing label fails
unless `--overwrite` is supplied explicitly.
