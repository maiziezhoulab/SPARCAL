# P4/P6 CalicoST evidence ablation

This workflow compares two explicitly named SPARCAL evidence modes on both P4
and P6 replicates. It reads the same UMI-deduplicated step-6 spot VCFs in both
arms.

- **full_calicost**: spatial clustering plus informative tumor-purity features
  and clone-resolved CNV consistency.
- **ablation_no_calicost**: spatial clustering (zeta) only. It does not load or
  calculate tumor purity, clone, CNV, or purity-by-spatial proxy evidence.

P4 rep2 has 619/648 missing CalicoST tumor-purity estimates and 29 zeros. Full
mode therefore excludes purity-derived votes for that section and uses zeta
plus CNV consistency; the other three sections use their informative purity.

Submit all four sections:

    sbatch run_slurm/p4p6_calicost_ablation/run_spatial_full_ablation_array.slurm

The two arms never share an output directory:

    data/{P4,P6}_tumor/{1,2}/spatial_filter_purity/
      baseQ0mapQ0_full_calicost/
      baseQ0mapQ0_ablation_no_calicost/

Matrices also have disjoint model tokens, SPARCAL_full_calicost and
SPARCAL_ablation_no_calicost.

After all array elements finish:

    python scripts/postanalyze/p4p6_calicost_ablation.py
    python scripts/preprocess/build_p4p6_calicost_ablation_viewer_profiles.py

The downstream comparison reports callset Jaccard, per-spot somatic-burden
concordance, and enrichment of shared/full-only/ablation-only SNVs in CalicoST
CNV/LOH segments. CNV enrichment is supporting internal evidence, not an
independent validation metric, because full mode consumes CalicoST CNV.

## CalicoST phylogeography

Run:

    /data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/python \
      scripts/calicost/run_phylogeography_postprocess.py

The current hg19 outputs realize one clone in P4 rep1, P6 rep1 and P6 rep2, so
those section-level trees are valid singleton Newick trees with no evolutionary
branch. P4 rep2 realizes clone0 and clone1 and its observed LOH matrix is an
exact perfect phylogeny. The generated METHOD.json files distinguish the
singleton and exact-perfect cases from a Startle reconstruction. Startle was
not used; it is only necessary if a future multi-clone LOH matrix needs
homoplasy correction.
