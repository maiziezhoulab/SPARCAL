# P4/P6 CalicoST hg19 rerun

This rerun uses one reference stack throughout: the non-deprecated hg19 Space Ranger BAMs, a chr-prefixed b37 SNP panel, hg19 1000 Genomes BCFs, the Eagle hg19 map, GENCODE v37lift37 gene coordinates, an hg19 CalicoST genetic map, and hg19 HLA intervals.

The previous hg38 outputs are retained. New outputs are written to `/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906`.

Array mapping:

- 0: P4 section 1
- 1: P6 section 1
- 2: P6 section 2
- 3: P4 section 2 (inherits P4 section 1 model/HMM settings)

The model/HMM parameters are inherited from each sample's previous configuration. Only build-dependent inputs and output paths are changed.

CNA mode is explicit in `samples.tsv`. P4 section 2 uses unsupervised CNA
because its LOH estimator returned no positive finite tumor-purity values
(619/648 NaN and the remaining 29 equal to zero); the other samples use
estimated tumor purity.

```bash
python scripts/calicost/build_hg19_resources.py
sbatch run_slurm/calicost_hg19/run_calicost_hg19_array.slurm
```
