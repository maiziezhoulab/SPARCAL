# Beagle architecture and default setting — 2026-09-16

The user requested an immediate final setting, prioritizing the architecture and
deferring additional proof experiments. This decision freezes the existing
screen comparator as the engineering default. It does not declare a winner of
the unfinished three-seed selection protocol.

## Fixed setting

| Parameter | Value |
|---|---|
| Engine | Beagle 4.1, `beagle.27Jul16.86a.jar` |
| Input | pooled genotype likelihoods (`gl=`), BQ0/MQ0 candidates |
| Reference | checksum-bound 1000 Genomes hg19/GRCh38 panels |
| `modelscale` / `niterations` / `maxlr` | 2 / 0 / 5000 |
| `seed` | 20260915 |
| `impute` / `gprobs` | false / true |
| Representation | single run; no ensemble |
| Hard-anchor rule | unrounded `GP(0/1) + GP(1/1) >= 0.90` |
| Execution | 30 threads, one chromosome at a time, Java heap 10g |

This is the prespecified fail-safe comparator. Alternatives have only
single-seed development evidence; their small improvements do not establish
replicated superiority. The fixed seed is the original screen seed, not a seed
chosen after comparing results.

## Architectural contribution

The module conserves every raw exact `CHROM/POS/REF/ALT` candidate and separates
four evidence states: non-reference high confidence, reference high confidence,
uncertain panel-present, and panel-absent. Only the first state becomes a hard
germline anchor. The other states remain available to downstream inference.

Two explicit output interfaces serve different consumers:

- Inference: `all_beagle_nonref_highconf.vcf.gz` supplies hard anchors;
  the complementary states retain uncertainty for later calling.
- Spatial representation: `all_panel_present.vcf.gz` conserves panel feature
  support for the established clustering consumer. It is not a germline callset.

Production inference accepts no WES, pathology, or cluster labels. A lossless
per-allele audit, output checksums, and exact panel/Beagle identity checks make
the representation inspectable. This is reference-informed genotype posterior
refinement; the output does not assert validated target haplotypes.

## Verification scope

The current freeze requires checksum and semantic validation of the completed
P4/P6 comparator runs, deterministic replay of the frozen router from their raw
VCFs and saved Beagle posteriors, exact audit and eight output-view equality,
candidate conservation, and executable profile contract tests. Replay verifies
routing; it does not rerun Beagle or measure seed stability.

The earlier `RELEASE_PROTOCOL.md` remains an immutable historical scientific
acceptance protocol. This dated user-directed engineering freeze does not emit
its acceptance markers or satisfy its remaining research gates by substitution.
The 20-run replication plan, ensemble comparison, final selected-module DLPFC
run, and additional spatial proof experiments are deferred. The earlier DLPFC
compatibility result remains compatibility evidence only.

The frozen bundle and explicit command-line entry point are the deliverable.
Historical outputs and the legacy pipeline entry point are preserved; adoption
by downstream pipeline consumers is a separate integration step.

## Frozen entry point

Bundle: `data/releases/beagle_v2_3_default_20260916_v1/`.
Verification: `data/experiments/beagle_default_verification_20260916_v1/`.

Use the compatible `snv_clustering` or `snv_caller` Python environment:

```bash
python data/releases/beagle_v2_3_default_20260916_v1/beagle.py verify
python data/releases/beagle_v2_3_default_20260916_v1/beagle.py run \
  --dataset P4_TUMOR --section-id 1 --raw-vcf /absolute/raw.vcf.gz \
  --output /absolute/new_section_root --dry-run
```

Remove `--dry-run` to execute. The output root must be new. The entry point
verifies its profile and runtime checksums, binds the reference ledger, and
copies its four runtime sources into the new run before executing. There are no
parameter override or evaluation-label options.
