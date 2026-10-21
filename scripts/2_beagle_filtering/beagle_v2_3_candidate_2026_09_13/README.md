# SPARCAL Beagle v2.3 candidate

This directory develops the fail-closed, WES-free replacement module without
modifying historical Beagle outputs or the live v2.2 runner.

Current contract additions:

- exact per-contig identity: raw alleles intersected with panel must equal raw
  alleles intersected with Beagle;
- biallelic raw/Beagle input enforcement and exact ALT matching;
- a real indexed query of one expected Beagle allele per completed contig;
- content hashes for Beagle VCFs and indexes;
- an output checksum ledger bound to both MANIFEST.json and RUN_COMPLETE.json.

The contract permits extra imputed records only when they are not raw input
alleles. Routing continues to use the observed raw exact-allele universe, so
an impute=true ablation cannot silently enlarge the candidate callset.

Parameter policy
----------------

Beagle 4.1 documentation states that the impute option affects reference-only
markers when gt and ref inputs are used. SPARCAL's evidence pass uses gl and
ref, so this module rejects an impute=true request instead of presenting it as
an effective ablation. The same manual defines niterations=5 as the default and
states that accuracy and compute time increase with iterations; niterations=0
requests the older phasing procedure. Therefore 0 versus 5 is being measured
across explicit seeds before selection.

Beagle's internal terminology must not be confused with this module's emitted
interface. The 4.1 manual states that `gl=`/`gtgl=` produces unphased estimated
genotypes and that phasing those calls requires a second Beagle run with `gt=`.
Completed v2.3 P6 chr22 outputs confirm this: all heterozygous target GTs were
slash-delimited (709/709 for modelscale=2,niterations=0 and 687/687 for
modelscale=2,niterations=5), even though the niterations=5 log reports five
internal phasing iterations. Production routing consumes unphased GT and
`GP=(P00,P01,P11)` and normalizes away allele order. The module therefore claims
reference-informed genotype-posterior refinement, not phased haplotypes.

The existing non-destructive P4 chr10 sidecar also tested the required second
`gt=` pass. It preserved all 2,131 exact records and unordered first-pass GTs
and phased 641/641 heterozygotes, but its records contained GT only rather than
the first-pass GP/DS evidence and had no PS field or independent phase truth.
Moreover, two historical direct-`gt` runs at niterations 0 versus 5 (without
pinned seeds) agreed on only 51.4% of orientations after the best global flip
and 64.6% of adjacent phase relations. This is mechanical phase emission, not
validated phase accuracy or stable caller evidence, so it is not admitted to
v2.3 production routing.

The manual also notes that increasing modelscale above its 0.8 default can
improve accuracy and runtime when posterior probabilities are estimated from
genotype likelihoods. Monopogen pins modelscale=2 and niterations=0. A compact
modelscale comparison is therefore justified after the iteration/input-quality
screen, while an unconstrained parameter search is not.

Primary Beagle documentation:
https://faculty.washington.edu/browning/beagle/beagle_4.1_21Jan17.pdf

Publication runs must supply a completed full-content reference checksum
ledger. This binds all hg19 and GRCh38 panel VCF/index files independently of
the small set of reference-overlapping raw alleles.

Routing and final selection
---------------------------

The biological interface is evidence-preserving rather than a binary
`in/out` interpretation. `P(nonref)>=0.90` is the only hard inherited
anchor. High-confidence reference, uncertain panel-present, and panel-absent
records remain separate non-anchor routes. The first three panel-present
states may be unioned only in the explicitly labeled backward-compatibility
feature view; that union is not a germline callset.

The clean selection workflow is frozen before results:

1. screen `modelscale={0.8,2} x niterations={0,5}` at seed 20260915;
2. compare the prespecified `modelscale=4,niterations=0` extension;
3. replicate only admitted configurations at seeds 20260916 and 20260917;
4. choose the stable base configuration with the gates in
   `BENCHMARK_PROTOCOL.md`;
5. compare the fixed three-seed mean/median P(nonref) ensembles with one
   selected seed, including their three-fold Beagle cost; and
6. after the mandatory final DLPFC gate passes, freeze a non-overwriting,
   replay-tested production bundle under `RELEASE_PROTOCOL.md` before changing
   any repository default.

The implementation lives in:

- `scripts/postanalyze/audit_beagle_v2_3_pre_replication.py`;
- `scripts/postanalyze/plan_beagle_v2_3_replication.py`;
- `scripts/postanalyze/assess_beagle_v2_3_replication.py`;
- `scripts/postanalyze/beagle_posterior_ensemble.py`;
- `scripts/postanalyze/build_selected_beagle_ensemble.py`; and
- `scripts/postanalyze/assess_beagle_posterior_ensemble.py`.

The release-side ensemble VCF adapter lives in
`scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/materialize_ensemble_routes.py`.
It is deliberately separate from the hash-pinned experimental aggregator and is
only eligible after representation selection. It preserves the directly
aggregated median `P(nonref)` rather than recomputing it from componentwise
median GP fields, then proves all eight compressed route/compatibility VCFs
against the raw VCF and aggregate audit before writing completion markers.

The independent pre-replication audit re-reads the canonical evidence rather
than trusting mutable assessment tables, requires one exact allele universe per
sample across all screen arms, and must reproduce every selector decision
exactly before the replication plan can be written.

All derived bundles refuse overwrite, bind source audits and code/protocol
snapshots, and declare that WES is evaluation-only. An ensemble is rejected if
it lacks material benefit, even if it is stable, because it requires three
Beagle runs. Downstream matrices are compared exactly first, but the one final
selected representation must undergo a new 12-section x 10-seed paired
STAGATE/fixed-EEE run even when its matrices are unchanged. The completed
120-vs-120 run is a compatibility rehearsal, not final-module acceptance.

Selection alone is not deployment. The release protocol requires a stable
profile-driven entry point and exact route-VCF/audit outputs for either a
single-seed or ensemble winner; the existing audit-only ensemble artifact is
insufficient as a production replacement.
