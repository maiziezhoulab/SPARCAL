# Frozen Beagle v2.3 production-release protocol

Frozen before the three-seed replication, ensemble selection, and final DLPFC
assessment were available. This protocol closes the gap between selecting an
experimental representation and shipping a module that can replace the legacy
Beagle step. It does not modify or overwrite the legacy runner or any historical
output tree.

## Admission

A release may be created only after all of the following immutable bundles pass
their own checksum and semantic validators:

1. the final three-seed base-configuration assessment;
2. the final single-seed-versus-ensemble representation selection;
3. the selected representation's 12-section DLPFC materialization; and
4. the exact-once terminal DLPFC assessment with `accept_replacement=true`.

The release freezer must revalidate these sources from primary ledgers. Marker
files, copied decision JSON, directory names, and an earlier compatibility run
are not sufficient. The representation, numeric Beagle parameters, fixed seeds,
reference ledger, code identities, materialization root, DLPFC claim key, and
terminal assessment must agree transitively.

## Production interface

The released module starts from a raw mpileup candidate VCF and a reference-panel
checksum ledger. It must not accept WES calls, pathology labels, spatial clusters,
or any other evaluation truth as inference input. It must emit:

- one conserved, exact-allele routing audit covering every raw candidate;
- mutually exclusive `beagle_nonref_highconf`, `beagle_ref_highconf`,
  `beagle_uncertain`, and `panel_absent` outputs;
- a primary `all_panel_present` feature-support view for legacy consumers;
- an anchor-only `P(nonref)>=0.90` view as a separately labeled inference signal;
- a manifest, completion record, and complete code/input/output checksum ledgers.

Only `beagle_nonref_highconf` is a hard germline anchor. `all_panel_present` is a
compatibility feature view and is not a germline callset. The production Beagle
operation is reference-panel-informed genotype-posterior refinement with
`impute=false`; it does not claim target haplotype phasing or reference-only
marker imputation. Optional two-pass phased output remains a separate sidecar.

For a selected single-seed representation, the four-route VCFs and audit produced
by the validated v2.3 runner are the canonical outputs. For a selected posterior
ensemble, the release must not stop at an aggregate TSV audit: it must materialize
the same four exact-allele route VCFs and both downstream views from the frozen
mean or median posterior rule. The materializer must recompute every aggregate
row from all three seed audits, preserve the original raw record identity, and
prove route-VCF/audit equality before completion.

## Non-overwriting release bundle

The release freezer writes once to a new versioned directory and fails if that
path already exists. The bundle must contain:

- a machine-readable production profile with the selected representation,
  base configuration, exact numeric parameters, seed or seeds, anchor rule,
  route semantics, and Beagle run multiplier;
- immutable copies of every runtime source required for single-seed execution
  and, when selected, posterior aggregation and route materialization;
- identities for the Beagle JAR, Java executable, reference ledger, and the
  complete upstream evidence chain;
- a stable command-line entry point that reads the production profile rather
  than parsing parameters from a configuration name; and
- an independently verifiable release completion ledger.

No symbolic link, `current` pointer, legacy script, or historical output is
changed by freezing the bundle. Making the release the repository default is a
separate, explicit promotion commit after the bundle and its replay tests pass.

## Release verification

Before default promotion, test both branches of the entry point with synthetic
contract fixtures, even if only one representation was selected. Replay the
selected branch against the already frozen P4 and P6 evidence without using WES
as an input and require exact route-audit and route-VCF equality. Verify the
primary DLPFC interface against the accepted final materialization. A release
cannot be marked complete if an ensemble branch emits only counts/audit rows,
if any allele is lost or duplicated, if a route overlaps another route, if an
input/output checksum is missing, or if fallback behavior is observed.

The final paper figure and methods text must cite the released production profile
and evidence ledgers, not mutable live-code paths. Negative parameter arms and a
rejected ensemble remain visible as ablations but are not shipped as defaults.
