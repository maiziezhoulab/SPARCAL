# Bounded maxlr pre-screen

Frozen before the `maxlr=50000` output existed. This pre-screen asks whether
retaining genotype-likelihood components down to one tenth of Beagle 4.1's
default clipping threshold changes any usable posterior signal. It does not
use WES.

- Comparator: completed P4 chromosome 22 smoke run, `maxlr=5000`.
- Candidate: identical raw VCF, reference ledger, executable, seed
  (`20260915`), `modelscale=2`, `niterations=0`, and routing rule, changing
  only to `maxlr=50000`.
- Required: both v2.3 bundles validate, and raw exact-allele universes match.
- Reject without a full-genome run if anchor and four-route assignments are
  identical and the maximum absolute P(nonref) change is below `0.001`.
- If either route assignments change or the maximum absolute P(nonref) change
  is at least `0.001`, freeze a separate full P4/P6 comparison before looking
  at WES metrics.

This is a computational relevance screen, not an accuracy test. The genetic
map is not screened because Beagle recommends it specifically when imputing
ungentoyped markers, whereas this module forbids reference-only marker
imputation. Beagle 4.1 `27Jan18.7e1` is not screened because its post-`27Jul16`
changes concern `bref` reading with `impute=false` and low-frequency AR2/DR2;
this pipeline uses VCF reference panels and GP-based routing.
