# Frozen full-genome maxlr comparison

Frozen after the WES-free chromosome-22 relevance screen advanced and before
the full P4/P6 `maxlr=50000` outputs existed. P4/P6 WES is used only for
development-set assessment; it is never passed to Beagle or the production
router.

## Fixed comparison

- Comparator: `ms2_niter0_seed20260915` from the clean factorial screen.
- Candidate: `ms2_niter0_maxlr50000_seed20260915`.
- Both use the same P4/P6 `baseQ0mapQ0` raw VCFs, frozen 88-artifact reference
  ledger, Beagle 4.1 jar, routing code, `modelscale=2`, `niterations=0`,
  `impute=false`, `gprobs=true`, no genetic map, and `P(nonref)>=0.90` anchor
  rule. Only `maxlr` changes from 5000 to 50000.
- Every run must satisfy the fail-closed v2.3 semantic and checksum contract.
  The raw input, exact allele universe, executables, core code, and reference
  identities must match within sample.

## Advancement gate

The candidate advances to seeds `20260916` and `20260917` only if all of the
following hold at seed `20260915`:

1. WES-somatic anchor leakage does not increase in either P4 or P6;
2. mean P4/P6 normal-WES anchor sensitivity is no more than 0.005 below the
   comparator;
3. mean selected-anchor normal-WES support is no more than 0.005 below the
   comparator; and
4. it improves mean sensitivity or mean support by at least 0.002, or reduces
   total WES-somatic anchor leakage.

If it advances, it enters the same three-seed stability and final-selection
gate as all other candidates. The final selected Beagle setting must undergo
the DLPFC STAGATE plus mclust non-inferiority check once; screen failures do not
trigger this expensive downstream run.

P4/P6 WES remains a development-set selection endpoint, not independent or
prospective validation.
