# Prespecified Beagle v2.3 higher-modelscale extension

Frozen before the first full-genome factorial result was available. This is a
development-set extension, not a prospective preregistration. Matched P4/P6 WES
is evaluation-only and is never provided to Beagle or the production router.

## Rationale and fixed arm

The Beagle 4.1 manual states that increasing `modelscale` generally trades
phasing accuracy for runtime, but can improve both accuracy and runtime when
posterior probabilities are estimated from genotype-likelihood data. The main
factorial brackets the default 0.8 and Monopogen/SPARCAL value 2. To test whether
the GL-posterior signal continues beyond 2 without open-ended tuning, this
extension adds exactly one higher value:

- `modelscale=4`, `niterations=0`, seed `20260915`.

All other inputs and parameters are identical to the clean factorial:
`baseQ0mapQ0`, the same P4/P6 raw VCF hashes, the frozen 88-artifact reference
ledger, `P(nonref)>=0.90`, `maxlr=5000`, `gprobs=true`, no genetic map,
and the same fail-closed v2.3 code.

The extension stops at 4. A further value such as 8 has no independent prior
justification and would turn the same two-sample development set into an
open-ended hyperparameter search. Additional phasing iterations are not crossed
with modelscale 4 unless the main factorial first shows that `niterations=5`
passes its advancement gates.

## Advancement rule

Compare against `modelscale=2,niterations=0,seed=20260915` from the clean
factorial. The modelscale-4 arm advances only if:

1. WES-somatic anchor leakage does not increase in either P4 or P6;
2. mean P4/P6 normal-WES anchor sensitivity is no more than 0.005 lower;
3. mean selected-anchor normal-WES support is no more than 0.005 lower; and
4. it improves mean sensitivity or support by at least 0.002, or reduces
   somatic leakage.

If it advances, replicate it at seeds 20260916 and 20260917 and apply the same
stability gates as the main frozen protocol. Otherwise retain it as a negative
ablation only.

As in the main protocol, compare downstream matrices exactly first. Run STAGATE
only once if the ultimately selected configuration changes the accepted matrix.
