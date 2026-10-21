# Pre-implementation self-review — 2026-09-16

DESIGN SHA256: eacc36eaa192dea5abeb86a162083842e5590028d6b323ac3f322778eb14a58f

- PASS additive scope, exact-allele alignment, no production/Beagle writes.
- PASS GT restricted to post-freeze evaluator; no StudyData construction in inference.
- PASS rebase each iteration, fixed weights, no hard selection feedback: contraction <=.75.
- PASS missing CN/purity remains absent or explicitly marginalized by native phase module.
- CORRECTION: detector grid is 3^4=81 cells, not the typographical 243 in DESIGN.
- CLARIFICATION: iteration0 B is the anchor; its spot evidence is computed for diagnostics.
  Iteration k>=1 scores using B(k-1), then stores the damped viewer update B(k).
- CLARIFICATION: all-class A preserves all SNP candidates with unambiguous exact alleles;
  records missing valid alleles cannot define an exact-allele contract and are counted as
  excluded adapter records. New selection universe remains denovo SNPs with ALT support.
- CLARIFICATION: contextual frozen posterior ratio is neutral for missing context, not a
  reconstructed likelihood under fabricated CN. Phase priors perform actual marginalization.
- CLARIFICATION: swap tests must run for every frozen iteration in the evaluator, after
  real labels become available. Run completion is separate from evaluation completion.
- CLARIFICATION: bootstrap confidence intervals are conditional on these sections and detector;
  no independent-donor context-interaction claim is licensed by three sections.

Proceed to implementation. This review is separate so the original design remains immutable.
