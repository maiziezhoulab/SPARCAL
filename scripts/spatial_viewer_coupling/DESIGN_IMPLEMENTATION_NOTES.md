# Implementation details frozen before GT evaluation

These clarify the immutable DESIGN, without replacing historical text.

- Detector grid contains 81 settings. Iterations 0 and 3 and all comparators receive the
  full grid; iterations 1/2 receive the fixed default detector and every-iteration swap tests.
- Frozen B posterior is already the viewer segmentation score, so raw detection uses it
  directly (the existing phase-aware viewer convention). Matched ALT burden is additionally
  evaluated with exactly the historical raw/residual-then-smooth detector, separating
  reclassification from prior transfer. Smooth coverage uses one smoothing pass; its residual
  endpoint removes the linear UMI projection of that already-smoothed coverage map.
- Tissue footprints are irregular. The spatial null uses bijective permutation of complete
  32-spot spatially ordered blocks, fixing the final incomplete block. This preserves block
  size and within-block label patterns, but does not preserve clusters crossing boundaries.
  It is an approximate spatial null; the block bootstrap is the principal uncertainty result.
- DCIS1's aligned cached AD is checked exactly against current VCF AD while recovering true
  record-presence entries. Thus no cache-limited presence semantics are used in final bundles.
- Frozen P4/P6 xMHC flags use chr6:28–34Mb; the new DCIS hg38 adapter conservatively vetoes
  chr6:25–34Mb. Intrinsic QC must state these intervals and not claim identical coordinates.
  Editing is an A>G/T>C compatibility flag, not a database-confirmed editing site.
- DCIS has no supplied frozen five-state or QB selections, nor a native phase arm in this
  experiment. Its production selected set is reconstructed from existing `in_somatic` metadata
  restricted to ALT-supported denovo SNPs; other P4/P6 QB entries are unavailable, not fabricated.
- The native-phase source uses CalicoST, so a phase+Coff label would be misleading. Phase is
  compared as `phase_Con`, with its in-sample provenance explicit. The direct channel toggle
  comparison is coverage_Con versus coverage_Coff.
- Each inference run checks the full raw-file ledger again at completion; individual bundle
  readers verify immediate file/code/ledger hashes. The final audit checks input-chain hashes
  and cross-contract invariants. Actual GT swaps occur in the evaluator after inference is
  sealed; all reads/writes are forbidden during each numeric replay, with a positive control
  proving the guard blocks a WES file open. Evaluator completion is a separate sealed artifact.
- Native phase construction is not filter rerunning: it only consumes already-existing
  phased-bin counts, CN and purity, using the existing phase-aware implementation.
