#!/bin/bash
#SBATCH --job-name=beagle_default_final
#SBATCH --output=slurm_output/beagle_v2_3/default_final_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/default_final_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
set -euo pipefail
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_clustering/bin/python
BUNDLE="$PROJECT/data/releases/beagle_v2_3_default_20260916_v1"
VERIFY_ROOT="$PROJECT/data/experiments/beagle_default_verification_20260916_v1"
cd "$PROJECT"
test ! -e "$BUNDLE"
test ! -e "$VERIFY_ROOT"
mkdir "$VERIFY_ROOT"
"$PYTHON" -m pytest -q tests/test_freeze_beagle_default.py \
  tests/test_beagle_v2_3_contracts.py tests/test_beagle_v2_3_run_validation.py \
  tests/test_beagle_v2_3_runner.py tests/test_beagle_evidence_router.py \
  tests/test_beagle_reference_ledger_contract.py \
  > "$VERIFY_ROOT/contract_tests.txt" 2>&1
"$PYTHON" scripts/2_beagle_filtering/freeze_beagle_default.py freeze \
  --project "$PROJECT" --bundle "$BUNDLE" \
  --decision "$PROJECT/doc/BEAGLE_FINAL_DECISION_2026-09-16.md"
"$PYTHON" "$BUNDLE/beagle.py" replay --sample P4 --output "$VERIFY_ROOT/P4"
"$PYTHON" "$BUNDLE/beagle.py" replay --sample P6 --output "$VERIFY_ROOT/P6"
"$PYTHON" - "$BUNDLE" "$VERIFY_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
bundle, root = map(Path, sys.argv[1:])
rows = {}
for sample in ("P4", "P6"):
    sample_root = root / sample
    result = json.loads((sample_root / "REPLAY_VALIDATION.json").read_text())
    assert result["status"] == "passed" and result["sample"] == sample
    ledger = sample_root / "REPLAY_SHA256SUMS.txt"
    completion = json.loads((sample_root / "REPLAY_COMPLETE.json").read_text())
    assert completion["status"] == "passed"
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == completion["ledger_sha256"]
    rows[sample] = result
summary = dict(status="engineering_validation_passed", bundle=str(bundle.resolve()),
    parameters=json.loads((bundle / "PROFILE.json").read_text())["parameters"],
    source="existing complete P4/P6 Beagle runs plus full routing replay",
    scientific_final_validation_complete=False,
    total_raw_candidates=sum(row["raw_candidates"] for row in rows.values()),
    replay=rows, contract_tests=(root / "contract_tests.txt").read_text(),
    deferred=["three_seed_selection", "posterior_ensemble", "final_module_dlpfc"])
with (root / "FINAL_VALIDATION.json").open("x") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
print("Engineering validation passed:", summary["total_raw_candidates"], "raw candidates")
print(summary["contract_tests"])
PY
