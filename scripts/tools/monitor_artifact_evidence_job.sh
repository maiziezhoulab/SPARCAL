#!/usr/bin/env bash
# Monitor one non-destructive artifact-evidence SLURM run and inspect its
# summaries once it reaches a terminal state.  Run from a cluster login node:
#   bash scripts/tools/monitor_artifact_evidence_job.sh --job-id 12571729

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
JOB_ID=12571729
INTERVAL_SECONDS=3600
ONCE=0
JOB_NAME=sparcal_artifact_v2_p4
RUN_ROOT="${PROJECT}/data/P4_tumor/1/artifact_evidence/v2_pilot_2026-07-15_p4_batch"
MONITOR_DIR="${PROJECT}/run_slurm/slurm_output/monitor"

usage() {
    cat <<'EOF'
Usage: monitor_artifact_evidence_job.sh [--job-id ID] [--interval-seconds N] [--once]

Polls squeue once per hour by default.  On completion, queries sacct and
prints the feature/evidence summaries; it never changes the job or its data.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-id) JOB_ID="$2"; shift 2 ;;
        --interval-seconds) INTERVAL_SECONDS="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--interval-seconds must be a positive integer" >&2
    exit 2
fi

mkdir -p "$MONITOR_DIR"
LOG_FILE="${MONITOR_DIR}/artifact_evidence_${JOB_ID}.log"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_FILE"
}

analyze_outputs() {
    local feature_summary="${RUN_ROOT}/features/summary.json"
    local evidence_summary="${RUN_ROOT}/evidence/summary.json"
    local stdout_log="${PROJECT}/run_slurm/slurm_output/${JOB_NAME}_${JOB_ID}.out"
    local stderr_log="${PROJECT}/run_slurm/slurm_output/${JOB_NAME}_${JOB_ID}.err"

    if [[ ! -f "$feature_summary" || ! -f "$evidence_summary" ]]; then
        log "RESULT INCOMPLETE: expected summaries are missing under ${RUN_ROOT}"
        [[ -f "$stdout_log" ]] && { log "Last stdout lines:"; tail -n 40 "$stdout_log" | tee -a "$LOG_FILE"; }
        [[ -f "$stderr_log" ]] && { log "Last stderr lines:"; tail -n 80 "$stderr_log" | tee -a "$LOG_FILE"; }
        return 1
    fi

    python - "$feature_summary" "$evidence_summary" <<'PY' | tee -a "$LOG_FILE"
import json
import sys

features = json.load(open(sys.argv[1]))
evidence = json.load(open(sys.argv[2]))
print("Artifact-evidence pilot summary")
print("  processed_candidates:", features.get("processed_candidates"))
print("  unique_candidates_after_sampling:", features.get("unique_candidates_after_sampling"))
print("  covered_candidates:", features.get("covered_candidates"))
print("  uncovered_candidates:", features.get("uncovered_candidates"))
print("  pileup_capped_candidates:", features.get("pileup_capped_candidates"))
print("  sampled per source:", features.get("candidate_sources_after_sampling"))
print("  n_sites_scored:", evidence.get("n_sites"))
print("  evidence_status_counts:", evidence.get("status_counts"))
print("  calibration:", evidence.get("calibration"))
PY
}

while true; do
    if ! queue_line="$(squeue --noheader -j "$JOB_ID" -o '%T|%M|%R' 2>&1)"; then
        # Once a job leaves squeue, this SLURM controller returns an error
        # rather than an empty result.  That is the expected hand-off to sacct.
        if [[ "$queue_line" == *"Invalid job id specified"* ]]; then
            queue_line=""
        else
            log "SQUEUE UNAVAILABLE: ${queue_line}"
            if [[ "$ONCE" -eq 1 ]]; then
                exit 2
            fi
            sleep "$INTERVAL_SECONDS"
            continue
        fi
    fi

    if [[ -n "$queue_line" ]]; then
        log "ACTIVE: ${queue_line}"
        if [[ "$ONCE" -eq 1 ]]; then
            exit 0
        fi
        sleep "$INTERVAL_SECONDS"
        continue
    else
        if ! accounting="$(sacct -n -X -P -j "$JOB_ID" --format=JobIDRaw,State,ExitCode,Elapsed,Start,End 2>&1)"; then
            log "SACCT UNAVAILABLE: ${accounting}"
        elif [[ -z "$accounting" ]]; then
            log "No squeue or sacct record yet; waiting for accounting propagation."
        else
            record="$(printf '%s\n' "$accounting" | awk -F'|' -v id="$JOB_ID" '$1 == id {print; exit}')"
            if [[ -z "$record" ]]; then
                log "No top-level accounting record yet: ${accounting}"
            else
                state="$(printf '%s' "$record" | cut -d'|' -f2 | sed 's/[[:space:]]//g')"
                exit_code="$(printf '%s' "$record" | cut -d'|' -f3 | sed 's/[[:space:]]//g')"
                case "$state" in
                    COMPLETED*)
                        log "TERMINAL: ${record}"
                        if [[ "$exit_code" == 0:0 ]]; then
                            analyze_outputs
                            exit $?
                        fi
                        analyze_outputs || true
                        exit 1
                        ;;
                    FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*)
                        log "TERMINAL: ${record}"
                        analyze_outputs || true
                        exit 1
                        ;;
                    *)
                        log "ACCOUNTING NON-TERMINAL: ${record}"
                        ;;
                esac
            fi
        fi
    fi

    if [[ "$ONCE" -eq 1 ]]; then
        exit 1
    fi
    sleep "$INTERVAL_SECONDS"
done
