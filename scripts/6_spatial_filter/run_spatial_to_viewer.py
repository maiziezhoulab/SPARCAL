#!/usr/bin/env python3
"""Run spatial filtering, matrix generation, and viewer packaging in one command.

Outputs live in a run-labelled experimental tree and therefore never overwrite
the canonical spatial-filter or matrix products.  Resume with ``--start-at``;
publish directly into SPARCALViewer by setting ``--viewer-output-root`` to a
viewer ``dist/studies/<release-label>`` directory.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALICOST_ROOT = Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906")
STAGES = ("filter", "matrix", "viewer")


def safe_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run label may contain only letters, numbers, '.', '_' and '-'")
    return value


def calicost_inputs(dataset: str, section_id: str) -> dict[str, Path]:
    cohort = dataset.split("_")[0]
    clone_request = "2" if cohort == "P4" else "3"
    run = CALICOST_ROOT / f"{cohort}_sec{section_id}"
    clone_dir = run / "calicost" / f"clone{clone_request}_rectangle0_w1.0"
    canonical_cnv = clone_dir / "cnv_seglevel.tsv"
    fallback_cnv = clone_dir / "cnv_diploid_seglevel.tsv"
    return {
        "purity": run / "estimate_tumor_prop" / "loh_estimator_tumor_prop.tsv",
        "clones": clone_dir / "clone_labels.tsv",
        "cnv": canonical_cnv if canonical_cnv.is_file() else fallback_cnv,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def command_text(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=["P4_TUMOR", "P6_TUMOR", "p4_tumor", "p6_tumor"])
    parser.add_argument("--section-id", required=True, choices=["1", "2"])
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    parser.add_argument("--run-label", required=True,
                        help="Unique output/profile label, e.g. ad_vaf_shadow_v1")
    parser.add_argument("--evidence-mode", default="full",
                        choices=["full", "spatial_only_no_calicost"])
    parser.add_argument("--mechanistic-vaf-mode", default="shadow",
                        choices=["off", "shadow", "vote"])
    parser.add_argument("--matrix-class", default="somatic",
                        choices=["1000G", "UPV", "germline", "somatic", "merged"])
    parser.add_argument("--output-base", type=Path,
                        default=PROJECT_ROOT / "data" / "spatial_filter_runs")
    parser.add_argument("--viewer-output-root", type=Path, default=None)
    parser.add_argument("--start-at", choices=STAGES, default="filter")
    parser.add_argument("--stop-after", choices=STAGES, default="viewer")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.upper()
    run_label = safe_label(args.run_label)
    if STAGES.index(args.start_at) > STAGES.index(args.stop_after):
        parser.error("--start-at must not come after --stop-after")
    if args.evidence_mode == "spatial_only_no_calicost" and args.mechanistic_vaf_mode != "off":
        parser.error("the no-CalicoST ablation requires --mechanistic-vaf-mode off")

    cohort = dataset.split("_")[0]
    run_root = args.output_base.resolve() / run_label
    section_root = run_root / f"{cohort}_tumor" / args.section_id
    spatial_root = section_root / "spatial_filter_purity" / args.quality_filter
    matrix_model = f"SPARCAL_{run_label}"
    viewer_root = (
        args.viewer_output_root.resolve()
        if args.viewer_output_root else run_root / "viewer_studies"
    )
    manifest_path = section_root / "spatial_to_viewer_manifest.json"

    calicost = calicost_inputs(dataset, args.section_id)
    if args.evidence_mode == "full":
        missing = [str(path) for path in calicost.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing corrected-hg19 CalicoST input(s):\n  " + "\n  ".join(missing))
    vcf_dir = (
        PROJECT_ROOT / "data" / f"{cohort}_tumor" / args.section_id
        / "output_VCFs" / "spotprofiles" / args.quality_filter / "vcf_by_spot"
    )
    n_vcfs = len(list(vcf_dir.glob("*.vcf.gz")))
    if n_vcfs == 0:
        raise FileNotFoundError(f"No per-spot VCFs found in {vcf_dir}")
    if args.start_at == "filter" and section_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run-labelled section already exists: {section_root}. Choose a new --run-label "
            "or pass --overwrite explicitly."
        )

    filter_command = [
        sys.executable, str(PROJECT_ROOT / "scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py"),
        "--dataset", dataset.lower(), "--section_id", args.section_id,
        "--quality_filter", args.quality_filter, "--evidence_mode", args.evidence_mode,
        "--mechanistic_vaf_mode", args.mechanistic_vaf_mode,
        "--output_dir", str(spatial_root), "--min_expression_germline", "2",
        "--min_expression_somatic", "1", "--neighbor_distance", "2.0",
    ]
    if args.evidence_mode == "full":
        filter_command.extend([
            "--tumor_purity_file", str(calicost["purity"]),
            "--clone_labels", str(calicost["clones"]),
            "--cnv_segments", str(calicost["cnv"]),
        ])
    matrix_command = [
        sys.executable, str(PROJECT_ROOT / "scripts/6_spatial_filter/generate_sparcal_matrices.py"),
        "--dataset", dataset, "--section_id", args.section_id,
        "--quality_filter", args.quality_filter, "--section-root", str(section_root),
        "--model", matrix_model, "--classes", args.matrix_class,
    ]
    viewer_command = [
        sys.executable, str(PROJECT_ROOT / "scripts/preprocess/build_sparcalviewer_profile.py"),
        "--dataset", dataset, "--section-id", args.section_id,
        "--section-root", str(section_root), "--quality-filter", args.quality_filter,
        "--model", matrix_model, "--matrix-class", args.matrix_class,
        "--run-label", run_label, "--output-root", str(viewer_root),
    ]
    if args.overwrite:
        viewer_command.append("--overwrite")
    commands = {"filter": filter_command, "matrix": matrix_command, "viewer": viewer_command}

    selected = STAGES[STAGES.index(args.start_at):STAGES.index(args.stop_after) + 1]
    manifest = {
        "dataset": dataset,
        "section_id": args.section_id,
        "run_label": run_label,
        "evidence_mode": args.evidence_mode,
        "mechanistic_vaf_mode": args.mechanistic_vaf_mode,
        "quality_filter": args.quality_filter,
        "matrix_class": args.matrix_class,
        "section_root": str(section_root),
        "viewer_output_root": str(viewer_root),
        "corrected_calicost_inputs": (
            {key: str(path) for key, path in calicost.items()}
            if args.evidence_mode == "full" else None
        ),
        "per_spot_vcf_count": n_vcfs,
        "selected_stages": list(selected),
        "commands": {stage: command_text(command) for stage, command in commands.items()},
        "status": "dry_run" if args.dry_run else "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    write_manifest(manifest_path, manifest)
    try:
        for stage in selected:
            print(f"\n=== {stage.upper()} ===", flush=True)
            print(command_text(commands[stage]), flush=True)
            started = datetime.now(timezone.utc).isoformat()
            subprocess.run(commands[stage], cwd=PROJECT_ROOT, check=True)
            manifest["stages"][stage] = {
                "status": "complete", "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            write_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print(f"\nComplete. Manifest: {manifest_path}")
    print(f"Viewer studies: {viewer_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
