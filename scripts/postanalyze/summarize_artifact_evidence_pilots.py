#!/usr/bin/env python3
"""Summarize fixed artifact-evidence pilots without modifying their outputs.

Each input is a ``LABEL=RUN_ROOT`` pair produced by
``run_artifact_evidence_workflow.py``.  Missing/failed runs are recorded so a
dependent QC job remains informative even if one array task fails.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


STATUSES = (
    "external_annotation",
    "low_information",
    "multiple_technical_flags",
    "read_evidence_pass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_ROOT")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--run must be LABEL=RUN_ROOT, got: {value}")
    label, root = value.split("=", 1)
    if not label or not root:
        raise ValueError(f"--run must be LABEL=RUN_ROOT, got: {value}")
    return label, Path(root)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing summary: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pilot_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for raw_run in args.run:
        label, root = parse_run(raw_run)
        feature_path = root / "features" / "summary.json"
        evidence_path = root / "evidence" / "summary.json"
        table_path = root / "evidence" / "site_artifact_evidence.tsv.gz"
        row: dict[str, object] = {"section": label, "run_root": str(root), "status": "missing"}
        if not feature_path.is_file() or not evidence_path.is_file() or not table_path.is_file():
            missing = [str(path.relative_to(root)) for path in (feature_path, evidence_path, table_path) if not path.is_file()]
            row["missing_outputs"] = ",".join(missing)
            pilot_rows.append(row)
            continue

        features = read_json(feature_path)
        evidence = read_json(evidence_path)
        row.update({
            "status": "complete",
            "processed_candidates": features.get("processed_candidates"),
            "unique_candidates_after_sampling": features.get("unique_candidates_after_sampling"),
            "covered_candidates": features.get("covered_candidates"),
            "uncovered_candidates": features.get("uncovered_candidates"),
            "pileup_capped_candidates": features.get("pileup_capped_candidates"),
            "n_sites_scored": evidence.get("n_sites"),
            "calibration_source": evidence.get("calibration", {}).get("source"),
            "n_calibration_sites": evidence.get("calibration", {}).get("n_calibration_sites"),
            "used_empirical_quantiles": evidence.get("calibration", {}).get("used_empirical_quantiles"),
        })
        for source, count in sorted(features.get("candidate_sources_after_sampling", {}).items()):
            row[f"sampled_{source}"] = count
        for status, count in evidence.get("status_counts", {}).items():
            row[f"n_{status}"] = count
        pilot_rows.append(row)

        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        support: defaultdict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        with gzip.open(table_path, "rt", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for record in reader:
                source = record["candidate_sources"]
                counts[source][record["artifact_evidence_status"]] += 1
                for metric in ("alt_n", "alt_umi_n", "n_spots_alt"):
                    try:
                        support[source][metric].append(float(record[metric]))
                    except (KeyError, TypeError, ValueError):
                        pass
        for source in sorted(counts):
            total = sum(counts[source].values())
            source_row: dict[str, object] = {"section": label, "source": source, "n_sites": total}
            for status in STATUSES:
                count = counts[source][status]
                source_row[f"n_{status}"] = count
                source_row[f"pct_{status}"] = round(100 * count / total, 3) if total else None
            for metric, values in support[source].items():
                values.sort()
                source_row[f"median_{metric}"] = values[len(values) // 2] if values else None
            source_rows.append(source_row)

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.out_dir / "pilot_qc.csv", pilot_rows)
    write_csv(args.out_dir / "source_status.csv", source_rows)

    with (args.out_dir / "README.md").open("w") as handle:
        handle.write("# Artifact-evidence cross-section pilot QC\n\n")
        handle.write("These are descriptive, non-filtering evidence summaries.\n\n")
        handle.write("| section | output status | processed | covered | pileup capped | scored |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for row in pilot_rows:
            handle.write(
                f"| {row['section']} | {row['status']} | {row.get('processed_candidates', '')} | "
                f"{row.get('covered_candidates', '')} | {row.get('pileup_capped_candidates', '')} | "
                f"{row.get('n_sites_scored', '')} |\n"
            )
        handle.write("\nSee `pilot_qc.csv` and `source_status.csv` for the machine-readable summaries.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
