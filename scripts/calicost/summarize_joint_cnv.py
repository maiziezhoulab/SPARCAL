#!/usr/bin/env python3
"""Summarize joint CalicoST arms and compare them with single-section runs."""

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_JOINT_ROOT = Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013")
DEFAULT_OUT_DIR = DEFAULT_JOINT_ROOT / "qc"
OLD_ROOT = Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906")
PATIENTS = {
    "P4": (("sec1", "P41"), ("sec2", "P42")),
    "P6": (("sec1", "P61"), ("sec2", "P62")),
}
OLD_CLONE_PREFIX = {"P4": "clone2_", "P6": "clone3_"}
ARMS = ("calicost", "calicost_unsup")


def normalize_chrom(value: str) -> str:
    chrom = value.strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom.upper() == "M":
        return "MT"
    return chrom.upper()


def chrom_key(chrom: str) -> tuple[int, object]:
    if chrom.isdigit():
        return (0, int(chrom))
    return (1, chrom)


def format_number(value: float) -> str:
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def label_key(label: str) -> tuple[int, object]:
    try:
        return (0, float(label))
    except ValueError:
        return (1, label)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def select_segment_file(arm_dir: Path, clone_prefix: Optional[str] = None) -> Optional[Path]:
    if clone_prefix is None:
        clone_dirs = sorted(arm_dir.glob("clone*_rectangle0_w1.0"))
    else:
        clone_dirs = sorted(
            path for path in arm_dir.glob(f"{clone_prefix}*") if path.is_dir()
        )
    for filename in ("cnv_seglevel.tsv", "cnv_diploid_seglevel.tsv"):
        for clone_dir in clone_dirs:
            candidate = clone_dir / filename
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def read_clone_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"BARCODES", "clone_label"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{path} lacks {sorted(required)}")
        return list(reader)


def section_label_counts(
    label_rows: Iterable[dict[str, str]], sample_id: str
) -> Counter[str]:
    suffix = "_" + sample_id
    return Counter(
        row["clone_label"]
        for row in label_rows
        if row["BARCODES"].endswith(suffix)
    )


def majority_clone(label_rows: list[dict[str, str]], sample_id: str) -> Optional[str]:
    counts = section_label_counts(label_rows, sample_id)
    if not counts:
        return None
    return sorted(counts, key=lambda label: (-counts[label], label_key(label)))[0]


def read_purity(path: Path, sample_ids: Iterable[str]) -> dict[str, list[float]]:
    result = {sample_id: [] for sample_id in sample_ids}
    if not path.is_file():
        return result
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            barcode = row.get("BARCODES", "")
            for sample_id in result:
                if barcode.endswith("_" + sample_id):
                    try:
                        result[sample_id].append(float(row.get("Tumor", "nan")))
                    except (TypeError, ValueError):
                        result[sample_id].append(float("nan"))
                    break
    return result


def purity_summary(values: list[float]) -> dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    denominator = len(values)
    return {
        "purity_total": denominator,
        "purity_finite": len(finite),
        "purity_finite_fraction": f"{len(finite) / denominator:.6f}" if denominator else "NA",
        "purity_mean": f"{statistics.fmean(finite):.6f}" if finite else "NA",
        "purity_median": f"{statistics.median(finite):.6f}" if finite else "NA",
        "purity_fraction_0.05": (
            f"{sum(value == 0.05 for value in finite) / denominator:.6f}"
            if denominator
            else "NA"
        ),
        "purity_fraction_0": (
            f"{sum(value == 0.0 for value in finite) / denominator:.6f}"
            if denominator
            else "NA"
        ),
    }


def read_segments(
    path: Path,
) -> tuple[list[str], list[tuple[str, int, int, dict[str, tuple[float, float]]]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        clones = []
        for field in fields:
            match = re.fullmatch(r"clone(.+) A", field)
            if match and f"clone{match.group(1)} B" in fields:
                clones.append(match.group(1))
        if not clones:
            raise RuntimeError(f"no clone A/B columns in {path}")
        rows = []
        for row in reader:
            start = int(float(row["START"]))
            end = int(float(row["END"]))
            if end <= start:
                continue
            states = {
                clone: (
                    float(row[f"clone{clone} A"]),
                    float(row[f"clone{clone} B"]),
                )
                for clone in clones
            }
            rows.append((normalize_chrom(row["CHR"]), start, end, states))
    return clones, rows


def summarize_genome(
    patient: str, arm: str, segment_file: Path
) -> list[dict[str, object]]:
    clones, segments = read_segments(segment_file)
    output: list[dict[str, object]] = []
    for clone in sorted(clones, key=label_key):
        state_bp: Counter[str] = Counter()
        chromosome = defaultdict(lambda: {"covered": 0, "altered": 0, "loh": 0})
        for chrom, start, end, states in segments:
            length = end - start
            allele_a, allele_b = states[clone]
            state = f"{format_number(allele_a)}|{format_number(allele_b)}"
            state_bp[state] += length
            chromosome[chrom]["covered"] += length
            if allele_a != 1 or allele_b != 1:
                chromosome[chrom]["altered"] += length
            if allele_a == 0 or allele_b == 0:
                chromosome[chrom]["loh"] += length
        total = sum(state_bp.values())
        for state in sorted(state_bp):
            output.append(
                {
                    "patient": patient,
                    "arm": arm,
                    "clone_label": clone,
                    "row_type": "state_fraction",
                    "chromosome": "",
                    "state": state,
                    "bp": state_bp[state],
                    "genome_fraction": f"{state_bp[state] / total:.6f}" if total else "NA",
                    "covered_mb": f"{total / 1e6:.6f}",
                    "altered_mb": "",
                    "loh_mb": "",
                    "segment_file": segment_file,
                }
            )
        for chrom in sorted(chromosome, key=chrom_key):
            values = chromosome[chrom]
            output.append(
                {
                    "patient": patient,
                    "arm": arm,
                    "clone_label": clone,
                    "row_type": "chromosome_summary",
                    "chromosome": chrom,
                    "state": "",
                    "bp": values["covered"],
                    "genome_fraction": "",
                    "covered_mb": f"{values['covered'] / 1e6:.6f}",
                    "altered_mb": f"{values['altered'] / 1e6:.6f}",
                    "loh_mb": f"{values['loh'] / 1e6:.6f}",
                    "segment_file": segment_file,
                }
            )
    return output


def clone_rows_for_arm(
    patient: str,
    arm: str,
    labels: list[dict[str, str]],
    purity_by_sample: dict[str, list[float]],
) -> list[dict[str, object]]:
    output = []
    for section, sample_id in PATIENTS[patient]:
        counts = section_label_counts(labels, sample_id)
        summary = purity_summary(purity_by_sample[sample_id])
        if not counts:
            output.append(
                {
                    "patient": patient,
                    "arm": arm,
                    "section": section,
                    "sample_id": sample_id,
                    "clone_label": "NA",
                    "spot_count": 0,
                    **summary,
                }
            )
            continue
        for clone, count in sorted(counts.items(), key=lambda item: label_key(item[0])):
            output.append(
                {
                    "patient": patient,
                    "arm": arm,
                    "section": section,
                    "sample_id": sample_id,
                    "clone_label": clone,
                    "spot_count": count,
                    **summary,
                }
            )
    return output


def clone_profile(
    segment_file: Path, clone: str
) -> dict[str, list[tuple[int, int, bool, bool]]]:
    clones, segments = read_segments(segment_file)
    profile = defaultdict(list)
    for chrom, start, end, states in segments:
        if clone in states:
            allele_a, allele_b = states[clone]
        elif clone == "0":
            # Some CalicoST tables omit clone 0 when it is the diploid normal clone.
            allele_a, allele_b = 1.0, 1.0
        else:
            raise RuntimeError(f"clone {clone} is absent from {segment_file}")
        profile[chrom].append(
            (start, end, allele_a != 1 or allele_b != 1, allele_a == 0 or allele_b == 0)
        )
    for chrom in profile:
        profile[chrom].sort()
    return profile


def profile_agreement(
    left: dict[str, list[tuple[int, int, bool, bool]]],
    right: dict[str, list[tuple[int, int, bool, bool]]],
) -> tuple[int, int, int]:
    covered = altered_agree = loh_agree = 0
    for chrom in set(left).intersection(right):
        left_rows = left[chrom]
        right_rows = right[chrom]
        i = j = 0
        while i < len(left_rows) and j < len(right_rows):
            l_start, l_end, l_altered, l_loh = left_rows[i]
            r_start, r_end, r_altered, r_loh = right_rows[j]
            overlap = min(l_end, r_end) - max(l_start, r_start)
            if overlap > 0:
                covered += overlap
                if l_altered == r_altered:
                    altered_agree += overlap
                if l_loh == r_loh:
                    loh_agree += overlap
            if l_end <= r_end:
                i += 1
            if r_end <= l_end:
                j += 1
    return covered, altered_agree, loh_agree


def old_profile(patient: str, section: str, sample_id: str) -> Optional[tuple[Path, str]]:
    arm_dir = OLD_ROOT / f"{patient}_{section}" / "calicost"
    segment_file = select_segment_file(arm_dir, OLD_CLONE_PREFIX[patient])
    if segment_file is None:
        return None
    labels_file = segment_file.parent / "clone_labels.tsv"
    if not labels_file.is_file():
        return None
    clone = majority_clone(read_clone_labels(labels_file), sample_id)
    if clone is None:
        return None
    return segment_file, clone


def comparison_row(
    patient: str,
    comparison: str,
    arm: str,
    left_section: str,
    left_file: Path,
    left_clone: str,
    right_section: str,
    right_file: Path,
    right_clone: str,
) -> dict[str, object]:
    covered, altered_agree, loh_agree = profile_agreement(
        clone_profile(left_file, left_clone),
        clone_profile(right_file, right_clone),
    )
    return {
        "patient": patient,
        "comparison": comparison,
        "arm": arm,
        "left_section": left_section,
        "left_segment_file": left_file,
        "left_clone_label": left_clone,
        "right_section": right_section,
        "right_segment_file": right_file,
        "right_clone_label": right_clone,
        "covered_mb": f"{covered / 1e6:.6f}",
        "altered_neutral_agreement": f"{altered_agree / covered:.6f}" if covered else "NA",
        "loh_nonloh_agreement": f"{loh_agree / covered:.6f}" if covered else "NA",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-root", type=Path, default=DEFAULT_JOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, object]] = []
    old_profiles: dict[tuple[str, str], tuple[Path, str]] = {}
    for patient, sections in PATIENTS.items():
        for section, sample_id in sections:
            selected = old_profile(patient, section, sample_id)
            if selected is None:
                print(f"WARNING: old profile missing for {patient} {section}")
            else:
                old_profiles[(patient, section)] = selected
        if all((patient, section) in old_profiles for section, _ in sections):
            left_file, left_clone = old_profiles[(patient, "sec1")]
            right_file, right_clone = old_profiles[(patient, "sec2")]
            row = comparison_row(
                patient,
                "old_sec1_vs_old_sec2",
                "baseline",
                "sec1",
                left_file,
                left_clone,
                "sec2",
                right_file,
                right_clone,
            )
            comparisons.append(row)
            print(
                f"Baseline {patient} old sec1 vs old sec2: "
                f"covered={row['covered_mb']} Mb, "
                f"altered/neutral agreement={row['altered_neutral_agreement']}, "
                f"LOH/non-LOH agreement={row['loh_nonloh_agreement']}"
            )

    completed_arms = 0
    for patient, sections in PATIENTS.items():
        patient_dir = args.joint_root / patient
        sample_ids = [sample_id for _, sample_id in sections]
        purity_by_sample = read_purity(
            patient_dir / "estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
            sample_ids,
        )
        for arm in ARMS:
            arm_dir = patient_dir / arm
            segment_file = select_segment_file(arm_dir)
            if segment_file is None:
                print(f"Joint arm not complete; skipping {patient}/{arm}")
                continue
            labels_file = segment_file.parent / "clone_labels.tsv"
            if not labels_file.is_file():
                print(f"Joint arm lacks clone labels; skipping {patient}/{arm}: {labels_file}")
                continue
            completed_arms += 1
            labels = read_clone_labels(labels_file)
            arm_out = args.out_dir / patient / arm
            clone_rows = clone_rows_for_arm(
                patient, arm, labels, purity_by_sample
            )
            write_tsv(
                arm_out / "clones_by_section.tsv",
                [
                    "patient",
                    "arm",
                    "section",
                    "sample_id",
                    "clone_label",
                    "spot_count",
                    "purity_total",
                    "purity_finite",
                    "purity_finite_fraction",
                    "purity_mean",
                    "purity_median",
                    "purity_fraction_0.05",
                    "purity_fraction_0",
                ],
                clone_rows,
            )
            genome_rows = summarize_genome(patient, arm, segment_file)
            write_tsv(
                arm_out / "genome_state.tsv",
                [
                    "patient",
                    "arm",
                    "clone_label",
                    "row_type",
                    "chromosome",
                    "state",
                    "bp",
                    "genome_fraction",
                    "covered_mb",
                    "altered_mb",
                    "loh_mb",
                    "segment_file",
                ],
                genome_rows,
            )
            for section, sample_id in sections:
                old = old_profiles.get((patient, section))
                joint_clone = majority_clone(labels, sample_id)
                if old is None or joint_clone is None:
                    print(
                        f"WARNING: cannot compare {patient} {section} with {arm}; "
                        "old profile or joint section clone is missing"
                    )
                    continue
                old_file, old_clone = old
                comparisons.append(
                    comparison_row(
                        patient,
                        "old_section_vs_joint",
                        arm,
                        section,
                        old_file,
                        old_clone,
                        section,
                        segment_file,
                        joint_clone,
                    )
                )

    write_tsv(
        args.out_dir / "vs_single_section.tsv",
        [
            "patient",
            "comparison",
            "arm",
            "left_section",
            "left_segment_file",
            "left_clone_label",
            "right_section",
            "right_segment_file",
            "right_clone_label",
            "covered_mb",
            "altered_neutral_agreement",
            "loh_nonloh_agreement",
        ],
        comparisons,
    )
    if completed_arms == 0:
        print("No completed joint arms found; wrote available old-section baselines only.")
    else:
        print(f"Summarized {completed_arms} completed joint arm(s).")
    print(f"Wrote QC under {args.out_dir}")


if __name__ == "__main__":
    main()
