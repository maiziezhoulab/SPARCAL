#!/usr/bin/env python3
"""Materialize a two-section, chr-prefixed hg19 CalicoST run."""

import argparse
import csv
import re
from pathlib import Path

from prepare_hg19_rerun import replace_keys, write_checked


CALICOST_DIR = Path("/data/maiziezhou_lab/leiy4/CalicoST")
RESOURCE_DIR = CALICOST_DIR / "hg19_resources"
DEFAULT_OUTPUT_ROOT = CALICOST_DIR / "hg19_joint_20261013"


def load_patient(manifest: Path, array_id: int) -> dict[str, str]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if int(row["array_id"]) == array_id]
    if len(matches) != 1:
        raise RuntimeError(f"array_id {array_id} has {len(matches)} manifest rows")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    row = load_patient(args.manifest, args.array_id)
    patient = row["patient"]
    source_dir = CALICOST_DIR / row["source_config_dir"]
    patient_dir = args.output_root / patient
    snp_dir = patient_dir / "snpinfo"
    purity_dir = patient_dir / "estimate_tumor_prop"
    cna_dir = patient_dir / "calicost"
    cna_unsup_dir = patient_dir / "calicost_unsup"

    input_rows = []
    for section in ("sec1", "sec2"):
        input_rows.append(
            "\t".join(
                (
                    row[f"bam_{section}"],
                    row[f"sample_id_{section}"],
                    row[f"spaceranger_outs_{section}"].rstrip("/") + "/",
                )
            )
        )
    write_checked(patient_dir / "input_filelist.tsv", "\n".join(input_rows) + "\n")

    eagle = (source_dir / "config_eagle2.yaml").read_text()
    eagle = replace_keys(
        eagle,
        {
            "samtools_sorting_mem": '"8G"',
            "region_vcf": str(
                RESOURCE_DIR / "genome1K.phase3.SNP_AF5e4.chr1toX.hg19.chr.vcf.gz"
            ),
            "phasing_panel": "/data/maiziezhou_lab/Datasets/1000G_hg19",
            "bamlist": str(patient_dir / "input_filelist.tsv"),
            "output_snpinfo": str(snp_dir),
        },
    )
    eagle_map = CALICOST_DIR / "external/Eagle_v2.4.1/tables/genetic_map_hg19_withX.txt.gz"
    if re.search(r"^eagle_genetic_map_file\s*:", eagle, re.MULTILINE):
        eagle = replace_keys(eagle, {"eagle_genetic_map_file": str(eagle_map)})
    else:
        marker = f"eagledir: {CALICOST_DIR / 'external/Eagle_v2.4.1'}\n"
        if marker not in eagle:
            raise RuntimeError("cannot place eagle_genetic_map_file in Eagle config")
        eagle = eagle.replace(
            marker, marker + f"eagle_genetic_map_file: {eagle_map}\n", 1
        )
    write_checked(patient_dir / "config_eagle2.yaml", eagle)

    common = {
        "input_filelist": str(patient_dir / "input_filelist.tsv"),
        "snp_dir": str(snp_dir),
        "geneticmap_file": str(RESOURCE_DIR / "genetic_map_hg19_merged.calicost.tsv.gz"),
        "hgtable_file": str(RESOURCE_DIR / "hgTables_hg19_gencode_v37lift37.txt"),
        "filtergenelist_file": str(RESOURCE_DIR / "ig_gene_list.txt"),
        "filterregion_file": str(RESOURCE_DIR / "HLA_regions_hg19.bed"),
    }
    purity = replace_keys(
        (source_dir / "configuration_purity").read_text(),
        {**common, "output_dir": str(purity_dir), "tumorprop_file": "None"},
    )
    source_cna = (source_dir / "configuration_cna").read_text()
    cna = replace_keys(
        source_cna,
        {
            **common,
            "output_dir": str(cna_dir),
            "tumorprop_file": str(purity_dir / "loh_estimator_tumor_prop.tsv"),
        },
    )
    cna_unsup = replace_keys(
        source_cna,
        {**common, "output_dir": str(cna_unsup_dir), "tumorprop_file": "None"},
    )
    write_checked(patient_dir / "configuration_purity", purity)
    write_checked(patient_dir / "configuration_cna", cna)
    write_checked(patient_dir / "configuration_cna_unsup", cna_unsup)
    (args.output_root / "slurm_output").mkdir(parents=True, exist_ok=True)
    print(patient_dir)
    print("Sections: " + ", ".join(row[f"sample_id_{s}"] for s in ("sec1", "sec2")))


if __name__ == "__main__":
    main()
