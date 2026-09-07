#!/usr/bin/env python3
"""Materialize build-consistent CalicoST configs while preserving model settings."""

import argparse
import csv
import re
from pathlib import Path

CALICOST_DIR = Path("/data/maiziezhou_lab/leiy4/CalicoST")
RESOURCE_DIR = CALICOST_DIR / "hg19_resources"


def replace_keys(text: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*).*$", re.MULTILINE)
        text, count = pattern.subn(rf"\g<1>{value}", text)
        if count != 1:
            raise RuntimeError(f"expected one {key!r} entry, found {count}")
    return text


def write_checked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"refusing to overwrite changed config: {path}")
    path.write_text(text)


def load_sample(manifest: Path, array_id: int) -> dict[str, str]:
    with open(manifest, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if int(row["array_id"]) == array_id]
    if len(matches) != 1:
        raise RuntimeError(f"array_id {array_id} has {len(matches)} manifest rows")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    row = load_sample(args.manifest, args.array_id)
    sample = row["sample"]
    source_dir = CALICOST_DIR / row["source_config_dir"]
    sample_dir = args.output_root / sample
    (args.output_root / "slurm_output").mkdir(parents=True, exist_ok=True)
    snp_dir = sample_dir / "snpinfo"
    purity_dir = sample_dir / "estimate_tumor_prop"
    cna_dir = sample_dir / "calicost"

    input_filelist = f'{row["bam"]}\t{row["sample_id"]}\t{row["spaceranger_outs"]}/\n'
    write_checked(sample_dir / "input_filelist.tsv", input_filelist)

    eagle = (source_dir / "config_eagle2.yaml").read_text()
    eagle = replace_keys(eagle, {
        "region_vcf": str(RESOURCE_DIR / "genome1K.phase3.SNP_AF5e4.chr1toX.hg19.chr.vcf.gz"),
        "phasing_panel": "/data/maiziezhou_lab/Datasets/1000G_hg19",
        "bamlist": str(sample_dir / "input_filelist.tsv"),
        "output_snpinfo": str(snp_dir),
    })
    if re.search(r"^eagle_genetic_map_file\s*:", eagle, re.MULTILINE):
        eagle = replace_keys(eagle, {"eagle_genetic_map_file": str(CALICOST_DIR / "external/Eagle_v2.4.1/tables/genetic_map_hg19_withX.txt.gz")})
    else:
        marker = f"eagledir: {CALICOST_DIR / 'external/Eagle_v2.4.1'}\n"
        if marker not in eagle:
            raise RuntimeError("cannot place eagle_genetic_map_file in Eagle config")
        eagle = eagle.replace(marker, marker + f"eagle_genetic_map_file: {CALICOST_DIR / 'external/Eagle_v2.4.1/tables/genetic_map_hg19_withX.txt.gz'}\n", 1)
    write_checked(sample_dir / "config_eagle2.yaml", eagle)

    common = {
        "input_filelist": str(sample_dir / "input_filelist.tsv"),
        "snp_dir": str(snp_dir),
        "geneticmap_file": str(RESOURCE_DIR / "genetic_map_hg19_merged.calicost.tsv.gz"),
        "hgtable_file": str(RESOURCE_DIR / "hgTables_hg19_gencode_v37lift37.txt"),
        "filtergenelist_file": str(RESOURCE_DIR / "ig_gene_list.txt"),
        "filterregion_file": str(RESOURCE_DIR / "HLA_regions_hg19.bed"),
    }
    purity = replace_keys((source_dir / "configuration_purity").read_text(), {
        **common,
        "output_dir": str(purity_dir),
        "tumorprop_file": "None",
    })
    cna_mode = row.get("cna_mode", "estimated_purity").strip().lower()
    if cna_mode not in {"estimated_purity", "unsupervised"}:
        raise RuntimeError(f"unsupported cna_mode {cna_mode!r} for {sample}")
    cna_tumorprop_file = (
        "None" if cna_mode == "unsupervised"
        else str(purity_dir / "loh_estimator_tumor_prop.tsv")
    )
    cna = replace_keys((source_dir / "configuration_cna").read_text(), {
        **common,
        "output_dir": str(cna_dir),
        "tumorprop_file": cna_tumorprop_file,
    })
    write_checked(sample_dir / "configuration_purity", purity)
    write_checked(sample_dir / "configuration_cna", cna)
    (sample_dir / "slurm_output").mkdir(parents=True, exist_ok=True)
    print(sample_dir)
    print(f"CNA mode: {cna_mode}; tumorprop_file: {cna_tumorprop_file}")


if __name__ == "__main__":
    main()
