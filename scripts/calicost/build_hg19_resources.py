#!/usr/bin/env python3
"""Build the hg19 resources required by the P4/P6 CalicoST rerun."""

import argparse
import gzip
import os
import re
import subprocess
from pathlib import Path

AUTOSOMES = tuple(str(i) for i in range(1, 23))
HLA_FILTER_GENES = {
    "BRD2", "HLA-A", "HLA-B", "HLA-C", "HLA-DMA", "HLA-DMB", "HLA-DOA",
    "HLA-DOB", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQA2", "HLA-DQB1",
    "HLA-DQB1-AS1", "HLA-DQB2", "HLA-DRA", "HLA-DRB1", "HLA-DRB5", "HLA-E",
    "HLA-F", "HLA-F-AS1", "HLA-G", "LINC02571",
}


def atomic_text(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w") as handle:
        writer(handle)
    os.replace(tmp, path)


def build_gene_resources(gtf: Path, hgtable: Path, hla_bed: Path) -> None:
    gene_name_re = re.compile(r'gene_name "([^\"]+)"')
    genes = []
    with gzip.open(gtf, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene" or not fields[0].startswith("chr"):
                continue
            chrom = fields[0][3:]
            if chrom not in AUTOSOMES:
                continue
            match = gene_name_re.search(fields[8])
            if not match:
                continue
            # GTF is 1-based closed; the CalicoST hgTable convention is 0-based half-open.
            genes.append((int(chrom), int(fields[3]) - 1, int(fields[4]), match.group(1)))
    genes.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    def write_hgtable(handle):
        handle.write("\tname2\tchrom\tcdsStart\tcdsEnd\n")
        for idx, (chrom, start, end, name) in enumerate(genes):
            handle.write(f"{idx}\t{name}\tchr{chrom}\t{start}\t{end}\n")

    def write_hla(handle):
        for chrom, start, end, name in genes:
            if name in HLA_FILTER_GENES:
                handle.write(f"chr{chrom}\t{start}\t{end}\n")

    atomic_text(hgtable, write_hgtable)
    atomic_text(hla_bed, write_hla)
    print(f"wrote {hgtable} ({len(genes)} autosomal genes)")
    print(f"wrote {hla_bed}")


def build_genetic_map(map_dir: Path, output: Path) -> None:
    def writer(handle):
        handle.write("chrom\tpos\trecomb_rate\tpos_cm\n")
        for chrom in AUTOSOMES:
            source = map_dir / f"genetic_map_GRCh37_chr{chrom}.txt.gz"
            with gzip.open(source, "rt") as source_handle:
                next(source_handle)
                for line in source_handle:
                    fields = line.split()
                    if len(fields) < 4:
                        continue
                    handle.write(f"chr{chrom}\t{fields[1]}\t{fields[2]}\t{fields[3]}\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_plain = output.with_name(output.name.removesuffix(".gz") + f".tmp.{os.getpid()}")
    with open(tmp_plain, "w") as handle:
        writer(handle)
    tmp_gz = Path(str(tmp_plain) + ".gz")
    with open(tmp_plain, "rb") as src, gzip.open(tmp_gz, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    tmp_plain.unlink()
    os.replace(tmp_gz, output)
    print(f"wrote {output}")


def first_vcf_contig(path: Path) -> str:
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                return line.split("\t", 1)[0]
    raise RuntimeError(f"no records in {path}")


def build_prefixed_panel(source: Path, output: Path, bcftools: Path, tabix: Path) -> None:
    if output.exists() and output.with_suffix(output.suffix + ".tbi").exists():
        if first_vcf_contig(output) != "chr1":
            raise RuntimeError(f"existing panel has wrong contig convention: {output}")
        print(f"reusing {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    rename = output.parent / "chrom_rename_hg19.tsv"
    rename.write_text("".join(f"{chrom}\tchr{chrom}\n" for chrom in (*AUTOSOMES, "X", "Y")))
    tmp = output.with_name(output.name + f".tmp.{os.getpid()}.vcf.gz")
    subprocess.run([str(bcftools), "annotate", "--rename-chrs", str(rename), "-Oz", "-o", str(tmp), str(source)], check=True)
    subprocess.run([str(tabix), "-f", "-p", "vcf", str(tmp)], check=True)
    if first_vcf_contig(tmp) != "chr1":
        raise RuntimeError("renamed hg19 panel does not begin with chr1")
    os.replace(tmp, output)
    os.replace(Path(str(tmp) + ".tbi"), Path(str(output) + ".tbi"))
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calicost-dir", type=Path, default=Path("/data/maiziezhou_lab/leiy4/CalicoST"))
    parser.add_argument("--bcftools", type=Path, default=Path("/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools"))
    parser.add_argument("--tabix", type=Path, default=Path("/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix"))
    args = parser.parse_args()
    resource_dir = args.calicost_dir / "hg19_resources"
    build_gene_resources(
        resource_dir / "gencode.v37lift37.annotation.gtf.gz",
        resource_dir / "hgTables_hg19_gencode_v37lift37.txt",
        resource_dir / "HLA_regions_hg19.bed",
    )
    build_genetic_map(
        resource_dir / "geneticMap-GRCh37",
        resource_dir / "genetic_map_hg19_merged.calicost.tsv.gz",
    )
    (resource_dir / "ig_gene_list.txt").write_text(
        (args.calicost_dir / "GRCh38_resources/ig_gene_list.txt").read_text()
    )
    print(f"wrote {resource_dir / 'ig_gene_list.txt'}")
    build_prefixed_panel(
        args.calicost_dir / "panel/genome1K.phase3.SNP_AF5e4.chr1toX.hg19.vcf.gz",
        resource_dir / "genome1K.phase3.SNP_AF5e4.chr1toX.hg19.chr.vcf.gz",
        args.bcftools,
        args.tabix,
    )


if __name__ == "__main__":
    main()
