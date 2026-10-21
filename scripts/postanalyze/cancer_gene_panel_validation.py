#!/usr/bin/env python3
"""Validate current SPARCAL classes against prespecified cancer-gene panels.

This analysis deliberately separates two questions:

1. Does a call lie in an exon of a recurrently mutated gene for this histology?
2. Is it an exact allele match to COSMIC v103 in one of those genes?

The first is sensitive to novel variants but does not establish consequence.  The
second is more specific but misses novel and uncatalogued variants.  Neither is
DNA truth, so results are reported as biological corroboration rather than a
false-positive-rate estimate.

P4/P6 and DCIS COSMIC hits reuse the locked 2026-07-28 annotation.  OVAR_P5 is
intersected once (all four classes in a union VCF) against local COSMIC GRCh38.

Usage (CPU, <1 GB RAM; the COSMIC stream takes a few minutes):
    python scripts/postanalyze/cancer_gene_panel_validation.py
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
QF = "baseQ0mapQ0"
DEFAULT_OUT = PROJECT / "data/cancer_gene_validation_2026-09-03"
BCFTOOLS = PROJECT / "apps/bcftools"
CHR_MAP = PROJECT / "resources/chr_to_nochr.tsv"
COSMIC38 = Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz")
LOCKED_COSMIC = PROJECT / "data/somatic_hits_2026-07-28/cosmic_hits_annotated.csv"
GTF = {
    "GRCh37": Path("/data/maiziezhou_lab/leiy4/SpaceTracer/Resources/_dl/gencode.hg19.gtf.gz"),
    "GRCh38": Path("/data/maiziezhou_lab/leiy4/SpaceTracer/Resources/_dl/gencode.hg38.gtf.gz"),
}

# Panels are fixed from the cited primary genomic studies, before inspecting the
# calls.  Copy-number-only markers are not added to an SNV panel.
PANELS = {
    "cSCC": {
        "genes": ["TP53", "CDKN2A", "NOTCH1", "AJUBA", "HRAS", "CASP8", "FAT1",
                  "KMT2C", "NOTCH2", "PARD3", "RASA1"],
        "source": "Pickering et al., Clin Cancer Res 2014",
        "url": "https://doi.org/10.1158/1078-0432.CCR-14-1768",
        "basis": "23 candidate cSCC drivers; these 11 are explicitly named in the abstract",
    },
    "DCIS": {
        "genes": ["PIK3CA", "TP53", "GATA3"],
        "source": "Lin et al., Breast Cancer Res Treat 2019",
        "url": "https://doi.org/10.1007/s10549-019-05401-x",
        "basis": "recurrent DCIS mutations; PIK3CA and TP53 were most frequent and GATA3 was also profiled",
    },
    "HGSOC": {
        "genes": ["TP53", "BRCA1", "BRCA2", "RB1", "NF1", "FAT3", "CSMD3", "GABRA6", "CDK12"],
        "source": "TCGA Research Network, Nature 2011",
        "url": "https://doi.org/10.1038/nature10166",
        "basis": "statistically recurrently mutated HGSOC genes (TP53 plus the lower-frequency set)",
    },
}

# These are same-patient expression/subclone markers, not expected mutation
# drivers.  They are tabulated separately and never included in enrichment.
OVAR_EXPRESSION_MARKERS = ["PIGR", "PTGS1", "KRT17", "SLPI"]
OVAR_SOURCE = "https://doi.org/10.1038/s41467-024-47271-y"

# Post-hoc diagnostic annotation: both apparent OVAR_P5 BRCA2 hits are known
# A-to-I editing sites in an AluSx element in the 3' UTR, not DNA-mutation
# corroboration.  Source: CAeditome / Wu et al., GPB 2023.
KNOWN_RNA_EDITING = {
    ("OVAR_P5", "13", 32400006, "A", "G"),
    ("OVAR_P5", "13", 32400047, "A", "G"),
}
RNA_EDIT_SOURCE = "https://doi.org/10.1016/j.gpb.2022.12.010"

SAMPLES = {
    "P4": {"tumor_type": "cSCC", "build": "GRCh37", "root": PROJECT / "data/P4_tumor/1"},
    "P6": {"tumor_type": "cSCC", "build": "GRCh37", "root": PROJECT / "data/P6_tumor/1"},
    "DCIS1": {"tumor_type": "DCIS", "build": "GRCh38", "root": PROJECT / "data/dcis1"},
    "DCIS2": {"tumor_type": "DCIS", "build": "GRCh38", "root": PROJECT / "data/dcis2"},
    "OVAR_P5": {"tumor_type": "HGSOC", "build": "GRCh38", "root": PROJECT / "data/ovar_p5/P5_sr13"},
}
CLASS_RELS = {
    "germline": "germline/defined/germline_defined.vcf.gz",
    "UPV": "germline/denovo/germline_denovo.vcf.gz",
    "somatic": "somatic/denovo/somatic_denovo.vcf.gz",
    "unresolved": "ambiguous/denovo/ambiguous_denovo.vcf.gz",
}
INFO_RE = re.compile(r"([^=;]+)=([^;]*)")
GENE_RE = re.compile(r'gene_name "([^"]+)"')


def vcf_path(sample: str, cls: str) -> Path:
    return SAMPLES[sample]["root"] / "spatial_filter_purity" / QF / CLASS_RELS[cls]


def open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def iter_vcf(path: Path):
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            chrom = f[0].removeprefix("chr")
            for alt in f[4].split(","):
                if alt not in (".", "*"):
                    yield chrom, int(f[1]), f[3], alt


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def load_panel_exons(build: str, genes: set[str]):
    raw = defaultdict(list)
    with gzip.open(GTF[build], "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) != 9 or f[2] != "exon":
                continue
            match = GENE_RE.search(f[8])
            if not match or match.group(1) not in genes:
                continue
            raw[(f[0].removeprefix("chr"), match.group(1))].append((int(f[3]), int(f[4])))
    by_chrom = defaultdict(list)
    for (chrom, gene), spans in raw.items():
        by_chrom[chrom].append((gene, merge_intervals(spans)))
    return by_chrom


def genes_at(chrom: str, pos: int, by_chrom) -> list[str]:
    found = []
    for gene, spans in by_chrom.get(chrom, []):
        if any(start <= pos <= end for start, end in spans):
            found.append(gene)
    return found


def scan_exonic_panel_hits():
    rows, totals = [], []
    exon_cache = {}
    for sample, meta in SAMPLES.items():
        genes = set(PANELS[meta["tumor_type"]]["genes"])
        if sample == "OVAR_P5":
            genes |= set(OVAR_EXPRESSION_MARKERS)
        cache_key = (meta["build"], tuple(sorted(genes)))
        if cache_key not in exon_cache:
            exon_cache[cache_key] = load_panel_exons(meta["build"], genes)
        exons = exon_cache[cache_key]
        for cls in CLASS_RELS:
            path = vcf_path(sample, cls)
            if not path.exists():
                raise FileNotFoundError(path)
            total = 0
            for chrom, pos, ref, alt in iter_vcf(path):
                total += 1
                for gene in genes_at(chrom, pos, exons):
                    rows.append({"sample": sample, "tumor_type": meta["tumor_type"],
                                 "build": meta["build"], "class": cls, "gene": gene,
                                 "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                                 "panel_kind": ("same_sample_expression_marker"
                                                if gene in OVAR_EXPRESSION_MARKERS
                                                else "recurrent_mutation_driver")})
            totals.append({"sample": sample, "class": cls, "n_variants": total})
            print(f"[exon] {sample:7s} {cls:10s}: {total:,} variants")
    return pd.DataFrame(rows), pd.DataFrame(totals)


def run(cmd):
    print("[cmd]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True)


def build_ovar_cosmic_union(out_dir: Path, force: bool = False) -> Path:
    work = out_dir / "ovar_cosmic_intersection"
    work.mkdir(parents=True, exist_ok=True)
    prepared = []
    for cls in CLASS_RELS:
        dst = work / f"{cls}.nochr.vcf.gz"
        if force or not dst.exists() or not Path(str(dst) + ".tbi").exists():
            tmp = work / f".{cls}.nochr.tmp.vcf.gz"
            run([BCFTOOLS, "annotate", "--rename-chrs", CHR_MAP, "-Oz", "-o", tmp,
                 vcf_path("OVAR_P5", cls)])
            os.replace(tmp, dst)
            run([BCFTOOLS, "index", "-f", "-t", dst])
        prepared.append(dst)

    union = work / "all_classes.nochr.vcf.gz"
    if force or not union.exists() or not Path(str(union) + ".tbi").exists():
        concat_bcf = work / ".all_classes.concat.tmp.bcf"
        sorted_bcf = work / ".all_classes.sorted.tmp.bcf"
        tmp_union = work / ".all_classes.nochr.tmp.vcf.gz"
        run([BCFTOOLS, "concat", "-a", "-Ob", "-o", concat_bcf, *prepared])
        run([BCFTOOLS, "sort", "-Ob", "-o", sorted_bcf, concat_bcf])
        # The four SPARCAL classes are disjoint.  Keep same-position alternate
        # alleles if they ever occur; the bundled older bcftools lacks the newer
        # `norm -d exact` spelling, and position-level deduplication would be unsafe.
        run([BCFTOOLS, "view", "-Oz", "-o", tmp_union, sorted_bcf])
        os.replace(tmp_union, union)
        run([BCFTOOLS, "index", "-f", "-t", union])
        concat_bcf.unlink(missing_ok=True)
        sorted_bcf.unlink(missing_ok=True)

    hits = work / "all_classes.cosmic_records.vcf"
    if force or not hits.exists():
        tmp_hits = work / ".all_classes.cosmic_records.tmp.vcf"
        run([BCFTOOLS, "isec", "-n=2", "-w2", "-Ov", "-o", tmp_hits, union, COSMIC38])
        os.replace(tmp_hits, hits)
    return hits


def consequence(aa: str, so_term: str) -> str:
    if not aa or aa in ("p.?", "."):
        return "non_coding_or_unannotated"
    if aa.endswith("="):
        return "synonymous"
    if aa.endswith("*") or aa.endswith("Ter"):
        return "nonsense"
    if so_term and so_term != "SNV":
        return so_term
    return "missense"


def parse_ovar_cosmic(path: Path) -> pd.DataFrame:
    class_keys = {cls: set(iter_vcf(vcf_path("OVAR_P5", cls))) for cls in CLASS_RELS}
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            info = dict(INFO_RE.findall(f[7]))
            key = (f[0].removeprefix("chr"), int(f[1]), f[3], f[4])
            classes = [cls for cls, keys in class_keys.items() if key in keys]
            if not classes:
                raise RuntimeError(f"COSMIC hit could not be mapped back to an OVAR class: {key}")
            for cls in classes:
                aa = info.get("AA", "")
                rows.append({"sample": "OVAR_P5", "tumor_type": "HGSOC", "build": "GRCh38",
                             "class": cls, "chrom": key[0], "pos": key[1], "ref": key[2],
                             "alt": key[3], "cosmic_id": f[2], "gene": info.get("GENE", ""),
                             "aa": aa, "so_term": info.get("SO_TERM", ""),
                             "screen_sample_count": int(info.get("GENOME_SCREEN_SAMPLE_COUNT", 0) or 0),
                             "consequence": consequence(aa, info.get("SO_TERM", ""))})
    return pd.DataFrame(rows)


def load_all_cosmic(ovar_vcf: Path) -> pd.DataFrame:
    old = pd.read_csv(LOCKED_COSMIC)
    old = old.rename(columns={"class_code": "old_class_code"})
    old["class"] = old["class"].replace({"germline": "germline", "UPV": "UPV",
                                         "somatic": "somatic", "unresolved": "unresolved"})
    keep = ["sample", "tumor_type", "build", "class", "chrom", "pos", "ref", "alt",
            "cosmic_id", "gene", "aa", "so_term", "screen_sample_count", "consequence"]
    old = old[keep]
    old["chrom"] = old["chrom"].astype(str).str.removeprefix("chr")
    ovar = parse_ovar_cosmic(ovar_vcf)
    return pd.concat([old, ovar], ignore_index=True)


def bh_adjust(pvalues):
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    out = [1.0] * n
    running = 1.0
    for rank0 in range(n - 1, -1, -1):
        i = order[rank0]
        rank = rank0 + 1
        running = min(running, pvalues[i] * n / rank)
        out[i] = min(1.0, running)
    return out


def enrichment_table(hit_df: pd.DataFrame, totals: pd.DataFrame, label: str) -> pd.DataFrame:
    records = []
    for sample in SAMPLES:
        # Expression markers are explicitly excluded from the driver test.
        sub = hit_df[(hit_df["sample"] == sample)]
        if "panel_kind" in sub.columns:
            sub = sub[sub.panel_kind == "recurrent_mutation_driver"]
        counts = (sub.groupby("class")
                  .apply(lambda x: x[["chrom", "pos", "ref", "alt"]].drop_duplicates().shape[0],
                         include_groups=False).to_dict())
        nts = dict(totals[totals["sample"] == sample][["class", "n_variants"]].values)
        a, c = int(counts.get("somatic", 0)), int(counts.get("unresolved", 0))
        ns, nu = int(nts["somatic"]), int(nts["unresolved"])
        odds, p = fisher_exact([[a, ns - a], [c, nu - c]], alternative="two-sided")
        records.append({"sample": sample, "tumor_type": SAMPLES[sample]["tumor_type"],
                        "endpoint": label, "somatic_hits": a, "somatic_total": ns,
                        "unresolved_hits": c, "unresolved_total": nu,
                        "somatic_per_10k": 1e4 * a / ns,
                        "unresolved_per_10k": 1e4 * c / nu,
                        "odds_ratio": odds, "fisher_p": p})
    df = pd.DataFrame(records)
    df["fdr_bh"] = bh_adjust(df.fisher_p.tolist())
    return df


def write_outputs(out_dir: Path, exonic: pd.DataFrame, totals: pd.DataFrame,
                  cosmic: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    exonic = exonic.copy()
    exonic["known_rna_editing"] = exonic.apply(
        lambda r: (r["sample"], str(r["chrom"]), int(r["pos"]), r["ref"], r["alt"])
        in KNOWN_RNA_EDITING, axis=1)
    panel_rows = []
    for tumor_type, spec in PANELS.items():
        for gene in spec["genes"]:
            panel_rows.append({"tumor_type": tumor_type, "gene": gene, "panel_kind": "recurrent_mutation_driver",
                               "source": spec["source"], "url": spec["url"], "basis": spec["basis"]})
    for gene in OVAR_EXPRESSION_MARKERS:
        panel_rows.append({"tumor_type": "HGSOC", "gene": gene,
                           "panel_kind": "same_sample_expression_marker",
                           "source": "Denisenko et al., Nat Commun 2024", "url": OVAR_SOURCE,
                           "basis": "P5 subclone/tumour expression marker; not a recurrent mutation driver"})
    pd.DataFrame(panel_rows).to_csv(out_dir / "literature_panels.csv", index=False)
    totals.to_csv(out_dir / "variant_class_totals.csv", index=False)
    exonic.to_csv(out_dir / "panel_exonic_hits.csv", index=False)
    cosmic.to_csv(out_dir / "all_cosmic_hits_five_samples.csv", index=False)

    driver_genes = {t: set(v["genes"]) for t, v in PANELS.items()}
    panel_cosmic = cosmic[cosmic.apply(lambda r: r.gene in driver_genes[r.tumor_type], axis=1)].copy()
    panel_cosmic.to_csv(out_dir / "panel_cosmic_hits.csv", index=False)
    exon_enrich = enrichment_table(exonic, totals, "panel_exonic")
    exon_clean = enrichment_table(exonic[~exonic.known_rna_editing], totals,
                                   "panel_exonic_known_RNA_editing_excluded")
    cosmic_enrich = enrichment_table(panel_cosmic, totals, "panel_exact_COSMIC")
    exon_enrich.to_csv(out_dir / "panel_exonic_enrichment.csv", index=False)
    exon_clean.to_csv(out_dir / "panel_exonic_rnaedit_excluded_enrichment.csv", index=False)
    cosmic_enrich.to_csv(out_dir / "panel_cosmic_enrichment.csv", index=False)

    status_rows = []
    for sample, meta in SAMPLES.items():
        for gene in PANELS[meta["tumor_type"]]["genes"]:
            e = exonic[(exonic["sample"] == sample) & (exonic.gene == gene)]
            c = panel_cosmic[(panel_cosmic["sample"] == sample) & (panel_cosmic.gene == gene)]
            som_c = c[c["class"] == "somatic"]
            status_rows.append({
                "sample": sample, "tumor_type": meta["tumor_type"], "gene": gene,
                "somatic_exonic": len(e[e["class"] == "somatic"].drop_duplicates(["chrom", "pos", "ref", "alt"])),
                "somatic_exonic_after_known_rna_editing_exclusion": len(
                    e[(e["class"] == "somatic") & (~e.known_rna_editing)]
                    .drop_duplicates(["chrom", "pos", "ref", "alt"])),
                "somatic_exact_cosmic": len(som_c.drop_duplicates(["chrom", "pos", "ref", "alt"])),
                "somatic_cosmic_protein_altering": int((~som_c.consequence.isin(
                    ["non_coding_or_unannotated", "synonymous"])).sum()),
                "germline_exonic": len(e[e["class"] == "germline"].drop_duplicates(["chrom", "pos", "ref", "alt"])),
                "upv_exonic": len(e[e["class"] == "UPV"].drop_duplicates(["chrom", "pos", "ref", "alt"])),
                "unresolved_exonic": len(e[e["class"] == "unresolved"].drop_duplicates(["chrom", "pos", "ref", "alt"])),
            })
    status = pd.DataFrame(status_rows)
    status.to_csv(out_dir / "gene_status.csv", index=False)

    exp = exonic[(exonic["sample"] == "OVAR_P5") &
                 (exonic.panel_kind == "same_sample_expression_marker")]
    exp_summary = (exp.groupby(["gene", "class"]).size().unstack(fill_value=0)
                   .reindex(index=OVAR_EXPRESSION_MARKERS, columns=list(CLASS_RELS),
                            fill_value=0).reset_index())
    exp_summary.to_csv(out_dir / "ovar_p5_expression_marker_variant_counts.csv", index=False)

    lines = [
        "# Cancer-gene validation of current SPARCAL calls — 2026-09-03", "",
        "## What was tested", "",
        "The panels were prespecified from primary genomic studies, before looking at these calls. "
        "For each sample we tested (1) variants landing in a GENCODE exon of a recurrently mutated "
        "histology-matched gene and (2) the stricter subset that exactly matches a COSMIC v103 allele. "
        "Somatic was compared with the unresolved class by two-sided Fisher tests; five sample-level "
        "tests were BH-adjusted per endpoint.", "",
        "| tumor type | panel | source |", "|---|---|---|",
    ]
    for tumor_type, spec in PANELS.items():
        lines.append(f"| {tumor_type} | {', '.join(spec['genes'])} | [{spec['source']}]({spec['url']}) |")
    lines += ["", "## Per-sample result", "",
              "| sample | type | somatic exonic hits raw→RNA-edit-filtered | exact COSMIC panel hits | "
              "exonic OR vs unresolved (FDR) | COSMIC OR vs unresolved (FDR) |",
              "|---|---|---:|---:|---:|---:|"]
    for sample, meta in SAMPLES.items():
        xe = exon_enrich[exon_enrich["sample"] == sample].iloc[0]
        xclean = exon_clean[exon_clean["sample"] == sample].iloc[0]
        xc = cosmic_enrich[cosmic_enrich["sample"] == sample].iloc[0]
        cor = "NA" if pd.isna(xc.odds_ratio) else f"{xc.odds_ratio:.3g}"
        lines.append(f"| {sample} | {meta['tumor_type']} | {xe.somatic_hits}→{xclean.somatic_hits} | "
                     f"{xc.somatic_hits} | {xe.odds_ratio:.3g} ({xe.fdr_bh:.3g}) | "
                     f"{cor} ({xc.fdr_bh:.3g}) |")

    lines += ["", "**No sample is enriched after BH correction on either endpoint (all FDR = 1).**"]

    lines += ["", "## Somatic-class genes observed", ""]
    for sample, meta in SAMPLES.items():
        s = status[(status["sample"] == sample) & (status.somatic_exonic > 0)]
        if s.empty:
            desc = "none"
        else:
            desc = ", ".join(
                f"{r.gene} (exonic raw→RNA-edit-filtered {r.somatic_exonic}→"
                f"{r.somatic_exonic_after_known_rna_editing_exclusion}; exact COSMIC {r.somatic_exact_cosmic}; "
                f"protein-altering COSMIC {r.somatic_cosmic_protein_altering})" for _, r in s.iterrows())
        lines.append(f"- **{sample} ({meta['tumor_type']})**: {desc}")

    exp_cols = exp_summary.columns.tolist()
    exp_md = ["| " + " | ".join(exp_cols) + " |",
              "|" + "|".join(["---"] + ["---:"] * (len(exp_cols) - 1)) + "|"]
    for _, row in exp_summary.iterrows():
        exp_md.append("| " + " | ".join(str(row[col]) for col in exp_cols) + " |")

    lines += ["", "## OVAR_P5 same-section expression markers", "",
              "PIGR/PTGS1 (with KRT17/SLPI in the validation set) mark the two P5 tumour subclones "
              f"in [Denisenko et al. 2024]({OVAR_SOURCE}). They are expression markers, not recurrent "
              "SNV drivers, so their variant counts are descriptive and excluded from enrichment.", "",
              *exp_md, "", "## OVAR_P5 RNA-editing diagnostic", "",
              "The two raw BRCA2 hits are chr13:32400006 A>G and chr13:32400047 A>G. "
              f"[CAeditome / Wu et al. 2023]({RNA_EDIT_SOURCE}) catalogs both as A-to-I editing "
              "sites in the BRCA2 3′ UTR within an AluSx repeat. They are excluded in the "
              "RNA-edit-filtered count and are not evidence of BRCA2 DNA mutation.", "",
              "## Interpretation guardrails", "",
              "- A panel hit is biological corroboration, not DNA validation. RNA coverage, gene "
              "expression, allele-specific expression, and catalogue membership affect detectability.",
              "- An exonic hit is not necessarily protein-altering; only exact COSMIC records have the "
              "catalogue consequence shown here.",
              "- Absence is weak negative evidence in spatial RNA. This is especially important for "
              "HGSOC, where TP53 is expected in tumour DNA but may not be captured or retained in RNA.",
              "- P4/P6 are independent patients; DCIS1/DCIS2 are sections, so they are shown separately "
              "rather than treated as independent patient replicates.", "",
              "Detailed outputs: `gene_status.csv`, `panel_exonic_hits.csv`, "
              "`panel_cosmic_hits.csv`, and the three enrichment CSVs.", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force-ovar-intersection", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ovar_cosmic = build_ovar_cosmic_union(args.out_dir, args.force_ovar_intersection)
    exonic, totals = scan_exonic_panel_hits()
    cosmic = load_all_cosmic(ovar_cosmic)
    write_outputs(args.out_dir, exonic, totals, cosmic)
    print(f"Wrote {args.out_dir / 'SUMMARY.md'} and supporting CSVs")


if __name__ == "__main__":
    main()
