# Why the COSMIC cascade descends, and how concentrated it is in the xMHC — 2026-08-20

Two questions raised on the Figure 4 (COSMIC) panels:

1. **Panel a.** The 1KGP-resolved germline class has by far the highest COSMIC hit
   rate (6.3–12.6%) — an order of magnitude above the somatic class. Why, and does
   the descending cascade germline > UPV > somatic > unresolved validate the calls?
2. **Panel b.** Is there a concentration of calls in the extended MHC, and how
   prominent is it?

Reproduce with `xmhc_prominence.py` (reads `cosmic_amb/isec_*/000{0,2}.vcf`) and
`cosmic_germline_content.py`. The two key files the latter needs are intermediates,
deleted to keep this package small; regenerate with:

```
C=/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz
P=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19
bcftools query -r 1 -f '%POS:%REF:%ALT\n' $C | awk -F: 'length($2)==1&&length($3)==1' | sort -u > cosmic_chr1.keys
bcftools query -r chr1 -f '%POS:%REF:%ALT\t%AF\n' $P/hg19_chr1.vcf.gz \
  | awk -F'\t' '{split($1,a,":"); if(length(a[2])==1&&length(a[3])==1) print}' | sort -u -k1,1 > panel_chr1.keys
python cosmic_germline_content.py panel_chr1.keys cosmic_chr1.keys
```

## 1. The catalog itself carries common germline polymorphism

Measured on chr1, allele-exact, COSMIC v103 Genome Screens Mutant vs the 1000G hg19
panel Beagle uses:

| quantity | value |
|---|---|
| COSMIC chr1 unique biallelic SNV alleles | 1,216,026 |
| ... that are also a 1000G panel allele | 89,567 (7.37%) |
| ... common (panel AF ≥ 1%) | 41,158 (3.38% of COSMIC) |
| **P(in COSMIC \| common 1000G allele)** | **4.32%** |
| P(in COSMIC \| rare 1000G allele) | 0.923% |
| **enrichment, common over rare** | **4.7×** |

If COSMIC were a clean somatic catalog, being a *common polymorphism* should not
raise the chance of being catalogued. It raises it 4.7-fold. Genome Screens Mutant
aggregates tumour screens, many unmatched, so germline variants are deposited as
"mutations". **This alone puts any 1KGP-resolved class an order of magnitude above
a non-panel class, independent of anything SPARCAL does.**

## 2. It is not a coverage effect in our data

Median spot count (`NS`) of COSMIC-hit variants, germline vs somatic class:

| sample | germline | somatic |
|---|---|---|
| P4 | 9 | 13 |
| P6 | 6 | 15 |
| DCIS1 | 20 | 33 |
| DCIS2 | 15 | 25 |

The somatic class is detected in **more** spots than the germline class in all four
samples (Stage 2 ranks on spatial clustering, so retained candidates are multi-spot
by construction). The germline class therefore does not top the cascade because it
is better covered — it tops it because of what COSMIC is.

## 3. What the cascade can and cannot support

- **Cannot** be read as validating the somatic calls. A cascade that validated
  somatic classification would put *somatic* highest; ours puts germline ~10× above
  it, for the catalog reason above. The rung ordering measures how much common
  polymorphism each class contains, which is the classifier's *input*, not a test.
- **Can** be read as validating the **germline** classification. The 1KGP class
  behaves exactly as a correctly identified germline class should against a catalog
  known to be contaminated by common polymorphism, and it does so at *lower* spot
  coverage than the somatic class. This is a coherence check on the classifier
  using an external resource that played no part in calling, and it is the same
  argument as the SpatialSNV/Mutect2 germline-leakage result.
- The only internally controlled comparison remains somatic vs unresolved (same
  candidate pool, differing only in the cascade decision): 1.16–1.33×, which nulls
  under depth adjustment. Unchanged by this analysis.

## 4. xMHC concentration and prominence

xMHC = chr6:28–34 Mb = **0.19% of the genome**. `prominence` = COSMIC hit rate
inside xMHC ÷ rate outside.

| sample | class | % of set in xMHC | % of COSMIC hits from xMHC | rate in | rate out | prominence | Fisher p |
|---|---|---:|---:|---:|---:|---:|---:|
| P4 | germline | 1.60 | 4.0 | 31.32 | 12.275 | 2.6× | 2.9e-38 |
| P4 | UPV | 5.37 | 22.8 | 12.94 | 2.494 | 5.2× | 1.0e-24 |
| P4 | **somatic** | 1.67 | **12.8** | 9.51 | 1.104 | **8.6×** | 1.2e-18 |
| P4 | unresolved | 0.65 | 4.1 | 5.96 | 0.905 | 6.6× | 9.2e-33 |
| P6 | germline | 1.17 | 2.6 | 14.26 | 6.224 | 2.3× | 5.8e-28 |
| P6 | UPV | 1.55 | 19.0 | 29.63 | 1.980 | 15.0× | 9.0e-08 |
| P6 | **somatic** | 1.95 | **11.7** | 4.21 | 0.635 | **6.6×** | 1.2e-25 |
| P6 | unresolved | 0.36 | 1.6 | 2.45 | 0.539 | 4.5× | 1.7e-18 |
| DCIS1 | germline | 0.99 | 1.7 | 15.09 | 8.991 | 1.7× | 1.9e-13 |
| DCIS1 | UPV | 2.48 | 18.1 | 13.26 | 1.525 | 8.7× | 3.0e-97 |
| DCIS1 | **somatic** | 0.28 | **6.9** | 25.49 | 0.947 | **26.9×** | 2.7e-15 |
| DCIS1 | unresolved | 0.26 | 2.9 | 9.72 | 0.851 | 11.4× | 3.3e-30 |
| DCIS2 | germline | 0.96 | 1.3 | 12.90 | 9.657 | 1.3× | 1.3e-04 |
| DCIS2 | UPV | 3.46 | 25.6 | 15.06 | 1.568 | 9.6× | 2.2e-109 |
| DCIS2 | **somatic** | 0.96 | **18.5** | 19.01 | 0.815 | **23.3×** | 1.4e-45 |
| DCIS2 | unresolved | 0.33 | 3.4 | 8.66 | 0.820 | 10.6× | 1.5e-43 |

**The concentration is real, large, and significant in every sample × class.**
It is *weakest* in the germline class (1.3–2.6×) and *strongest* in the non-panel
classes (somatic 6.6–26.9×). A region carrying 0.19% of the genome supplies
6.9–25.6% of the COSMIC hits.

## 5. This is exactly what drives the panel-b sensitivity

Splitting the somatic/unresolved ratio by region:

| sample | inside xMHC | outside xMHC |
|---|---|---|
| P4 | 1.60× (p=0.019) | 1.22× (p=0.0042) |
| P6 | 1.72× (p=0.003) | 1.18× (p=0.0011) |
| DCIS1 | 2.62× (p=0.002) | 1.11× (p=0.099) |
| DCIS2 | 2.20× (p=0.000) | 0.99× (p=0.547) |

The outside-xMHC column reproduces the manuscript's published xMHC-excluded ratios
(1.22 / 1.18 / 1.11 / 0.99) and p-values exactly, which validates this pipeline
against the existing result. The new information is the **inside** column: the
somatic/unresolved separation is 1.6–2.6× within the xMHC and 0.99–1.22× outside
it. The class separation reported in the paper is therefore substantially an
MHC phenomenon, which is the mechanism behind the tumour-type-specific behaviour
already reported (it survives in the high-TMB cSCC cases and vanishes in DCIS2).

Related: `snv_calling/data/somatic_hits_2026-07-28/SUMMARY.md`,
`PAPER_PLAN.md` Story C3, Decision D6.
