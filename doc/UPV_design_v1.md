# UPV 设计 v1

**日期**：2026-09-12
**状态**：默认路由。成员定义沿用 Stage-1 gate，v1 新增经验证的分层标注。
**代码**：[`scripts/6_spatial_filter/upv_v1_annotate.py`](../scripts/6_spatial_filter/upv_v1_annotate.py)

---

## 1. UPV 是什么

**Ubiquitous Private Variant**：在 1000 Genomes panel 中不存在（de novo），但在组织切片
中**广泛且均匀分布**的变异。它由行为定义（spatial behavior），不由来源定义。

Stage-1 gate：
```
alpha = spatial uniformity   （空间均匀性）
beta  = global prevalence    （全局普遍度）
UPV  <=>  alpha > T_alpha  AND  beta > T_beta
```
生产阈值：P4/P6 用 0.5/0.4，DCIS1/DCIS2 用 0.5/0.2。

---

## 2. 我们本轮做了什么（实验与结果）

### 2.1 matched WES 验证（P4/P6）— 起点，结论为负
UPV 在 matched-normal WES 的支持率仅 **1.24–1.94%**，而 1KGP panel germline 是 **80%**。
同一套 pileup、同样阈值，阳性对照通过 → 机制可靠，结论成立。

**局限**：仅 P4/P6 有 WES；exome 靶区只覆盖 ~20% 的 UPV；DCIS（占全部 UPV 的 89%）
完全无 WES。→ WES 路线最多覆盖 ~4% 的 UPV，必须转 DNA-free。

### 2.2 Substitution spectrum（DNA-free 工具一）
真 germline 基准（四样本一致）：**Ti/Tv 2.31，(C>T|G>A)/Tv 1.24**；均匀随机错误 = 0.25。

| 样本 | 类别 | Ti/Tv | A>G\|T>C | (C>T\|G>A)/Tv | edit% |
|---|---|---:|---:|---:|---:|
| P4 | panel | 2.31 | 32.3% | **1.24** | 3.1% |
| P4 | UPV | 0.59 | 19.5% | 0.28 | 5.8% |
| DCIS1 | UPV | 2.16 | **49.3%** | 0.60 | 29.6% |

**结论**：cSCC 的 UPV 是随机错误主导（Ti/Tv 0.59，接近随机的 0.5）；DCIS 的 UPV 是
**RNA editing 主导**（A>G|T>C 49.3%，29.6% 命中 editing catalogue）。

**关键陷阱**：Ti/Tv 会被 editing 欺骗（A>G 是 transition），DCIS 的 Ti/Tv 2.16 看似像
germline，实为假象。**必须用 editing-免疫的 (C>T|G>A)/Tv。**

### 2.3 gnomAD corroboration（DNA-free 工具二）
gnomAD 只作 **ground truth / 更大的 panel 用于检查**，**不作 Beagle 的 reference panel**
——扩大 panel 会让 out-of-panel 这个目标类别本身消失。

全基因组 88/88 染色体。chance-correction：
`f = (exact − null) / (panel_exact − null)`，null = 同位点 permuted-ALT 命中率。

| 样本 | 类别 | n | allele-exact | null | enrich |
|---|---|---:|---:|---:|---:|
| DCIS1 | panel | 140,773 | 100.0% | 15.1% | 6.62x |
| DCIS1 | UPV | 53,120 | 27.2% | 11.0% | **2.48x** |
| P4 | UPV | 9,491 | 9.4% | 4.2% | **2.23x** |

命中集中在**罕见变异**：AF<0.1% 占 87.9%（DCIS1）、60.3%（P4）——正是 2,504 人的
panel 漏掉、80 万人的 gnomAD 捕获的那一类。

### 2.4 期望携带数校正（**推翻了初版数量级**）
一个人不可能携带上万个 out-of-panel germline。对 AF=f 的 allele，期望携带 `1−(1−f)²≈2f`。

| 样本 | 类别 | hits | **E[carried]** | hits/E |
|---|---|---:|---:|---:|
| DCIS1 | **panel（校准基准）** | 140,707 | 69,863 | **2.0** |
| DCIS1 | UPV | 14,442 | **717** | **20.1** |
| P4 | UPV | 894 | **117** | 7.6 |

真 germline 的 hits/E = 1.6–2.0；UPV 是 5.7–20.1，**过量 3–10 倍**。
按 panel ratio 重标定后：

| 样本 | 初版报告 | **校正后** | 占比 | WES 独立估计 |
|---|---:|---:|---:|---:|
| DCIS1 | 9,688 (18.2%) | **~1,430** | 2.7% | 无 |
| P4 | 543 (5.7%) | **~190** | 2.0% | 1.45% |
| P6 | 110 (6.3%) | **~50** | 2.9% | 1.94% |

校正后四样本收敛到 **2–4%**，且 **P4/P6 与 matched-normal WES 吻合**（校正前不吻合）。
根因：DCIS1 命中里 80.2% 是 AF<1e-4，绝大多数是**位置巧合**而非该个体真的携带。

### 2.5 Allelic dropout 机制（决定了 confident set 的位置）
**问题**：germline het 在 DNA 里就是 0.5，为什么 VAF 会散开？

用已知真 germline（1KGP panel）做深度分层：

| ADP | n | VAF 中位数 | het-like | hom-like |
|---:|---:|---:|---:|---:|
| 1–3 | 74,251 | **1.000** | **4.7%** | 91.1% |
| 10–19 | 15,762 | 1.000 | 13.4% | 68.8% |
| 50–199 | 6,546 | 0.768 | 36.4% | 44.0% |
| 200+ | 2,739 | 0.623 | **51.1%** | 40.5% |

**单调收敛 → 是采样问题（allelic dropout），不是生物学（ASE/imprinting）。**
het 位点在 ADP=1 时 100% 读作 hom，ADP=2 时 50%，ADP=3 时 25%；
而 **53% 的 panel germline 位点 ADP 只有 1–3**。数据本身干净（已 UMI 去重），
根因是 10x 3' 化学每位点捕获分子数太少。

**推论**：ADP < 50 时 VAF 不可解释 → confident set 定在 **ADP ≥ 50**。

### 2.6 editing 冒充 het-like（决定了过滤顺序）
DCIS ambiguous 在 ADP≥100 时 het-like **62.4%**，**高于** panel 的 47.6%——但其中
**80.7% 是 editing catalogue 命中**，(C>T|G>A)/Tv 塌到 0.18。
editing 效率通常 10–50%，VAF 落在中间，正好进入 het 窗口。
**→ editing 过滤必须在 allelic 判据之前。**

### 2.7 `uncalled` 层（干净的阴性结果）
SeqErrModel 阶段丢弃的候选（每样本抽样 6 万）：germline 含量 **2.5–2.7%（DCIS）/
0.0%（cSCC）**，ADP≥50 时降到 **0.0–0.1%**。
**→ SeqErrModel 没有丢真 germline，召回问题不在这一环。**

### 2.8 可分性与模型尝试（为什么 v1 不改成员）
训练 gnomAD-corroborated 分类器（`data/germline_model_2026-09-12/`）：

| 分层 | n | AUC |
|---|---:|---:|
| 全体 (ADP≥1) | 43,305 | 0.890 |
| **ADP≥50** | 2,990 | **0.665** |
| **ADP≥100** | 1,155 | **0.678** |

**在 confident set 上 AUC 塌到 0.67**；高分主要来自低深度位点的深度/质量差异。
且训练标签 **~95% 是噪声**（DCIS1 有 30,353 个 PASS 正样本，但携带期望仅 1,432）。
**→ 没有证据支持改变 UPV 成员，v1 不改。**

### 2.9 Clustering（250 kb bin，12 个 DLPFC sections）

| 方法 | mean ARI |
|---|---:|
| gene expression（上界） | 0.4116 |
| **SPARCAL 1kG-defined** | **0.3626** |
| GATK | 0.3536 |
| **SPARCAL 1kG+UPV** | **0.3124** |
| Strelka2 | 0.2571 |

配对 Wilcoxon：1kG+UPV vs GATK **P=0.0425（显著更差）**；1kG+UPV vs Strelka2
P=0.0049（更好）；**1kG-defined vs GATK P=0.68（打平）**，vs Strelka2 P=0.0010（更好）。

**解读**：UPV 由空间均匀性定义，本身不携带空间结构，对聚类只能稀释。
**聚类的结论必须建立在 1kG-defined 上**；"UPV 不增益聚类"与其定义一致，是自洽性论据。

---

## 3. UPV v1 的设计

### 3.1 成员：不变
沿用 Stage-1 gate。理由见 §2.8——学习型分类器在 confident set 上未能胜出。

### 3.2 新增：三层标注（不丢弃任何变异）

```
IF   editing catalogue 精确命中        -> UPV_TIER = editing
ELIF ADP >= 50                        -> UPV_TIER = confident
ELSE                                  -> UPV_TIER = low_depth
```

顺序不可颠倒（§2.6）。

### 3.3 新增 INFO 字段

| 字段 | 含义 | 为什么需要 |
|---|---|---|
| `ADP` | 由 I16 重算的 allelic depth（ref+alt，双链） | VCF 的 DP 不是决定 VAF 可解释性的那个量 |
| `VAFR` | 重算的 ALT 比例 | 存储的 FORMAT BAF 在 homozygous-ALT 记录上不可靠 |
| `IS_EDIT` | editing catalogue 命中 | §2.6 |
| `UPV_TIER` | confident / editing / low_depth | §2.5 |

### 3.4 各层的使用规则（2026-09-12 定）
- **confident**：报备、比对、校准的唯一对象。
- **editing / low_depth**：照常 call 出来，尽可能 verify，**无法 verify 也不做断言**。

### 3.5 实际产出

| 样本 | total | confident | editing | low_depth |
|---|---:|---:|---:|---:|
| DCIS1 | 53,120 | 3,242 (6.1%) | 15,743 (29.6%) | 34,135 (64.3%) |
| DCIS2 | 36,323 | 3,002 (8.3%) | 10,212 (28.1%) | 23,109 (63.6%) |
| P4 | 9,491 | 1,472 (15.5%) | 549 (5.8%) | 7,470 (78.7%) |
| P6 | 1,744 | 256 (14.7%) | 96 (5.5%) | 1,392 (79.8%) |

注意 cSCC 与 DCIS 的 editing 比例差一个数量级（5.5–5.8% vs 28.1–29.6%），
与 §2.2 的 spectrum 诊断一致。

---

## 4. 运行方式

```bash
PY=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
$PY scripts/6_spatial_filter/upv_v1_annotate.py --root data/dcis1     --build hg38
$PY scripts/6_spatial_filter/upv_v1_annotate.py --root data/dcis2     --build hg38
$PY scripts/6_spatial_filter/upv_v1_annotate.py --root data/P4_tumor/1 --build hg19
$PY scripts/6_spatial_filter/upv_v1_annotate.py --root data/P6_tumor/1 --build hg19
```

输出 `<root>/spatial_filter_purity/<qf>/germline/denovo/upv_v1.vcf.gz` 与
`upv_v1_summary.tsv`。**绝不覆盖 `germline_denovo.vcf.gz`。**

---

## 5. 图（现有 + 建议新增）

### 现有可直接用的
- `data/dlpfc/clustering_benchmark/ari_matrix_mean.csv` → 250 kb bin ARI 箱线图
  （生成器 `scripts/postanalyze/suppA_dlpfc_germline_ari.py`，当前只画了 1kG-defined，
  需加 1kG+UPV 一列以完整呈现 §2.9）

### 建议新增（数据均已就位）
1. **VAF vs 深度收敛图**（§2.5）——panel germline 的 het-like 比例随 ADP 上升曲线。
   这是 allelic dropout 机制最直观的一张图，也是 confident set 阈值的依据。
   数据：`data/dnafree_2026-09-08/validation/vaf_by_depth_out.txt`
2. **gnomAD 富集 + AF 组成图**（§2.3）——各类 enrichment 柱状 + 命中 AF 分布。
   数据：`data/dnafree_2026-09-08/track_b/gnomad/`
3. **期望携带数校正图**（§2.4）——hits vs E[carried]，panel 作对角参考线。
   数据：`data/germline_model_2026-09-12/expected_carriage_out.txt`
4. **substitution spectrum 对比图**（§2.2）——四样本各类的 96/6 通道谱。
   数据：`data/dnafree_2026-09-08/track_c_somatic/out/*.features.tsv.gz`

绘图须遵循 `SPARCAL_pnas_2026/CLAUDE.md` 的 house style（Arial gate、无标题、
dpi=300、pdf+png、方法配色固定）。

---

## 6. 已知局限（必须随结论一起陈述）

1. **gnomAD 标签是 one-sided（PU）**：存在=真 germline；**不存在 ≠ artifact**
   （真正私有的变异按定义不在 gnomAD 中）。所有由此得到的比率都是**下界**。
2. **gnomAD 命中 ≠ germline**：gnomAD 自身含 artifact，我们的命中仅 **46% 是 PASS**，
   且 artifact 与 gnomAD 低质量记录富集在同一批困难区域。
3. **spectrum meter 在 editing 重的样本上有偏**：DCIS 报 36% 而 gnomAD 报 18.2%
   （校正后 2.7%）。**只在 cSCC 上使用。**
4. **confident 层只占 6–16%**：纯度换召回，绝大多数 UPV 无法断言。
5. **跨个体 recurrence 不能单独作 artifact 证据**：常见 SNP 在个体间共享是预期行为，
   仅在 `AF<1e-3 或不在 gnomAD` 条件下可用。v1 未使用。

---

## 7. 下一版的方向

v2 拟与 spatial filter 联合重新设计，但**不做大改**。候选方向：
- 把 `ADP` 与 editing flag 前移到 Stage-1，使 gate 本身具备 allelic 感知
  （当前 gate 只看 0/1 检出矩阵，故 alpha/beta 实测的是**表达量**而非变异来源）
- Panel holdout 实验（从 1KGP panel 扣出一部分混入 de novo 池）——目前唯一能给出
  真正 precision/recall 的设计，且完全不依赖 gnomAD
