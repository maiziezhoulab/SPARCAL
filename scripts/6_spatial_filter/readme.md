# SNV Classification Methodology - Detailed Explanation

## Overview

Your pipeline uses a **probabilistic dual-scoring framework** to classify SNVs into:
- **Germline**: Present across most/all tissue, independent of tumor regions
- **Somatic**: Correlated with tumor purity, spatially clustered in tumor regions  
- **Ambiguous**: Mixed signals, unclear classification

---

## Classification Thresholds

### **Primary Decision Thresholds** (Default Values)
```python
germline_threshold = 0.5  # Default
somatic_threshold = 0.4   # Default
```

### **Expression Thresholds** (Pre-filters)
```python
min_expression_germline = 2  # Must appear in ≥2 spots
min_expression_somatic = 1   # Must appear in ≥1 spot
```

---

## Classification Logic

### **Decision Tree:**

```
For each variant:
  1. Check expression threshold
     - If < min_expression for both categories → SKIP variant
  
  2. Calculate both scores
     - germline_score (0-1 scale)
     - somatic_score (0-1 scale)
  
  3. Apply decision rules:
     
     IF germline_score > 0.5 AND somatic_score < 0.4:
        → GERMLINE
     
     ELIF somatic_score > 0.4 AND germline_score < 0.5:
        → SOMATIC
     
     ELSE:
        → AMBIGUOUS
```

### **Visual Representation:**

```
Germline Score
     ↑
 1.0 |  A   A   A   A   A
     |  A   A   A   A   A
 0.5 |  A   A   G   G   G
     |  A   A   G   G   G
 0.4 |  S   S   A   G   G
     |  S   S   A   G   G
 0.0 |  S   S   S   A   A
     |________________________→ Somatic Score
     0.0     0.4    0.5    1.0

Legend:
G = Germline
S = Somatic  
A = Ambiguous
```

---

## Germline Score Calculation

### **Formula:**
```
Germline Score = α × S_uniform + β × S_prevalence + γ × S_independence
```

### **Weights:**
```python
α (alpha)  = 0.4  # Spatial uniformity weight
β (beta)   = 0.3  # Global prevalence weight
γ (gamma)  = 0.3  # Purity independence weight
```

### **Component Metrics:**

#### **1. Spatial Uniformity Score (S_uniform)** - α = 0.4

**Purpose:** Germline variants should be evenly distributed across tissue

**Calculation:**
```python
# Step 1: Get all spots with this variant
spots_with_variant = [barcodes where variant is present]

# Step 2: Calculate pairwise distances between all spots
positions = [(x, y) for each spot]
distances = distance_matrix(positions, positions)

# Step 3: Calculate coefficient of variation (CV) of distances
cv = std(distances) / mean(distances)

# Step 4: Convert CV to uniformity score
uniformity_score = 1.0 / (1.0 + cv)
```

**Interpretation:**
- **High CV** (large variation in distances) → **Low uniformity** → **Low score** (0.1-0.3)
  - Variant is clustered or patchy
- **Low CV** (consistent spacing) → **High uniformity** → **High score** (0.7-0.9)
  - Variant is evenly spread

**Example Values:**
- Perfect grid distribution: uniformity ≈ 0.85
- Highly clustered: uniformity ≈ 0.25
- Random distribution: uniformity ≈ 0.50

---

#### **2. Global Prevalence Score (S_prevalence)** - β = 0.3

**Purpose:** Germline variants should appear in many spots

**Calculation:**
```python
# Step 1: Calculate proportion of spots with variant
prevalence = spots_with_variant / total_spots

# Step 2: Apply sigmoid transformation (centered at 10%)
score = 1.0 / (1.0 + exp(-20 × (prevalence - 0.1)))
```

**Interpretation:**
- **prevalence < 5%**: score ≈ 0.05 (very rare, likely not germline)
- **prevalence = 10%**: score = 0.50 (threshold)
- **prevalence = 20%**: score ≈ 0.98 (common, likely germline)
- **prevalence > 30%**: score ≈ 1.00 (very common, strong germline signal)

**The sigmoid shape:**
```
 1.0 |        _______________
     |      /
S    |    /
c  0.5|  /
o    | /
r    |/___________________
e    0.0
         0%    10%   20%   30%
             Prevalence
```

---

#### **3. Purity Independence Score (S_independence)** - γ = 0.3

**Purpose:** Germline variants should NOT correlate with tumor purity

**Calculation:**
```python
# Step 1: Separate tumor purity values
purities_with_variant = [purity for spots WITH variant]
purities_without_variant = [purity for spots WITHOUT variant]

# Step 2: Kolmogorov-Smirnov 2-sample test
statistic, pvalue = ks_2samp(purities_with, purities_without)

# Step 3: Return p-value as score
independence_score = pvalue
```

**Statistical Test - Kolmogorov-Smirnov (KS) Test:**
- **Null hypothesis**: The two distributions are the same
- **High p-value (>0.3)**: Distributions are similar → Variant is independent of purity → **Germline-like**
- **Low p-value (<0.05)**: Distributions differ significantly → Variant depends on purity → **Not germline**

**Interpretation:**
- **p-value = 0.9**: Strong independence → **Score = 0.9** → Likely germline
- **p-value = 0.5**: Moderate independence → **Score = 0.5** → Neutral
- **p-value = 0.01**: Strong dependence → **Score = 0.01** → Likely somatic

**Minimum Sample Size:** Requires ≥3 spots with AND ≥3 without variant, else returns 0.5 (neutral)

---

### **Germline Score Example:**

```python
# Example variant found in 80 out of 200 spots
S_uniform = 0.75        # Evenly distributed (low CV)
S_prevalence = 1.00     # 40% prevalence (well above 10%)
S_independence = 0.85   # p-value = 0.85 (independent of purity)

Germline_Score = 0.4 × 0.75 + 0.3 × 1.00 + 0.3 × 0.85
               = 0.30 + 0.30 + 0.255
               = 0.855

→ Strong germline signal!
```

---

## Somatic Score Calculation

### **Formula:**
```
Somatic Score = δ × S_purity + ε × S_clone + ζ × S_cluster

where: S_clone = S_cluster × S_purity  (simplified)
```

### **Weights:**
```python
δ (delta)   = 0.5  # Purity correlation weight
ε (epsilon) = 0.2  # Clone-specific weight
ζ (zeta)    = 0.3  # Spatial clustering weight
```

### **Component Metrics:**

#### **1. Purity Correlation Score (S_purity)** - δ = 0.5

**Purpose:** Somatic variants should correlate with tumor purity

**Calculation:**
```python
# Step 1: Create binary variable for variant presence
variant_presence = [1 if variant in spot else 0 for each spot]
purities = [tumor_purity for each spot]

# Step 2: Point-biserial correlation
correlation, pvalue = pointbiserialr(variant_presence, purities)

# Step 3: Take positive correlation only
purity_score = max(0, correlation)
```

**Statistical Test - Point-Biserial Correlation:**
- Tests correlation between **binary variable** (has variant?) and **continuous variable** (purity)
- Range: -1 to +1
- **Positive correlation**: Variant appears MORE in high-purity spots → Somatic
- **Negative correlation**: Variant appears MORE in low-purity spots → NOT somatic (set to 0)
- **Zero correlation**: No relationship → Neutral

**Interpretation:**
- **r = 0.7-0.9**: Strong positive correlation → **Somatic**
- **r = 0.3-0.5**: Moderate correlation → **Possible somatic**
- **r = -0.3**: Negative correlation → **Score = 0** (not somatic)
- **r = 0.05**: No correlation → **Score = 0.05** (not somatic)

---

#### **2. Spatial Clustering Score (S_cluster)** - ζ = 0.3

**Purpose:** Somatic variants should be clustered in tumor regions

**Calculation:**
```python
# Step 1: For each spot WITH the variant
clustering_scores = []
for spot in spots_with_variant:
    # Step 2: Get spatial neighbors (within distance threshold)
    neighbors = get_neighbors(spot, distance=2.0)
    
    # Step 3: Calculate local clustering coefficient
    neighbors_with_variant = count(neighbors with variant)
    cluster_coef = neighbors_with_variant / total_neighbors
    clustering_scores.append(cluster_coef)

# Step 4: Average across all spots
clustering_score = mean(clustering_scores)
```

**Interpretation:**
- **Clustering = 0.9**: 90% of neighbors also have variant → **Highly clustered** → Somatic-like
- **Clustering = 0.5**: 50% of neighbors have variant → **Moderate clustering**
- **Clustering = 0.1**: 10% of neighbors have variant → **Scattered** → Not somatic

**Neighbor Distance:** Default = 2.0 spatial units (typically spots)

---

#### **3. Clone-Specific Score (S_clone)** - ε = 0.2

**Purpose:** Combine clustering and purity to identify tumor clones

**Calculation:**
```python
S_clone = S_cluster × S_purity
```

**Rationale:**
- True somatic variants should be:
  1. **Clustered** (not scattered)
  2. **Correlated with purity** (in tumor regions)
- This metric captures variants that satisfy BOTH conditions

**Interpretation:**
- **S_clone = 0.81** (0.9 × 0.9): Strong cluster + strong purity → **Strong clone signal**
- **S_clone = 0.45** (0.9 × 0.5): Strong cluster but weak purity → **Moderate signal**
- **S_clone = 0.09** (0.3 × 0.3): Weak cluster + weak purity → **Weak signal**

---

### **Somatic Score Example:**

```python
# Example variant found in 25 out of 200 spots, clustered in high-purity region
S_purity = 0.75         # r = 0.75, strong positive correlation
S_cluster = 0.85        # 85% of neighbors also have variant
S_clone = 0.85 × 0.75   # = 0.6375

Somatic_Score = 0.5 × 0.75 + 0.2 × 0.6375 + 0.3 × 0.85
              = 0.375 + 0.1275 + 0.255
              = 0.7575

→ Strong somatic signal!
```

---

## Classification Examples

### **Example 1: Clear Germline**
```
Variant appears in 150/200 spots, evenly distributed

Germline Score:
  S_uniform = 0.82      (evenly spread)
  S_prevalence = 1.00   (75% prevalence)
  S_independence = 0.88 (p = 0.88, independent of purity)
  
  Total = 0.4×0.82 + 0.3×1.00 + 0.3×0.88 = 0.892

Somatic Score:
  S_purity = 0.05       (r = 0.05, no correlation)
  S_cluster = 0.45      (scattered neighbors)
  S_clone = 0.0225
  
  Total = 0.5×0.05 + 0.2×0.0225 + 0.3×0.45 = 0.164

Classification: GERMLINE (0.892 > 0.5 AND 0.164 < 0.4) ✅
```

---

### **Example 2: Clear Somatic**
```
Variant appears in 30/200 spots, clustered in high-purity region

Germline Score:
  S_uniform = 0.25      (highly clustered, high CV)
  S_prevalence = 0.12   (15% prevalence, low sigmoid)
  S_independence = 0.02 (p = 0.02, depends on purity)
  
  Total = 0.4×0.25 + 0.3×0.12 + 0.3×0.02 = 0.142

Somatic Score:
  S_purity = 0.82       (r = 0.82, strong correlation)
  S_cluster = 0.88      (88% of neighbors have it)
  S_clone = 0.7216
  
  Total = 0.5×0.82 + 0.2×0.7216 + 0.3×0.88 = 0.818

Classification: SOMATIC (0.818 > 0.4 AND 0.142 < 0.5) ✅
```

---

### **Example 3: Ambiguous**
```
Variant appears in 40/200 spots, partially clustered, moderate purity correlation

Germline Score:
  S_uniform = 0.45      (somewhat uneven)
  S_prevalence = 0.45   (20% prevalence)
  S_independence = 0.35 (p = 0.35, slight dependence)
  
  Total = 0.4×0.45 + 0.3×0.45 + 0.3×0.35 = 0.42

Somatic Score:
  S_purity = 0.42       (r = 0.42, moderate correlation)
  S_cluster = 0.55      (moderately clustered)
  S_clone = 0.231
  
  Total = 0.5×0.42 + 0.2×0.231 + 0.3×0.55 = 0.421

Classification: AMBIGUOUS (both 0.42 are in ambiguous zone) ❓
```

---

## Parameter Tuning Guidance

### **When to Adjust Thresholds:**

#### **Increase germline_threshold** (e.g., 0.5 → 0.6):
- Use when: Too many false positive germlines
- Effect: Stricter germline criteria, fewer germlines, more ambiguous

#### **Decrease germline_threshold** (e.g., 0.5 → 0.4):
- Use when: Missing true germlines
- Effect: Looser germline criteria, more germlines

#### **Increase somatic_threshold** (e.g., 0.4 → 0.5):
- Use when: Too many false positive somatics
- Effect: Stricter somatic criteria, fewer somatics, more ambiguous

#### **Decrease somatic_threshold** (e.g., 0.4 → 0.35):
- Use when: Missing true somatics
- Effect: Looser somatic criteria, more somatics

---

### **When to Adjust Weights:**

#### **Increase α (spatial uniformity)** in germline:
- Use when: Clustered patterns are problematic
- Effect: Penalizes non-uniform distributions more

#### **Increase δ (purity correlation)** in somatic:
- Use when: Purity signal is strong and reliable
- Effect: Emphasizes purity correlation more

#### **Increase ζ (spatial clustering)** in somatic:
- Use when: Spatial patterns are very distinct
- Effect: Emphasizes clustering more than purity

---

## Statistical Rigor

### **Tests Used:**
1. **Kolmogorov-Smirnov Test** (germline independence)
   - Non-parametric
   - Compares entire distributions
   - Robust to non-normal data

2. **Point-Biserial Correlation** (somatic purity)
   - Special case of Pearson correlation
   - Designed for binary × continuous variables
   - Standard statistical test

3. **Coefficient of Variation** (spatial uniformity)
   - Standard spatial statistics measure
   - Scale-invariant

4. **Local Clustering Coefficient** (somatic clustering)
   - Network/graph theory metric
   - Standard in spatial analysis

---

## Summary Table

| Metric | Purpose | Test/Calculation | Germline Target | Somatic Target |
|--------|---------|-----------------|-----------------|----------------|
| **Spatial Uniformity** | Even distribution | CV of distances | **High (>0.7)** | Low (<0.3) |
| **Global Prevalence** | Common presence | Sigmoid(proportion) | **High (>0.9)** | Low (<0.2) |
| **Purity Independence** | No tumor correlation | KS test p-value | **High (>0.7)** | Low (<0.05) |
| **Purity Correlation** | Tumor correlation | Point-biserial r | Low (<0.1) | **High (>0.6)** |
| **Spatial Clustering** | Neighbor similarity | Local clustering | Low (<0.3) | **High (>0.8)** |
| **Clone-Specific** | Combined signal | Cluster × Purity | Low (<0.1) | **High (>0.5)** |

---

## Key Numbers Summary

### **Thresholds:**
- **Germline threshold**: 0.5
- **Somatic threshold**: 0.4
- **Min expression (germline)**: 2 spots
- **Min expression (somatic)**: 1 spot
- **Neighbor distance**: 2.0 spatial units

### **Weights:**
```
Germline = 0.4×uniform + 0.3×prevalence + 0.3×independence
Somatic  = 0.5×purity + 0.2×clone + 0.3×cluster
```

### **Score Ranges:**
- All scores normalized to [0, 1]
- **>0.7**: Strong signal
- **0.4-0.7**: Moderate signal
- **<0.4**: Weak signal

---

## Validation Recommendations

To validate these numbers work well for your data:

1. **Compare against known variants**
   - Use matched WES data
   - Use validated germline/somatic calls

2. **Check classification distribution**
   - Should get ~60-80% germline, ~10-30% somatic, ~10-20% ambiguous
   - Adjust thresholds if distribution is very skewed

3. **Examine ambiguous variants**
   - Review spatial patterns
   - Check tumor purity profiles
   - May indicate biological heterogeneity

4. **Benchmark against other callers**
   - Compare with GATK, Mutect2, Strelka
   - Measure precision/recall

5. **Stratify by race**
   - Compare defined vs denovo performance
   - May reveal systematic biases in Beagle or Classifier

## Spatial Filter

python run_spatial_snv_filter_enhanced.py \
  --dataset p4_tumor \
  --section_id 1 \
  --quality_filter baseQ0mapQ0 \
  --tumor_purity_file /data/maiziezhou_lab/leiy4/CalicoST/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv

# Plot spatial filter result

./plot_variant_scores.py     --input /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/all_variant_scores.txt --output_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/plots/ --all