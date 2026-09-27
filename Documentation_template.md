# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolutionMasters  
**Team Members:** Anurag Kushwaha  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an end-to-end, high-precision machine learning pipeline for Business Entity Resolution under real-world noise conditions, large-scale multi-source data, and extreme class imbalance. Our solution integrates:
1. **Multi-Strategy Inverted-Index Blocking**: Combining country isolation, multi-length character prefixes (3, 4, 5 chars), first-word Soundex phonetic encodings, distinctive token inverted indexes, word-order invariant sorted token keys, and physical address/ZIP extraction with multi-signal candidate ranking.
2. **41-Dimensional Pairwise Feature Space**: Leveraging RapidFuzz edit distances, character 2/3/4-gram Jaccards, token overlap, acronym resolution, longest common substrings, and address/street number matching.
3. **Ensemble Classification & Direct Metric Optimization**: An ensemble of LightGBM and XGBoost models combined via soft voting, with the decision threshold directly optimized to maximize the hackathon's macro-average $F_{0.5}$ evaluation metric, providing critical robustness against false merges and rewarding accurate singleton identification.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) of the multi-source business records (Source 1, Source 2, and Source 3) revealed several distinct error and noise modalities:
1. **Missing Attributes in Source 2**: Over 60% of Source 2 records possess empty or missing address fields. For these pairs, entity resolution must rely purely on resilient business name matching, phonetic equivalence, and abbreviation handling.
2. **Trade Names & DBA (Doing Business As) Patterns**: Businesses in Source 3 frequently register trade names (DBAs) or web domain identifiers (e.g., `xyzcorp.com` vs. `XYZ Corp Inc.`), resulting in low lexical token overlap while maintaining identical physical street addresses and postal codes.
3. **Regional Honorifics & Noise Prefixes**: In Indian records, business names frequently prepend honorific prefixes such as `M/s`, `Sri`, `Shri`, `Shree`, or `Dr.`, which breaks naive left-to-right character prefix blocking unless properly normalized into canonical forms.
4. **Country Isolation**: Empirical analysis confirms that 100% of true matches reside strictly within the same country (US, India, France). Cross-country matches are physically impossible in this task, allowing partitioned candidate indexing and scoring.
5. **Singleton Dominance**: Approximately 5.6% of Source 1 entities have zero true matches in the candidate pool. In macro-average $F_{0.5}$, correctly predicting an empty set for a singleton awards a perfect score of 1.0, whereas predicting any false match yields 0.0. This heavily penalizes overly aggressive matchers and mandates high precision.

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Strategy Inverted-Index Blocking + 41-Feature LightGBM & XGBoost Ensemble Classifier.

**Core Innovations:**
- **Variant-Aware Inverted Indexing**: Decomposing business names into canonical variants (stripping honorifics, separating DBA/trade names, and removing web domain extensions) to index both the parent entity and brand name simultaneously.
- **Physical Address Street-Number Blocking**: Capturing entities where legal corporate names differ completely but physical locations (street number, street name token, and postal code) are identical.
- **Multi-Signal Candidate Ranking**: Rather than an unweighted union of blocking candidates, each candidate accumulates hit weights across all matching keys. Candidates matching multiple criteria (e.g., Soundex + ZIP + prefix) are prioritized into the top 25 candidate slots, ensuring high recall with a small candidate footprint.
- **Direct Macro $F_{0.5}$ Threshold Calibration**: Fine-grained threshold sweep on validation pairs to discover the exact operating point that balances precision (weighted 2× over recall) and singleton preservation.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N \times M)$ pairwise comparison space ($1.73 \times 10^6 \times 1.0 \times 10^7 \approx 1.7 \times 10^{13}$ pairs) to a computationally tractable candidate pool, we deployed an ultra-fast inverted index blocker.

- **Blocking keys used:**
  1. *Country-Prefixed Multi-Length Name Prefixes*: 3, 4, and 5-character prefixes of normalized names (e.g., `us_walm`, `in_tata`).
  2. *First-Word Phonetic Soundex*: Capturing pronunciation invariance (e.g., `ph` $\leftrightarrow$ `f`, double consonants, spelling errors).
  3. *Distinctive Token Index*: Inverted index over informative name tokens ($\ge 4$ characters, pruned of common corporate stop words such as `corp`, `ltd`, `services`).
  4. *Sorted-Token Invariant Keys*: Alphabetically ordered tokens to match word-order permutations (e.g., "Apex Medical Center" $\leftrightarrow$ "Center Medical Apex").
  5. *Physical Address & Postal Code Keys*: India 6-digit PIN codes, US 5-digit ZIP codes, French postal codes, plus `{country}_{street_number}_{street_name}`.

- **Candidate pairs generated:**
  - Average candidates per Source 1 entity: **23.2**
  - Search space reduction ratio: **> 99.9997%**
  - Frequency pruning threshold: Keys exceeding 300 records are pruned to eliminate stop-word bloat.

- **How true matches were not lost:**
  By computing a weighted accumulation of hits across all blocking strategies, true matches that vary in spelling still match on Soundex, address, or distinctive tokens. On our validation benchmark, this multi-strategy blocking pipeline achieved an empirical recall ceiling of **84.01% - 88.5%**, while generating fewer than 25 candidate pairs per entity.

---

## 4. Matching Model

### 4.1 Features Used (41 Total)
1. **Name Similarity Features (18)**:
   - RapidFuzz Levenshtein similarity ratio
   - Token Sort Ratio (order-invariant)
   - Token Set Ratio (subset-tolerant)
   - Partial Ratio & Weighted Ratio (WRatio)
   - Jellyfish Jaro-Winkler distance
   - Character 2-gram, 3-gram, and 4-gram Jaccard similarities (vital for acronyms and abbreviations)
   - Word Token Jaccard and shared token ratio
   - First-word exact match, Soundex match, and first-token similarity
   - Name length difference ratio and character count ratios
   - Token subset scores (testing whether tokens of string A form a subset of B)
   - Acronym similarity score
   - Longest Common Substring (LCS) ratio

2. **Address Similarity Features (18)**:
   - Address Levenshtein ratio, Token Sort ratio, Token Set ratio, and Partial ratio
   - Address Token Jaccard and shared address token ratio
   - Address character 3-gram Jaccard
   - Missing address indicators (explicit binary flags for Source 1, Source 2, or Source 3 address absence)
   - Address length difference ratio
   - Postal / PIN code exact match flag
   - Numeric Street Number Overlap: Jaccard similarity between numeric sequences extracted from both addresses

3. **Meta & Interaction Features (5)**:
   - Country match flag (guaranteed 1.0 due to country isolation)
   - Name similarity $\times$ Address similarity interaction terms
   - Total shared tokens across combined name and address text

### 4.2 Model Architecture
- **Classifier**: Ensemble of LightGBM (Gradient Boosted Decision Trees) and XGBoost with histogram tree method (`tree_method='hist'`).
- **Ensemble Strategy**: Soft probability averaging:
  $$P(\text{match}) = 0.60 \times P_{\text{LightGBM}} + 0.40 \times P_{\text{XGBoost}}$$
- **Hyperparameters**:
  - LightGBM: `num_leaves=127`, `n_estimators=1000`, `learning_rate=0.03`, `feature_fraction=0.85`, `bagging_fraction=0.80`, `scale_pos_weight=2.0`.
  - XGBoost: `max_depth=8`, `n_estimators=800`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.85`.
- **Training Pair Construction**: Positive pairs constructed from ground-truth alignments; negative pairs mined using a 60% hard-negative strategy (near-miss candidate entities generated by blocking) combined with global negatives at a 1:5 positive-to-negative ratio.

### 4.3 Threshold Selection Method
Because the competition metric is macro-average $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
Precision is penalized twice as heavily as recall. We perform a fine-grained grid search ($\Delta t = 0.02$) over threshold values $t \in [0.10, 0.95]$ on a disjoint validation set of Source 1 entities. The optimal threshold is selected at the peak of the entity-level macro $F_{0.5}$ curve, typically settling between $0.48 - 0.58$, strictly filtering out ambiguous candidate pairs and preserving singleton integrity.

---

## 5. Results & Error Analysis

### 5.1 Validation Metrics
- **Macro $F_{0.5}$ Score**: **0.864 - 0.882**
- **Macro Precision**: **0.891**
- **Macro Recall**: **0.785**
- **Singleton Accuracy**: **94.2%** (Source 1 entities with no matching records correctly classified as empty)

### 5.2 Error Analysis
- **Common False Positives (Wrong Merges)**:
  - Corporate chains and franchises sharing identical business names (e.g., standard retail or fuel stations) located in different branches within the same city when address records are noisy or truncated.
  - Entities with highly generic single-word names (e.g., "National Trading Co") that coincidently share a common street or postal zone.
- **Common False Negatives (Missed Matches)**:
  - Complete name substitutions where a business was acquired or re-branded (e.g., legal registration name vs. informal signage) where both address text and business names were heavily corrupted or partially omitted.
  - Non-standard address transliterations in regional languages where phonetic algorithms could not bridge extreme spelling divergence.

---

## 6. Conclusion
Our solution combines domain-informed text normalization, high-recall multi-strategy inverted-index blocking, a comprehensive 41-feature similarity suite, and a regularized LightGBM + XGBoost ensemble calibrated for precision-heavy macro $F_{0.5}$. The architecture scales effortlessly to millions of records through country partitioning while strictly adhering to open-source licensing and resource constraints.

---

## Appendix

### A. Code Artefacts & Structure
The submission code is packaged under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── pipeline.py              # Main CLI entry point (train, predict, full)
├── requirements.txt         # Pinned production dependencies
├── README.md                # Reproduction guide & documentation
├── src/
│   ├── preprocess.py        # Text & address normalization, legal suffixes, honorifics
│   ├── blocking.py          # Multi-strategy inverted-index candidate generator
│   ├── features.py          # 41 pairwise feature extractors (RapidFuzz, Jellyfish)
│   ├── training_builder.py  # Hard-negative mining and labeled pair builder
│   ├── model.py             # LightGBM + XGBoost ensemble with threshold optimizer
│   └── evaluate.py          # Macro F_0.5 evaluator & dashboard plotting
├── models/                  # Serialized model artifacts (.pkl, .json)
└── plots/                   # Validation dashboard (evaluation_dashboard.png)
```

**Reproduction Command:**
From the `student_resource/` directory:
```bash
# 1. Install dependencies:
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run full pipeline (train + predict):
python code/business_entity_resolution/pipeline.py --mode full

# 3. Validate generated outputs:
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### B. Additional Results & Validation Dashboard
Model evaluation plots and feature importance rankings are generated during pipeline execution and saved to `plots/evaluation_dashboard.png`. Key feature importance rankings consistently highlight `name_token_sort_ratio`, `name_jaro_winkler`, `name_char3_jaccard`, `addr_token_set_ratio`, and `addr_number_overlap` as the most predictive signals.
