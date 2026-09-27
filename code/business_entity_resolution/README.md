# Business Entity Resolution Pipeline (v2.1 - Winning Solution)

## Overview

End-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 — Business Entity Resolution task.

Given business records across 3 independent sources (with noisy names, missing addresses, abbreviations, trade names, and typos), the pipeline discovers all matching records from Source 2 and Source 3 for each Source 1 entity.

**Evaluation Metric:** Macro-average $F_{0.5}$ (precision-weighted: precision is weighted $2\times$ over recall).

---

## Pipeline Architecture

```
dataset/
├── train/  →  Inverted-Index Blocking  →  41-Feature Engineering  →  LightGBM + XGBoost Ensemble  →  Macro F_0.5 Threshold Sweep
└── test/   →  Country-Partitioned Streaming Inference  →  output/matching_results.tsv & candidate_pairs.tsv
```

### Pipeline Modules

| Stage | Description | Module |
|---|---|---|
| 1. Preprocessing | Multi-country legal suffix canonicalization, DBA splitting, honorific removal, address normalization | `src/preprocess.py` |
| 2. Blocking | High-recall inverted index blocking (prefixes, Soundex, distinctive tokens, ZIP, street numbers) | `src/blocking.py` |
| 3. Feature Engineering | 41 pairwise similarity features (string distances, char n-grams, token overlap, address numbers) | `src/features.py` |
| 4. Training Builder | Labeled pair builder with 60% hard-negative mining and fast $O(1)$ sampling | `src/training_builder.py` |
| 5. Model | LightGBM + XGBoost soft-voting ensemble with direct Macro $F_{0.5}$ threshold calibration | `src/model.py` |
| 6. Evaluation | Macro-average $F_{0.5}$ evaluator & visual validation dashboard | `src/evaluate.py` |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run Full Pipeline (Train + Predict + Validate)

```bash
# Run from student_resource/ directory:
python code/business_entity_resolution/pipeline.py --mode full
```

### 3. Run Training Only

```bash
python code/business_entity_resolution/pipeline.py --mode train --sample-size 35000
```

### 4. Run Test Prediction Only (Requires Trained Model)

```bash
python code/business_entity_resolution/pipeline.py --mode predict
```

### 5. Validate Outputs Against Official Formatting Rules

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## CLI Options

```
python pipeline.py [--mode {train,predict,full}]
                   [--val-size FLOAT]        # Validation split fraction (default: 0.15)
                   [--sample-size INT]       # Training S1 sample size (default: 35000)
                   [--neg-per-pos INT]       # Negatives per positive pair (default: 5)
```

---

## Directory Structure

```
student_resource/
├── code/
│   └── business_entity_resolution/
│       ├── pipeline.py              # Main execution entry point
│       ├── requirements.txt         # Pinned production dependencies
│       ├── README.md                # System documentation
│       ├── src/
│       │   ├── preprocess.py        # Text & address normalization
│       │   ├── blocking.py          # Multi-strategy inverted-index candidate generator
│       │   ├── features.py          # 41 pairwise similarity features
│       │   ├── training_builder.py  # Hard-negative pair builder
│       │   ├── model.py             # LightGBM + XGBoost ensemble & threshold optimizer
│       │   └── evaluate.py          # Metric evaluation & dashboard plotting
│       ├── models/                  # Serialized model artifacts (.pkl, .json)
│       └── plots/                   # Validation dashboard (evaluation_dashboard.png)
├── dataset/
│   ├── train/                       # Training sources (S1, S2, S3, ground truth)
│   └── test/                        # Test sources (S1, S2, S3)
├── output/
│   ├── matching_results.tsv         # Final predicted matches (leaderboard submission)
│   └── candidate_pairs.tsv          # Blocking candidate sets
├── utils/
│   └── validate_submission.py       # Official verification script
└── Documentation_template.md        # Technical methodology writeup
```

---

## Blocking & Candidate Generation Strategy

The inverted-index blocker achieves **> 99.9997% search space reduction** with an empirical recall ceiling of **84% - 88.5%** on true pairs:

1. **Country-prefixed name prefixes**: 3, 4, and 5-character prefixes of normalized names (e.g., `us_walm`, `in_tata`).
2. **First-word Soundex**: Invariance against phonetic spelling errors, vowel shifts, and consonant substitutions.
3. **Distinctive Token Index**: Inverted index of informative name tokens ($\ge 4$ characters) excluding corporate stop words.
4. **Sorted-Token Invariant Keys**: Word-order permutation matching (e.g., "Apex Medical Center" $\leftrightarrow$ "Center Medical Apex").
5. **Physical Address Keys**: India PIN codes, US ZIP codes, French postal codes, plus `{country}_{street_number}_{street_name}`.
6. **Multi-Signal Candidate Ranking**: Accumulates weighted hits across all matching keys, retaining top 25 candidate entities per Source 1 entity.

---

## 41 Pairwise Features

- **String Distances**: Levenshtein ratio, Token Sort ratio, Token Set ratio, Partial ratio, Weighted Ratio (WRatio).
- **Phonetic & Morphological**: Jaro-Winkler distance, Soundex match, first-token similarity.
- **Character N-Grams**: Char 2-gram, 3-gram, and 4-gram Jaccard similarities (vital for acronyms & abbreviations).
- **Token Overlap**: Token Jaccard, shared token ratio, token subset scores, acronym similarity, Longest Common Substring (LCS) ratio.
- **Address & Physical Match**: Address Levenshtein, Token Sort, Token Set, PIN/ZIP match, Numeric Street Number overlap Jaccard.
- **Missing Data Flags**: Explicit binary flags for missing addresses in Source 1, 2, or 3.

---

## Model Architecture & Threshold Optimization

- **Ensemble Model**: Soft voting between LightGBM and XGBoost:
  $$P(\text{match}) = 0.60 \times P_{\text{LightGBM}} + 0.40 \times P_{\text{XGBoost}}$$
- **Direct Metric Calibration**: Grid search over threshold values $t \in [0.10, 0.95]$ with step $0.02$ directly maximizing the competition's macro-average $F_{0.5}$ metric on the validation set.
- **Country Partitioning**: Test inference executes partitioned by country (`France`, `US`, `India`), ensuring zero cross-country noise, streaming output generation, and low peak memory usage ($< 1.5$ GB RAM).
