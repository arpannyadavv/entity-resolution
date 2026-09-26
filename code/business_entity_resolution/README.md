# Business Entity Resolution Pipeline

## Overview

End-to-end ML pipeline for the Amazon ML Challenge 2026 — Business Entity Resolution task.

Given business records from 3 independent sources (with noisy names and addresses), the pipeline finds all matching records from Source 2 and Source 3 for each Source 1 entity.

**Evaluation Metric:** Macro-average F_0.5 (precision-heavy)

---

## Pipeline Architecture

```
dataset/
├── train/  →  Blocking  →  Feature Engineering  →  LightGBM Classifier  →  Threshold Optimization
└── test/   →  (Re-fit Blocker on test pool)  →  Scoring  →  output/
```

### Stages

| Stage | Description | Module |
|-------|-------------|--------|
| 1. Preprocessing | Name/address normalization, legal suffix canonicalization | `src/preprocess.py` |
| 2. Blocking | Multi-strategy candidate generation (TF-IDF + inverted indexes) | `src/blocking.py` |
| 3. Feature Engineering | 25+ similarity features for each candidate pair | `src/features.py` |
| 4. Training Builder | Positive/negative pair construction with hard-negative mining | `src/training_builder.py` |
| 5. Model | LightGBM binary classifier with early stopping | `src/model.py` |
| 6. Evaluation | Macro-average F_0.5 + evaluation dashboard | `src/evaluate.py` |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Train + Predict)

```bash
# From student_resource/ directory:
python code/business_entity_resolution/pipeline.py --mode full
```

### 3. Train Only

```bash
python code/business_entity_resolution/pipeline.py --mode train
```

### 4. Predict Only (requires trained model)

```bash
python code/business_entity_resolution/pipeline.py --mode predict
```

---

## CLI Options

```
python pipeline.py [--mode {train,predict,full}]
                   [--val-size FLOAT]        # Validation split (default: 0.15)
                   [--top-k-tfidf INT]       # TF-IDF top-K candidates (default: 15)
                   [--neg-per-pos INT]       # Negatives per positive pair (default: 3)
                   [--no-tfidf]              # Disable TF-IDF blocking (faster)
```

---

## Directory Structure

```
code/business_entity_resolution/
├── pipeline.py              # Main entry point
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── src/
│   ├── preprocess.py        # Text normalization
│   ├── blocking.py          # Multi-strategy blocking (TF-IDF, prefix, soundex, ZIP)
│   ├── features.py          # Pair similarity features (rapidfuzz, jellyfish)
│   ├── training_builder.py  # Training pair construction
│   ├── model.py             # LightGBM EntityMatcher
│   └── evaluate.py          # F_0.5 evaluation + dashboard plots
├── models/                  # Saved model artifacts (created at runtime)
└── plots/                   # Evaluation dashboard (created at runtime)
```

**Output files** (in `output/` at project root):
- `matching_results.tsv` — Final entity matches (submitted to leaderboard)
- `candidate_pairs.tsv` — Blocking candidate set (before ML scoring)

---

## Blocking Strategy

Multi-strategy blocking per entity (union of all signals):

1. **Country-prefixed name prefix** — 3-char and 4-char prefix blocking, keyed by country
2. **ZIP/PIN code** — Exact postal code match (handles US ZIP, India PIN, France postal)
3. **Soundex phonetic** — First-word Soundex encoding (country-prefixed)
4. **First-word exact** — Country-prefixed first meaningful token match
5. **TF-IDF cosine** — Top-K most similar names via char/word n-gram TF-IDF

Blocking recall is logged during training. Target: >95% recall with manageable candidate set size.

---

## Features (25+)

**Name features** (rapidfuzz + jellyfish):
- Levenshtein ratio, Token Sort ratio, Token Set ratio, Partial ratio, WRatio
- Jaro-Winkler similarity
- Token Jaccard, shared token ratio, length difference
- Soundex match of first word
- Token count features (shared, max, min)

**Address features** (rapidfuzz):
- Levenshtein, Token Sort, Token Set, Partial ratio
- Jaccard, shared token ratio, length difference
- ZIP/PIN exact match
- Shared address tokens
- Missing address flags

**Meta features**:
- Country exact match / mismatch flags

---

## Model

**Algorithm:** LightGBM (GBDT)  
**Hyperparameters:**
- `num_leaves=127`, `learning_rate=0.05`, `n_estimators=500`
- `feature_fraction=0.8`, `bagging_fraction=0.8`
- `reg_alpha=0.1`, `reg_lambda=1.0`
- Early stopping on validation binary logloss (patience=50)
- Class imbalance handled via `scale_pos_weight`

**Threshold:** Optimized on validation set by sweeping [0.10, 0.95] to maximize entity-level macro F_0.5.

---

## Reproduce Results

```bash
# From student_resource/ root:
python code/business_entity_resolution/pipeline.py --mode full --val-size 0.15 --top-k-tfidf 15 --neg-per-pos 3

# Validate output before submission:
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
