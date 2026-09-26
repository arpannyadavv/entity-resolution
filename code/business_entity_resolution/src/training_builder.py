"""
Step 4: Training Data Builder
Creates labeled (positive/negative) pair dataset from ground truth.

For each S1 entity with matches:
  - Positive pairs: (S1, matched_entity) → label=1
  - Negative pairs: (S1, non-matching candidate from blocking) → label=0

Hard negative mining: negatives are sampled from the blocking candidate pool
(hardest examples), not randomly — this trains the model to distinguish
near-matches from true matches.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import compute_pair_features, get_feature_names


def parse_matched_ids(matched_str):
    """Parse comma-separated matched IDs from ground truth."""
    if not isinstance(matched_str, str) or not matched_str.strip():
        return []
    return [mid.strip() for mid in matched_str.split(',') if mid.strip()]


def build_lookup(df: pd.DataFrame) -> dict:
    """Build entity_id -> row dict for fast lookup."""
    return {row['entity_id']: row.to_dict() for _, row in df.iterrows()}


def build_training_pairs(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    candidates: dict,
    neg_per_pos: int = 3,
    max_pos_per_entity: int = None,
    random_seed: int = 42
) -> tuple:
    """
    Build labeled training pairs.

    Args:
        s1, s2, s3: Source dataframes
        gt: Ground truth dataframe
        candidates: dict {s1_entity_id -> set(candidate_ids)}
        neg_per_pos: How many negatives to sample per positive
        max_pos_per_entity: Cap positives per S1 entity (for class balance)
        random_seed: For reproducibility

    Returns:
        (X: np.array of shape [n_pairs, n_features],
         y: np.array of shape [n_pairs],
         pair_ids: list of (s1_id, cand_id) tuples)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("[TrainingBuilder] Building lookup tables...")
    s1_lookup = build_lookup(s1)
    s2_lookup = build_lookup(s2)
    s3_lookup = build_lookup(s3)
    cand_lookup = {**s2_lookup, **s3_lookup}

    # Parse ground truth
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    gt_dict = {}
    for _, row in gt.iterrows():
        gt_dict[row['source1_entity_id']] = set(parse_matched_ids(row['matched_entity_ids']))

    feature_names = get_feature_names()
    X_rows = []
    y_rows = []
    pair_ids = []

    print(f"[TrainingBuilder] Building training pairs (neg_per_pos={neg_per_pos})...")
    
    skipped_no_s1 = 0
    skipped_no_cands = 0
    
    for s1_id, true_matches in tqdm(gt_dict.items(), desc="  Building pairs", ncols=80):
        if s1_id not in s1_lookup:
            skipped_no_s1 += 1
            continue
        
        s1_row = s1_lookup[s1_id]
        cand_set = candidates.get(s1_id, set())
        
        # Add all true matches to candidates (ensure recall ceiling is 100% on train)
        cand_set = cand_set | true_matches
        
        if not cand_set:
            skipped_no_cands += 1
            continue

        # Positive pairs
        pos_cands = [mid for mid in true_matches if mid in cand_lookup]
        if max_pos_per_entity and len(pos_cands) > max_pos_per_entity:
            pos_cands = random.sample(pos_cands, max_pos_per_entity)
        
        # Negative pairs (from candidates that are NOT true matches)
        neg_pool = [mid for mid in cand_set if mid not in true_matches and mid in cand_lookup]
        n_neg = min(len(neg_pool), len(pos_cands) * neg_per_pos)
        neg_cands = random.sample(neg_pool, n_neg) if n_neg > 0 else []

        for cand_id in pos_cands:
            cand_row = cand_lookup[cand_id]
            feats = compute_pair_features(s1_row, cand_row)
            X_rows.append([feats[f] for f in feature_names])
            y_rows.append(1)
            pair_ids.append((s1_id, cand_id))

        for cand_id in neg_cands:
            cand_row = cand_lookup[cand_id]
            feats = compute_pair_features(s1_row, cand_row)
            X_rows.append([feats[f] for f in feature_names])
            y_rows.append(0)
            pair_ids.append((s1_id, cand_id))

    print(f"  Total pairs: {len(X_rows):,} ({sum(y_rows):,} positive, {len(y_rows)-sum(y_rows):,} negative)")
    print(f"  Skipped (no S1 lookup): {skipped_no_s1}, Skipped (no candidates): {skipped_no_cands}")
    
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), pair_ids, feature_names


if __name__ == '__main__':
    DTYPES = {'entity_id': str, 'business_name': str, 'business_address': str, 'country': str}
    print("Quick test of training pair builder...")
    s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=DTYPES, nrows=200)
    s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=DTYPES, nrows=5000)
    s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=DTYPES, nrows=5000)
    gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t',
                     dtype={'source1_entity_id': str, 'matched_entity_ids': str}, nrows=200)
    
    # Dummy candidates (all S2+S3 for quick test)
    candidates = {row['entity_id']: set(s2['entity_id'].tolist()[:20] + s3['entity_id'].tolist()[:20])
                  for _, row in s1.iterrows()}
    
    X, y, pairs, feat_names = build_training_pairs(s1, s2, s3, gt, candidates, neg_per_pos=2)
    print(f"\nX shape: {X.shape}")
    print(f"y distribution: pos={y.sum()}, neg={len(y)-y.sum()}")
    print(f"Feature names ({len(feat_names)}):", feat_names)
