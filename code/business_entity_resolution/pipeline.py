"""
Main Pipeline: Business Entity Resolution
==========================================
End-to-end pipeline:
  1. Load data
  2. Build blocking indexes
  3. Generate candidates (candidate_pairs.tsv)
  4. Build training pairs with feature engineering
  5. Train LightGBM matcher
  6. Optimize threshold on validation
  7. Predict on test set
  8. Generate output TSVs
  9. Plot evaluation dashboard

Usage:
    python src/pipeline.py --mode train   # Train on training data
    python src/pipeline.py --mode predict # Generate test predictions
    python src/pipeline.py --mode full    # Train + predict (default)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import pickle
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local modules
from src.preprocess import clean_name, clean_address
from src.blocking import EntityBlocker
from src.features import compute_pair_features, get_feature_names
from src.training_builder import build_training_pairs, parse_matched_ids, build_lookup
from src.model import EntityMatcher
from src.evaluate import evaluate_predictions, load_ground_truth, plot_evaluation_dashboard

# ---- Paths ----
BASE_DIR = Path(__file__).parent.parent.parent  # student_resource/
DATA_DIR = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = Path(__file__).parent / 'models'
PLOTS_DIR = Path(__file__).parent / 'plots'

DTYPES = {'entity_id': str, 'business_name': str, 'business_address': str, 'country': str}


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"[Pipeline] Output dir: {OUTPUT_DIR}")
    print(f"[Pipeline] Model dir: {MODEL_DIR}")
    print(f"[Pipeline] Plots dir: {PLOTS_DIR}")


def load_data(split: str = 'train') -> tuple:
    """Load source files for given split."""
    print(f"\n[Pipeline] Loading {split} data...")
    s1 = pd.read_csv(DATA_DIR / split / f'{split}_source1.tsv', sep='\t', dtype=DTYPES)
    s2 = pd.read_csv(DATA_DIR / split / f'{split}_source2.tsv', sep='\t', dtype=DTYPES)
    s3 = pd.read_csv(DATA_DIR / split / f'{split}_source3.tsv', sep='\t', dtype=DTYPES)
    print(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")
    return s1, s2, s3


def load_ground_truth_df() -> pd.DataFrame:
    gt = pd.read_csv(
        DATA_DIR / 'train' / 'train_ground_truth.tsv',
        sep='\t',
        dtype={'source1_entity_id': str, 'matched_entity_ids': str}
    )
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    print(f"  GT: {len(gt):,} S1 entities")
    return gt


def score_blocking(candidates: dict, gt_dict: dict, all_s1_ids: list):
    """Compute blocking recall (how many true matches are in candidate set)."""
    total_true = 0
    total_recalled = 0
    total_cands = 0
    
    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        cand_set = candidates.get(s1_id, set())
        total_true += len(true_set)
        total_recalled += len(true_set & cand_set)
        total_cands += len(cand_set)
    
    recall = total_recalled / total_true if total_true > 0 else 0.0
    avg_cands = total_cands / len(all_s1_ids) if all_s1_ids else 0.0
    print(f"\n[Blocking Stats]")
    print(f"  Recall ceiling: {recall:.4f} ({total_recalled}/{total_true} true matches in candidates)")
    print(f"  Avg candidates per S1: {avg_cands:.1f}")
    print(f"  Total candidate pairs: {total_cands:,}")
    return recall, avg_cands


def save_candidate_pairs(candidates: dict, all_s1_ids: list, output_path: Path):
    """Save candidate_pairs.tsv."""
    rows = []
    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, set())
        # Filter to only S2/S3 IDs
        valid_cands = [c for c in cands if c.startswith('S2-') or c.startswith('S3-')]
        # Deduplicate
        valid_cands = list(dict.fromkeys(valid_cands))
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(valid_cands)
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"[Pipeline] Saved candidate_pairs.tsv → {output_path} ({len(df):,} rows)")


def save_matching_results(predictions: dict, all_s1_ids: list, output_path: Path):
    """Save matching_results.tsv."""
    rows = []
    for s1_id in all_s1_ids:
        matches = predictions.get(s1_id, [])
        valid = [m for m in matches if m.startswith('S2-') or m.startswith('S3-')]
        valid = list(dict.fromkeys(valid))  # deduplicate
        rows.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(valid)
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"[Pipeline] Saved matching_results.tsv → {output_path} ({len(df):,} rows)")


def score_all_pairs(
    s1: pd.DataFrame,
    candidates: dict,
    cand_lookup: dict,
    matcher: EntityMatcher,
    feature_names: list,
    threshold: float,
    batch_size: int = 5000
) -> dict:
    """
    Score all candidate pairs and return predictions dict.
    Memory-efficient: processes in batches.
    """
    predictions = {}
    s1_lookup = build_lookup(s1)
    
    print(f"\n[Scoring] Scoring candidates with threshold={threshold:.3f}...")
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"  Total pairs to score: {total_pairs:,}")
    
    batch_s1_ids = []
    batch_cand_ids = []
    batch_X = []
    
    def flush_batch():
        if not batch_X:
            return
        X_batch = np.array(batch_X, dtype=np.float32)
        probas = matcher.predict_proba(X_batch)
        preds = (probas >= threshold).astype(int)
        for i, (s1_id, cand_id) in enumerate(zip(batch_s1_ids, batch_cand_ids)):
            if preds[i] == 1:
                if s1_id not in predictions:
                    predictions[s1_id] = []
                predictions[s1_id].append(cand_id)
        batch_s1_ids.clear()
        batch_cand_ids.clear()
        batch_X.clear()
    
    with tqdm(total=total_pairs, desc="  Scoring", ncols=80) as pbar:
        for s1_id, cand_set in candidates.items():
            if s1_id not in s1_lookup:
                continue
            s1_row = s1_lookup[s1_id]
            for cand_id in cand_set:
                if cand_id not in cand_lookup:
                    continue
                cand_row = cand_lookup[cand_id]
                feats = compute_pair_features(s1_row, cand_row)
                batch_s1_ids.append(s1_id)
                batch_cand_ids.append(cand_id)
                batch_X.append([feats[f] for f in feature_names])
                pbar.update(1)
                
                if len(batch_X) >= batch_size:
                    flush_batch()
    
    flush_batch()  # Process remaining
    return predictions


def run_training_pipeline(
    val_size: float = 0.15,
    top_k_tfidf: int = 15,
    neg_per_pos: int = 3,
    use_tfidf: bool = True
):
    """
    Full training pipeline:
    1. Load training data
    2. Split train/val
    3. Build blocking indexes
    4. Generate candidate pairs
    5. Build labeled training data
    6. Train LightGBM
    7. Optimize threshold
    8. Evaluate on validation
    9. Save model + plots
    """
    ensure_dirs()
    start = time.time()
    
    print("=" * 60)
    print("STAGE 1: Loading Training Data")
    print("=" * 60)
    s1, s2, s3 = load_data('train')
    gt = load_ground_truth_df()
    
    # Parse GT dict
    gt_dict = {}
    for _, row in gt.iterrows():
        ids = parse_matched_ids(row['matched_entity_ids'])
        gt_dict[row['source1_entity_id']] = set(ids)
    
    # Train/val split on S1 entities
    all_s1_ids = s1['entity_id'].tolist()
    train_ids, val_ids = train_test_split(
        all_s1_ids, test_size=val_size, random_state=42
    )
    print(f"\n[Split] Train: {len(train_ids):,} S1 entities | Val: {len(val_ids):,} S1 entities")
    
    s1_train = s1[s1['entity_id'].isin(set(train_ids))].reset_index(drop=True)
    s1_val = s1[s1['entity_id'].isin(set(val_ids))].reset_index(drop=True)
    
    print("=" * 60)
    print("STAGE 2: Building Blocking Indexes")
    print("=" * 60)
    blocker = EntityBlocker(top_k_tfidf=top_k_tfidf)
    blocker.fit(s2, s3, use_tfidf=use_tfidf)
    
    print("=" * 60)
    print("STAGE 3: Generating Training Candidates")
    print("=" * 60)
    train_candidates = blocker.generate_candidates(s1_train)
    
    print("=" * 60)
    print("STAGE 4: Generating Validation Candidates")
    print("=" * 60)
    val_candidates = blocker.generate_candidates(s1_val)
    
    # Score blocking recall on train
    print("\n[Blocking] Training set recall:")
    score_blocking(train_candidates, gt_dict, train_ids)
    print("\n[Blocking] Validation set recall:")
    val_recall, _ = score_blocking(val_candidates, gt_dict, val_ids)
    
    print("=" * 60)
    print("STAGE 5: Building Training Pairs")
    print("=" * 60)
    X_train, y_train, train_pair_ids, feature_names = build_training_pairs(
        s1_train, s2, s3, gt[gt['source1_entity_id'].isin(set(train_ids))],
        train_candidates, neg_per_pos=neg_per_pos
    )
    
    print("=" * 60)
    print("STAGE 6: Building Validation Pairs")
    print("=" * 60)
    X_val, y_val, val_pair_ids, _ = build_training_pairs(
        s1_val, s2, s3, gt[gt['source1_entity_id'].isin(set(val_ids))],
        val_candidates, neg_per_pos=neg_per_pos * 2  # More negatives for val
    )
    
    print("=" * 60)
    print("STAGE 7: Training LightGBM Matcher")
    print("=" * 60)
    matcher = EntityMatcher(threshold=0.5)
    matcher.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    print("=" * 60)
    print("STAGE 8: Optimizing Classification Threshold")
    print("=" * 60)
    best_t, best_f05, threshold_curve = matcher.optimize_threshold(X_val, y_val, val_pair_ids)
    
    print("=" * 60)
    print("STAGE 9: Full Validation Evaluation")
    print("=" * 60)
    # Score all val candidates
    cand_lookup = {**build_lookup(s2), **build_lookup(s3)}
    val_predictions = score_all_pairs(
        s1_val, val_candidates, cand_lookup, matcher, feature_names, best_t
    )
    
    # Entity-level evaluation including singletons
    eval_results = evaluate_predictions(val_predictions, gt_dict, val_ids)
    print(f"\n{'='*60}")
    print(f"  VALIDATION RESULTS:")
    print(f"  Macro F_0.5:    {eval_results['macro_f05']:.4f}")
    print(f"  Macro Precision:{eval_results['macro_precision']:.4f}")
    print(f"  Macro Recall:   {eval_results['macro_recall']:.4f}")
    print(f"{'='*60}")
    
    # Feature importance
    feat_imp = matcher.feature_importance()
    print("\nTop 10 Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    # Save model
    matcher.save(MODEL_DIR / 'entity_matcher.pkl')
    with open(MODEL_DIR / 'blocker.pkl', 'wb') as f:
        pickle.dump(blocker, f)
    with open(MODEL_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    with open(MODEL_DIR / 'threshold.json', 'w') as f:
        json.dump({'threshold': float(best_t), 'val_f05': float(best_f05)}, f)
    
    # Save validation candidate pairs
    save_candidate_pairs(val_candidates, val_ids, PLOTS_DIR / 'val_candidate_pairs.tsv')
    
    # Generate evaluation dashboard
    plot_path = PLOTS_DIR / 'evaluation_dashboard.png'
    plot_evaluation_dashboard(eval_results, threshold_curve, feat_imp, str(plot_path))
    
    print(f"\n[Pipeline] Training complete in {(time.time()-start)/60:.1f} minutes")
    print(f"[Pipeline] Model saved to: {MODEL_DIR}")
    print(f"[Pipeline] Dashboard: {plot_path}")
    
    return matcher, blocker, feature_names, best_t, eval_results


def run_prediction_pipeline():
    """
    Prediction pipeline (uses saved model):
    1. Load test data
    2. Load saved blocker + model
    3. Generate test candidates
    4. Score all pairs
    5. Save output TSVs
    """
    ensure_dirs()
    start = time.time()
    
    print("=" * 60)
    print("STAGE 1: Loading Test Data")
    print("=" * 60)
    s1_test, s2_test, s3_test = load_data('test')
    all_test_s1_ids = s1_test['entity_id'].tolist()
    
    print("=" * 60)
    print("STAGE 2: Loading Saved Model")
    print("=" * 60)
    matcher = EntityMatcher.load(MODEL_DIR / 'entity_matcher.pkl')
    with open(MODEL_DIR / 'blocker.pkl', 'rb') as f:
        blocker = pickle.load(f)
    with open(MODEL_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)
    with open(MODEL_DIR / 'threshold.json') as f:
        thresh_data = json.load(f)
    threshold = thresh_data['threshold']
    print(f"  Loaded threshold: {threshold:.3f} (val F_0.5: {thresh_data['val_f05']:.4f})")
    
    # Rebuild blocker on test pool (re-fit to test S2/S3)
    print("=" * 60)
    print("STAGE 3: Rebuilding Blocker on Test Data")
    print("=" * 60)
    test_blocker = EntityBlocker(top_k_tfidf=blocker.top_k_tfidf)
    test_blocker.fit(s2_test, s3_test, use_tfidf=True)
    
    print("=" * 60)
    print("STAGE 4: Generating Test Candidates")
    print("=" * 60)
    test_candidates = test_blocker.generate_candidates(s1_test)
    
    # Save candidate_pairs.tsv
    save_candidate_pairs(test_candidates, all_test_s1_ids, OUTPUT_DIR / 'candidate_pairs.tsv')
    
    print("=" * 60)
    print("STAGE 5: Scoring Test Pairs")
    print("=" * 60)
    cand_lookup = {**build_lookup(s2_test), **build_lookup(s3_test)}
    test_predictions = score_all_pairs(
        s1_test, test_candidates, cand_lookup, matcher, feature_names, threshold
    )
    
    # Save matching_results.tsv
    save_matching_results(test_predictions, all_test_s1_ids, OUTPUT_DIR / 'matching_results.tsv')
    
    print(f"\n[Pipeline] Prediction complete in {(time.time()-start)/60:.1f} minutes")
    print(f"[Pipeline] Output files in: {OUTPUT_DIR}")
    
    # Stats
    n_with_match = sum(1 for sid in all_test_s1_ids if test_predictions.get(sid, []))
    n_singleton = len(all_test_s1_ids) - n_with_match
    print(f"\n[Stats] Entities predicted with match: {n_with_match:,}")
    print(f"[Stats] Entities predicted as singleton: {n_singleton:,}")


def main():
    parser = argparse.ArgumentParser(description='Business Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                        help='Pipeline mode: train, predict, or full (train+predict)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation split fraction (default: 0.15)')
    parser.add_argument('--top-k-tfidf', type=int, default=15,
                        help='Top-K candidates from TF-IDF per entity (default: 15)')
    parser.add_argument('--neg-per-pos', type=int, default=3,
                        help='Negative pairs per positive in training (default: 3)')
    parser.add_argument('--no-tfidf', action='store_true',
                        help='Skip TF-IDF blocking (faster but lower recall)')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Business Entity Resolution Pipeline")
    print(f"  Mode: {args.mode.upper()}")
    print(f"{'='*60}\n")
    
    if args.mode in ('train', 'full'):
        run_training_pipeline(
            val_size=args.val_size,
            top_k_tfidf=args.top_k_tfidf,
            neg_per_pos=args.neg_per_pos,
            use_tfidf=not args.no_tfidf
        )
    
    if args.mode in ('predict', 'full'):
        run_prediction_pipeline()
    
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
