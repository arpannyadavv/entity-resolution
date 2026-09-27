"""
Main Pipeline: Business Entity Resolution (v2.1 - Winning Solution)
===================================================================
End-to-end pipeline:
  1. High-recall multi-strategy blocking (Prefixes, Phonetic Soundex, Tokens, Address & ZIP)
  2. 41 Rich pairwise string, phonetic, token, and address similarity features
  3. LightGBM + XGBoost ensemble classifier with soft voting
  4. Decision threshold optimization directly maximizing Macro-average F_0.5
  5. Country-partitioned streaming test inference (memory-safe, ultra-fast)
  6. Automatic validation against submission format rules
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import gc
import json
import time
import pickle
import random
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rapidfuzz import fuzz as rfuzz

# Local modules
from src.preprocess import clean_name, clean_address
from src.blocking import EntityBlocker
from src.features import compute_pair_features, get_feature_names
from src.training_builder import build_training_pairs, parse_matched_ids, build_lookup
from src.model import EntityMatcher
from src.evaluate import evaluate_predictions, plot_evaluation_dashboard

# ---- Paths ----
BASE_DIR   = Path(__file__).parent.parent.parent   # student_resource/
DATA_DIR   = BASE_DIR / 'dataset'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR  = Path(__file__).parent / 'models'
PLOTS_DIR  = Path(__file__).parent / 'plots'

DTYPES = {'entity_id': str, 'business_name': str,
          'business_address': str, 'country': str}


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"[Pipeline] Output dir: {OUTPUT_DIR}")
    print(f"[Pipeline] Model  dir: {MODEL_DIR}")
    print(f"[Pipeline] Plots  dir: {PLOTS_DIR}")


def load_training_sample(sample_size: int = 35000):
    """
    Load a representative sample of training data with all corresponding
    true matches and sufficient candidate background records.
    """
    print(f"\n[Pipeline] Loading training sample (sample_size={sample_size:,})...")
    gt_path = DATA_DIR / 'train' / 'train_ground_truth.tsv'
    gt_df = pd.read_csv(gt_path, sep='\t', nrows=sample_size, dtype={'source1_entity_id': str, 'matched_entity_ids': str})
    gt_df['matched_entity_ids'] = gt_df['matched_entity_ids'].fillna('')

    target_s1_ids = set(gt_df['source1_entity_id'])
    all_matched_ids = set()
    for _, r in gt_df.iterrows():
        mids = parse_matched_ids(r['matched_entity_ids'])
        all_matched_ids.update(mids)

    print(f"  Target S1 entities: {len(target_s1_ids):,} | True matched S2/S3 entities: {len(all_matched_ids):,}")

    # Load S1 rows
    s1_rows = []
    with open(DATA_DIR / 'train' / 'train_source1.tsv', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in target_s1_ids:
                s1_rows.append(parts)
                if len(s1_rows) == len(target_s1_ids):
                    break
    s1_df = pd.DataFrame(s1_rows, columns=header)
    print(f"  Loaded S1 sample: {len(s1_df):,} records")

    # Load S2 rows: all true matches + 50,000 background
    s2_rows = []
    with open(DATA_DIR / 'train' / 'train_source2.tsv', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts[0] in all_matched_ids or i < 50000:
                s2_rows.append(parts)
    s2_df = pd.DataFrame(s2_rows, columns=header)
    print(f"  Loaded S2 pool:   {len(s2_df):,} records")

    # Load S3 rows: all true matches + 50,000 background
    s3_rows = []
    with open(DATA_DIR / 'train' / 'train_source3.tsv', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if parts[0] in all_matched_ids or i < 50000:
                s3_rows.append(parts)
    s3_df = pd.DataFrame(s3_rows, columns=header)
    print(f"  Loaded S3 pool:   {len(s3_df):,} records")

    return s1_df, s2_df, s3_df, gt_df


def score_blocking(candidates: dict, gt_dict: dict, all_s1_ids: list):
    total_true = total_recalled = total_cands = 0
    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        cand_set = candidates.get(s1_id, set())
        total_true     += len(true_set)
        total_recalled += len(true_set & cand_set)
        total_cands    += len(cand_set)
    recall    = total_recalled / total_true if total_true > 0 else 0.0
    avg_cands = total_cands / len(all_s1_ids) if all_s1_ids else 0.0
    print(f"\n[Blocking Stats]")
    print(f"  Recall ceiling: {recall:.4f} ({total_recalled:,}/{total_true:,} true matches in candidates)")
    print(f"  Avg candidates per S1: {avg_cands:.1f}")
    print(f"  Total candidate pairs:  {total_cands:,}")
    return recall, avg_cands


def run_training_pipeline(
    val_size: float = 0.15,
    sample_size: int = 35000,
    neg_per_pos: int = 5,
):
    ensure_dirs()
    start_time = time.time()

    print("=" * 60)
    print("STAGE 1: Loading Training Sample")
    print("=" * 60)
    s1, s2, s3, gt = load_training_sample(sample_size=sample_size)

    gt_dict = {}
    for _, row in gt.iterrows():
        gt_dict[row['source1_entity_id']] = set(parse_matched_ids(row['matched_entity_ids']))

    all_s1_ids = s1['entity_id'].tolist()
    train_ids, val_ids = train_test_split(all_s1_ids, test_size=val_size, random_state=42)
    print(f"\n[Split] Train S1: {len(train_ids):,} | Val S1: {len(val_ids):,}")

    s1_train = s1[s1['entity_id'].isin(set(train_ids))].reset_index(drop=True)
    s1_val   = s1[s1['entity_id'].isin(set(val_ids))].reset_index(drop=True)

    print("=" * 60)
    print("STAGE 2: Building Blocker Indexes")
    print("=" * 60)
    blocker = EntityBlocker(max_candidates=25, max_block_size=300)
    blocker.fit(s2, s3)

    print("=" * 60)
    print("STAGE 3: Generating Training Candidates")
    print("=" * 60)
    train_candidates = blocker.generate_candidates(s1_train)

    print("=" * 60)
    print("STAGE 4: Generating Validation Candidates")
    print("=" * 60)
    val_candidates = blocker.generate_candidates(s1_val)

    print("\n[Blocking Evaluation]")
    score_blocking(train_candidates, gt_dict, train_ids)
    val_recall, val_avg_cands = score_blocking(val_candidates, gt_dict, val_ids)

    print("=" * 60)
    print("STAGE 5: Building Labeled Training Pairs")
    print("=" * 60)
    X_train, y_train, train_pair_ids, feature_names = build_training_pairs(
        s1_train, s2, s3,
        gt[gt['source1_entity_id'].isin(set(train_ids))],
        train_candidates,
        neg_per_pos=neg_per_pos,
        hard_negative_fraction=0.6
    )

    print("=" * 60)
    print("STAGE 6: Building Validation Pairs")
    print("=" * 60)
    X_val, y_val, val_pair_ids, _ = build_training_pairs(
        s1_val, s2, s3,
        gt[gt['source1_entity_id'].isin(set(val_ids))],
        val_candidates,
        neg_per_pos=neg_per_pos * 2,
        hard_negative_fraction=0.7
    )

    print("=" * 60)
    print("STAGE 7: Training Ensemble Model (LightGBM + XGBoost)")
    print("=" * 60)
    matcher = EntityMatcher(threshold=0.5)
    matcher.train(X_train, y_train, X_val, y_val,
                  feature_names=feature_names, val_pair_ids=val_pair_ids)

    print("=" * 60)
    print("STAGE 8: Optimizing F_0.5 Threshold")
    print("=" * 60)
    best_t, best_f05, threshold_curve = matcher.optimize_threshold(X_val, y_val, val_pair_ids)
    print(f"\n[Threshold] Optimal Decision Threshold: {best_t:.3f} (Val F_0.5: {best_f05:.4f})")

    # Save model artifacts
    matcher.save(MODEL_DIR / 'entity_matcher.pkl')
    with open(MODEL_DIR / 'blocker.pkl', 'wb') as f:
        pickle.dump(blocker, f)
    with open(MODEL_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    with open(MODEL_DIR / 'threshold.json', 'w') as f:
        json.dump({'threshold': float(best_t), 'val_f05': float(best_f05)}, f)

    feat_imp = matcher.feature_importance()
    print("\nTop 15 Most Important Features:")
    print(feat_imp.head(15).to_string(index=False))

    plot_path = PLOTS_DIR / 'evaluation_dashboard.png'
    val_preds = matcher.predict(X_val, threshold=best_t)
    pred_dict = defaultdict(list)
    for idx, (s1_id, cand_id) in enumerate(val_pair_ids):
        if val_preds[idx] == 1:
            pred_dict[s1_id].append(cand_id)
    eval_summary = evaluate_predictions(pred_dict, gt_dict, s1_ids=val_ids)
    plot_evaluation_dashboard(eval_summary, threshold_curve, feat_imp, str(plot_path))

    elapsed = (time.time() - start_time) / 60
    print(f"\n[Pipeline] Training complete in {elapsed:.1f} minutes")
    print(f"[Pipeline] Model artifacts saved to: {MODEL_DIR}")
    print(f"[Pipeline] Evaluation dashboard:     {plot_path}")

    return matcher, blocker, feature_names, best_t


def run_prediction_pipeline():
    """
    Run memory-safe, country-partitioned candidate generation and scoring
    for the entire test set (1.73M entities).
    """
    ensure_dirs()
    start_time = time.time()

    print("=" * 60)
    print("STAGE 1: Loading Trained Ensemble & Threshold")
    print("=" * 60)
    matcher = EntityMatcher.load(MODEL_DIR / 'entity_matcher.pkl')
    with open(MODEL_DIR / 'feature_names.json') as f:
        feature_names = json.load(f)
    with open(MODEL_DIR / 'threshold.json') as f:
        thresh_data = json.load(f)
    threshold = float(thresh_data['threshold'])
    print(f"  Loaded threshold: {threshold:.3f} (Val F_0.5: {thresh_data['val_f05']:.4f})")

    # Output paths
    matching_tsv = OUTPUT_DIR / 'matching_results.tsv'
    candidate_tsv = OUTPUT_DIR / 'candidate_pairs.tsv'

    # Check already processed entities for seamless resume
    already_processed = set()
    if matching_tsv.exists() and matching_tsv.stat().st_size > 0:
        with open(matching_tsv, encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if parts and parts[0]:
                    already_processed.add(parts[0])
        print(f"  [Resume] Found {len(already_processed):,} already processed entities in {matching_tsv}")

    mode = 'a' if already_processed else 'w'
    f_match = open(matching_tsv, mode, encoding='utf-8')
    f_cand  = open(candidate_tsv, mode, encoding='utf-8')
    if mode == 'w':
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    countries = ['France', 'US', 'India']
    total_s1_processed = len(already_processed)
    total_matched_count = 0
    total_singleton_count = 0

    for country in countries:
        c_start = time.time()
        print(f"\n{'='*60}")
        print(f"PROCESSING COUNTRY: {country.upper()}")
        print(f"{'='*60}")

        # Check unprocessed S1 records for this country
        unprocessed_s1 = []
        with open(DATA_DIR / 'test' / 'test_source1.tsv', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3].strip() == country:
                    if parts[0] not in already_processed:
                        unprocessed_s1.append(parts[:4])

        if not unprocessed_s1:
            print(f"  [Skip] {country} is already 100% processed. Skipping to next country.")
            continue

        print(f"  [Resume] {country} has {len(unprocessed_s1):,} records remaining to process.")

        # 1. Load S2 candidate records for this country
        print(f"  Reading test S2 records for {country}...")
        s2_records = []
        cand_lookup = {}
        with open(DATA_DIR / 'test' / 'test_source2.tsv', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3].strip() == country:
                    s2_records.append(parts[:4])
                    cand_lookup[parts[0]] = {
                        'entity_id': parts[0],
                        'business_name': parts[1],
                        'business_address': parts[2],
                        'country': parts[3]
                    }
        print(f"  Loaded {len(s2_records):,} S2 records for {country}")

        # 2. Load S3 candidate records for this country
        print(f"  Reading test S3 records for {country}...")
        s3_records = []
        with open(DATA_DIR / 'test' / 'test_source3.tsv', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4 and parts[3].strip() == country:
                    s3_records.append(parts[:4])
                    cand_lookup[parts[0]] = {
                        'entity_id': parts[0],
                        'business_name': parts[1],
                        'business_address': parts[2],
                        'country': parts[3]
                    }
        print(f"  Loaded {len(s3_records):,} S3 records for {country}")

        # 3. Fit Blocker on candidate pool
        blocker = EntityBlocker(max_candidates=20, max_block_size=300)
        blocker.fit(s2_records, s3_records)

        # 4. Stream S1 entities in memory-light chunks
        print(f"  Streaming & scoring S1 entities for {country}...")
        chunk_size = 2000
        current_chunk = []

        def process_s1_chunk(chunk):
            nonlocal total_s1_processed, total_matched_count, total_singleton_count
            if not chunk:
                return

            chunk_X = []
            chunk_pairs = []
            chunk_candidates = {}

            for s1_tuple in chunk:
                s1_id = s1_tuple[0]
                cands = blocker.get_candidates(s1_tuple)
                valid_cands = [c for c in cands if c in cand_lookup]
                chunk_candidates[s1_id] = valid_cands

                s1_clean_n = clean_name(s1_tuple[1])
                s1_dict = {
                    'entity_id': s1_tuple[0],
                    'business_name': s1_tuple[1],
                    'business_address': s1_tuple[2],
                    'country': s1_tuple[3]
                }
                for cand_id in valid_cands:
                    cand_dict = cand_lookup[cand_id]
                    cand_clean_n = clean_name(cand_dict['business_name'])
                    if rfuzz.ratio(s1_clean_n, cand_clean_n) < 25 and (s1_clean_n not in cand_clean_n and cand_clean_n not in s1_clean_n):
                        continue
                    feats = compute_pair_features(s1_dict, cand_dict)
                    chunk_X.append([feats[f] for f in feature_names])
                    chunk_pairs.append((s1_id, cand_id))

            # Batch scoring with ensemble
            chunk_matches = defaultdict(list)
            if chunk_X:
                X_arr = np.array(chunk_X, dtype=np.float32)
                probas = matcher.predict_proba(X_arr)
                for idx, (s1_id, cand_id) in enumerate(chunk_pairs):
                    if probas[idx] >= threshold:
                        chunk_matches[s1_id].append(cand_id)

            # Write chunk directly to files
            for s1_tuple in chunk:
                s1_id = s1_tuple[0]
                cand_list = chunk_candidates.get(s1_id, [])
                match_list = list(dict.fromkeys(chunk_matches.get(s1_id, [])))
                match_list = [m for m in match_list if m in set(cand_list)]

                f_cand.write(f"{s1_id}\t{','.join(cand_list)}\n")
                f_match.write(f"{s1_id}\t{','.join(match_list)}\n")

                if match_list:
                    total_matched_count += 1
                else:
                    total_singleton_count += 1
                total_s1_processed += 1

                if total_s1_processed % 10000 == 0:
                    print(f"    [{country}] {total_s1_processed:,} S1 entities processed | Matched: {total_matched_count:,} | Singletons: {total_singleton_count:,}")

        # Stream unprocessed S1 records in chunks
        for i in range(0, len(unprocessed_s1), chunk_size):
            process_s1_chunk(unprocessed_s1[i:i + chunk_size])

        f_match.flush()
        f_cand.flush()

        c_elapsed = (time.time() - c_start) / 60
        print(f"  {country} completed in {c_elapsed:.1f} minutes")

        # Cleanup memory for next country
        del unprocessed_s1, s2_records, s3_records, blocker, cand_lookup
        gc.collect()

    f_match.close()
    f_cand.close()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"PREDICTION COMPLETE!")
    print(f"  Total S1 entities processed: {total_s1_processed:,}")
    print(f"  Entities with >= 1 match:    {total_matched_count:,} ({total_matched_count/total_s1_processed*100:.1f}%)")
    print(f"  Singletons (no match):       {total_singleton_count:,} ({total_singleton_count/total_s1_processed*100:.1f}%)")
    print(f"  Time taken:                  {elapsed:.1f} minutes")
    print(f"  Matching output:             {matching_tsv}")
    print(f"  Candidate output:            {candidate_tsv}")
    print(f"{'='*60}\n")

    # Run submission validator
    print("=" * 60)
    print("VALIDATING SUBMISSION ARTIFACTS")
    print("=" * 60)
    validator_cmd = [
        sys.executable,
        str(BASE_DIR / 'utils' / 'validate_submission.py'),
        '--matching', str(matching_tsv),
        '--candidate', str(candidate_tsv),
        '--test-dir', str(DATA_DIR / 'test')
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator STDERR:", res.stderr)
    if res.returncode == 0:
        print("[PASS] Submission files passed all formatting and validation checks!")
    else:
        print(f"[FAIL] Submission validation exited with code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description='Business Entity Resolution Pipeline v2.1')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'],
                        default='full', help='Pipeline execution mode')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation split fraction (default: 0.15)')
    parser.add_argument('--sample-size', type=int, default=35000,
                        help='Number of S1 training records to sample (default: 35000)')
    parser.add_argument('--neg-per-pos', type=int, default=5,
                        help='Negatives per positive in training (default: 5)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Business Entity Resolution Pipeline v2.1")
    print(f"  Mode: {args.mode.upper()}")
    print(f"{'='*60}\n")

    if args.mode in ('train', 'full'):
        run_training_pipeline(
            val_size=args.val_size,
            sample_size=args.sample_size,
            neg_per_pos=args.neg_per_pos
        )

    if args.mode in ('predict', 'full'):
        run_prediction_pipeline()

    print(f"\n{'='*60}")
    print("  ALL PIPELINE STAGES FINISHED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
