"""
Step 5: Model Training & Evaluation
LightGBM classifier for entity pair matching.

Trained on positive/negative pairs → predicts match probability.
Threshold optimized for F_0.5 (precision-heavy).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def f05_score(precision: float, recall: float) -> float:
    """F_0.5 = (1.25 * P * R) / (0.25 * P + R)"""
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_entity_level_f05(
    s1_ids: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_ids: list
) -> float:
    """
    Compute macro-average F_0.5 at the entity level.
    pair_ids: list of (s1_id, cand_id)
    """
    from collections import defaultdict
    
    # Group by S1 entity
    entity_true = defaultdict(set)
    entity_pred = defaultdict(set)
    
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if y_true[i] == 1:
            entity_true[s1_id].add(cand_id)
        if y_pred[i] == 1:
            entity_pred[s1_id].add(cand_id)
    
    all_s1_ids = set(entity_true.keys()) | set(entity_pred.keys())
    
    scores = []
    for s1_id in all_s1_ids:
        true_set = entity_true.get(s1_id, set())
        pred_set = entity_pred.get(s1_id, set())
        
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not pred_set:
            scores.append(0.0)  # missed all
        elif not true_set:
            scores.append(0.0)  # all false positives
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            scores.append(f05_score(p, r))
    
    return np.mean(scores) if scores else 0.0


def find_optimal_threshold(
    probas: np.ndarray,
    y_true: np.ndarray,
    pair_ids: list,
    thresholds: np.ndarray = None
) -> tuple:
    """
    Find threshold that maximizes entity-level F_0.5.
    Returns (best_threshold, best_f05, threshold_curve_df)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    results = []
    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        f05 = compute_entity_level_f05([], y_true, y_pred, pair_ids)
        results.append({'threshold': t, 'f05': f05})
    
    df = pd.DataFrame(results)
    best_row = df.loc[df['f05'].idxmax()]
    return best_row['threshold'], best_row['f05'], df


class EntityMatcher:
    """
    LightGBM-based entity matching classifier.
    """

    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': -1,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
        val_pair_ids: list = None
    ):
        """Train the LightGBM model."""
        self.feature_names = feature_names
        
        # Scale positive weight for class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        params = dict(self.LGBM_PARAMS)
        params['scale_pos_weight'] = scale_pos_weight
        
        print(f"[Matcher] Training LightGBM with {len(X_train):,} pairs "
              f"(pos={pos_count:,}, neg={neg_count:,})")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        callbacks = [lgb.log_evaluation(period=50), lgb.early_stopping(50, verbose=False)]
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            feature_name=feature_names if feature_names else 'auto',
            callbacks=callbacks
        )
        
        print(f"  Best iteration: {self.model.best_iteration_}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probability."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict binary match labels."""
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_pair_ids: list
    ) -> float:
        """Find and set the optimal F_0.5 threshold on validation set."""
        print("[Matcher] Optimizing threshold on validation set...")
        probas = self.predict_proba(X_val)
        best_t, best_f05, curve_df = find_optimal_threshold(probas, y_val, val_pair_ids)
        self.threshold = best_t
        print(f"  Optimal threshold: {best_t:.2f} → F_0.5 = {best_f05:.4f}")
        return best_t, best_f05, curve_df

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance dataframe."""
        if self.model is None or self.feature_names is None:
            return pd.DataFrame()
        imp = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': imp
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[Matcher] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'EntityMatcher':
        """Load model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[Matcher] Model loaded from {path}")
        return model


if __name__ == '__main__':
    print("Test EntityMatcher with dummy data...")
    X = np.random.rand(1000, 25).astype(np.float32)
    y = (np.random.rand(1000) > 0.7).astype(np.int32)
    pair_ids = [(f'S1-{i}', f'S2-{i}') for i in range(1000)]
    
    matcher = EntityMatcher()
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]
    val_pairs = pair_ids[800:]
    
    matcher.train(X_train, y_train, X_val, y_val)
    best_t, best_f05, curve = matcher.optimize_threshold(X_val, y_val, val_pairs)
    
    print(f"\nBest threshold: {best_t:.2f}, F_0.5: {best_f05:.4f}")
    print("\nTop features:")
    # No feature names in dummy, skip
    print("Done!")
