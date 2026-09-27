"""
Step 5: Model Training & Evaluation (v2 - Enhanced)
=====================================================
Ensemble of LightGBM + XGBoost for entity pair matching.

Improvements over v1:
  - Much better LightGBM hyperparameters (more leaves, more estimators)
  - XGBoost added as second classifier
  - Soft-voting ensemble: 0.60 * lgbm_proba + 0.40 * xgb_proba
  - Finer threshold sweep (0.02 step instead of 0.05)
  - Feature importance combined across both models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    """F_0.5 = (1.25 * P * R) / (0.25 * P + R)"""
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_entity_level_f05(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pair_ids: list
) -> float:
    """Macro-average F_0.5 at the entity level."""
    from collections import defaultdict
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
            scores.append(0.0)
        elif not true_set:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            scores.append(f05_score(p, r))
    return float(np.mean(scores)) if scores else 0.0


def find_optimal_threshold(
    probas: np.ndarray,
    y_true: np.ndarray,
    pair_ids: list,
    thresholds: np.ndarray = None
) -> tuple:
    """Find threshold maximising entity-level macro F_0.5."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.96, 0.02)   # finer grid than v1

    results = []
    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        f05 = compute_entity_level_f05(y_true, y_pred, pair_ids)
        results.append({'threshold': float(t), 'f05': f05})

    df = pd.DataFrame(results)
    best_row = df.loc[df['f05'].idxmax()]
    return float(best_row['threshold']), float(best_row['f05']), df


# ---------------------------------------------------------------------------
# Ensemble EntityMatcher
# ---------------------------------------------------------------------------

class EntityMatcher:
    """
    LightGBM + XGBoost soft-voting ensemble for entity pair classification.
    
    Final probability = LGBM_WEIGHT * lgbm_proba + XGB_WEIGHT * xgb_proba
    """

    LGBM_WEIGHT = 0.60
    XGB_WEIGHT  = 0.40

    # ---- LightGBM params (v2: more leaves, more trees, lower lr) ----
    LGBM_PARAMS = {
        'objective':        'binary',
        'metric':           'binary_logloss',
        'boosting_type':    'gbdt',
        'num_leaves':       255,        # 127 → 255
        'max_depth':        -1,
        'learning_rate':    0.03,       # 0.05 → 0.03
        'n_estimators':     2000,       # 500 → 2000
        'min_child_samples':15,         # 20 → 15
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq':     5,
        'reg_alpha':        0.05,
        'reg_lambda':       0.50,
        'min_split_gain':   0.0,
        'random_state':     42,
        'n_jobs':           -1,
        'verbose':          -1,
    }

    # ---- XGBoost params ----
    XGB_PARAMS = {
        'objective':        'binary:logistic',
        'eval_metric':      'logloss',
        'max_depth':        8,
        'learning_rate':    0.05,
        'n_estimators':     1500,
        'min_child_weight': 5,
        'subsample':        0.80,
        'colsample_bytree': 0.80,
        'reg_alpha':        0.05,
        'reg_lambda':       1.0,
        'random_state':     42,
        'n_jobs':           -1,
        'verbosity':        0,
        'tree_method':      'hist',     # fast on CPU
    }

    def __init__(self, threshold: float = 0.5):
        self.threshold    = threshold
        self.lgbm_model   = None
        self.xgb_model    = None
        self.feature_names = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val:  np.ndarray = None,
        feature_names: list = None,
        val_pair_ids:  list = None
    ):
        self.feature_names = feature_names

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        spw = neg_count / pos_count if pos_count > 0 else 1.0

        print(f"[Matcher] Training on {len(X_train):,} pairs "
              f"(pos={pos_count:,}, neg={neg_count:,})")
        print(f"  scale_pos_weight: {spw:.2f}")

        # ---- LightGBM ----
        print("\n[Matcher] Fitting LightGBM...")
        lgbm_params = dict(self.LGBM_PARAMS)
        lgbm_params['scale_pos_weight'] = spw
        callbacks = [lgb.log_evaluation(period=100),
                     lgb.early_stopping(stopping_rounds=100, verbose=False)]
        eval_set_lgbm = [(X_val, y_val)] if (X_val is not None) else None
        self.lgbm_model = lgb.LGBMClassifier(**lgbm_params)
        self.lgbm_model.fit(
            X_train, y_train,
            eval_set=eval_set_lgbm,
            feature_name=feature_names if feature_names else 'auto',
            callbacks=callbacks
        )
        print(f"  LGBM best iteration: {self.lgbm_model.best_iteration_}")

        # ---- XGBoost ----
        print("\n[Matcher] Fitting XGBoost...")
        xgb_params = dict(self.XGB_PARAMS)
        xgb_params['scale_pos_weight'] = spw
        eval_set_xgb = [(X_val, y_val)] if (X_val is not None) else None
        if eval_set_xgb:
            xgb_params['early_stopping_rounds'] = 100
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set_xgb,
            verbose=False
        )
        best_xgb = getattr(self.xgb_model, 'best_iteration', 'N/A')
        print(f"  XGB best iteration: {best_xgb}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble probability."""
        lgbm_p = self.lgbm_model.predict_proba(X)[:, 1]
        xgb_p  = self.xgb_model.predict_proba(X)[:, 1]
        return self.LGBM_WEIGHT * lgbm_p + self.XGB_WEIGHT * xgb_p

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_pair_ids: list
    ) -> tuple:
        print("[Matcher] Optimizing threshold on validation set...")
        probas = self.predict_proba(X_val)
        best_t, best_f05, curve_df = find_optimal_threshold(probas, y_val, val_pair_ids)
        self.threshold = best_t
        print(f"  Optimal threshold: {best_t:.2f} -> F_0.5 = {best_f05:.4f}")
        return best_t, best_f05, curve_df

    def feature_importance(self) -> pd.DataFrame:
        """Return combined feature importance (mean of LGBM + XGB normalized)."""
        if self.lgbm_model is None or self.feature_names is None:
            return pd.DataFrame()
        lgbm_imp = self.lgbm_model.feature_importances_.astype(float)
        xgb_imp  = self.xgb_model.feature_importances_.astype(float)
        # Normalize each to [0, 1] then average
        lgbm_imp = lgbm_imp / (lgbm_imp.max() + 1e-9)
        xgb_imp  = xgb_imp  / (xgb_imp.max()  + 1e-9)
        combined = 0.5 * lgbm_imp + 0.5 * xgb_imp
        return pd.DataFrame({
            'feature':    self.feature_names,
            'importance': combined
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[Matcher] Ensemble model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'EntityMatcher':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"[Matcher] Ensemble model loaded from {path}")
        return model


if __name__ == '__main__':
    print("Test EntityMatcher ensemble with dummy data...")
    n_feat = 40
    X = np.random.rand(2000, n_feat).astype(np.float32)
    y = (np.random.rand(2000) > 0.7).astype(np.int32)
    pair_ids = [(f'S1-{i:05d}', f'S2-{i:05d}') for i in range(2000)]

    X_tr, X_vl = X[:1600], X[1600:]
    y_tr, y_vl = y[:1600], y[1600:]
    vp = pair_ids[1600:]

    matcher = EntityMatcher()
    matcher.train(X_tr, y_tr, X_vl, y_vl,
                  feature_names=[f'f{i}' for i in range(n_feat)],
                  val_pair_ids=vp)
    best_t, best_f05, _ = matcher.optimize_threshold(X_vl, y_vl, vp)
    print(f"\nBest threshold={best_t:.2f}, F_0.5={best_f05:.4f}")
