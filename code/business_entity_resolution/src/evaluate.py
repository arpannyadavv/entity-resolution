"""
Step 6: Evaluation & Scoring
Computes F_0.5 metric exactly as the leaderboard does.
Also generates analysis graphs with matplotlib.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


def f05_score(precision: float, recall: float) -> float:
    """F_0.5 = (1.25 * P * R) / (0.25 * P + R)"""
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    pred_dict: dict,   # {s1_id -> list of predicted match ids}
    gt_dict: dict,     # {s1_id -> set of true match ids}
    all_s1_ids: list   # all S1 entity IDs (including singletons)
) -> dict:
    """
    Evaluate predictions against ground truth using macro-average F_0.5.
    Singletons (no true match) are included: score 1.0 if correctly empty.
    
    Returns dict with:
      - 'macro_f05': overall score
      - 'macro_precision': overall precision  
      - 'macro_recall': overall recall
      - 'per_entity': DataFrame with per-entity scores
    """
    entity_scores = []
    
    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_list = pred_dict.get(s1_id, [])
        pred_set = set(pred_list)
        
        is_singleton = len(true_set) == 0
        
        if is_singleton and not pred_set:
            p, r, f = 1.0, 1.0, 1.0  # Correctly predicted singleton
        elif is_singleton and pred_set:
            p, r, f = 0.0, 1.0, 0.0  # False merge on singleton
        elif not pred_set:
            p, r, f = 0.0, 0.0, 0.0  # Missed all matches
        else:
            tp = len(true_set & pred_set)
            p = tp / len(pred_set)
            r = tp / len(true_set) if true_set else 0.0
            f = f05_score(p, r)
        
        entity_scores.append({
            's1_id': s1_id,
            'is_singleton': is_singleton,
            'n_true': len(true_set),
            'n_pred': len(pred_set),
            'precision': p,
            'recall': r,
            'f05': f,
        })
    
    df = pd.DataFrame(entity_scores)
    
    return {
        'macro_f05': df['f05'].mean(),
        'macro_precision': df['precision'].mean(),
        'macro_recall': df['recall'].mean(),
        'per_entity': df,
    }


def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth into {s1_id -> set of matched_ids}."""
    gt = pd.read_csv(gt_path, sep='\t', dtype=str)
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('')
    result = {}
    for _, row in gt.iterrows():
        ids = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]
        result[row['source1_entity_id']] = set(ids)
    return result


def plot_evaluation_dashboard(
    eval_results: dict,
    threshold_curve: pd.DataFrame = None,
    feature_importance: pd.DataFrame = None,
    output_path: str = 'evaluation_dashboard.png'
):
    """
    Generate a comprehensive matplotlib dashboard with:
    1. F_0.5 / Precision / Recall bar chart
    2. Threshold optimization curve
    3. Per-entity F_0.5 distribution
    4. Feature importance
    5. Singleton vs Non-singleton performance
    6. Match count distribution vs performance
    """
    if 'per_entity' in eval_results and isinstance(eval_results['per_entity'], pd.DataFrame):
        df = eval_results['per_entity']
    else:
        df = pd.DataFrame([{'f05': eval_results.get('macro_f05', 0.85), 'is_singleton': False, 'n_true': 1, 'precision': 0.9, 'recall': 0.8}])
    
    n_plots = 6
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    colors = {
        'primary': '#58a6ff',
        'success': '#3fb950',
        'warning': '#d29922',
        'danger': '#f85149',
        'purple': '#bc8cff',
        'teal': '#39d353',
        'bg': '#161b22',
        'text': '#c9d1d9',
        'grid': '#21262d',
    }
    
    def style_ax(ax, title):
        ax.set_facecolor(colors['bg'])
        ax.title.set_color(colors['text'])
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(colors=colors['text'], labelsize=8)
        ax.spines['bottom'].set_color(colors['grid'])
        ax.spines['top'].set_color(colors['grid'])
        ax.spines['left'].set_color(colors['grid'])
        ax.spines['right'].set_color(colors['grid'])
        ax.xaxis.label.set_color(colors['text'])
        ax.yaxis.label.set_color(colors['text'])
        ax.grid(True, color=colors['grid'], alpha=0.5, linewidth=0.5)

    # ---- Plot 1: Overall Metrics ----
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, '📊 Overall Performance (Macro)')
    metrics = ['F_0.5', 'Precision', 'Recall']
    values = [eval_results['macro_f05'], eval_results['macro_precision'], eval_results['macro_recall']]
    bar_colors = [colors['primary'], colors['success'], colors['warning']]
    bars = ax1.bar(metrics, values, color=bar_colors, alpha=0.85, width=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color=colors['text'])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score', color=colors['text'])
    
    # ---- Plot 2: Threshold Curve ----
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, '🎯 Threshold Optimization Curve')
    if threshold_curve is not None and len(threshold_curve) > 0:
        ax2.plot(threshold_curve['threshold'], threshold_curve['f05'],
                 color=colors['primary'], linewidth=2, marker='o', markersize=4)
        best_idx = threshold_curve['f05'].idxmax()
        best_t = threshold_curve.loc[best_idx, 'threshold']
        best_f = threshold_curve.loc[best_idx, 'f05']
        ax2.axvline(x=best_t, color=colors['danger'], linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.scatter([best_t], [best_f], color=colors['danger'], s=100, zorder=5)
        ax2.text(best_t + 0.01, best_f - 0.03,
                 f'Best: t={best_t:.2f}\nF0.5={best_f:.3f}',
                 color=colors['danger'], fontsize=8)
    ax2.set_xlabel('Threshold', color=colors['text'])
    ax2.set_ylabel('F_0.5', color=colors['text'])
    ax2.set_ylim(0, 1.05)

    # ---- Plot 3: Per-Entity F_0.5 Distribution ----
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, '📈 Per-Entity F_0.5 Distribution')
    ax3.hist(df['f05'], bins=25, color=colors['purple'], alpha=0.8, edgecolor='none')
    ax3.axvline(x=df['f05'].mean(), color=colors['warning'], linestyle='--',
                linewidth=1.5, label=f'Mean={df["f05"].mean():.3f}')
    ax3.legend(fontsize=8, facecolor=colors['bg'], labelcolor=colors['text'])
    ax3.set_xlabel('F_0.5 Score', color=colors['text'])
    ax3.set_ylabel('Count', color=colors['text'])

    # ---- Plot 4: Feature Importance ----
    ax4 = fig.add_subplot(gs[1, :2])
    style_ax(ax4, '🏆 Top Feature Importances (LightGBM)')
    if feature_importance is not None and len(feature_importance) > 0:
        top_feats = feature_importance.head(20)
        y_pos = range(len(top_feats))
        bar_data = top_feats['importance'].values
        ax4.barh(y_pos, bar_data, color=colors['teal'], alpha=0.8, edgecolor='none')
        ax4.set_yticks(list(y_pos))
        ax4.set_yticklabels(top_feats['feature'].tolist(), fontsize=8)
        ax4.set_xlabel('Importance', color=colors['text'])
        ax4.invert_yaxis()
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnot available',
                 ha='center', va='center', color=colors['text'], fontsize=12)

    # ---- Plot 5: Singleton vs Non-singleton ----
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, '🔍 Singleton vs Multi-match Performance')
    singleton_df = df[df['is_singleton']]
    non_singleton_df = df[~df['is_singleton']]
    labels = ['Singletons\n(no match)', 'Entities with\ntrue matches']
    means = [singleton_df['f05'].mean(), non_singleton_df['f05'].mean()]
    counts = [len(singleton_df), len(non_singleton_df)]
    bar_c = [colors['warning'], colors['primary']]
    bars5 = ax5.bar(labels, means, color=bar_c, alpha=0.85, width=0.4, edgecolor='none')
    for bar, val, cnt in zip(bars5, means, counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}\n(n={cnt:,})', ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color=colors['text'])
    ax5.set_ylim(0, 1.15)
    ax5.set_ylabel('Mean F_0.5', color=colors['text'])

    # ---- Plot 6: F05 by number of true matches ----
    ax6 = fig.add_subplot(gs[2, :])
    style_ax(ax6, '📉 F_0.5 Score vs. Number of True Matches per Entity')
    non_sing = df[~df['is_singleton']].copy()
    non_sing['n_true_capped'] = non_sing['n_true'].clip(upper=10)
    grp = non_sing.groupby('n_true_capped')['f05'].agg(['mean', 'count', 'std'])
    grp = grp[grp['count'] >= 5]  # Only groups with enough samples
    x = grp.index.values
    y_m = grp['mean'].values
    y_e = grp['std'].fillna(0).values
    ax6.plot(x, y_m, color=colors['primary'], linewidth=2, marker='o', markersize=6)
    ax6.fill_between(x, y_m - y_e, y_m + y_e, color=colors['primary'], alpha=0.2)
    for xi, yi, cnt in zip(x, y_m, grp['count'].values):
        ax6.text(xi, yi + 0.01, f'n={cnt}', ha='center', va='bottom',
                 fontsize=7, color=colors['text'])
    ax6.set_xlabel('Number of True Matches (capped at 10)', color=colors['text'])
    ax6.set_ylabel('Mean F_0.5', color=colors['text'])
    ax6.set_xticks(x)
    ax6.set_ylim(0, 1.1)

    # Title
    fig.suptitle(
        f'Business Entity Resolution — Evaluation Dashboard\n'
        f'Macro F_0.5: {eval_results["macro_f05"]:.4f} | '
        f'Precision: {eval_results["macro_precision"]:.4f} | '
        f'Recall: {eval_results["macro_recall"]:.4f}',
        fontsize=14, fontweight='bold', color=colors['text'], y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Evaluator] Dashboard saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    print("Test evaluator with dummy data...")
    all_ids = [f'S1-{i:05d}' for i in range(200)]
    gt_dict = {}
    for i, sid in enumerate(all_ids):
        if i % 4 == 0:
            gt_dict[sid] = set()  # singleton
        else:
            gt_dict[sid] = {f'S2-{i:05d}', f'S3-{i:05d}'}
    
    pred_dict = {}
    import random
    for sid in all_ids:
        true = gt_dict.get(sid, set())
        if true:
            pred_dict[sid] = list(true)[:1]  # miss some
        else:
            if random.random() < 0.1:
                pred_dict[sid] = [f'S2-99999']  # false merge
    
    results = evaluate_predictions(pred_dict, gt_dict, all_ids)
    print(f"Macro F_0.5: {results['macro_f05']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    
    # Dummy threshold curve
    thresholds = np.arange(0.1, 0.95, 0.05)
    dummy_f05 = np.sin(thresholds * np.pi) * 0.8
    curve_df = pd.DataFrame({'threshold': thresholds, 'f05': dummy_f05})
    
    # Dummy feature importance
    feat_imp = pd.DataFrame({
        'feature': [f'feat_{i}' for i in range(25)],
        'importance': np.random.rand(25) * 100
    }).sort_values('importance', ascending=False)
    
    output = plot_evaluation_dashboard(results, curve_df, feat_imp, 'test_dashboard.png')
    print(f"Dashboard saved: {output}")
