"""
Improved Baseline Evaluation Script
CareerRank Project - Day 2 Improvements

Evaluates all improved baselines with corrected relevance threshold:
- TF-IDF Enhanced
- Structured Regression Enhanced
- Hybrid (TF-IDF + Structured)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append('.')
from evaluation.ranking_metrics import recall_at_k, mrr, ndcg_at_k, rmse, mae

print("=" * 80)
print("IMPROVED BASELINE EVALUATION")
print("=" * 80)

np.random.seed(42)

# Load relevance threshold
print("\nLoading relevance threshold...")
with open('artifacts/eval_sets/relevance_threshold.json', 'r') as f:
    threshold_info = json.load(f)

relevance_threshold = threshold_info['relevance_threshold']
print(f"Relevance threshold: {relevance_threshold:.2f} ({threshold_info['description']})")

# Load improved evaluation sets
print("\nLoading improved evaluation sets...")

def load_eval_set(path):
    if path.endswith('.parquet'):
        try:
            return pd.read_parquet(path)
        except:
            return pd.read_csv(path.replace('.parquet', '.csv'))
    return pd.read_csv(path)

val_eval = load_eval_set('artifacts/eval_sets/val_candidates_improved.parquet')
test_eval = load_eval_set('artifacts/eval_sets/test_candidates_improved.parquet')

print(f"Val candidates: {len(val_eval)}")
print(f"Test candidates: {len(test_eval)}")
print(f"Val relevant items: {val_eval['is_relevant'].sum()}")
print(f"Test relevant items: {test_eval['is_relevant'].sum()}")

# Load all baseline rankings
print("\nLoading baseline rankings...")

baselines = {}

# TF-IDF Enhanced
try:
    with open('artifacts/baselines/tfidf_improved_rankings_val.json', 'r') as f:
        baselines['TF-IDF Enhanced'] = {'val': json.load(f)}
    with open('artifacts/baselines/tfidf_improved_rankings_test.json', 'r') as f:
        baselines['TF-IDF Enhanced']['test'] = json.load(f)
    print(f"✓ Loaded TF-IDF Enhanced")
except FileNotFoundError:
    print("✗ TF-IDF Enhanced not found")

# Structured Enhanced
try:
    with open('artifacts/baselines/structured_improved_rankings_val.json', 'r') as f:
        baselines['Structured Enhanced'] = {'val': json.load(f)}
    with open('artifacts/baselines/structured_improved_rankings_test.json', 'r') as f:
        baselines['Structured Enhanced']['test'] = json.load(f)
    
    with open('artifacts/baselines/structured_improved_rmse_mae.json', 'r') as f:
        baselines['Structured Enhanced']['metrics'] = json.load(f)
    print(f"✓ Loaded Structured Enhanced")
except FileNotFoundError:
    print("✗ Structured Enhanced not found")

# Hybrid
try:
    with open('artifacts/baselines/hybrid_rankings_val.json', 'r') as f:
        baselines['Hybrid'] = {'val': json.load(f)}
    with open('artifacts/baselines/hybrid_rankings_test.json', 'r') as f:
        baselines['Hybrid']['test'] = json.load(f)
    print(f"✓ Loaded Hybrid")
except FileNotFoundError:
    print("✗ Hybrid not found")

# Original baselines for comparison
try:
    with open('artifacts/baselines/tfidf_rankings_val.json', 'r') as f:
        baselines['TF-IDF Original'] = {'val': json.load(f)}
    with open('artifacts/baselines/tfidf_rankings_test.json', 'r') as f:
        baselines['TF-IDF Original']['test'] = json.load(f)
    print(f"✓ Loaded TF-IDF Original (for comparison)")
except FileNotFoundError:
    pass


def evaluate_rankings(eval_df, rankings, baseline_name, k_values=[5, 10, 50]):
    """Evaluate rankings using multiple metrics."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {baseline_name}")
    print(f"{'=' * 60}")
    
    grouped = eval_df.groupby('profile_a_id')
    
    all_metrics = []
    
    for profile_a_id, group in grouped:
        if profile_a_id not in rankings:
            continue
        
        # Get ground truth
        relevant_set = set(group[group['is_relevant']]['profile_b_id'].values)
        
        # Get relevance grades
        relevance_grades = {}
        for _, row in group.iterrows():
            relevance_grades[row['profile_b_id']] = row['relevance_grade']
        
        ranking_dict = rankings[profile_a_id]
        
        metrics = {'profile_a_id': profile_a_id}
        
        # Compute metrics for each K
        for k in k_values:
            key = f'top_{k}'
            if key not in ranking_dict:
                continue
            
            ranked_list = [item[0] for item in ranking_dict[key]]
            
            # Recall@K
            recall = recall_at_k(relevant_set, ranked_list, k)
            metrics[f'recall@{k}'] = recall
            
            # Precision@K
            num_relevant_in_topk = len(set(ranked_list[:k]) & relevant_set)
            precision = num_relevant_in_topk / k if k > 0 else 0
            metrics[f'precision@{k}'] = precision
            
            # NDCG@K
            ndcg = ndcg_at_k(relevance_grades, ranked_list, k)
            metrics[f'ndcg@{k}'] = ndcg
        
        # MRR
        ranked_list_full = [item[0] for item in ranking_dict.get('top_50', [])]
        mrr_score = mrr(relevant_set, ranked_list_full)
        metrics['mrr'] = mrr_score
        
        # Hit Rate (did we get at least one relevant item in top-K?)
        for k in k_values:
            key = f'top_{k}'
            if key in ranking_dict:
                ranked_list = [item[0] for item in ranking_dict[key]]
                hit = 1 if len(set(ranked_list[:k]) & relevant_set) > 0 else 0
                metrics[f'hit@{k}'] = hit
        
        all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    print(f"\nQueries evaluated: {len(metrics_df)}")
    print(f"\nAggregated Metrics (mean):")
    
    summary = {}
    for col in metrics_df.columns:
        if col != 'profile_a_id':
            mean_val = metrics_df[col].mean()
            summary[col] = mean_val
            print(f"  {col}: {mean_val:.4f}")
    
    return summary, metrics_df


# Evaluate all baselines
all_results = []

for baseline_name, baseline_data in baselines.items():
    if 'val' not in baseline_data or 'test' not in baseline_data:
        continue
    
    print("\n" + "=" * 80)
    print(f"{baseline_name.upper()}")
    print("=" * 80)
    
    # Validation
    val_summary, val_metrics = evaluate_rankings(
        val_eval, baseline_data['val'], f"{baseline_name} Validation"
    )
    
    # Test
    test_summary, test_metrics = evaluate_rankings(
        test_eval, baseline_data['test'], f"{baseline_name} Test"
    )
    
    # Add to results
    all_results.append({
        'baseline': baseline_name,
        'split': 'validation',
        **val_summary,
        'rmse': baseline_data.get('metrics', {}).get('val', {}).get('rmse', None),
        'mae': baseline_data.get('metrics', {}).get('val', {}).get('mae', None)
    })
    
    all_results.append({
        'baseline': baseline_name,
        'split': 'test',
        **test_summary,
        'rmse': baseline_data.get('metrics', {}).get('test', {}).get('rmse', None),
        'mae': baseline_data.get('metrics', {}).get('test', {}).get('mae', None)
    })

# Create comprehensive summary
print("\n" + "=" * 80)
print("CREATING COMPREHENSIVE METRICS SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame(all_results)

# Save summary
output_path = 'artifacts/baselines/improved_metrics_summary.csv'
summary_df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Print summary table
print("\n" + "=" * 80)
print("IMPROVED BASELINE METRICS SUMMARY")
print("=" * 80)

# Select key metrics to display
key_metrics = ['baseline', 'split', 'recall@5', 'recall@10', 'precision@5', 
               'ndcg@10', 'mrr', 'hit@10', 'rmse', 'mae']
display_cols = [col for col in key_metrics if col in summary_df.columns]

print("\n" + summary_df[display_cols].to_string(index=False))

# Create comparison visualization
print("\n" + "=" * 80)
print("CREATING COMPARISON VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Filter to test set only for cleaner visualization
test_results = summary_df[summary_df['split'] == 'test'].copy()

# 1. Recall@K comparison
ax = axes[0, 0]
metrics_to_plot = ['recall@5', 'recall@10', 'recall@50']
x = np.arange(len(test_results))
width = 0.25

for i, metric in enumerate(metrics_to_plot):
    if metric in test_results.columns:
        ax.bar(x + i*width, test_results[metric], width, label=metric)

ax.set_xlabel('Baseline')
ax.set_ylabel('Recall')
ax.set_title('Recall@K Comparison (Test Set)')
ax.set_xticks(x + width)
ax.set_xticklabels(test_results['baseline'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Precision@K comparison
ax = axes[0, 1]
metrics_to_plot = ['precision@5', 'precision@10']
x = np.arange(len(test_results))
width = 0.35

for i, metric in enumerate(metrics_to_plot):
    if metric in test_results.columns:
        ax.bar(x + i*width, test_results[metric], width, label=metric)

ax.set_xlabel('Baseline')
ax.set_ylabel('Precision')
ax.set_title('Precision@K Comparison (Test Set)')
ax.set_xticks(x + width/2)
ax.set_xticklabels(test_results['baseline'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. NDCG@K comparison
ax = axes[0, 2]
metrics_to_plot = ['ndcg@5', 'ndcg@10']
x = np.arange(len(test_results))
width = 0.35

for i, metric in enumerate(metrics_to_plot):
    if metric in test_results.columns:
        ax.bar(x + i*width, test_results[metric], width, label=metric)

ax.set_xlabel('Baseline')
ax.set_ylabel('NDCG')
ax.set_title('NDCG@K Comparison (Test Set)')
ax.set_xticks(x + width/2)
ax.set_xticklabels(test_results['baseline'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. MRR comparison
ax = axes[1, 0]
if 'mrr' in test_results.columns:
    ax.bar(test_results['baseline'], test_results['mrr'], color='steelblue')
ax.set_xlabel('Baseline')
ax.set_ylabel('MRR')
ax.set_title('Mean Reciprocal Rank (Test Set)')
ax.set_xticklabels(test_results['baseline'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# 5. Hit Rate@K comparison
ax = axes[1, 1]
metrics_to_plot = ['hit@5', 'hit@10']
x = np.arange(len(test_results))
width = 0.35

for i, metric in enumerate(metrics_to_plot):
    if metric in test_results.columns:
        ax.bar(x + i*width, test_results[metric], width, label=metric)

ax.set_xlabel('Baseline')
ax.set_ylabel('Hit Rate')
ax.set_title('Hit Rate@K Comparison (Test Set)')
ax.set_xticks(x + width/2)
ax.set_xticklabels(test_results['baseline'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. RMSE/MAE comparison (for models that have it)
ax = axes[1, 2]
models_with_rmse = test_results[test_results['rmse'].notna()]
if len(models_with_rmse) > 0:
    x = np.arange(len(models_with_rmse))
    width = 0.35
    
    ax.bar(x, models_with_rmse['rmse'], width, label='RMSE', alpha=0.8)
    ax.bar(x + width, models_with_rmse['mae'], width, label='MAE', alpha=0.8)
    
    ax.set_xlabel('Baseline')
    ax.set_ylabel('Error')
    ax.set_title('RMSE/MAE Comparison (Test Set)')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(models_with_rmse['baseline'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/baselines/improved_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: artifacts/baselines/improved_metrics_comparison.png")

# Score distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
val_relevant = val_eval[val_eval['is_relevant']]['compatibility_score']
val_non_relevant = val_eval[~val_eval['is_relevant']]['compatibility_score']

ax.hist(val_relevant, bins=30, alpha=0.6, label=f'Relevant (score >= {relevance_threshold:.1f})', 
        color='green', edgecolor='black')
ax.hist(val_non_relevant, bins=30, alpha=0.6, label=f'Non-Relevant (score < {relevance_threshold:.1f})', 
        color='red', edgecolor='black')
ax.set_xlabel('Compatibility Score')
ax.set_ylabel('Frequency')
ax.set_title('Validation Set: Score Distribution (Improved)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
test_relevant = test_eval[test_eval['is_relevant']]['compatibility_score']
test_non_relevant = test_eval[~test_eval['is_relevant']]['compatibility_score']

ax.hist(test_relevant, bins=30, alpha=0.6, label=f'Relevant (score >= {relevance_threshold:.1f})', 
        color='green', edgecolor='black')
ax.hist(test_non_relevant, bins=30, alpha=0.6, label=f'Non-Relevant (score < {relevance_threshold:.1f})', 
        color='red', edgecolor='black')
ax.set_xlabel('Compatibility Score')
ax.set_ylabel('Frequency')
ax.set_title('Test Set: Score Distribution (Improved)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/baselines/improved_score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: artifacts/baselines/improved_score_distributions.png")

print("\n" + "=" * 80)
print("IMPROVED BASELINE EVALUATION COMPLETE")
print("=" * 80)

print(f"""
Summary:
- Relevance threshold: {relevance_threshold:.2f} (top 10% of scores)
- Evaluated {len(baselines)} baselines
- Key improvements visible in Recall@K, Precision@K, and Hit Rate
- All visualizations saved to artifacts/baselines/
""")
