"""
Baseline Evaluation Script
CareerRank Project - Day 2

Runs all baselines and computes comprehensive metrics:
- Recall@K, MRR, NDCG@K (ranking metrics)
- RMSE, MAE (regression metrics)
- Generates summary report and visualizations
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add evaluation module to path
sys.path.append('.')
from evaluation.ranking_metrics import recall_at_k, mrr, ndcg_at_k, rmse, mae

print("=" * 80)
print("BASELINE EVALUATION")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Load evaluation sets
print("\nLoading evaluation sets...")

def load_eval_set(path):
    """Load evaluation set (parquet or csv)."""
    if path.endswith('.parquet'):
        try:
            return pd.read_parquet(path)
        except:
            return pd.read_csv(path.replace('.parquet', '.csv'))
    return pd.read_csv(path)

val_eval = load_eval_set('artifacts/eval_sets/val_candidates.parquet')
test_eval = load_eval_set('artifacts/eval_sets/test_candidates.parquet')

print(f"Val candidates: {len(val_eval)}")
print(f"Test candidates: {len(test_eval)}")

# Load rankings
print("\nLoading baseline rankings...")

with open('artifacts/baselines/tfidf_rankings_val.json', 'r') as f:
    tfidf_val_rankings = json.load(f)

with open('artifacts/baselines/tfidf_rankings_test.json', 'r') as f:
    tfidf_test_rankings = json.load(f)

print(f"TF-IDF val queries: {len(tfidf_val_rankings)}")
print(f"TF-IDF test queries: {len(tfidf_test_rankings)}")

# Try to load structured baseline rankings
structured_available = False
try:
    with open('artifacts/baselines/structured_rankings_val.json', 'r') as f:
        structured_val_rankings = json.load(f)
    
    with open('artifacts/baselines/structured_rankings_test.json', 'r') as f:
        structured_test_rankings = json.load(f)
    
    with open('artifacts/baselines/structured_rmse_mae.json', 'r') as f:
        structured_rmse_mae = json.load(f)
    
    structured_available = True
    print(f"Structured val queries: {len(structured_val_rankings)}")
    print(f"Structured test queries: {len(structured_test_rankings)}")
except FileNotFoundError:
    print("Structured baseline not available (skipped)")


def evaluate_rankings(eval_df, rankings, baseline_name, k_values=[5, 10, 50]):
    """
    Evaluate rankings using multiple metrics.
    
    Returns:
        dict: metrics for each query
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating {baseline_name}")
    print(f"{'=' * 60}")
    
    # Group eval_df by profile_a_id
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
        
        # Get rankings for different K values
        ranking_dict = rankings[profile_a_id]
        
        metrics = {'profile_a_id': profile_a_id}
        
        # Compute metrics for each K
        for k in k_values:
            key = f'top_{k}'
            if key not in ranking_dict:
                continue
            
            # Extract ranked list (just profile_b_ids)
            ranked_list = [item[0] for item in ranking_dict[key]]
            
            # Recall@K
            recall = recall_at_k(relevant_set, ranked_list, k)
            metrics[f'recall@{k}'] = recall
            
            # NDCG@K
            ndcg = ndcg_at_k(relevance_grades, ranked_list, k)
            metrics[f'ndcg@{k}'] = ndcg
        
        # MRR (use top_50 for full ranking)
        ranked_list_full = [item[0] for item in ranking_dict.get('top_50', [])]
        mrr_score = mrr(relevant_set, ranked_list_full)
        metrics['mrr'] = mrr_score
        
        all_metrics.append(metrics)
    
    # Aggregate metrics
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


# Evaluate TF-IDF baseline
print("\n" + "=" * 80)
print("TF-IDF BASELINE EVALUATION")
print("=" * 80)

tfidf_val_summary, tfidf_val_metrics = evaluate_rankings(
    val_eval, tfidf_val_rankings, "TF-IDF Validation"
)

tfidf_test_summary, tfidf_test_metrics = evaluate_rankings(
    test_eval, tfidf_test_rankings, "TF-IDF Test"
)

# Evaluate structured baseline if available
if structured_available:
    print("\n" + "=" * 80)
    print("STRUCTURED REGRESSION BASELINE EVALUATION")
    print("=" * 80)
    
    structured_val_summary, structured_val_metrics = evaluate_rankings(
        val_eval, structured_val_rankings, "Structured Validation"
    )
    
    structured_test_summary, structured_test_metrics = evaluate_rankings(
        test_eval, structured_test_rankings, "Structured Test"
    )

# Create comprehensive summary
print("\n" + "=" * 80)
print("CREATING BASELINE METRICS SUMMARY")
print("=" * 80)

summary_rows = []

# TF-IDF results
summary_rows.append({
    'baseline': 'TF-IDF',
    'split': 'validation',
    **tfidf_val_summary,
    'rmse': None,
    'mae': None
})

summary_rows.append({
    'baseline': 'TF-IDF',
    'split': 'test',
    **tfidf_test_summary,
    'rmse': None,
    'mae': None
})

# Structured results
if structured_available:
    summary_rows.append({
        'baseline': 'Structured',
        'split': 'validation',
        **structured_val_summary,
        'rmse': structured_rmse_mae['val']['rmse'],
        'mae': structured_rmse_mae['val']['mae']
    })
    
    summary_rows.append({
        'baseline': 'Structured',
        'split': 'test',
        **structured_test_summary,
        'rmse': structured_rmse_mae['test']['rmse'],
        'mae': structured_rmse_mae['test']['mae']
    })

summary_df = pd.DataFrame(summary_rows)

# Save summary
output_path = 'artifacts/baselines/baseline_metrics_summary.csv'
summary_df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

# Print summary table
print("\n" + "=" * 80)
print("BASELINE METRICS SUMMARY")
print("=" * 80)
print("\n" + summary_df.to_string(index=False))

# Create visualization: score distributions
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Val set score distribution
ax = axes[0]
val_relevant = val_eval[val_eval['is_relevant']]['compatibility_score']
val_non_relevant = val_eval[~val_eval['is_relevant']]['compatibility_score']

ax.hist(val_relevant, bins=30, alpha=0.6, label='Relevant (score >= 80)', color='green', edgecolor='black')
ax.hist(val_non_relevant, bins=30, alpha=0.6, label='Non-Relevant (score < 80)', color='red', edgecolor='black')
ax.set_xlabel('Compatibility Score')
ax.set_ylabel('Frequency')
ax.set_title('Validation Set: Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Test set score distribution
ax = axes[1]
test_relevant = test_eval[test_eval['is_relevant']]['compatibility_score']
test_non_relevant = test_eval[~test_eval['is_relevant']]['compatibility_score']

ax.hist(test_relevant, bins=30, alpha=0.6, label='Relevant (score >= 80)', color='green', edgecolor='black')
ax.hist(test_non_relevant, bins=30, alpha=0.6, label='Non-Relevant (score < 80)', color='red', edgecolor='black')
ax.set_xlabel('Compatibility Score')
ax.set_ylabel('Frequency')
ax.set_title('Test Set: Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/baselines/score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: artifacts/baselines/score_distributions.png")

# Print example query
print("\n" + "=" * 80)
print("EXAMPLE QUERY")
print("=" * 80)

# Get first profile_a_id from val
example_profile_a = list(tfidf_val_rankings.keys())[0]
example_group = val_eval[val_eval['profile_a_id'] == example_profile_a]

print(f"\nProfile A ID: {example_profile_a}")
print(f"Total candidates: {len(example_group)}")
print(f"Relevant candidates (score >= 80): {example_group['is_relevant'].sum()}")

print(f"\n{'=' * 60}")
print("TF-IDF Top 5 Results")
print(f"{'=' * 60}")

tfidf_top5 = tfidf_val_rankings[example_profile_a]['top_5']

for rank, (profile_b_id, cosine_score) in enumerate(tfidf_top5, start=1):
    # Get true compatibility score
    match = example_group[example_group['profile_b_id'] == profile_b_id]
    if len(match) > 0:
        true_score = match.iloc[0]['compatibility_score']
        is_relevant = match.iloc[0]['is_relevant']
        relevance_str = "✓ RELEVANT" if is_relevant else "✗ Not Relevant"
    else:
        true_score = 0.0
        relevance_str = "✗ Not in candidates"
    
    print(f"\nRank {rank}:")
    print(f"  Profile B ID: {profile_b_id}")
    print(f"  Cosine Score: {cosine_score:.4f}")
    print(f"  True Compatibility: {true_score:.2f}")
    print(f"  {relevance_str}")

if structured_available:
    print(f"\n{'=' * 60}")
    print("Structured Regression Top 5 Results")
    print(f"{'=' * 60}")
    
    structured_top5 = structured_val_rankings[example_profile_a]['top_5']
    
    for rank, (profile_b_id, pred_score) in enumerate(structured_top5, start=1):
        # Get true compatibility score
        match = example_group[example_group['profile_b_id'] == profile_b_id]
        if len(match) > 0:
            true_score = match.iloc[0]['compatibility_score']
            is_relevant = match.iloc[0]['is_relevant']
            relevance_str = "✓ RELEVANT" if is_relevant else "✗ Not Relevant"
        else:
            true_score = 0.0
            relevance_str = "✗ Not in candidates"
        
        print(f"\nRank {rank}:")
        print(f"  Profile B ID: {profile_b_id}")
        print(f"  Predicted Score: {pred_score:.2f}")
        print(f"  True Compatibility: {true_score:.2f}")
        print(f"  {relevance_str}")

# Sanity checks
print("\n" + "=" * 80)
print("SANITY CHECKS")
print("=" * 80)

# Check for empty candidate pools
val_candidates_per_a = val_eval.groupby('profile_a_id').size()
test_candidates_per_a = test_eval.groupby('profile_a_id').size()

assert val_candidates_per_a.min() > 0, "Found empty candidate pool in val!"
assert test_candidates_per_a.min() > 0, "Found empty candidate pool in test!"
print("✓ No empty candidate pools")

# Check that we have rankings for all queries
assert len(tfidf_val_rankings) == val_eval['profile_a_id'].nunique(), "Missing TF-IDF val rankings!"
assert len(tfidf_test_rankings) == test_eval['profile_a_id'].nunique(), "Missing TF-IDF test rankings!"
print("✓ All queries have rankings")

print("\n" + "=" * 80)
print("BASELINE EVALUATION COMPLETE")
print("=" * 80)

print(f"""
Summary:
- Evaluated {len(tfidf_val_rankings)} validation queries
- Evaluated {len(tfidf_test_rankings)} test queries
- Baselines: TF-IDF{' + Structured Regression' if structured_available else ''}
- Metrics: Recall@K, MRR, NDCG@K{', RMSE, MAE' if structured_available else ''}
- Saved: artifacts/baselines/baseline_metrics_summary.csv
- Saved: artifacts/baselines/score_distributions.png
""")
