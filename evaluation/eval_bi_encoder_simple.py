"""
Evaluate Bi-Encoder Retrieval (Simplified)
CareerRank Project - Day 3
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')
from sentence_transformers import SentenceTransformer
from evaluation.ranking_metrics import recall_at_k, mrr, ndcg_at_k

print("=" * 80)
print("BI-ENCODER RETRIEVAL EVALUATION")
print("=" * 80)

# Load model
print("\nLoading bi-encoder model...")
model_path = 'artifacts/models/bi_encoder'
model = SentenceTransformer(model_path)
print(f"✓ Model loaded")

# Load evaluation sets (use improved versions)
print("\nLoading evaluation sets...")

def load_eval_set(path):
    if path.endswith('.parquet'):
        try:
            return pd.read_parquet(path)
        except:
            return pd.read_csv(path.replace('.parquet', '.csv'))
    return pd.read_csv(path)

try:
    val_eval = load_eval_set('artifacts/eval_sets/val_candidates_improved.parquet')
    test_eval = load_eval_set('artifacts/eval_sets/test_candidates_improved.parquet')
    print("✓ Using improved evaluation sets")
except:
    val_eval = load_eval_set('artifacts/eval_sets/val_candidates.parquet')
    test_eval = load_eval_set('artifacts/eval_sets/test_candidates.parquet')
    print("✓ Using original evaluation sets")

print(f"Val candidates: {len(val_eval)}")
print(f"Test candidates: {len(test_eval)}")

# Load profile texts
print("\nLoading profile texts...")
profiles_df = pd.read_csv('artifacts/serialized_profiles.csv')
profile_to_text = dict(zip(profiles_df['profile_id'], profiles_df['serialized_text']))
print(f"Loaded {len(profile_to_text)} profile texts")


def rank_candidates_bi_encoder(profile_a_id, candidates_df, profile_to_text, model, k_values=[5, 10, 50]):
    """Rank candidates using bi-encoder."""
    if profile_a_id not in profile_to_text:
        return {k: [] for k in k_values}
    
    query_text = profile_to_text[profile_a_id]
    
    # Get candidate texts
    candidate_profile_b_ids = candidates_df['profile_b_id'].values
    candidate_texts = []
    valid_candidate_ids = []
    
    for cand_id in candidate_profile_b_ids:
        if cand_id in profile_to_text:
            candidate_texts.append(profile_to_text[cand_id])
            valid_candidate_ids.append(cand_id)
    
    if len(candidate_texts) == 0:
        return {k: [] for k in k_values}
    
    # Encode
    query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
    candidate_embeddings = model.encode(candidate_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    
    # Compute similarities
    similarities = np.dot(query_embedding, candidate_embeddings.T).flatten()
    
    # Sort by similarity
    sorted_indices = np.argsort(-similarities)
    
    # Create rankings
    rankings = {}
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        top_k_results = [(valid_candidate_ids[idx], float(similarities[idx])) 
                         for idx in top_k_indices]
        rankings[k] = top_k_results
    
    return rankings


def evaluate_retrieval(eval_df, set_name, model, profile_to_text):
    """Evaluate bi-encoder retrieval."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {set_name}")
    print(f"{'=' * 60}")
    
    grouped = eval_df.groupby('profile_a_id')
    
    all_rankings = {}
    all_metrics = []
    
    k_values = [5, 10, 50]
    
    for profile_a_id, group in tqdm(grouped, desc=f"Ranking {set_name}"):
        rankings = rank_candidates_bi_encoder(
            profile_a_id, group, profile_to_text, model, k_values
        )
        
        all_rankings[profile_a_id] = {
            'top_5': rankings[5],
            'top_10': rankings[10],
            'top_50': rankings[50]
        }
        
        # Compute metrics
        relevant_set = set(group[group['is_relevant']]['profile_b_id'].values)
        relevance_grades = {}
        for _, row in group.iterrows():
            relevance_grades[row['profile_b_id']] = row['relevance_grade']
        
        metrics = {'profile_a_id': profile_a_id}
        
        for k in k_values:
            ranked_list = [item[0] for item in rankings[k]]
            
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
            
            # Hit@K
            hit = 1 if len(set(ranked_list[:k]) & relevant_set) > 0 else 0
            metrics[f'hit@{k}'] = hit
        
        # MRR
        ranked_list_full = [item[0] for item in rankings[50]]
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
    
    return summary, all_rankings


# Evaluate
val_summary, val_rankings = evaluate_retrieval(val_eval, "Validation", model, profile_to_text)
test_summary, test_rankings = evaluate_retrieval(test_eval, "Test", model, profile_to_text)

# Save metrics
print("\n" + "=" * 80)
print("SAVING METRICS")
print("=" * 80)

Path('artifacts/bi_encoder').mkdir(parents=True, exist_ok=True)

with open('artifacts/bi_encoder/bi_encoder_metrics_val.json', 'w') as f:
    json.dump(val_summary, f, indent=2)
print(f"✓ Saved: artifacts/bi_encoder/bi_encoder_metrics_val.json")

with open('artifacts/bi_encoder/bi_encoder_metrics_test.json', 'w') as f:
    json.dump(test_summary, f, indent=2)
print(f"✓ Saved: artifacts/bi_encoder/bi_encoder_metrics_test.json")

with open('artifacts/bi_encoder/bi_encoder_rankings_val.json', 'w') as f:
    json.dump(val_rankings, f, indent=2)
print(f"✓ Saved: artifacts/bi_encoder/bi_encoder_rankings_val.json")

with open('artifacts/bi_encoder/bi_encoder_rankings_test.json', 'w') as f:
    json.dump(test_rankings, f, indent=2)
print(f"✓ Saved: artifacts/bi_encoder/bi_encoder_rankings_test.json")

# Create comparison
print("\n" + "=" * 80)
print("COMPARISON AGAINST BASELINES")
print("=" * 80)

try:
    baseline_df = pd.read_csv('artifacts/baselines/improved_metrics_summary.csv')
    print("✓ Using improved baseline metrics")
except:
    baseline_df = pd.read_csv('artifacts/baselines/baseline_metrics_summary.csv')
    print("✓ Using original baseline metrics")

comparison_data = []

# Get baselines
for _, row in baseline_df.iterrows():
    if row['split'] in ['validation', 'test']:
        comparison_data.append({
            'model': row['baseline'],
            'split': row['split'],
            'recall@5': row.get('recall@5', 0),
            'recall@10': row.get('recall@10', 0),
            'recall@50': row.get('recall@50', 0),
            'ndcg@10': row.get('ndcg@10', 0),
            'mrr': row.get('mrr', 0),
            'precision@5': row.get('precision@5', 0)
        })

# Add bi-encoder
comparison_data.append({
    'model': 'Bi-Encoder',
    'split': 'validation',
    'recall@5': val_summary.get('recall@5', 0),
    'recall@10': val_summary.get('recall@10', 0),
    'recall@50': val_summary.get('recall@50', 0),
    'ndcg@10': val_summary.get('ndcg@10', 0),
    'mrr': val_summary.get('mrr', 0),
    'precision@5': val_summary.get('precision@5', 0)
})

comparison_data.append({
    'model': 'Bi-Encoder',
    'split': 'test',
    'recall@5': test_summary.get('recall@5', 0),
    'recall@10': test_summary.get('recall@10', 0),
    'recall@50': test_summary.get('recall@50', 0),
    'ndcg@10': test_summary.get('ndcg@10', 0),
    'mrr': test_summary.get('mrr', 0),
    'precision@5': test_summary.get('precision@5', 0)
})

comparison_df = pd.DataFrame(comparison_data)

comparison_path = 'artifacts/bi_encoder/bi_encoder_vs_baselines.csv'
comparison_df.to_csv(comparison_path, index=False)
print(f"\n✓ Saved comparison to: {comparison_path}")

print("\n" + "=" * 80)
print("BI-ENCODER VS BASELINES COMPARISON")
print("=" * 80)

print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS (Test Set)")
print("=" * 80)

test_comparison = comparison_df[comparison_df['split'] == 'test']
bi_encoder_test = test_comparison[test_comparison['model'] == 'Bi-Encoder'].iloc[0]

for _, baseline_row in test_comparison.iterrows():
    if baseline_row['model'] == 'Bi-Encoder':
        continue
    
    print(f"\nBi-Encoder vs {baseline_row['model']}:")
    
    for metric in ['recall@10', 'ndcg@10', 'mrr', 'precision@5']:
        baseline_val = baseline_row[metric]
        bi_encoder_val = bi_encoder_test[metric]
        
        if baseline_val > 0:
            improvement = ((bi_encoder_val - baseline_val) / baseline_val) * 100
            symbol = "↑" if improvement > 0 else "↓"
            print(f"  {metric}: {baseline_val:.4f} → {bi_encoder_val:.4f} ({improvement:+.1f}% {symbol})")
        else:
            print(f"  {metric}: {baseline_val:.4f} → {bi_encoder_val:.4f}")

# Print example
print("\n" + "=" * 80)
print("EXAMPLE QUERY")
print("=" * 80)

example_profile_a = list(val_rankings.keys())[0]
example_ranking = val_rankings[example_profile_a]

print(f"\nQuery Profile A: {example_profile_a}")

example_group = val_eval[val_eval['profile_a_id'] == example_profile_a]
print(f"Total candidates: {len(example_group)}")
print(f"Relevant candidates: {example_group['is_relevant'].sum()}")

print(f"\nTop 5 Bi-Encoder Results:")
for rank, (profile_b_id, similarity) in enumerate(example_ranking['top_5'], start=1):
    match = example_group[example_group['profile_b_id'] == profile_b_id]
    if len(match) > 0:
        true_score = match.iloc[0]['compatibility_score']
        is_relevant = match.iloc[0]['is_relevant']
        relevance_str = "✓ Relevant" if is_relevant else "✗ Not Relevant"
    else:
        true_score = 0.0
        relevance_str = "✗ Not in candidates"
    
    print(f"\n  Rank {rank}:")
    print(f"    Profile B ID: {profile_b_id}")
    print(f"    Similarity: {similarity:.4f}")
    print(f"    True Compatibility: {true_score:.2f}")
    print(f"    {relevance_str}")

print("\n" + "=" * 80)
print("BI-ENCODER EVALUATION COMPLETE")
print("=" * 80)

print(f"""
Summary:
- Evaluated on validation and test sets
- Metrics: Recall@K, Precision@K, NDCG@K, MRR, Hit@K
- Comparison against baselines saved
- All results saved to artifacts/bi_encoder/
""")
