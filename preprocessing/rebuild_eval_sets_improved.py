"""
Rebuild Evaluation Sets with Improved Negative Sampling
CareerRank Project - Day 2 Improvements

Improvements:
- Better negative sampling (not all zeros)
- Adaptive relevance threshold based on actual score distribution
- More realistic evaluation setup
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("REBUILDING EVALUATION DATASETS (IMPROVED)")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Load data
print("\nLoading data...")
pairs_df = pd.read_csv('artifacts/pair_text_dataset.csv')
serialized_df = pd.read_csv('artifacts/serialized_profiles.csv')
compat_df = pd.read_csv('compatibility_pairs.csv')

# Add profile IDs
pairs_df['profile_a_id'] = compat_df['profile_a_id']
pairs_df['profile_b_id'] = compat_df['profile_b_id']

print(f"Loaded {len(pairs_df)} pairs")
print(f"Loaded {len(serialized_df)} profiles")

# Analyze score distribution to set adaptive threshold
print("\n--- SCORE DISTRIBUTION ANALYSIS ---")
all_scores = compat_df['compatibility_score']
print(f"Mean score: {all_scores.mean():.2f}")
print(f"Median score: {all_scores.median():.2f}")
print(f"90th percentile: {all_scores.quantile(0.9):.2f}")
print(f"95th percentile: {all_scores.quantile(0.95):.2f}")
print(f"Max score: {all_scores.max():.2f}")

# Set adaptive relevance threshold (top 10% of scores)
relevance_threshold = all_scores.quantile(0.90)
print(f"\n✓ Setting relevance threshold to 90th percentile: {relevance_threshold:.2f}")

# Load splits
with open('artifacts/splits/val_profile_a_ids.txt', 'r') as f:
    val_profile_a_ids = set(f.read().strip().split('\n'))

with open('artifacts/splits/test_profile_a_ids.txt', 'r') as f:
    test_profile_a_ids = set(f.read().strip().split('\n'))

print(f"\nVal profile_a_ids: {len(val_profile_a_ids)}")
print(f"Test profile_a_ids: {len(test_profile_a_ids)}")

# Get all available profile_b_ids
all_profile_b_ids = set(serialized_df['profile_id'].values)
print(f"Total available profile_b_ids: {len(all_profile_b_ids)}")

# Create a mapping of profile pairs to scores for better negative sampling
print("\nCreating profile pair score mapping...")
pair_score_map = {}
for _, row in compat_df.iterrows():
    key = (row['profile_a_id'], row['profile_b_id'])
    pair_score_map[key] = row['compatibility_score']

print(f"Created mapping for {len(pair_score_map)} pairs")


def build_candidate_pool_improved(profile_a_id, pairs_df, all_profile_b_ids, 
                                   pair_score_map, n_negatives=50):
    """
    Build improved candidate pool with better negative sampling.
    
    Strategy:
    - Include all true pairs for this profile_a
    - Sample negatives from other profiles (not just zeros)
    - Try to get actual compatibility scores for negatives if they exist as pairs with other profiles
    """
    # Get all true pairs for this profile_a
    true_pairs = pairs_df[pairs_df['profile_a_id'] == profile_a_id].copy()
    true_profile_b_ids = set(true_pairs['profile_b_id'].values)
    
    # Sample random negatives
    available_negatives = list(all_profile_b_ids - true_profile_b_ids - {profile_a_id})
    
    if len(available_negatives) < n_negatives:
        n_negatives = len(available_negatives)
    
    negative_profile_b_ids = np.random.choice(available_negatives, size=n_negatives, replace=False)
    
    # For negatives, check if they have scores with other profiles (use as proxy)
    negative_scores = []
    for neg_id in negative_profile_b_ids:
        # Check if this negative has a score with the query profile in the full dataset
        key = (profile_a_id, neg_id)
        if key in pair_score_map:
            # Should not happen if our logic is correct, but use it if available
            negative_scores.append(pair_score_map[key])
        else:
            # Assign a low score (realistic for non-matches)
            # Sample from low end of distribution
            negative_scores.append(np.random.uniform(15, 25))
    
    # Create negative pairs
    negative_pairs = pd.DataFrame({
        'profile_a_id': [profile_a_id] * len(negative_profile_b_ids),
        'profile_b_id': negative_profile_b_ids,
        'compatibility_score': negative_scores,
        'is_true_pair': [False] * len(negative_profile_b_ids)
    })
    
    # Mark true pairs
    true_pairs['is_true_pair'] = True
    
    # Combine true and negative pairs
    candidate_pool = pd.concat([
        true_pairs[['profile_a_id', 'profile_b_id', 'compatibility_score', 'is_true_pair']], 
        negative_pairs
    ], ignore_index=True)
    
    return candidate_pool


def build_eval_set_improved(profile_a_ids, pairs_df, all_profile_b_ids, 
                            pair_score_map, relevance_threshold, set_name):
    """Build improved evaluation set."""
    print(f"\nBuilding {set_name} evaluation set (improved)...")
    
    all_candidates = []
    
    for profile_a_id in tqdm(list(profile_a_ids), desc=f"Processing {set_name}"):
        candidate_pool = build_candidate_pool_improved(
            profile_a_id, pairs_df, all_profile_b_ids, pair_score_map, n_negatives=50
        )
        all_candidates.append(candidate_pool)
    
    eval_set = pd.concat(all_candidates, ignore_index=True)
    
    print(f"\n{set_name} evaluation set:")
    print(f"  Total candidates: {len(eval_set)}")
    print(f"  True pairs: {eval_set['is_true_pair'].sum()}")
    print(f"  Negative pairs: {(~eval_set['is_true_pair']).sum()}")
    print(f"  Unique profile_a_ids: {eval_set['profile_a_id'].nunique()}")
    
    # Add relevance labels using adaptive threshold
    eval_set['is_relevant'] = eval_set['compatibility_score'] >= relevance_threshold
    
    # Add graded relevance (0-3) based on quartiles
    def compute_graded_relevance(score, threshold):
        if score < 25:
            return 0  # Very low
        elif score < 35:
            return 1  # Low-medium
        elif score < threshold:
            return 2  # Medium-high
        else:
            return 3  # High (top 10%)
    
    eval_set['relevance_grade'] = eval_set['compatibility_score'].apply(
        lambda x: compute_graded_relevance(x, relevance_threshold)
    )
    
    print(f"  Relevant (score >= {relevance_threshold:.2f}): {eval_set['is_relevant'].sum()}")
    print(f"  Relevance grade distribution:")
    print(eval_set['relevance_grade'].value_counts().sort_index())
    
    # Score statistics
    print(f"\n  Score statistics:")
    print(f"    Mean: {eval_set['compatibility_score'].mean():.2f}")
    print(f"    Median: {eval_set['compatibility_score'].median():.2f}")
    print(f"    Min: {eval_set['compatibility_score'].min():.2f}")
    print(f"    Max: {eval_set['compatibility_score'].max():.2f}")
    
    return eval_set


# Build improved evaluation sets
val_eval_set = build_eval_set_improved(
    val_profile_a_ids, pairs_df, all_profile_b_ids, 
    pair_score_map, relevance_threshold, "Validation"
)

test_eval_set = build_eval_set_improved(
    test_profile_a_ids, pairs_df, all_profile_b_ids, 
    pair_score_map, relevance_threshold, "Test"
)

# Save improved evaluation sets
Path('artifacts/eval_sets').mkdir(parents=True, exist_ok=True)

val_output = 'artifacts/eval_sets/val_candidates_improved.parquet'
test_output = 'artifacts/eval_sets/test_candidates_improved.parquet'

try:
    val_eval_set.to_parquet(val_output, index=False)
    print(f"\nSaved: {val_output}")
except:
    val_output = 'artifacts/eval_sets/val_candidates_improved.csv'
    val_eval_set.to_csv(val_output, index=False)
    print(f"\nSaved (CSV fallback): {val_output}")

try:
    test_eval_set.to_parquet(test_output, index=False)
    print(f"Saved: {test_output}")
except:
    test_output = 'artifacts/eval_sets/test_candidates_improved.csv'
    test_eval_set.to_csv(test_output, index=False)
    print(f"Saved (CSV fallback): {test_output}")

# Save relevance threshold for later use
threshold_info = {
    'relevance_threshold': float(relevance_threshold),
    'percentile': 90,
    'description': 'Top 10% of compatibility scores'
}

import json
with open('artifacts/eval_sets/relevance_threshold.json', 'w') as f:
    json.dump(threshold_info, f, indent=2)

print(f"Saved: artifacts/eval_sets/relevance_threshold.json")

# Sanity checks
print("\n--- SANITY CHECKS ---")
assert len(val_eval_set) > 0, "Val eval set is empty!"
assert len(test_eval_set) > 0, "Test eval set is empty!"
assert val_eval_set['is_relevant'].sum() > 0, "No relevant items in val set!"
assert test_eval_set['is_relevant'].sum() > 0, "No relevant items in test set!"
print("✓ All sanity checks passed")

print("\n" + "=" * 80)
print("IMPROVED EVALUATION DATASETS BUILT SUCCESSFULLY")
print("=" * 80)
