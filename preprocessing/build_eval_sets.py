"""
Build Ranking Evaluation Datasets
CareerRank Project - Day 2

For each profile_a_id in val/test:
- Include all true pairs
- Add 50 random negative candidates
- Define ground-truth relevance using compatibility_score
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("BUILDING RANKING EVALUATION DATASETS")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Load data
print("\nLoading data...")
pairs_df = pd.read_csv('artifacts/pair_text_dataset.csv')
serialized_df = pd.read_csv('artifacts/serialized_profiles.csv')

# Load compatibility_pairs.csv to get profile IDs
compat_df = pd.read_csv('compatibility_pairs.csv')
pairs_df['profile_a_id'] = compat_df['profile_a_id']
pairs_df['profile_b_id'] = compat_df['profile_b_id']

print(f"Loaded {len(pairs_df)} pairs")
print(f"Loaded {len(serialized_df)} profiles")

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


def build_candidate_pool(profile_a_id, pairs_df, all_profile_b_ids, n_negatives=50):
    """
    Build candidate pool for a given profile_a_id.
    
    Returns:
        DataFrame with columns: profile_a_id, profile_b_id, compatibility_score, is_true_pair
    """
    # Get all true pairs for this profile_a
    true_pairs = pairs_df[pairs_df['profile_a_id'] == profile_a_id].copy()
    true_profile_b_ids = set(true_pairs['profile_b_id'].values)
    
    # Sample random negatives (profile_b_ids not in true pairs)
    available_negatives = list(all_profile_b_ids - true_profile_b_ids - {profile_a_id})
    
    if len(available_negatives) < n_negatives:
        n_negatives = len(available_negatives)
    
    negative_profile_b_ids = np.random.choice(available_negatives, size=n_negatives, replace=False)
    
    # Create negative pairs
    negative_pairs = pd.DataFrame({
        'profile_a_id': [profile_a_id] * len(negative_profile_b_ids),
        'profile_b_id': negative_profile_b_ids,
        'compatibility_score': [0.0] * len(negative_profile_b_ids),  # Assume 0 for negatives
        'is_true_pair': [False] * len(negative_profile_b_ids)
    })
    
    # Mark true pairs
    true_pairs['is_true_pair'] = True
    
    # Combine true and negative pairs
    candidate_pool = pd.concat([true_pairs[['profile_a_id', 'profile_b_id', 'compatibility_score', 'is_true_pair']], 
                                negative_pairs], ignore_index=True)
    
    return candidate_pool


def build_eval_set(profile_a_ids, pairs_df, all_profile_b_ids, set_name):
    """Build evaluation set for a list of profile_a_ids."""
    print(f"\nBuilding {set_name} evaluation set...")
    
    all_candidates = []
    
    for profile_a_id in tqdm(list(profile_a_ids), desc=f"Processing {set_name}"):
        candidate_pool = build_candidate_pool(profile_a_id, pairs_df, all_profile_b_ids, n_negatives=50)
        all_candidates.append(candidate_pool)
    
    eval_set = pd.concat(all_candidates, ignore_index=True)
    
    print(f"\n{set_name} evaluation set:")
    print(f"  Total candidates: {len(eval_set)}")
    print(f"  True pairs: {eval_set['is_true_pair'].sum()}")
    print(f"  Negative pairs: {(~eval_set['is_true_pair']).sum()}")
    print(f"  Unique profile_a_ids: {eval_set['profile_a_id'].nunique()}")
    
    # Add relevance labels
    eval_set['is_relevant'] = eval_set['compatibility_score'] >= 80
    
    # Add graded relevance (0-3)
    def compute_graded_relevance(score):
        if score < 30:
            return 0
        elif score < 60:
            return 1
        elif score < 80:
            return 2
        else:
            return 3
    
    eval_set['relevance_grade'] = eval_set['compatibility_score'].apply(compute_graded_relevance)
    
    print(f"  Relevant (score >= 80): {eval_set['is_relevant'].sum()}")
    print(f"  Relevance grade distribution:")
    print(eval_set['relevance_grade'].value_counts().sort_index())
    
    return eval_set


# Build val evaluation set
val_eval_set = build_eval_set(val_profile_a_ids, pairs_df, all_profile_b_ids, "Validation")

# Build test evaluation set
test_eval_set = build_eval_set(test_profile_a_ids, pairs_df, all_profile_b_ids, "Test")

# Create output directory
Path('artifacts/eval_sets').mkdir(parents=True, exist_ok=True)

# Save evaluation sets
val_output = 'artifacts/eval_sets/val_candidates.parquet'
test_output = 'artifacts/eval_sets/test_candidates.parquet'

try:
    val_eval_set.to_parquet(val_output, index=False)
    print(f"\nSaved: {val_output}")
except:
    # Fallback to CSV if parquet not available
    val_output = 'artifacts/eval_sets/val_candidates.csv'
    val_eval_set.to_csv(val_output, index=False)
    print(f"\nSaved (CSV fallback): {val_output}")

try:
    test_eval_set.to_parquet(test_output, index=False)
    print(f"Saved: {test_output}")
except:
    # Fallback to CSV if parquet not available
    test_output = 'artifacts/eval_sets/test_candidates.csv'
    test_eval_set.to_csv(test_output, index=False)
    print(f"Saved (CSV fallback): {test_output}")

# Sanity checks
print("\n--- SANITY CHECKS ---")
assert len(val_eval_set) > 0, "Val eval set is empty!"
assert len(test_eval_set) > 0, "Test eval set is empty!"
print("✓ No empty candidate pools")

# Check that each profile_a has candidates
val_candidates_per_a = val_eval_set.groupby('profile_a_id').size()
test_candidates_per_a = test_eval_set.groupby('profile_a_id').size()

print(f"\nCandidates per profile_a (val):")
print(f"  Min: {val_candidates_per_a.min()}")
print(f"  Max: {val_candidates_per_a.max()}")
print(f"  Mean: {val_candidates_per_a.mean():.1f}")

print(f"\nCandidates per profile_a (test):")
print(f"  Min: {test_candidates_per_a.min()}")
print(f"  Max: {test_candidates_per_a.max()}")
print(f"  Mean: {test_candidates_per_a.mean():.1f}")

print("\n" + "=" * 80)
print("EVALUATION DATASETS BUILT SUCCESSFULLY")
print("=" * 80)
