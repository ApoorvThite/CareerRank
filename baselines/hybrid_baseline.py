"""
Hybrid Baseline: TF-IDF + Structured Features
CareerRank Project - Day 2 Improvements

Combines text similarity with structured features for better ranking.
Strategy: weighted combination of TF-IDF scores and structured model predictions
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("HYBRID BASELINE (TF-IDF + STRUCTURED)")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Load TF-IDF rankings
print("\nLoading TF-IDF improved rankings...")
with open('artifacts/baselines/tfidf_improved_rankings_val.json', 'r') as f:
    tfidf_val_rankings = json.load(f)

with open('artifacts/baselines/tfidf_improved_rankings_test.json', 'r') as f:
    tfidf_test_rankings = json.load(f)

print(f"TF-IDF val queries: {len(tfidf_val_rankings)}")
print(f"TF-IDF test queries: {len(tfidf_test_rankings)}")

# Load structured rankings
print("\nLoading structured improved rankings...")
with open('artifacts/baselines/structured_improved_rankings_val.json', 'r') as f:
    structured_val_rankings = json.load(f)

with open('artifacts/baselines/structured_improved_rankings_test.json', 'r') as f:
    structured_test_rankings = json.load(f)

print(f"Structured val queries: {len(structured_val_rankings)}")
print(f"Structured test queries: {len(structured_test_rankings)}")


def normalize_scores(scores_dict):
    """Normalize scores to 0-1 range using min-max scaling."""
    if not scores_dict:
        return scores_dict
    
    scores = list(scores_dict.values())
    if len(scores) == 0:
        return scores_dict
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return {k: 0.5 for k in scores_dict.keys()}
    
    normalized = {}
    for k, v in scores_dict.items():
        normalized[k] = (v - min_score) / (max_score - min_score)
    
    return normalized


def create_hybrid_rankings(tfidf_rankings, structured_rankings, 
                          tfidf_weight=0.4, structured_weight=0.6):
    """
    Create hybrid rankings by combining TF-IDF and structured scores.
    
    Args:
        tfidf_rankings: dict of TF-IDF rankings
        structured_rankings: dict of structured rankings
        tfidf_weight: weight for TF-IDF scores (default 0.4)
        structured_weight: weight for structured scores (default 0.6)
    
    Returns:
        dict: hybrid rankings
    """
    print(f"\nCreating hybrid rankings...")
    print(f"  TF-IDF weight: {tfidf_weight}")
    print(f"  Structured weight: {structured_weight}")
    
    hybrid_rankings = {}
    
    for profile_a_id in tqdm(tfidf_rankings.keys(), desc="Combining rankings"):
        if profile_a_id not in structured_rankings:
            continue
        
        # Get top-50 from both models
        tfidf_top50 = tfidf_rankings[profile_a_id]['top_50']
        structured_top50 = structured_rankings[profile_a_id]['top_50']
        
        # Create score dictionaries
        tfidf_scores = {item[0]: item[1] for item in tfidf_top50}
        structured_scores = {item[0]: item[1] for item in structured_top50}
        
        # Get all unique candidates
        all_candidates = set(tfidf_scores.keys()) | set(structured_scores.keys())
        
        # Normalize scores
        tfidf_scores_norm = normalize_scores(tfidf_scores)
        structured_scores_norm = normalize_scores(structured_scores)
        
        # Combine scores
        hybrid_scores = {}
        for candidate in all_candidates:
            tfidf_score = tfidf_scores_norm.get(candidate, 0.0)
            struct_score = structured_scores_norm.get(candidate, 0.0)
            
            # Weighted combination
            hybrid_score = (tfidf_weight * tfidf_score + 
                          structured_weight * struct_score)
            
            hybrid_scores[candidate] = hybrid_score
        
        # Sort by hybrid score
        sorted_candidates = sorted(hybrid_scores.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
        
        # Create rankings for different K values
        hybrid_rankings[profile_a_id] = {
            'top_5': sorted_candidates[:5],
            'top_10': sorted_candidates[:10],
            'top_50': sorted_candidates[:50]
        }
    
    return hybrid_rankings


# Create hybrid rankings for validation set
print("\n" + "=" * 60)
print("VALIDATION SET")
print("=" * 60)

hybrid_val_rankings = create_hybrid_rankings(
    tfidf_val_rankings, 
    structured_val_rankings,
    tfidf_weight=0.4,
    structured_weight=0.6
)

# Save validation rankings
output_path = 'artifacts/baselines/hybrid_rankings_val.json'
with open(output_path, 'w') as f:
    json.dump(hybrid_val_rankings, f, indent=2)

print(f"\nSaved: {output_path}")
print(f"Total queries: {len(hybrid_val_rankings)}")

# Create hybrid rankings for test set
print("\n" + "=" * 60)
print("TEST SET")
print("=" * 60)

hybrid_test_rankings = create_hybrid_rankings(
    tfidf_test_rankings, 
    structured_test_rankings,
    tfidf_weight=0.4,
    structured_weight=0.6
)

# Save test rankings
output_path = 'artifacts/baselines/hybrid_rankings_test.json'
with open(output_path, 'w') as f:
    json.dump(hybrid_test_rankings, f, indent=2)

print(f"\nSaved: {output_path}")
print(f"Total queries: {len(hybrid_test_rankings)}")

# Print example
print("\n" + "=" * 80)
print("EXAMPLE HYBRID RANKING")
print("=" * 80)

example_profile_a = list(hybrid_val_rankings.keys())[0]
example_ranking = hybrid_val_rankings[example_profile_a]

print(f"\nQuery Profile A: {example_profile_a}")
print(f"\nTop 5 Hybrid Results:")
for rank, (profile_b_id, hybrid_score) in enumerate(example_ranking['top_5'], start=1):
    # Get individual scores
    tfidf_items = tfidf_val_rankings[example_profile_a]['top_50']
    struct_items = structured_val_rankings[example_profile_a]['top_50']
    
    tfidf_score = next((s for pid, s in tfidf_items if pid == profile_b_id), 0.0)
    struct_score = next((s for pid, s in struct_items if pid == profile_b_id), 0.0)
    
    print(f"\n  Rank {rank}: {profile_b_id}")
    print(f"    Hybrid Score: {hybrid_score:.4f}")
    print(f"    TF-IDF: {tfidf_score:.4f} | Structured: {struct_score:.2f}")

print("\n" + "=" * 80)
print("HYBRID BASELINE COMPLETE")
print("=" * 80)
