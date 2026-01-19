"""
Enhanced TF-IDF Retrieval Baseline
CareerRank Project - Day 2 Improvements

Improvements:
- Increased max_features (5000 → 10000)
- Added character n-grams for better skill matching
- BM25-like weighting
- Better text preprocessing
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

print("=" * 80)
print("ENHANCED TF-IDF RETRIEVAL BASELINE")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Load serialized profiles
print("\nLoading serialized profiles...")
serialized_df = pd.read_csv('artifacts/serialized_profiles.csv')
print(f"Loaded {len(serialized_df)} profiles")

# Create profile_id to text mapping
profile_to_text = dict(zip(serialized_df['profile_id'], serialized_df['serialized_text']))

# Build enhanced TF-IDF vectorizer with better parameters
print("\nBuilding enhanced TF-IDF vectors...")
print("Improvements:")
print("  - max_features: 5000 → 10000")
print("  - Added character n-grams (3-5) for skill matching")
print("  - Lower min_df for rare skills")
print("  - Sublinear TF scaling (BM25-like)")

vectorizer = TfidfVectorizer(
    max_features=10000,  # Increased from 5000
    ngram_range=(1, 2),  # Word unigrams and bigrams
    analyzer='word',
    min_df=1,  # Lower threshold to capture rare skills
    max_df=0.7,  # Slightly lower to filter common words
    stop_words='english',
    sublinear_tf=True,  # Use log(tf) instead of tf (BM25-like)
    norm='l2'
)

# Also create character-level vectorizer for skills
char_vectorizer = TfidfVectorizer(
    max_features=5000,
    analyzer='char',
    ngram_range=(3, 5),  # Character trigrams to 5-grams
    min_df=2,
    sublinear_tf=True
)

# Fit on all profiles
all_texts = serialized_df['serialized_text'].values

print("\nFitting word-level TF-IDF...")
word_tfidf_matrix = vectorizer.fit_transform(all_texts)
print(f"Word TF-IDF matrix shape: {word_tfidf_matrix.shape}")

print("Fitting character-level TF-IDF...")
char_tfidf_matrix = char_vectorizer.fit_transform(all_texts)
print(f"Character TF-IDF matrix shape: {char_tfidf_matrix.shape}")

# Combine both representations (weighted combination)
print("\nCombining word and character representations...")
from scipy.sparse import hstack

# Weight: 70% word-level, 30% character-level
word_weight = 0.7
char_weight = 0.3

# Normalize and combine
combined_matrix = hstack([
    word_tfidf_matrix * word_weight,
    char_tfidf_matrix * char_weight
])

print(f"Combined matrix shape: {combined_matrix.shape}")

# Create profile_id to index mapping
profile_to_idx = {pid: idx for idx, pid in enumerate(serialized_df['profile_id'])}


def rank_candidates_tfidf_enhanced(profile_a_id, candidate_profile_b_ids, 
                                   profile_to_idx, combined_matrix, k_values=[5, 10, 50]):
    """
    Rank candidates using enhanced TF-IDF cosine similarity.
    """
    if profile_a_id not in profile_to_idx:
        return {k: [] for k in k_values}
    
    query_idx = profile_to_idx[profile_a_id]
    query_vector = combined_matrix[query_idx]
    
    # Get candidate vectors
    candidate_indices = []
    valid_candidate_ids = []
    
    for cand_id in candidate_profile_b_ids:
        if cand_id in profile_to_idx:
            candidate_indices.append(profile_to_idx[cand_id])
            valid_candidate_ids.append(cand_id)
    
    if len(candidate_indices) == 0:
        return {k: [] for k in k_values}
    
    candidate_vectors = combined_matrix[candidate_indices]
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_vector, candidate_vectors).flatten()
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(-similarities)
    
    # Create rankings for different K values
    rankings = {}
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        top_k_results = [(valid_candidate_ids[idx], float(similarities[idx])) 
                         for idx in top_k_indices]
        rankings[k] = top_k_results
    
    return rankings


def process_eval_set(eval_set_path, output_path, set_name):
    """Process evaluation set and generate rankings."""
    print(f"\n{'=' * 60}")
    print(f"Processing {set_name} set")
    print(f"{'=' * 60}")
    
    # Load evaluation set
    if eval_set_path.endswith('.parquet'):
        try:
            eval_df = pd.read_parquet(eval_set_path)
        except:
            eval_df = pd.read_csv(eval_set_path.replace('.parquet', '.csv'))
    else:
        eval_df = pd.read_csv(eval_set_path)
    
    print(f"Loaded {len(eval_df)} candidates")
    
    # Group by profile_a_id
    grouped = eval_df.groupby('profile_a_id')
    
    all_rankings = {}
    k_values = [5, 10, 50]
    
    for profile_a_id, group in tqdm(grouped, desc=f"Ranking {set_name}"):
        candidate_profile_b_ids = group['profile_b_id'].values
        
        # Get rankings
        rankings = rank_candidates_tfidf_enhanced(
            profile_a_id, candidate_profile_b_ids, 
            profile_to_idx, combined_matrix, k_values
        )
        
        # Store rankings
        all_rankings[profile_a_id] = {
            'top_5': rankings[5],
            'top_10': rankings[10],
            'top_50': rankings[50]
        }
    
    # Save rankings
    Path('artifacts/baselines').mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_rankings, f, indent=2)
    
    print(f"\nSaved: {output_path}")
    print(f"Total queries: {len(all_rankings)}")
    
    return all_rankings


# Process improved evaluation sets
val_rankings = process_eval_set(
    'artifacts/eval_sets/val_candidates_improved.parquet',
    'artifacts/baselines/tfidf_improved_rankings_val.json',
    'Validation'
)

test_rankings = process_eval_set(
    'artifacts/eval_sets/test_candidates_improved.parquet',
    'artifacts/baselines/tfidf_improved_rankings_test.json',
    'Test'
)

# Print example
print("\n" + "=" * 80)
print("EXAMPLE RANKING")
print("=" * 80)

example_profile_a = list(val_rankings.keys())[0]
example_ranking = val_rankings[example_profile_a]

print(f"\nQuery Profile A: {example_profile_a}")
print(f"\nQuery Text:")
print(profile_to_text.get(example_profile_a, 'N/A')[:300] + "...")

print(f"\nTop 5 Results:")
for rank, (profile_b_id, cosine_score) in enumerate(example_ranking['top_5'], start=1):
    print(f"\n  Rank {rank}: {profile_b_id}")
    print(f"  Cosine Score: {cosine_score:.4f}")
    print(f"  Text: {profile_to_text.get(profile_b_id, 'N/A')[:150]}...")

print("\n" + "=" * 80)
print("ENHANCED TF-IDF BASELINE COMPLETE")
print("=" * 80)
