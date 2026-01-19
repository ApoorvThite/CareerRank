"""
Retrieve Similar Profiles using Bi-Encoder and FAISS
CareerRank Project - Day 3

CLI tool for retrieving top-N similar profiles for a given query profile.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.append('.')
from models.bi_encoder import BiEncoder

try:
    import faiss
except ImportError:
    print("FAISS not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss


def load_retrieval_system():
    """Load all components needed for retrieval."""
    print("Loading retrieval system...")
    
    # Load FAISS index
    index_path = 'artifacts/index/faiss_profiles.index'
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run build_faiss_index.py first.")
    
    index = faiss.read_index(index_path)
    print(f"✓ Loaded FAISS index: {index.ntotal} profiles")
    
    # Load profile IDs
    profile_ids_path = 'artifacts/embeddings/profile_id_order.npy'
    profile_ids = np.load(profile_ids_path)
    print(f"✓ Loaded profile IDs: {len(profile_ids)} IDs")
    
    # Load serialized profiles
    profiles_df = pd.read_csv('artifacts/serialized_profiles.csv')
    profile_id_to_text = dict(zip(profiles_df['profile_id'], profiles_df['serialized_text']))
    print(f"✓ Loaded profile texts: {len(profile_id_to_text)} profiles")
    
    # Load bi-encoder model
    model_path = 'artifacts/models/bi_encoder'
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model = BiEncoder.load_pretrained(model_path, device=device)
    print(f"✓ Loaded bi-encoder model")
    
    return index, profile_ids, profile_id_to_text, model


def retrieve_similar_profiles(query_profile_id, top_n=20, 
                              index=None, profile_ids=None, 
                              profile_id_to_text=None, model=None):
    """
    Retrieve top-N similar profiles for a query profile.
    
    Args:
        query_profile_id: Profile ID to query
        top_n: Number of results to return
        index: FAISS index
        profile_ids: Array of profile IDs
        profile_id_to_text: Dict mapping profile ID to text
        model: BiEncoder model
    
    Returns:
        DataFrame with matched_profile_id and similarity_score
    """
    # Get query text
    if query_profile_id not in profile_id_to_text:
        raise ValueError(f"Query profile ID {query_profile_id} not found in database")
    
    query_text = profile_id_to_text[query_profile_id]
    
    # Encode query
    query_embedding = model.encode_texts([query_text], batch_size=1, show_progress=False)
    
    # Search FAISS index
    # Request top_n + 1 to account for self-match
    k = min(top_n + 1, index.ntotal)
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Build results
    results = []
    for idx, similarity in zip(indices[0], distances[0]):
        matched_profile_id = profile_ids[idx]
        
        # Skip self-match
        if matched_profile_id == query_profile_id:
            continue
        
        results.append({
            'matched_profile_id': matched_profile_id,
            'similarity_score': float(similarity)
        })
        
        # Stop when we have enough results
        if len(results) >= top_n:
            break
    
    results_df = pd.DataFrame(results)
    
    return results_df


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Retrieve similar profiles using bi-encoder')
    parser.add_argument('--profile_id', type=str, required=True,
                       help='Profile ID to query')
    parser.add_argument('--top_n', type=int, default=20,
                       help='Number of results to return (default: 20)')
    parser.add_argument('--show_text', action='store_true',
                       help='Show profile texts in output')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BI-ENCODER PROFILE RETRIEVAL")
    print("=" * 80)
    
    # Load system
    index, profile_ids, profile_id_to_text, model = load_retrieval_system()
    
    # Retrieve
    print(f"\nQuery Profile ID: {args.profile_id}")
    print(f"Retrieving top {args.top_n} matches...\n")
    
    results_df = retrieve_similar_profiles(
        args.profile_id,
        top_n=args.top_n,
        index=index,
        profile_ids=profile_ids,
        profile_id_to_text=profile_id_to_text,
        model=model
    )
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if args.show_text:
        # Show query text
        print(f"\nQuery Profile ({args.profile_id}):")
        print("-" * 80)
        print(profile_id_to_text[args.profile_id][:500] + "...")
        print()
    
    print(f"\nTop {len(results_df)} matches:\n")
    
    for i, row in results_df.iterrows():
        print(f"{i+1}. Profile: {row['matched_profile_id']}")
        print(f"   Similarity: {row['similarity_score']:.4f}")
        
        if args.show_text:
            text = profile_id_to_text.get(row['matched_profile_id'], 'N/A')
            print(f"   Text: {text[:200]}...")
        
        print()
    
    # Save results
    output_path = f"artifacts/bi_encoder/retrieval_{args.profile_id}.csv"
    Path('artifacts/bi_encoder').mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
