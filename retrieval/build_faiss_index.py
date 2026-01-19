"""
Build FAISS Index for Fast Profile Retrieval
CareerRank Project - Day 3

Steps:
1. Load trained bi-encoder model
2. Embed all profiles from serialized_profiles.csv
3. Build FAISS index (Inner Product on normalized vectors)
4. Save embeddings and index
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append('.')
from models.bi_encoder import BiEncoder

print("=" * 80)
print("BUILDING FAISS INDEX")
print("=" * 80)

# Check if FAISS is available
try:
    import faiss
    print("✓ FAISS is available")
except ImportError:
    print("✗ FAISS not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss
    print("✓ FAISS installed successfully")

# Load serialized profiles
print("\nLoading serialized profiles...")
profiles_df = pd.read_csv('artifacts/serialized_profiles.csv')
print(f"Loaded {len(profiles_df)} profiles")

# Extract profile IDs and texts
profile_ids = profiles_df['profile_id'].values
profile_texts = profiles_df['serialized_text'].values

print(f"Profile IDs shape: {profile_ids.shape}")
print(f"Profile texts: {len(profile_texts)} texts")

# Load trained bi-encoder model
print("\n" + "=" * 80)
print("LOADING TRAINED BI-ENCODER MODEL")
print("=" * 80)

model_path = 'artifacts/models/bi_encoder'
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BiEncoder.load_pretrained(model_path, device=device)

# Embed all profiles
print("\n" + "=" * 80)
print("EMBEDDING ALL PROFILES")
print("=" * 80)

print(f"Encoding {len(profile_texts)} profiles...")
print("This may take a few minutes...")

embeddings = model.encode_texts(
    profile_texts.tolist(),
    batch_size=64,
    max_length=256,
    show_progress=True
)

print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Verify embeddings are normalized
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embeddings normalized: {np.allclose(norms, 1.0, atol=1e-5)}")
print(f"Mean norm: {norms.mean():.6f}, Std: {norms.std():.6f}")

# Check for NaN or Inf
if np.any(np.isnan(embeddings)):
    raise ValueError("NaN values found in embeddings!")
if np.any(np.isinf(embeddings)):
    raise ValueError("Inf values found in embeddings!")

print("✓ Embeddings are valid")

# Save embeddings
print("\n" + "=" * 80)
print("SAVING EMBEDDINGS")
print("=" * 80)

embeddings_dir = Path('artifacts/embeddings')
embeddings_dir.mkdir(parents=True, exist_ok=True)

embeddings_path = embeddings_dir / 'profile_embeddings.npy'
profile_ids_path = embeddings_dir / 'profile_id_order.npy'

np.save(embeddings_path, embeddings)
np.save(profile_ids_path, profile_ids)

print(f"Saved embeddings to: {embeddings_path}")
print(f"Saved profile IDs to: {profile_ids_path}")

# Build FAISS index
print("\n" + "=" * 80)
print("BUILDING FAISS INDEX")
print("=" * 80)

embedding_dim = embeddings.shape[1]
num_profiles = embeddings.shape[0]

print(f"Embedding dimension: {embedding_dim}")
print(f"Number of profiles: {num_profiles}")

# Use Inner Product index (since embeddings are normalized, this is equivalent to cosine similarity)
# IndexFlatIP is exact search with inner product
index = faiss.IndexFlatIP(embedding_dim)

print(f"Created FAISS index: {type(index).__name__}")

# Add embeddings to index
print("Adding embeddings to index...")
index.add(embeddings.astype('float32'))

print(f"Index size: {index.ntotal}")

# Verify index
assert index.ntotal == num_profiles, "Index size mismatch!"
print("✓ Index built successfully")

# Test search
print("\n--- TEST SEARCH ---")
test_query_idx = 0
test_query_embedding = embeddings[test_query_idx:test_query_idx+1].astype('float32')
test_profile_id = profile_ids[test_query_idx]

k = 5
distances, indices = index.search(test_query_embedding, k)

print(f"\nTest query: {test_profile_id}")
print(f"Top {k} matches:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"  {i}. Profile: {profile_ids[idx]}, Similarity: {dist:.4f}")

# Save FAISS index
print("\n" + "=" * 80)
print("SAVING FAISS INDEX")
print("=" * 80)

index_dir = Path('artifacts/index')
index_dir.mkdir(parents=True, exist_ok=True)

index_path = index_dir / 'faiss_profiles.index'
faiss.write_index(index, str(index_path))

print(f"Saved FAISS index to: {index_path}")

# Save metadata
import json
metadata = {
    'num_profiles': int(num_profiles),
    'embedding_dim': int(embedding_dim),
    'index_type': 'IndexFlatIP',
    'normalized': True,
    'model_path': model_path
}

metadata_path = index_dir / 'index_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to: {metadata_path}")

print("\n" + "=" * 80)
print("FAISS INDEX BUILD COMPLETE")
print("=" * 80)

print(f"""
Summary:
- Embedded {num_profiles} profiles
- Embedding dimension: {embedding_dim}
- Index type: Inner Product (cosine similarity for normalized vectors)
- Saved to: {index_path}

Files created:
- {embeddings_path}
- {profile_ids_path}
- {index_path}
- {metadata_path}
""")
