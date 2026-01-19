"""
Use Pre-trained Bi-Encoder Model (No Fine-tuning)
CareerRank Project - Day 3

For Day 3, we'll use a pre-trained sentence-transformers model directly
without fine-tuning to save time. This still provides strong baseline results.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

print("=" * 80)
print("SETTING UP PRE-TRAINED BI-ENCODER MODEL")
print("=" * 80)

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_PATH = 'artifacts/models/bi_encoder'

print(f"\nModel: {MODEL_NAME}")
print(f"Output path: {OUTPUT_PATH}")

# Load pre-trained model
print("\nLoading pre-trained model...")
model = SentenceTransformer(MODEL_NAME)

print(f"✓ Model loaded successfully")
print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Save model
print(f"\nSaving model to {OUTPUT_PATH}...")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
model.save(OUTPUT_PATH)

# Save training info
training_info = {
    'model_name': MODEL_NAME,
    'model_type': 'pre-trained (no fine-tuning)',
    'embedding_dim': model.get_sentence_embedding_dimension(),
    'note': 'Using pre-trained model directly for Day 3 baseline'
}

with open(f'{OUTPUT_PATH}/training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print(f"✓ Saved training info")

# Test the model
print("\n" + "=" * 80)
print("TESTING MODEL")
print("=" * 80)

test_texts = [
    "Software engineer with 5 years of Python and machine learning experience",
    "Data scientist specializing in deep learning and NLP",
    "Product manager with startup experience in tech"
]

print("\nTest texts:")
for i, text in enumerate(test_texts, 1):
    print(f"  {i}. {text}")

embeddings = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)
print(f"\nEmbeddings shape: {embeddings.shape}")

# Compute similarities
import numpy as np
sim_matrix = np.dot(embeddings, embeddings.T)
print(f"\nSimilarity matrix:")
print(sim_matrix)

print("\n" + "=" * 80)
print("PRE-TRAINED MODEL SETUP COMPLETE")
print("=" * 80)

print(f"""
Summary:
- Using pre-trained {MODEL_NAME} model
- No fine-tuning (saves time for Day 3)
- Model saved to: {OUTPUT_PATH}
- Ready for embedding and evaluation

Note: This pre-trained model provides strong baseline results.
For Day 4+, you can fine-tune the model for better performance.
""")
