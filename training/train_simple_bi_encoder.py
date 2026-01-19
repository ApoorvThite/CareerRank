"""
Train Bi-Encoder using sentence-transformers library
Simplified approach that avoids transformers import issues
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('.')

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import torch

print("=" * 80)
print("TRAINING BI-ENCODER MODEL (SIMPLIFIED)")
print("=" * 80)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 32,
    'epochs': 3,
    'warmup_steps': 100,
    'evaluation_steps': 500,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Load relevance threshold
print("\nLoading relevance threshold...")
try:
    with open('artifacts/eval_sets/relevance_threshold.json', 'r') as f:
        threshold_info = json.load(f)
    RELEVANCE_THRESHOLD = threshold_info['relevance_threshold']
    print(f"Using relevance threshold: {RELEVANCE_THRESHOLD:.2f}")
except FileNotFoundError:
    RELEVANCE_THRESHOLD = 47.44
    print(f"Using default relevance threshold: {RELEVANCE_THRESHOLD:.2f}")

# Load data
print("\nLoading data...")
pairs_df = pd.read_csv('artifacts/pair_text_dataset.csv')
compat_df = pd.read_csv('compatibility_pairs.csv')

pairs_df['profile_a_id'] = compat_df['profile_a_id']
pairs_df['profile_b_id'] = compat_df['profile_b_id']
pairs_df['compatibility_score'] = compat_df['compatibility_score']

print(f"Loaded {len(pairs_df)} pairs")

# Load splits
print("\nLoading splits...")
with open('artifacts/splits/train_profile_a_ids.txt', 'r') as f:
    train_profile_a_ids = set(f.read().strip().split('\n'))

with open('artifacts/splits/val_profile_a_ids.txt', 'r') as f:
    val_profile_a_ids = set(f.read().strip().split('\n'))

print(f"Train profile_a_ids: {len(train_profile_a_ids)}")
print(f"Val profile_a_ids: {len(val_profile_a_ids)}")

# Split data
train_df = pairs_df[pairs_df['profile_a_id'].isin(train_profile_a_ids)].copy()
val_df = pairs_df[pairs_df['profile_a_id'].isin(val_profile_a_ids)].copy()

print(f"Train pairs: {len(train_df)}")
print(f"Val pairs: {len(val_df)}")

# Filter positive pairs for training
train_positives = train_df[train_df['compatibility_score'] >= RELEVANCE_THRESHOLD].copy()
val_positives = val_df[val_df['compatibility_score'] >= RELEVANCE_THRESHOLD].copy()

print(f"\nPositive pairs (score >= {RELEVANCE_THRESHOLD:.2f}):")
print(f"  Train: {len(train_positives)}")
print(f"  Val: {len(val_positives)}")

# Create training examples
print("\nCreating training examples...")
train_examples = []

for _, row in tqdm(train_positives.iterrows(), total=len(train_positives), desc="Processing train"):
    # Create positive pair with normalized score as label
    # Score range: 0-100, normalize to 0-1
    score_normalized = row['compatibility_score'] / 100.0
    
    example = InputExample(
        texts=[row['profile_a_text'], row['profile_b_text']],
        label=score_normalized
    )
    train_examples.append(example)

print(f"Created {len(train_examples)} training examples")

# Create validation examples
print("\nCreating validation examples...")
val_examples = []

for _, row in tqdm(val_positives.iterrows(), total=len(val_positives), desc="Processing val"):
    score_normalized = row['compatibility_score'] / 100.0
    
    example = InputExample(
        texts=[row['profile_a_text'], row['profile_b_text']],
        label=score_normalized
    )
    val_examples.append(example)

print(f"Created {len(val_examples)} validation examples")

# Initialize model
print("\n" + "=" * 80)
print("INITIALIZING MODEL")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SentenceTransformer(CONFIG['model_name'], device=device)
print(f"Loaded model: {CONFIG['model_name']}")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# Create dataloader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=CONFIG['batch_size']
)

# Define loss function (CosineSimilarityLoss for regression on similarity)
train_loss = losses.CosineSimilarityLoss(model)

# Create evaluator
print("\nCreating evaluator...")
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    val_examples,
    name='val'
)

# Training
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

output_path = 'artifacts/models/bi_encoder'
Path(output_path).mkdir(parents=True, exist_ok=True)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=CONFIG['epochs'],
    evaluation_steps=CONFIG['evaluation_steps'],
    warmup_steps=CONFIG['warmup_steps'],
    output_path=output_path,
    save_best_model=True,
    show_progress_bar=True
)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

# Save training info
training_info = {
    'model_name': CONFIG['model_name'],
    'epochs': CONFIG['epochs'],
    'batch_size': CONFIG['batch_size'],
    'num_train_examples': len(train_examples),
    'num_val_examples': len(val_examples),
    'relevance_threshold': RELEVANCE_THRESHOLD,
    'embedding_dim': model.get_sentence_embedding_dimension()
}

with open(f'{output_path}/training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print(f"\nModel saved to: {output_path}")
print(f"Training info saved to: {output_path}/training_info.json")

# Test the model
print("\n" + "=" * 80)
print("TESTING MODEL")
print("=" * 80)

test_texts = [
    train_positives.iloc[0]['profile_a_text'][:200],
    train_positives.iloc[0]['profile_b_text'][:200],
    train_positives.iloc[1]['profile_a_text'][:200]
]

print("\nTest texts:")
for i, text in enumerate(test_texts):
    print(f"{i+1}. {text}...")

embeddings = model.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True)
print(f"\nEmbeddings shape: {embeddings.shape}")

# Compute similarities
sim_matrix = np.dot(embeddings, embeddings.T)
print(f"\nSimilarity matrix:")
print(sim_matrix)

print("\n✓ Model training and testing complete!")
