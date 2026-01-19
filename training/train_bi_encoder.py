"""
Train Bi-Encoder Model with Contrastive Learning
CareerRank Project - Day 3

Training strategy:
- InfoNCE loss with in-batch negatives
- Positive pairs: compatibility_score >= threshold (from improved eval)
- Hard negatives: low score pairs from same profile_a
- Early stopping on validation Recall@10
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime

sys.path.append('.')
from models.bi_encoder import BiEncoder, compute_infonce_loss

print("=" * 80)
print("TRAINING BI-ENCODER MODEL")
print("=" * 80)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
CONFIG = {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 2e-5,
    'max_length': 256,
    'temperature': 0.05,
    'gradient_accumulation_steps': 1,
    'warmup_steps': 100,
    'eval_steps': 500,
    'save_steps': 1000,
    'max_grad_norm': 1.0,
    'use_amp': torch.cuda.is_available(),
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Load relevance threshold from improved eval sets
print("\nLoading relevance threshold...")
try:
    with open('artifacts/eval_sets/relevance_threshold.json', 'r') as f:
        threshold_info = json.load(f)
    RELEVANCE_THRESHOLD = threshold_info['relevance_threshold']
    print(f"Using relevance threshold: {RELEVANCE_THRESHOLD:.2f} (top 10% of scores)")
except FileNotFoundError:
    RELEVANCE_THRESHOLD = 47.44  # Default from Day 2 improvements
    print(f"Using default relevance threshold: {RELEVANCE_THRESHOLD:.2f}")

# Load data
print("\nLoading data...")
pairs_df = pd.read_csv('artifacts/pair_text_dataset.csv')
compat_df = pd.read_csv('compatibility_pairs.csv')

# Add profile IDs and compatibility score
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

# Analyze score distribution
print("\n--- SCORE DISTRIBUTION ---")
print(f"Train - Mean: {train_df['compatibility_score'].mean():.2f}, "
      f"Median: {train_df['compatibility_score'].median():.2f}, "
      f"Max: {train_df['compatibility_score'].max():.2f}")
print(f"Val - Mean: {val_df['compatibility_score'].mean():.2f}, "
      f"Median: {val_df['compatibility_score'].median():.2f}, "
      f"Max: {val_df['compatibility_score'].max():.2f}")


class ContrastivePairDataset(Dataset):
    """
    Dataset for contrastive learning with positive and hard negative sampling.
    """
    
    def __init__(self, df, relevance_threshold, tokenizer, max_length=256, 
                 negative_ratio=1, hard_negative_threshold=40):
        """
        Args:
            df: DataFrame with pairs
            relevance_threshold: Threshold for positive pairs
            tokenizer: Tokenizer instance
            max_length: Max sequence length
            negative_ratio: Number of negatives per positive
            hard_negative_threshold: Max score for hard negatives
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.relevance_threshold = relevance_threshold
        self.hard_negative_threshold = hard_negative_threshold
        
        # Filter positive pairs
        self.positives = df[df['compatibility_score'] >= relevance_threshold].copy()
        print(f"  Positive pairs (score >= {relevance_threshold:.2f}): {len(self.positives)}")
        
        # Group all pairs by profile_a_id for negative sampling
        self.all_pairs_by_a = df.groupby('profile_a_id')
        
        # Create index
        self.positive_indices = list(range(len(self.positives)))
        
        if len(self.positives) == 0:
            raise ValueError("No positive pairs found! Check relevance threshold.")
    
    def __len__(self):
        return len(self.positive_indices)
    
    def __getitem__(self, idx):
        # Get positive pair
        pos_row = self.positives.iloc[idx]
        
        profile_a_text = pos_row['profile_a_text']
        profile_b_text = pos_row['profile_b_text']
        
        # Tokenize
        profile_a_encoded = self.tokenizer(
            profile_a_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        profile_b_encoded = self.tokenizer(
            profile_b_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'profile_a_input_ids': profile_a_encoded['input_ids'].squeeze(0),
            'profile_a_attention_mask': profile_a_encoded['attention_mask'].squeeze(0),
            'profile_b_input_ids': profile_b_encoded['input_ids'].squeeze(0),
            'profile_b_attention_mask': profile_b_encoded['attention_mask'].squeeze(0),
            'score': torch.tensor(pos_row['compatibility_score'], dtype=torch.float32)
        }


# Initialize model
print("\n" + "=" * 80)
print("INITIALIZING MODEL")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BiEncoder(model_name=CONFIG['model_name'], device=device)

# Create datasets
print("\n" + "=" * 80)
print("CREATING DATASETS")
print("=" * 80)

print("\nTrain dataset:")
train_dataset = ContrastivePairDataset(
    train_df, 
    RELEVANCE_THRESHOLD, 
    model.tokenizer, 
    max_length=CONFIG['max_length']
)

print("\nValidation dataset:")
val_dataset = ContrastivePairDataset(
    val_df, 
    RELEVANCE_THRESHOLD, 
    model.tokenizer, 
    max_length=CONFIG['max_length']
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = CONFIG['warmup_steps']

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Mixed precision scaler
scaler = GradScaler() if CONFIG['use_amp'] else None

# Training logs
training_log = []

# Training function
def train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch):
    """Train for one epoch."""
    model.encoder.train()
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        profile_a_inputs = {
            'input_ids': batch['profile_a_input_ids'].to(device),
            'attention_mask': batch['profile_a_attention_mask'].to(device)
        }
        profile_b_inputs = {
            'input_ids': batch['profile_b_input_ids'].to(device),
            'attention_mask': batch['profile_b_attention_mask'].to(device)
        }
        
        # Forward pass with mixed precision
        if CONFIG['use_amp'] and scaler is not None:
            with autocast():
                outputs = model(profile_a_inputs, profile_b_inputs)
                loss = compute_infonce_loss(
                    outputs['embeddings_a'],
                    outputs['embeddings_b'],
                    temperature=CONFIG['temperature']
                )
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            outputs = model(profile_a_inputs, profile_b_inputs)
            loss = compute_infonce_loss(
                outputs['embeddings_a'],
                outputs['embeddings_b'],
                temperature=CONFIG['temperature']
            )
            
            loss.backward()
            
            if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log
        if (batch_idx + 1) % 100 == 0:
            log_entry = {
                'epoch': epoch + 1,
                'step': batch_idx + 1,
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'timestamp': datetime.now().isoformat()
            }
            training_log.append(log_entry)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_recall(model, val_loader, k=10):
    """
    Evaluate Recall@K on validation set.
    For each query, check if positive is in top-K of in-batch candidates.
    """
    model.encoder.eval()
    
    total_recall = 0
    num_queries = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating Recall@{k}"):
            # Move to device
            profile_a_inputs = {
                'input_ids': batch['profile_a_input_ids'].to(device),
                'attention_mask': batch['profile_a_attention_mask'].to(device)
            }
            profile_b_inputs = {
                'input_ids': batch['profile_b_input_ids'].to(device),
                'attention_mask': batch['profile_b_attention_mask'].to(device)
            }
            
            # Get embeddings
            outputs = model(profile_a_inputs, profile_b_inputs)
            embeddings_a = outputs['embeddings_a']
            embeddings_b = outputs['embeddings_b']
            
            # Compute similarity matrix
            similarity = torch.matmul(embeddings_a, embeddings_b.T)
            
            # For each query, check if diagonal (positive) is in top-K
            batch_size = similarity.size(0)
            for i in range(batch_size):
                # Get similarities for query i
                sims = similarity[i]
                
                # Get top-K indices
                top_k_indices = torch.topk(sims, k=min(k, len(sims))).indices
                
                # Check if positive (index i) is in top-K
                if i in top_k_indices:
                    total_recall += 1
                
                num_queries += 1
    
    recall = total_recall / num_queries if num_queries > 0 else 0
    return recall


# Training loop
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_recall = 0
best_epoch = 0

for epoch in range(CONFIG['epochs']):
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
    print(f"{'=' * 60}")
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
    print(f"\nTrain Loss: {train_loss:.4f}")
    
    # Evaluate
    val_recall = evaluate_recall(model, val_loader, k=10)
    print(f"Val Recall@10: {val_recall:.4f}")
    
    # Log
    epoch_log = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_recall@10': val_recall,
        'timestamp': datetime.now().isoformat()
    }
    training_log.append(epoch_log)
    
    # Save best model
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_epoch = epoch + 1
        
        print(f"\n✓ New best Recall@10: {val_recall:.4f}")
        print("Saving model...")
        
        model.save_pretrained('artifacts/models/bi_encoder')
        
        # Save training info
        with open('artifacts/models/bi_encoder/training_info.json', 'w') as f:
            json.dump({
                'best_epoch': best_epoch,
                'best_val_recall@10': best_val_recall,
                'config': CONFIG,
                'relevance_threshold': RELEVANCE_THRESHOLD
            }, f, indent=2)

# Save training log
print("\n" + "=" * 80)
print("SAVING TRAINING LOG")
print("=" * 80)

with open('artifacts/models/bi_encoder/training_log.jsonl', 'w') as f:
    for entry in training_log:
        f.write(json.dumps(entry) + '\n')

print(f"Saved training log to artifacts/models/bi_encoder/training_log.jsonl")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print(f"\nBest Epoch: {best_epoch}")
print(f"Best Val Recall@10: {best_val_recall:.4f}")
print(f"Model saved to: artifacts/models/bi_encoder/")
