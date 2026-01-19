"""
Create Train/Val/Test Splits
CareerRank Project - Day 2

Implements a split strategy that avoids leakage:
- Split by profile_a_id (80% train, 10% val, 10% test)
- Fixed random seed: 42
- Save split IDs to disk
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Load pair dataset
print("\nLoading pair_text_dataset.csv...")
pairs_df = pd.read_csv('artifacts/pair_text_dataset.csv')
print(f"Loaded {len(pairs_df)} pairs")

# Check if profile_a_id and profile_b_id columns exist
# If not, we need to load from compatibility_pairs.csv
if 'profile_a_id' not in pairs_df.columns:
    print("\nNote: profile_a_id not in pair_text_dataset.csv")
    print("Loading compatibility_pairs.csv to get profile IDs...")
    compat_df = pd.read_csv('compatibility_pairs.csv')
    
    # Add profile IDs to pairs_df
    pairs_df['profile_a_id'] = compat_df['profile_a_id']
    pairs_df['profile_b_id'] = compat_df['profile_b_id']
    print(f"Added profile_a_id and profile_b_id columns")

# Get unique profile_a_ids
unique_profile_a = pairs_df['profile_a_id'].unique()
print(f"\nUnique profile_a_ids: {len(unique_profile_a)}")

# Shuffle profile_a_ids
np.random.shuffle(unique_profile_a)

# Split: 80% train, 10% val, 10% test
n_profiles = len(unique_profile_a)
n_train = int(0.8 * n_profiles)
n_val = int(0.1 * n_profiles)

train_profile_a_ids = unique_profile_a[:n_train]
val_profile_a_ids = unique_profile_a[n_train:n_train + n_val]
test_profile_a_ids = unique_profile_a[n_train + n_val:]

print(f"\nSplit sizes:")
print(f"  Train: {len(train_profile_a_ids)} profile_a_ids ({len(train_profile_a_ids)/n_profiles*100:.1f}%)")
print(f"  Val:   {len(val_profile_a_ids)} profile_a_ids ({len(val_profile_a_ids)/n_profiles*100:.1f}%)")
print(f"  Test:  {len(test_profile_a_ids)} profile_a_ids ({len(test_profile_a_ids)/n_profiles*100:.1f}%)")

# Verify no overlap
assert len(set(train_profile_a_ids) & set(val_profile_a_ids)) == 0, "Train/Val overlap!"
assert len(set(train_profile_a_ids) & set(test_profile_a_ids)) == 0, "Train/Test overlap!"
assert len(set(val_profile_a_ids) & set(test_profile_a_ids)) == 0, "Val/Test overlap!"
print("\n✓ No overlap between splits")

# Create output directory
Path('artifacts/splits').mkdir(parents=True, exist_ok=True)

# Save split IDs to disk
train_path = 'artifacts/splits/train_profile_a_ids.txt'
val_path = 'artifacts/splits/val_profile_a_ids.txt'
test_path = 'artifacts/splits/test_profile_a_ids.txt'

with open(train_path, 'w') as f:
    f.write('\n'.join(train_profile_a_ids))
print(f"\nSaved: {train_path}")

with open(val_path, 'w') as f:
    f.write('\n'.join(val_profile_a_ids))
print(f"Saved: {val_path}")

with open(test_path, 'w') as f:
    f.write('\n'.join(test_profile_a_ids))
print(f"Saved: {test_path}")

# Count pairs in each split
train_pairs = pairs_df[pairs_df['profile_a_id'].isin(train_profile_a_ids)]
val_pairs = pairs_df[pairs_df['profile_a_id'].isin(val_profile_a_ids)]
test_pairs = pairs_df[pairs_df['profile_a_id'].isin(test_profile_a_ids)]

print(f"\nPair counts:")
print(f"  Train: {len(train_pairs)} pairs")
print(f"  Val:   {len(val_pairs)} pairs")
print(f"  Test:  {len(test_pairs)} pairs")
print(f"  Total: {len(train_pairs) + len(val_pairs) + len(test_pairs)} pairs")

print("\n" + "=" * 80)
print("SPLITS CREATED SUCCESSFULLY")
print("=" * 80)
