"""
Structured Regression Baseline
CareerRank Project - Day 2

Implements a gradient boosting regression model using structured features:
- skill_match_score, skill_complementarity_score
- experience_gap, geographic_score, seniority_match
- industry_match, network_value_a_to_b, network_value_b_to_a
- career_alignment_score
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("STRUCTURED REGRESSION BASELINE")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Check if structured features are available
print("\nChecking for structured features in compatibility_pairs.csv...")
compat_df = pd.read_csv('compatibility_pairs.csv')

required_features = [
    'skill_match_score', 'skill_complementarity_score',
    'experience_gap', 'geographic_score', 'seniority_match',
    'industry_match', 'network_value_a_to_b', 'network_value_b_to_a',
    'career_alignment_score'
]

available_features = [f for f in required_features if f in compat_df.columns]

if len(available_features) == 0:
    print("\n" + "!" * 80)
    print("WARNING: No structured features found in compatibility_pairs.csv")
    print("Skipping structured regression baseline.")
    print("!" * 80)
    exit(0)

print(f"\nAvailable features: {len(available_features)}/{len(required_features)}")
print(f"Features: {available_features}")

# Load splits
print("\nLoading splits...")
with open('artifacts/splits/train_profile_a_ids.txt', 'r') as f:
    train_profile_a_ids = set(f.read().strip().split('\n'))

with open('artifacts/splits/val_profile_a_ids.txt', 'r') as f:
    val_profile_a_ids = set(f.read().strip().split('\n'))

with open('artifacts/splits/test_profile_a_ids.txt', 'r') as f:
    test_profile_a_ids = set(f.read().strip().split('\n'))

print(f"Train: {len(train_profile_a_ids)} profile_a_ids")
print(f"Val: {len(val_profile_a_ids)} profile_a_ids")
print(f"Test: {len(test_profile_a_ids)} profile_a_ids")

# Split data by profile_a_id
print("\nSplitting data...")
train_df = compat_df[compat_df['profile_a_id'].isin(train_profile_a_ids)].copy()
val_df = compat_df[compat_df['profile_a_id'].isin(val_profile_a_ids)].copy()
test_df = compat_df[compat_df['profile_a_id'].isin(test_profile_a_ids)].copy()

print(f"Train pairs: {len(train_df)}")
print(f"Val pairs: {len(val_df)}")
print(f"Test pairs: {len(test_df)}")

# Prepare features and target
X_train = train_df[available_features].fillna(0)
y_train = train_df['compatibility_score']

X_val = val_df[available_features].fillna(0)
y_val = val_df['compatibility_score']

X_test = test_df[available_features].fillna(0)
y_test = test_df['compatibility_score']

# Try to import gradient boosting libraries
model = None
model_name = None

try:
    import lightgbm as lgb
    print("\nUsing LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    model_name = "LightGBM"
except ImportError:
    try:
        import xgboost as xgb
        print("\nUsing XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        model_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print("\nUsing sklearn HistGradientBoostingRegressor...")
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model_name = "HistGradientBoosting"

# Train model
print(f"\nTraining {model_name} model...")
model.fit(X_train, y_train)
print("Training complete!")

# Make predictions
print("\nMaking predictions...")
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Compute RMSE and MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n{'=' * 60}")
print("REGRESSION METRICS")
print(f"{'=' * 60}")
print(f"\nTrain:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE:  {train_mae:.4f}")

print(f"\nValidation:")
print(f"  RMSE: {val_rmse:.4f}")
print(f"  MAE:  {val_mae:.4f}")

print(f"\nTest:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")

# Save RMSE/MAE results
rmse_mae_results = {
    'model': model_name,
    'features': available_features,
    'train': {'rmse': float(train_rmse), 'mae': float(train_mae)},
    'val': {'rmse': float(val_rmse), 'mae': float(val_mae)},
    'test': {'rmse': float(test_rmse), 'mae': float(test_mae)}
}

Path('artifacts/baselines').mkdir(parents=True, exist_ok=True)
with open('artifacts/baselines/structured_rmse_mae.json', 'w') as f:
    json.dump(rmse_mae_results, f, indent=2)

print(f"\nSaved: artifacts/baselines/structured_rmse_mae.json")

# Generate rankings for evaluation sets
print(f"\n{'=' * 60}")
print("GENERATING RANKINGS")
print(f"{'=' * 60}")


def generate_rankings_from_predictions(eval_set_path, split_df, predictions, output_path, set_name):
    """Generate rankings using predicted scores."""
    print(f"\nProcessing {set_name} set...")
    
    # Load evaluation set
    if eval_set_path.endswith('.parquet'):
        try:
            eval_df = pd.read_parquet(eval_set_path)
        except:
            eval_df = pd.read_csv(eval_set_path.replace('.parquet', '.csv'))
    else:
        eval_df = pd.read_csv(eval_set_path)
    
    # Add predictions to split_df
    split_df = split_df.copy()
    split_df['predicted_score'] = predictions
    
    # Create mapping from (profile_a_id, profile_b_id) to predicted score
    score_map = {}
    for _, row in split_df.iterrows():
        key = (row['profile_a_id'], row['profile_b_id'])
        score_map[key] = row['predicted_score']
    
    # Group by profile_a_id
    grouped = eval_df.groupby('profile_a_id')
    
    all_rankings = {}
    k_values = [5, 10, 50]
    
    for profile_a_id, group in tqdm(grouped, desc=f"Ranking {set_name}"):
        # Get predicted scores for candidates
        candidates = []
        for _, row in group.iterrows():
            profile_b_id = row['profile_b_id']
            key = (profile_a_id, profile_b_id)
            
            # Use predicted score if available, otherwise use 0
            pred_score = score_map.get(key, 0.0)
            candidates.append((profile_b_id, float(pred_score)))
        
        # Sort by predicted score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create rankings for different K values
        all_rankings[profile_a_id] = {
            'top_5': candidates[:5],
            'top_10': candidates[:10],
            'top_50': candidates[:50]
        }
    
    # Save rankings
    with open(output_path, 'w') as f:
        json.dump(all_rankings, f, indent=2)
    
    print(f"Saved: {output_path}")
    print(f"Total queries: {len(all_rankings)}")
    
    return all_rankings


# Generate rankings for validation set
val_rankings = generate_rankings_from_predictions(
    'artifacts/eval_sets/val_candidates.parquet',
    val_df,
    y_val_pred,
    'artifacts/baselines/structured_rankings_val.json',
    'Validation'
)

# Generate rankings for test set
test_rankings = generate_rankings_from_predictions(
    'artifacts/eval_sets/test_candidates.parquet',
    test_df,
    y_test_pred,
    'artifacts/baselines/structured_rankings_test.json',
    'Test'
)

# Print example
print("\n" + "=" * 80)
print("EXAMPLE RANKING")
print("=" * 80)

example_profile_a = list(val_rankings.keys())[0]
example_ranking = val_rankings[example_profile_a]

print(f"\nQuery Profile A: {example_profile_a}")
print(f"\nTop 5 Results (by predicted score):")
for rank, (profile_b_id, pred_score) in enumerate(example_ranking['top_5'], start=1):
    print(f"  Rank {rank}: {profile_b_id} - Predicted Score: {pred_score:.2f}")

print("\n" + "=" * 80)
print("STRUCTURED REGRESSION BASELINE COMPLETE")
print("=" * 80)
