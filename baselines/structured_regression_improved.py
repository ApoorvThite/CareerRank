"""
Enhanced Structured Regression Baseline
CareerRank Project - Day 2 Improvements

Improvements:
- Feature engineering (interactions, ratios, polynomials)
- Better hyperparameters
- More sophisticated model
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("ENHANCED STRUCTURED REGRESSION BASELINE")
print("=" * 80)

# Set random seed
np.random.seed(42)

# Load compatibility data
print("\nLoading compatibility data...")
compat_df = pd.read_csv('compatibility_pairs.csv')

base_features = [
    'skill_match_score', 'skill_complementarity_score',
    'experience_gap', 'geographic_score', 'seniority_match',
    'industry_match', 'network_value_a_to_b', 'network_value_b_to_a',
    'career_alignment_score'
]

print(f"Base features: {len(base_features)}")

# Feature Engineering
print("\n--- FEATURE ENGINEERING ---")

# 1. Interaction features
print("Creating interaction features...")
compat_df['skill_match_x_complementarity'] = (
    compat_df['skill_match_score'] * compat_df['skill_complementarity_score']
)
compat_df['skill_total'] = (
    compat_df['skill_match_score'] + compat_df['skill_complementarity_score']
)
compat_df['network_total'] = (
    compat_df['network_value_a_to_b'] + compat_df['network_value_b_to_a']
)
compat_df['network_asymmetry'] = abs(
    compat_df['network_value_a_to_b'] - compat_df['network_value_b_to_a']
)

# 2. Ratio features
print("Creating ratio features...")
compat_df['network_ratio'] = np.where(
    compat_df['network_value_b_to_a'] > 0,
    compat_df['network_value_a_to_b'] / (compat_df['network_value_b_to_a'] + 1e-6),
    0
)

# 3. Polynomial features for key variables
print("Creating polynomial features...")
compat_df['experience_gap_squared'] = compat_df['experience_gap'] ** 2
compat_df['career_alignment_squared'] = compat_df['career_alignment_score'] ** 2

# 4. Boolean interactions
print("Creating boolean interaction features...")
compat_df['same_industry_high_career'] = (
    (compat_df['industry_match'] == 100) & 
    (compat_df['career_alignment_score'] >= 80)
).astype(int)

compat_df['same_seniority_high_skills'] = (
    (compat_df['seniority_match'] >= 85) & 
    (compat_df['skill_match_score'] >= 10)
).astype(int)

# 5. Experience gap categories
print("Creating experience gap categories...")
compat_df['exp_gap_small'] = (compat_df['experience_gap'] <= 3).astype(int)
compat_df['exp_gap_medium'] = ((compat_df['experience_gap'] > 3) & 
                                (compat_df['experience_gap'] <= 10)).astype(int)
compat_df['exp_gap_large'] = (compat_df['experience_gap'] > 10).astype(int)

# All features
engineered_features = [
    'skill_match_x_complementarity', 'skill_total', 'network_total', 
    'network_asymmetry', 'network_ratio', 'experience_gap_squared',
    'career_alignment_squared', 'same_industry_high_career',
    'same_seniority_high_skills', 'exp_gap_small', 'exp_gap_medium', 'exp_gap_large'
]

all_features = base_features + engineered_features
print(f"\nTotal features: {len(all_features)} (base: {len(base_features)}, engineered: {len(engineered_features)})")

# Load splits
print("\nLoading splits...")
with open('artifacts/splits/train_profile_a_ids.txt', 'r') as f:
    train_profile_a_ids = set(f.read().strip().split('\n'))

with open('artifacts/splits/val_profile_a_ids.txt', 'r') as f:
    val_profile_a_ids = set(f.read().strip().split('\n'))

with open('artifacts/splits/test_profile_a_ids.txt', 'r') as f:
    test_profile_a_ids = set(f.read().strip().split('\n'))

# Split data
print("\nSplitting data...")
train_df = compat_df[compat_df['profile_a_id'].isin(train_profile_a_ids)].copy()
val_df = compat_df[compat_df['profile_a_id'].isin(val_profile_a_ids)].copy()
test_df = compat_df[compat_df['profile_a_id'].isin(test_profile_a_ids)].copy()

print(f"Train pairs: {len(train_df)}")
print(f"Val pairs: {len(val_df)}")
print(f"Test pairs: {len(test_df)}")

# Prepare features and target
X_train = train_df[all_features].fillna(0)
y_train = train_df['compatibility_score']

X_val = val_df[all_features].fillna(0)
y_val = val_df['compatibility_score']

X_test = test_df[all_features].fillna(0)
y_test = test_df['compatibility_score']

# Try to import gradient boosting libraries
model = None
model_name = None

try:
    import lightgbm as lgb
    print("\nUsing LightGBM with improved hyperparameters...")
    model = lgb.LGBMRegressor(
        n_estimators=200,  # Increased from 100
        learning_rate=0.05,  # Reduced for better generalization
        max_depth=8,  # Increased from 6
        num_leaves=64,  # More leaves for complex patterns
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=0.1,  # L2 regularization
        random_state=42,
        verbose=-1
    )
    model_name = "LightGBM_Enhanced"
except ImportError:
    try:
        import xgboost as xgb
        print("\nUsing XGBoost with improved hyperparameters...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=0
        )
        model_name = "XGBoost_Enhanced"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print("\nUsing sklearn HistGradientBoostingRegressor with improved hyperparameters...")
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
        model_name = "HistGradientBoosting_Enhanced"

# Train model
print(f"\nTraining {model_name} model...")
model.fit(X_train, y_train)
print("Training complete!")

# Feature importance
print("\n--- TOP 10 FEATURE IMPORTANCES ---")
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))

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

# Calculate improvement
print(f"\n--- IMPROVEMENT OVER BASELINE ---")
baseline_val_rmse = 0.2918
baseline_test_rmse = 0.2901
val_improvement = ((baseline_val_rmse - val_rmse) / baseline_val_rmse) * 100
test_improvement = ((baseline_test_rmse - test_rmse) / baseline_test_rmse) * 100
print(f"Val RMSE improvement: {val_improvement:.2f}%")
print(f"Test RMSE improvement: {test_improvement:.2f}%")

# Save RMSE/MAE results
rmse_mae_results = {
    'model': model_name,
    'num_features': len(all_features),
    'base_features': base_features,
    'engineered_features': engineered_features,
    'train': {'rmse': float(train_rmse), 'mae': float(train_mae)},
    'val': {'rmse': float(val_rmse), 'mae': float(val_mae)},
    'test': {'rmse': float(test_rmse), 'mae': float(test_mae)},
    'improvement_vs_baseline': {
        'val_rmse_pct': float(val_improvement),
        'test_rmse_pct': float(test_improvement)
    }
}

Path('artifacts/baselines').mkdir(parents=True, exist_ok=True)
with open('artifacts/baselines/structured_improved_rmse_mae.json', 'w') as f:
    json.dump(rmse_mae_results, f, indent=2)

print(f"\nSaved: artifacts/baselines/structured_improved_rmse_mae.json")

# Generate rankings for improved evaluation sets
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
        candidates = []
        for _, row in group.iterrows():
            profile_b_id = row['profile_b_id']
            key = (profile_a_id, profile_b_id)
            
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
    'artifacts/eval_sets/val_candidates_improved.parquet',
    val_df,
    y_val_pred,
    'artifacts/baselines/structured_improved_rankings_val.json',
    'Validation'
)

# Generate rankings for test set
test_rankings = generate_rankings_from_predictions(
    'artifacts/eval_sets/test_candidates_improved.parquet',
    test_df,
    y_test_pred,
    'artifacts/baselines/structured_improved_rankings_test.json',
    'Test'
)

print("\n" + "=" * 80)
print("ENHANCED STRUCTURED REGRESSION BASELINE COMPLETE")
print("=" * 80)
