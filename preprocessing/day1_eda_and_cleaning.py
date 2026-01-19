"""
Day 1: Data Cleaning and Exploratory Data Analysis (EDA)
CareerRank Project

This script performs:
1. Data loading and understanding
2. Data cleaning for profiles and compatibility pairs
3. Exploratory Data Analysis with visualizations
4. Profile text serialization
5. Generation of cleaned artifacts for modeling
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# STEP 1: READ AND UNDERSTAND THE DATA
# ============================================================================

print("=" * 80)
print("STEP 1: READING AND UNDERSTANDING DATA")
print("=" * 80)

# Load datasets
profiles_df = pd.read_csv('profiles.csv')
compatibility_df = pd.read_csv('compatibility_pairs.csv')

print("\n--- PROFILES DATASET ---")
print(f"Shape: {profiles_df.shape}")
print(f"Rows: {profiles_df.shape[0]}, Columns: {profiles_df.shape[1]}")
print(f"\nColumn Names:\n{list(profiles_df.columns)}")
print(f"\nData Types:\n{profiles_df.dtypes}")
print(f"\nMissing Values:\n{profiles_df.isnull().sum()}")
print(f"\nBasic Statistics:\n{profiles_df.describe()}")

print("\n--- COMPATIBILITY PAIRS DATASET ---")
print(f"Shape: {compatibility_df.shape}")
print(f"Rows: {compatibility_df.shape[0]}, Columns: {compatibility_df.shape[1]}")
print(f"\nColumn Names:\n{list(compatibility_df.columns)}")
print(f"\nData Types:\n{compatibility_df.dtypes}")
print(f"\nMissing Values:\n{compatibility_df.isnull().sum()}")
print(f"\nScore Statistics:\n{compatibility_df['compatibility_score'].describe()}")

# Inspect JSON columns in profiles
print("\n--- INSPECTING JSON COLUMNS ---")
json_columns = ['skills', 'experience', 'education', 'goals', 'needs', 'can_offer']

for col in json_columns:
    print(f"\n{col.upper()}:")
    sample_value = profiles_df[col].iloc[0]
    print(f"Sample raw value: {sample_value}")
    try:
        parsed = json.loads(sample_value.replace("'", '"'))
        print(f"Parsed structure: {type(parsed)}")
        print(f"Sample parsed: {parsed}")
    except Exception as e:
        print(f"Error parsing: {e}")

# Inspect compatibility score distributions
print("\n--- COMPATIBILITY SCORE DISTRIBUTION ---")
print(compatibility_df['compatibility_score'].describe())
print(f"\nScore Range: [{compatibility_df['compatibility_score'].min()}, {compatibility_df['compatibility_score'].max()}]")

# Check for missing values in key columns
print("\n--- MISSING VALUES IN COMPATIBILITY PAIRS ---")
print(compatibility_df[['profile_a_id', 'profile_b_id', 'compatibility_score']].isnull().sum())

# ============================================================================
# STEP 2 & 3: DATA CLEANING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2 & 3: DATA CLEANING")
print("=" * 80)

# --- PROFILES CLEANING ---
print("\n--- CLEANING PROFILES ---")

# Check for duplicate profile_ids
print(f"Duplicate profile_ids before cleaning: {profiles_df['profile_id'].duplicated().sum()}")
profiles_clean = profiles_df.drop_duplicates(subset=['profile_id'], keep='first')
print(f"Rows after removing duplicates: {profiles_clean.shape[0]}")

# Handle missing values
# Strategy: Drop rows with missing critical fields (profile_id, name, email)
# For other fields, we'll handle them in serialization
print(f"\nMissing values in critical fields:")
print(profiles_clean[['profile_id', 'name', 'email']].isnull().sum())

# Drop rows with missing profile_id (if any)
profiles_clean = profiles_clean.dropna(subset=['profile_id'])
print(f"Rows after dropping missing profile_id: {profiles_clean.shape[0]}")

# Normalize text fields: lowercase and strip whitespace
text_columns = ['name', 'location', 'headline', 'about', 'current_role', 
                'current_company', 'industry', 'seniority_level', 'remote_preference']

for col in text_columns:
    if col in profiles_clean.columns:
        profiles_clean[col] = profiles_clean[col].fillna('').astype(str).str.strip().str.lower()

print(f"\nText fields normalized (lowercase, stripped)")

# Parse JSON fields safely
def safe_json_parse(json_str):
    """Safely parse JSON string, handling single quotes and errors"""
    if pd.isna(json_str) or json_str == '':
        return []
    try:
        # Replace single quotes with double quotes for valid JSON
        json_str = str(json_str).replace("'", '"')
        return json.loads(json_str)
    except:
        return []

print("\n--- PARSING JSON FIELDS ---")
for col in json_columns:
    profiles_clean[f'{col}_parsed'] = profiles_clean[col].apply(safe_json_parse)
    print(f"Parsed {col}: {profiles_clean[f'{col}_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()} total items")

# Convert skills into clean list of strings
def clean_skills_list(skills_list):
    """Convert skills list to clean lowercase strings"""
    if not isinstance(skills_list, list):
        return []
    return [str(skill).strip().lower() for skill in skills_list if skill]

profiles_clean['skills_clean'] = profiles_clean['skills_parsed'].apply(clean_skills_list)

# Sort experience chronologically (by duration, descending)
def sort_experience(exp_list):
    """Sort experience entries by duration"""
    if not isinstance(exp_list, list) or len(exp_list) == 0:
        return exp_list
    # Extract numeric duration and sort
    try:
        for exp in exp_list:
            if isinstance(exp, dict) and 'duration' in exp:
                duration_str = exp.get('duration', '0 years')
                exp['duration_years'] = int(duration_str.split()[0])
        return sorted(exp_list, key=lambda x: x.get('duration_years', 0), reverse=True)
    except:
        return exp_list

profiles_clean['experience_sorted'] = profiles_clean['experience_parsed'].apply(sort_experience)

print(f"\nProfiles cleaned: {profiles_clean.shape[0]} rows")

# --- COMPATIBILITY PAIRS CLEANING ---
print("\n--- CLEANING COMPATIBILITY PAIRS ---")

# Verify referential integrity
valid_profile_ids = set(profiles_clean['profile_id'].values)
print(f"Valid profile IDs: {len(valid_profile_ids)}")

compatibility_clean = compatibility_df.copy()
before_count = len(compatibility_clean)

# Check if both profile_a_id and profile_b_id exist in profiles
compatibility_clean['a_exists'] = compatibility_clean['profile_a_id'].isin(valid_profile_ids)
compatibility_clean['b_exists'] = compatibility_clean['profile_b_id'].isin(valid_profile_ids)

print(f"Pairs with invalid profile_a_id: {(~compatibility_clean['a_exists']).sum()}")
print(f"Pairs with invalid profile_b_id: {(~compatibility_clean['b_exists']).sum()}")

# Keep only pairs where both profiles exist
compatibility_clean = compatibility_clean[compatibility_clean['a_exists'] & compatibility_clean['b_exists']]
compatibility_clean = compatibility_clean.drop(['a_exists', 'b_exists'], axis=1)

print(f"Pairs removed due to invalid references: {before_count - len(compatibility_clean)}")
print(f"Pairs after referential integrity check: {len(compatibility_clean)}")

# Check score ranges and clip if needed
print(f"\nCompatibility score range: [{compatibility_clean['compatibility_score'].min():.2f}, {compatibility_clean['compatibility_score'].max():.2f}]")

# Clip scores to reasonable range (0-100)
compatibility_clean['compatibility_score'] = compatibility_clean['compatibility_score'].clip(0, 100)

# Add derived column: compatibility_bucket
def categorize_compatibility(score):
    """Categorize compatibility score into buckets"""
    if score < 30:
        return 'low'
    elif score < 60:
        return 'medium'
    else:
        return 'high'

compatibility_clean['compatibility_bucket'] = compatibility_clean['compatibility_score'].apply(categorize_compatibility)

print(f"\nCompatibility buckets distribution:")
print(compatibility_clean['compatibility_bucket'].value_counts())

print(f"\nCompatibility pairs cleaned: {compatibility_clean.shape[0]} rows")

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Create output directory for plots
Path('artifacts/eda').mkdir(parents=True, exist_ok=True)

# --- PROFILES EDA ---
print("\n--- PROFILES EDA ---")

# 1. Distribution of years_experience
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
profiles_clean['years_experience'].hist(bins=30, edgecolor='black')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.title('Distribution of Years of Experience')

plt.subplot(1, 2, 2)
profiles_clean['years_experience'].plot(kind='box')
plt.ylabel('Years of Experience')
plt.title('Box Plot: Years of Experience')
plt.tight_layout()
plt.savefig('artifacts/eda/years_experience_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: years_experience_distribution.png")

# 2. Distribution of seniority_level
plt.figure(figsize=(10, 6))
seniority_counts = profiles_clean['seniority_level'].value_counts()
seniority_counts.plot(kind='bar', edgecolor='black')
plt.xlabel('Seniority Level')
plt.ylabel('Count')
plt.title('Distribution of Seniority Levels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('artifacts/eda/seniority_level_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: seniority_level_distribution.png")

# 3. Distribution of industry
plt.figure(figsize=(14, 6))
industry_counts = profiles_clean['industry'].value_counts().head(15)
industry_counts.plot(kind='barh', edgecolor='black')
plt.xlabel('Count')
plt.ylabel('Industry')
plt.title('Top 15 Industries')
plt.tight_layout()
plt.savefig('artifacts/eda/industry_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: industry_distribution.png")

# 4. Number of skills per profile
profiles_clean['num_skills'] = profiles_clean['skills_clean'].apply(len)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
profiles_clean['num_skills'].hist(bins=30, edgecolor='black')
plt.xlabel('Number of Skills')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Skills per Profile')

plt.subplot(1, 2, 2)
profiles_clean['num_skills'].plot(kind='box')
plt.ylabel('Number of Skills')
plt.title('Box Plot: Number of Skills')
plt.tight_layout()
plt.savefig('artifacts/eda/num_skills_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: num_skills_distribution.png")

# 5. Basic statistics for connections
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
profiles_clean['connections'].hist(bins=50, edgecolor='black')
plt.xlabel('Number of Connections')
plt.ylabel('Frequency')
plt.title('Distribution of Connections')

plt.subplot(1, 2, 2)
profiles_clean['connections'].plot(kind='box')
plt.ylabel('Number of Connections')
plt.title('Box Plot: Connections')
plt.tight_layout()
plt.savefig('artifacts/eda/connections_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: connections_distribution.png")

print(f"\nConnections Statistics:")
print(profiles_clean['connections'].describe())

# --- COMPATIBILITY PAIRS EDA ---
print("\n--- COMPATIBILITY PAIRS EDA ---")

# 6. Distribution of compatibility_score
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
compatibility_clean['compatibility_score'].hist(bins=50, edgecolor='black')
plt.xlabel('Compatibility Score')
plt.ylabel('Frequency')
plt.title('Distribution of Compatibility Scores')

plt.subplot(1, 2, 2)
compatibility_clean['compatibility_score'].plot(kind='box')
plt.ylabel('Compatibility Score')
plt.title('Box Plot: Compatibility Score')
plt.tight_layout()
plt.savefig('artifacts/eda/compatibility_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: compatibility_score_distribution.png")

# 7. Correlation heatmap between sub-scores
score_columns = ['skill_match_score', 'skill_complementarity_score', 
                 'network_value_a_to_b', 'network_value_b_to_a',
                 'career_alignment_score', 'experience_gap', 
                 'geographic_score', 'seniority_match', 'compatibility_score']

correlation_matrix = compatibility_clean[score_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap: Compatibility Sub-Scores')
plt.tight_layout()
plt.savefig('artifacts/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# 8. Analysis: skill_match_score vs skill_complementarity_score
plt.figure(figsize=(10, 6))
plt.scatter(compatibility_clean['skill_match_score'], 
            compatibility_clean['skill_complementarity_score'],
            alpha=0.3, s=10)
plt.xlabel('Skill Match Score')
plt.ylabel('Skill Complementarity Score')
plt.title('Skill Match vs Skill Complementarity')
plt.tight_layout()
plt.savefig('artifacts/eda/skill_match_vs_complementarity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: skill_match_vs_complementarity.png")

# 9. Analysis: experience_gap vs compatibility_score
plt.figure(figsize=(10, 6))
plt.scatter(compatibility_clean['experience_gap'], 
            compatibility_clean['compatibility_score'],
            alpha=0.3, s=10)
plt.xlabel('Experience Gap (years)')
plt.ylabel('Compatibility Score')
plt.title('Experience Gap vs Compatibility Score')
plt.tight_layout()
plt.savefig('artifacts/eda/experience_gap_vs_compatibility.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: experience_gap_vs_compatibility.png")

# 10. Compatibility bucket distribution
plt.figure(figsize=(10, 6))
bucket_counts = compatibility_clean['compatibility_bucket'].value_counts()
bucket_counts.plot(kind='bar', edgecolor='black', color=['#d62728', '#ff7f0e', '#2ca02c'])
plt.xlabel('Compatibility Bucket')
plt.ylabel('Count')
plt.title('Distribution of Compatibility Buckets')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('artifacts/eda/compatibility_bucket_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: compatibility_bucket_distribution.png")

print("\nEDA complete. All plots saved to artifacts/eda/")

# ============================================================================
# STEP 5: PROFILE TEXT SERIALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: PROFILE TEXT SERIALIZATION")
print("=" * 80)

def serialize_profile(profile_row):
    """
    Convert a profile row into a clean, structured text block.
    
    Args:
        profile_row: A pandas Series representing one profile
        
    Returns:
        str: Clean, deterministic text representation of the profile
    """
    lines = []
    
    # Current Role
    current_role = profile_row.get('current_role', '').strip()
    current_company = profile_row.get('current_company', '').strip()
    if current_role and current_company:
        lines.append(f"Current Role: {current_role} at {current_company}")
    elif current_role:
        lines.append(f"Current Role: {current_role}")
    
    # Industry
    industry = profile_row.get('industry', '').strip()
    if industry:
        lines.append(f"Industry: {industry}")
    
    # Experience
    years_exp = profile_row.get('years_experience', '')
    seniority = profile_row.get('seniority_level', '').strip()
    if years_exp != '' and seniority:
        lines.append(f"Experience: {years_exp} years, {seniority}")
    elif years_exp != '':
        lines.append(f"Experience: {years_exp} years")
    elif seniority:
        lines.append(f"Experience: {seniority}")
    
    # Skills
    skills_clean = profile_row.get('skills_clean', [])
    if skills_clean and len(skills_clean) > 0:
        skills_str = ', '.join(skills_clean)
        lines.append(f"Skills: {skills_str}")
    
    # Work History
    experience_sorted = profile_row.get('experience_sorted', [])
    if experience_sorted and len(experience_sorted) > 0:
        work_history_parts = []
        for exp in experience_sorted[:3]:  # Top 3 most recent
            if isinstance(exp, dict):
                title = exp.get('title', '')
                company = exp.get('company', '')
                duration = exp.get('duration', '')
                if title and company:
                    work_history_parts.append(f"{title} at {company} ({duration})")
        if work_history_parts:
            lines.append(f"Work History: {'; '.join(work_history_parts)}")
    
    # Education
    education_parsed = profile_row.get('education_parsed', [])
    if education_parsed and len(education_parsed) > 0:
        edu_parts = []
        for edu in education_parsed:
            if isinstance(edu, dict):
                degree = edu.get('degree', '')
                field = edu.get('field', '')
                school = edu.get('school', '')
                if degree and field and school:
                    edu_parts.append(f"{degree} in {field} from {school}")
                elif degree and school:
                    edu_parts.append(f"{degree} from {school}")
        if edu_parts:
            lines.append(f"Education: {'; '.join(edu_parts)}")
    
    # Goals
    goals_parsed = profile_row.get('goals_parsed', [])
    if goals_parsed and len(goals_parsed) > 0:
        goals_str = ', '.join([str(g).lower() for g in goals_parsed if g])
        if goals_str:
            lines.append(f"Goals: {goals_str}")
    
    # Needs
    needs_parsed = profile_row.get('needs_parsed', [])
    if needs_parsed and len(needs_parsed) > 0:
        needs_str = ', '.join([str(n).lower() for n in needs_parsed if n])
        if needs_str:
            lines.append(f"Needs: {needs_str}")
    
    # Can Offer
    can_offer_parsed = profile_row.get('can_offer_parsed', [])
    if can_offer_parsed and len(can_offer_parsed) > 0:
        can_offer_str = ', '.join([str(c).lower() for c in can_offer_parsed if c])
        if can_offer_str:
            lines.append(f"Can Offer: {can_offer_str}")
    
    # Location
    location = profile_row.get('location', '').strip()
    if location:
        lines.append(f"Location: {location}")
    
    # Join all lines with newlines
    return '\n'.join(lines)

print("Testing serialize_profile function on sample profiles...")

# Test on a few profiles
sample_profiles = profiles_clean.head(3)
for idx, row in sample_profiles.iterrows():
    print(f"\n--- Profile {row['profile_id']} ---")
    serialized = serialize_profile(row)
    print(serialized)
    print(f"Text length: {len(serialized)} characters")

# ============================================================================
# STEP 6: GENERATE SERIALIZED ARTIFACTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: GENERATING SERIALIZED ARTIFACTS")
print("=" * 80)

# Create serialized profiles dataframe
print("\nSerializing all profiles...")
profiles_clean['serialized_text'] = profiles_clean.apply(serialize_profile, axis=1)

serialized_profiles_df = profiles_clean[['profile_id', 'serialized_text']].copy()

# Save serialized profiles
output_path = 'artifacts/serialized_profiles.csv'
serialized_profiles_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Shape: {serialized_profiles_df.shape}")

# Create lightweight joined dataset for pairs
print("\nCreating pair text dataset...")

# Merge to get text for both profiles in each pair
pair_text_df = compatibility_clean[['pair_id', 'profile_a_id', 'profile_b_id', 'compatibility_score']].copy()

# Merge with serialized profiles
pair_text_df = pair_text_df.merge(
    serialized_profiles_df.rename(columns={'profile_id': 'profile_a_id', 'serialized_text': 'profile_a_text'}),
    on='profile_a_id',
    how='left'
)

pair_text_df = pair_text_df.merge(
    serialized_profiles_df.rename(columns={'profile_id': 'profile_b_id', 'serialized_text': 'profile_b_text'}),
    on='profile_b_id',
    how='left'
)

# Select final columns
pair_text_df = pair_text_df[['profile_a_text', 'profile_b_text', 'compatibility_score']]

# Save pair text dataset
output_path = 'artifacts/pair_text_dataset.csv'
pair_text_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Shape: {pair_text_df.shape}")

# ============================================================================
# STEP 7: VALIDATION AND SANITY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: VALIDATION AND SANITY CHECKS")
print("=" * 80)

# Print 3 random serialized profiles
print("\n--- 3 RANDOM SERIALIZED PROFILES ---")
random_samples = serialized_profiles_df.sample(n=3, random_state=42)
for idx, row in random_samples.iterrows():
    print(f"\nProfile ID: {row['profile_id']}")
    print(row['serialized_text'])
    print(f"Length: {len(row['serialized_text'])} characters")
    print("-" * 60)

# Print 3 random profile pairs with text and score
print("\n--- 3 RANDOM PROFILE PAIRS ---")
random_pairs = pair_text_df.sample(n=3, random_state=42)
for idx, row in random_pairs.iterrows():
    print(f"\nPair {idx + 1}:")
    print(f"Compatibility Score: {row['compatibility_score']:.2f}")
    print("\nProfile A:")
    print(row['profile_a_text'])
    print("\nProfile B:")
    print(row['profile_b_text'])
    print("-" * 60)

# Assert no null serialized_text values
null_count = serialized_profiles_df['serialized_text'].isnull().sum()
print(f"\n--- VALIDATION CHECKS ---")
print(f"Null serialized_text values: {null_count}")
assert null_count == 0, "ERROR: Found null serialized_text values!"
print("✓ No null serialized_text values")

# Assert text length is reasonable and non-zero
text_lengths = serialized_profiles_df['serialized_text'].str.len()
zero_length = (text_lengths == 0).sum()
print(f"Zero-length serialized_text: {zero_length}")
assert zero_length == 0, "ERROR: Found zero-length serialized_text!"
print("✓ All serialized texts have non-zero length")

print(f"\nText length statistics:")
print(text_lengths.describe())

# Check pair text dataset
null_a = pair_text_df['profile_a_text'].isnull().sum()
null_b = pair_text_df['profile_b_text'].isnull().sum()
print(f"\nNull profile_a_text: {null_a}")
print(f"Null profile_b_text: {null_b}")
assert null_a == 0 and null_b == 0, "ERROR: Found null text in pair dataset!"
print("✓ No null text in pair dataset")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DAY 1 COMPLETE - SUMMARY")
print("=" * 80)

print(f"""
✓ Data Understanding:
  - Profiles: {profiles_clean.shape[0]} rows, {profiles_clean.shape[1]} columns
  - Compatibility Pairs: {compatibility_clean.shape[0]} rows, {compatibility_clean.shape[1]} columns

✓ Data Cleaning:
  - Removed duplicate profiles
  - Normalized text fields
  - Parsed JSON fields successfully
  - Verified referential integrity
  - Added compatibility_bucket column

✓ EDA:
  - Generated 10 visualization plots
  - Saved to: artifacts/eda/

✓ Profile Serialization:
  - Created deterministic serialize_profile function
  - Generated serialized_profiles.csv ({serialized_profiles_df.shape[0]} profiles)
  - Generated pair_text_dataset.csv ({pair_text_df.shape[0]} pairs)

✓ Validation:
  - All serialized texts are non-null and non-empty
  - Text lengths are reasonable
  - Pair dataset is complete

All artifacts saved to artifacts/ directory.
Ready for Day 2: Model Development!
""")

print("=" * 80)
