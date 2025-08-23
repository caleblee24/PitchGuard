#!/usr/bin/env python3
"""
Debug script to investigate feature engineering issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.mock_data_generator import MockDataGenerator
from etl.enhanced_features import EnhancedFeatureEngineer
from modeling.enhanced_model import EnhancedInjuryRiskModel

def debug_feature_engineering():
    """Debug the feature engineering process."""
    
    print("🔍 Debugging Feature Engineering Process")
    print("=" * 50)
    
    # Step 1: Generate test data
    print("\n1. Generating test data...")
    config = {
        'num_pitchers': 5,
        'start_date': datetime.now() - timedelta(days=90),
        'end_date': datetime.now(),
        'injury_probability': 0.15,
        'velocity_decline_threshold': -2.0,
        'pitch_count_threshold': 100
    }
    
    generator = MockDataGenerator(config)
    pitches_df, appearances_df, injuries_data = generator.generate_realistic_injury_scenarios()
    
    print(f"   Generated: {len(pitches_df)} pitches, {len(appearances_df)} appearances, {len(injuries_data)} injuries")
    
    # Step 2: Examine data quality
    print("\n2. Examining data quality...")
    print(f"   Unique pitchers: {pitches_df['pitcher_id'].nunique()}")
    print(f"   Date range: {pitches_df['game_date'].min()} to {pitches_df['game_date'].max()}")
    print(f"   Velocity range: {pitches_df['release_speed'].min():.1f} - {pitches_df['release_speed'].max():.1f}")
    
    # Step 3: Test feature engineering for one pitcher
    print("\n3. Testing feature engineering for one pitcher...")
    pitcher_id = pitches_df['pitcher_id'].iloc[0]
    print(f"   Testing pitcher ID: {pitcher_id}")
    
    # Filter data for this pitcher
    pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id]
    pitcher_appearances = appearances_df[appearances_df['pitcher_id'] == pitcher_id]
    
    print(f"   Pitcher pitches: {len(pitcher_pitches)}")
    print(f"   Pitcher appearances: {len(pitcher_appearances)}")
    
    # Step 4: Engineer features
    print("\n4. Engineering features...")
    feature_engineer = EnhancedFeatureEngineer()
    
    # Convert date columns to datetime
    pitcher_pitches['game_date'] = pd.to_datetime(pitcher_pitches['game_date'])
    pitcher_appearances['game_date'] = pd.to_datetime(pitcher_appearances['game_date'])
    
    # Use the latest date as as_of_date
    as_of_date = max(pitcher_appearances['game_date'].max(), pitcher_pitches['game_date'].max())
    
    features = feature_engineer.engineer_features(
        pitcher_appearances, pitcher_pitches, pd.DataFrame(), as_of_date
    )
    
    print(f"   Features computed: {len(features)}")
    print(f"   Feature columns: {list(features.columns)}")
    
    # Step 5: Examine feature values
    print("\n5. Examining feature values...")
    if not features.empty:
        feature_row = features.iloc[0]
        
        # Check for zero values
        zero_features = []
        non_zero_features = []
        
        for col in features.columns:
            if col not in ['pitcher_id', 'as_of_date']:
                value = feature_row[col]
                if value == 0.0 or pd.isna(value):
                    zero_features.append(col)
                else:
                    non_zero_features.append((col, value))
        
        print(f"   Zero/null features: {len(zero_features)}")
        print(f"   Non-zero features: {len(non_zero_features)}")
        
        if zero_features:
            print(f"   Zero features: {zero_features[:5]}...")  # Show first 5
        
        if non_zero_features:
            print(f"   Non-zero features (first 5):")
            for col, value in non_zero_features[:5]:
                print(f"     {col}: {value}")
    
    # Step 6: Test with multiple pitchers
    print("\n6. Testing with multiple pitchers...")
    all_features = []
    
    for pitcher_id in pitches_df['pitcher_id'].unique()[:3]:  # Test first 3 pitchers
        pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id]
        pitcher_appearances = appearances_df[appearances_df['pitcher_id'] == pitcher_id]
        
        if len(pitcher_appearances) > 0:
            pitcher_pitches['game_date'] = pd.to_datetime(pitcher_pitches['game_date'])
            pitcher_appearances['game_date'] = pd.to_datetime(pitcher_appearances['game_date'])
            
            as_of_date = max(pitcher_appearances['game_date'].max(), pitcher_pitches['game_date'].max())
            
            features = feature_engineer.engineer_features(
                pitcher_appearances, pitcher_pitches, pd.DataFrame(), as_of_date
            )
            
            if not features.empty:
                all_features.append(features)
    
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        print(f"   Combined features shape: {combined_features.shape}")
        
        # Check feature variance
        numeric_features = combined_features.select_dtypes(include=[np.number])
        zero_variance_features = []
        low_variance_features = []
        
        for col in numeric_features.columns:
            if col not in ['pitcher_id']:
                variance = numeric_features[col].var()
                if variance == 0:
                    zero_variance_features.append(col)
                elif variance < 0.01:
                    low_variance_features.append(col)
        
        print(f"   Zero variance features: {len(zero_variance_features)}")
        print(f"   Low variance features (< 0.01): {len(low_variance_features)}")
        
        if zero_variance_features:
            print(f"   Zero variance features: {zero_variance_features[:5]}...")
    
    print("\n✅ Feature engineering debug complete!")

if __name__ == "__main__":
    debug_feature_engineering()
