#!/usr/bin/env python3
"""
Simple Full-Scale Model Testing
Test PitchGuard model with enhanced injury data
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_data():
    """Load all available data"""
    conn = sqlite3.connect("pitchguard.db")
    
    # Load injury data
    injuries_df = pd.read_sql_query("""
        SELECT * FROM injury_records 
        WHERE confidence_score >= 0.3
    """, conn)
    
    # Load workload data
    workload_df = pd.read_sql_query("SELECT * FROM appearances", conn)
    
    # Load pitch data
    pitches_df = pd.read_sql_query("SELECT * FROM pitches", conn)
    
    conn.close()
    
    logger.info(f"Loaded {len(injuries_df)} injuries, {len(workload_df)} appearances, {len(pitches_df)} pitches")
    return injuries_df, workload_df, pitches_df

def create_training_data(injuries_df, workload_df, pitches_df):
    """Create training data with injury labels"""
    logger.info("Creating training data")
    
    # Convert dates
    injuries_df['injury_date'] = pd.to_datetime(injuries_df['injury_date'])
    workload_df['game_date'] = pd.to_datetime(workload_df['game_date'])
    pitches_df['game_date'] = pd.to_datetime(pitches_df['game_date'])
    
    training_data = []
    
    # Create positive cases (injured)
    for _, injury in injuries_df.iterrows():
        pitcher_id = injury['pitcher_id']
        injury_date = injury['injury_date']
        
        # Get pitcher's recent appearances
        pitcher_appearances = workload_df[workload_df['pitcher_id'] == pitcher_id]
        recent_appearances = pitcher_appearances[
            (pitcher_appearances['game_date'] >= injury_date - timedelta(days=21)) &
            (pitcher_appearances['game_date'] < injury_date)
        ]
        
        for _, appearance in recent_appearances.iterrows():
            training_data.append({
                'pitcher_id': pitcher_id,
                'game_date': appearance['game_date'],
                'label_injury_within_21d': 1,
                'injury_type': injury['injury_type'],
                'injury_location': injury['injury_location'],
                'severity': injury['severity']
            })
    
    # Create negative cases (healthy)
    for _, injury in injuries_df.iterrows():
        pitcher_id = injury['pitcher_id']
        injury_date = injury['injury_date']
        
        pitcher_appearances = workload_df[workload_df['pitcher_id'] == pitcher_id]
        healthy_appearances = pitcher_appearances[
            (pitcher_appearances['game_date'] >= injury_date - timedelta(days=120)) &
            (pitcher_appearances['game_date'] <= injury_date - timedelta(days=60))
        ]
        
        # Sample healthy appearances
        if len(healthy_appearances) > 0:
            sample_size = min(len(healthy_appearances), 2)
            healthy_sample = healthy_appearances.sample(n=sample_size, random_state=42)
            
            for _, appearance in healthy_sample.iterrows():
                training_data.append({
                    'pitcher_id': pitcher_id,
                    'game_date': appearance['game_date'],
                    'label_injury_within_21d': 0,
                    'injury_type': None,
                    'injury_location': None,
                    'severity': None
                })
    
    training_df = pd.DataFrame(training_data)
    logger.info(f"Created {len(training_df)} training samples ({training_df['label_injury_within_21d'].sum()} positive)")
    
    return training_df

def engineer_features(training_df, pitches_df):
    """Engineer features for training"""
    logger.info("Engineering features")
    
    feature_data = []
    
    for _, row in training_df.iterrows():
        pitcher_id = row['pitcher_id']
        game_date = row['game_date']
        
        # Get historical pitch data
        pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id]
        historical_pitches = pitcher_pitches[pitcher_pitches['game_date'] < game_date]
        
        features = {
            'pitcher_id': pitcher_id,
            'game_date': game_date,
            'label_injury_within_21d': row['label_injury_within_21d']
        }
        
        # Calculate rolling features
        for window in [7, 14, 30]:
            window_start = game_date - timedelta(days=window)
            window_pitches = historical_pitches[historical_pitches['game_date'] >= window_start]
            
            features[f'pitch_count_{window}d'] = len(window_pitches)
            features[f'games_{window}d'] = window_pitches['game_date'].nunique()
            
            if 'release_speed' in window_pitches.columns and len(window_pitches) > 0:
                features[f'avg_velocity_{window}d'] = window_pitches['release_speed'].mean()
                features[f'max_velocity_{window}d'] = window_pitches['release_speed'].max()
            else:
                features[f'avg_velocity_{window}d'] = 0
                features[f'max_velocity_{window}d'] = 0
        
        feature_data.append(features)
    
    feature_df = pd.DataFrame(feature_data)
    
    # Fill missing values
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)
    
    logger.info(f"Engineered {len(feature_df)} feature samples")
    return feature_df

def train_and_evaluate_model(feature_df):
    """Train and evaluate the model"""
    logger.info("Training and evaluating model")
    
    # Prepare features
    feature_cols = [col for col in feature_df.columns if col not in ['pitcher_id', 'game_date', 'label_injury_within_21d']]
    X = feature_df[feature_cols]
    y = feature_df['label_injury_within_21d']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    class_weights = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = xgb.XGBClassifier(scale_pos_weight=class_weights, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'metrics': {'roc_auc': roc_auc, 'pr_auc': pr_auc},
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }

def print_results(results, feature_df):
    """Print comprehensive results"""
    print("\n" + "="*80)
    print("FULL-SCALE MODEL TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"  ROC-AUC: {results['metrics']['roc_auc']:.3f}")
    print(f"  PR-AUC: {results['metrics']['pr_auc']:.3f}")
    
    print(f"\n📈 DATASET STATISTICS:")
    print(f"  Total Samples: {len(feature_df)}")
    print(f"  Positive Cases: {feature_df['label_injury_within_21d'].sum()}")
    print(f"  Negative Cases: {len(feature_df) - feature_df['label_injury_within_21d'].sum()}")
    print(f"  Positive Rate: {(feature_df['label_injury_within_21d'].sum() / len(feature_df)) * 100:.1f}%")
    
    print(f"\n🔍 TOP FEATURES:")
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
    ax1.plot(fpr, tpr, label=f'ROC-AUC = {results["metrics"]["roc_auc"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Feature Importance
    top_features = results['feature_importance'].head(10)
    ax2.barh(range(len(top_features)), top_features['importance'])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.savefig('full_scale_test_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'full_scale_test_results.png'")

def main():
    """Main execution function"""
    logger.info("Starting full-scale model testing")
    
    # Load data
    injuries_df, workload_df, pitches_df = load_enhanced_data()
    
    if len(injuries_df) == 0:
        logger.error("No injury data available")
        return
    
    # Create training data
    training_df = create_training_data(injuries_df, workload_df, pitches_df)
    
    if len(training_df) == 0:
        logger.error("No training data created")
        return
    
    # Engineer features
    feature_df = engineer_features(training_df, pitches_df)
    
    # Train and evaluate
    results = train_and_evaluate_model(feature_df)
    
    # Print results
    print_results(results, feature_df)
    
    logger.info("Full-scale testing completed")

if __name__ == "__main__":
    main()
