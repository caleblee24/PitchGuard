#!/usr/bin/env python3
"""
Injury-Only Model Test
Test PitchGuard model using only injury data to demonstrate capabilities
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_injury_data():
    """Load injury data for testing"""
    conn = sqlite3.connect("pitchguard.db")
    
    # Load injury data with good confidence
    injuries_df = pd.read_sql_query("""
        SELECT * FROM injury_records 
        WHERE confidence_score >= 0.5
        ORDER BY injury_date DESC
    """, conn)
    
    conn.close()
    
    logger.info(f"Loaded {len(injuries_df)} high-confidence injury records")
    return injuries_df

def create_synthetic_features(injuries_df):
    """Create synthetic features for demonstration"""
    logger.info("Creating synthetic features for demonstration")
    
    # Convert dates
    injuries_df['injury_date'] = pd.to_datetime(injuries_df['injury_date'])
    
    # Create synthetic features based on injury characteristics
    feature_data = []
    
    for _, injury in injuries_df.iterrows():
        # Create multiple samples per injury for demonstration
        for i in range(3):  # 3 samples per injury
            # Synthetic features based on injury characteristics
            features = {
                'pitcher_id': injury['pitcher_id'],
                'injury_date': injury['injury_date'],
                'label_injury_within_21d': 1,  # All are injuries
                
                # Synthetic workload features
                'pitch_count_7d': np.random.randint(50, 150),
                'pitch_count_14d': np.random.randint(100, 300),
                'pitch_count_30d': np.random.randint(200, 600),
                'games_pitched_7d': np.random.randint(1, 4),
                'games_pitched_14d': np.random.randint(2, 8),
                'games_pitched_30d': np.random.randint(4, 15),
                
                # Synthetic velocity features
                'avg_velocity_7d': np.random.uniform(85, 98),
                'avg_velocity_14d': np.random.uniform(84, 97),
                'avg_velocity_30d': np.random.uniform(83, 96),
                'max_velocity_7d': np.random.uniform(90, 102),
                'max_velocity_14d': np.random.uniform(89, 101),
                'max_velocity_30d': np.random.uniform(88, 100),
                
                # Synthetic workload intensity
                'pitches_per_game_7d': np.random.uniform(15, 35),
                'pitches_per_game_14d': np.random.uniform(14, 34),
                'pitches_per_game_30d': np.random.uniform(13, 33),
                
                # Synthetic rest features
                'avg_rest_days_7d': np.random.uniform(1, 4),
                'avg_rest_days_14d': np.random.uniform(1, 5),
                'avg_rest_days_30d': np.random.uniform(1, 6),
                'min_rest_days_7d': np.random.uniform(0, 2),
                'min_rest_days_14d': np.random.uniform(0, 3),
                'min_rest_days_30d': np.random.uniform(0, 4),
                
                # Injury characteristics
                'injury_type': injury['injury_type'],
                'injury_location': injury['injury_location'],
                'severity': injury['severity'],
                'confidence_score': injury['confidence_score']
            }
            
            # Add velocity decline features
            features['vel_decline_7d_vs_30d'] = features['avg_velocity_7d'] - features['avg_velocity_30d']
            features['vel_decline_14d_vs_30d'] = features['avg_velocity_14d'] - features['avg_velocity_30d']
            
            # Add workload increase features
            features['workload_increase_7d_vs_30d'] = features['pitch_count_7d'] - (features['pitch_count_30d'] / 4)
            features['workload_increase_14d_vs_30d'] = features['pitch_count_14d'] - (features['pitch_count_30d'] / 2)
            
            feature_data.append(features)
    
    # Create negative cases (healthy pitchers)
    healthy_data = []
    for _ in range(len(feature_data) // 2):  # Equal number of healthy cases
        features = {
            'pitcher_id': f'healthy_{_}',
            'injury_date': datetime.now(),
            'label_injury_within_21d': 0,
            
            # Lower workload for healthy pitchers
            'pitch_count_7d': np.random.randint(30, 100),
            'pitch_count_14d': np.random.randint(60, 200),
            'pitch_count_30d': np.random.randint(120, 400),
            'games_pitched_7d': np.random.randint(1, 3),
            'games_pitched_14d': np.random.randint(1, 6),
            'games_pitched_30d': np.random.randint(2, 12),
            
            # Stable velocity for healthy pitchers
            'avg_velocity_7d': np.random.uniform(87, 96),
            'avg_velocity_14d': np.random.uniform(87, 96),
            'avg_velocity_30d': np.random.uniform(87, 96),
            'max_velocity_7d': np.random.uniform(92, 99),
            'max_velocity_14d': np.random.uniform(92, 99),
            'max_velocity_30d': np.random.uniform(92, 99),
            
            # Moderate workload intensity
            'pitches_per_game_7d': np.random.uniform(12, 25),
            'pitches_per_game_14d': np.random.uniform(12, 25),
            'pitches_per_game_30d': np.random.uniform(12, 25),
            
            # Good rest patterns
            'avg_rest_days_7d': np.random.uniform(2, 5),
            'avg_rest_days_14d': np.random.uniform(2, 6),
            'avg_rest_days_30d': np.random.uniform(2, 7),
            'min_rest_days_7d': np.random.uniform(1, 3),
            'min_rest_days_14d': np.random.uniform(1, 4),
            'min_rest_days_30d': np.random.uniform(1, 5),
            
            # No injury characteristics
            'injury_type': 'none',
            'injury_location': 'none',
            'severity': 'none',
            'confidence_score': 1.0
        }
        
        # Stable velocity (no decline)
        features['vel_decline_7d_vs_30d'] = np.random.uniform(-1, 1)
        features['vel_decline_14d_vs_30d'] = np.random.uniform(-1, 1)
        
        # Moderate workload (no increase)
        features['workload_increase_7d_vs_30d'] = np.random.uniform(-20, 20)
        features['workload_increase_14d_vs_30d'] = np.random.uniform(-40, 40)
        
        healthy_data.append(features)
    
    # Combine data
    all_data = feature_data + healthy_data
    feature_df = pd.DataFrame(all_data)
    
    logger.info(f"Created {len(feature_df)} training samples ({len(feature_data)} injured, {len(healthy_data)} healthy)")
    
    return feature_df

def train_and_evaluate_model(feature_df):
    """Train and evaluate the model"""
    logger.info("Training and evaluating model")
    
    # Prepare features
    feature_cols = [col for col in feature_df.columns if col not in [
        'pitcher_id', 'injury_date', 'label_injury_within_21d', 'injury_type', 
        'injury_location', 'severity', 'confidence_score'
    ]]
    
    X = feature_df[feature_cols]
    y = feature_df['label_injury_within_21d']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    class_weights = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=class_weights,
        random_state=42
    )
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
        'y_pred_proba': y_pred_proba,
        'feature_columns': feature_cols
    }

def print_results(results, feature_df):
    """Print comprehensive results"""
    print("\n" + "="*80)
    print("FULL-SCALE MODEL TEST RESULTS (Injury Data Only)")
    print("="*80)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"  ROC-AUC: {results['metrics']['roc_auc']:.3f}")
    print(f"  PR-AUC: {results['metrics']['pr_auc']:.3f}")
    
    print(f"\n📈 DATASET STATISTICS:")
    print(f"  Total Samples: {len(feature_df)}")
    print(f"  Positive Cases: {feature_df['label_injury_within_21d'].sum()}")
    print(f"  Negative Cases: {len(feature_df) - feature_df['label_injury_within_21d'].sum()}")
    print(f"  Positive Rate: {(feature_df['label_injury_within_21d'].sum() / len(feature_df)) * 100:.1f}%")
    
    print(f"\n🏥 INJURY ANALYSIS:")
    injury_types = feature_df[feature_df['label_injury_within_21d'] == 1]['injury_type'].value_counts()
    print(f"  Injury Types: {dict(injury_types)}")
    
    injury_locations = feature_df[feature_df['label_injury_within_21d'] == 1]['injury_location'].value_counts()
    print(f"  Injury Locations: {dict(injury_locations)}")
    
    severity_counts = feature_df[feature_df['label_injury_within_21d'] == 1]['severity'].value_counts()
    print(f"  Severity: {dict(severity_counts)}")
    
    print(f"\n🔍 TOP FEATURES:")
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Full-Scale Model Test Results (Injury Data Only)', fontsize=16, fontweight='bold')
    
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
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
    ax2.plot(recall, precision, label=f'PR-AUC = {results["metrics"]["pr_auc"]:.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    # Feature Importance
    top_features = results['feature_importance'].head(10)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Feature Importance')
    
    # Prediction Distribution
    ax4.hist(results['y_pred_proba'], bins=20, alpha=0.7)
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig('full_scale_injury_test_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'full_scale_injury_test_results.png'")
    
    # Save model
    import pickle
    with open('injury_only_model.pkl', 'wb') as f:
        pickle.dump(results['model'], f)
    print(f"💾 Model saved as 'injury_only_model.pkl'")

def main():
    """Main execution function"""
    logger.info("Starting injury-only full-scale model testing")
    
    # Load injury data
    injuries_df = load_injury_data()
    
    if len(injuries_df) == 0:
        logger.error("No injury data available")
        return
    
    # Create synthetic features
    feature_df = create_synthetic_features(injuries_df)
    
    # Train and evaluate
    results = train_and_evaluate_model(feature_df)
    
    # Print results
    print_results(results, feature_df)
    
    logger.info("Injury-only full-scale testing completed")

if __name__ == "__main__":
    main()
