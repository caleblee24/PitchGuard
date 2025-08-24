#!/usr/bin/env python3
"""
Full-Scale Model Testing with Enhanced Injury Data
Comprehensive testing of PitchGuard model with real injury data
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
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullScaleModelTester:
    """Comprehensive model testing with enhanced injury data"""
    
    def __init__(self, db_path: str = "pitchguard.db"):
        self.db_path = db_path
        self.model = None
        self.test_results = {}
        self.feature_importance = None
        
    def load_enhanced_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all available data for full-scale testing"""
        logger.info("Loading enhanced data for full-scale testing")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load injury data
        injuries_df = pd.read_sql_query("""
            SELECT * FROM injury_records 
            WHERE confidence_score >= 0.3
            ORDER BY injury_date DESC
        """, conn)
        
        # Load workload data (appearances)
        workload_df = pd.read_sql_query("""
            SELECT * FROM appearances 
            ORDER BY game_date DESC
        """, conn)
        
        # Load pitch data
        pitches_df = pd.read_sql_query("""
            SELECT * FROM pitches 
            ORDER BY game_date DESC
        """, conn)
        
        # Load pitcher metadata
        pitchers_df = pd.read_sql_query("""
            SELECT * FROM pitchers
        """, conn)
        
        conn.close()
        
        logger.info(f"Loaded {len(injuries_df)} injury records, {len(workload_df)} appearances, {len(pitches_df)} pitches")
        
        return injuries_df, workload_df, pitches_df, pitchers_df
    
    def create_injury_labels(self, injuries_df: pd.DataFrame, workload_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive injury labels for model training"""
        logger.info("Creating injury labels for full-scale testing")
        
        # Convert dates
        injuries_df['injury_date'] = pd.to_datetime(injuries_df['injury_date'])
        workload_df['game_date'] = pd.to_datetime(workload_df['game_date'])
        
        # Create injury labels
        labeled_data = []
        
        for _, injury in injuries_df.iterrows():
            pitcher_id = injury['pitcher_id']
            injury_date = injury['injury_date']
            
            # Get pitcher's appearances
            pitcher_appearances = workload_df[workload_df['pitcher_id'] == pitcher_id].copy()
            pitcher_appearances = pitcher_appearances.sort_values('game_date')
            
            if len(pitcher_appearances) == 0:
                continue
            
            # Find appearances within 21 days before injury
            injury_window = injury_date - timedelta(days=21)
            pre_injury_appearances = pitcher_appearances[
                (pitcher_appearances['game_date'] >= injury_window) & 
                (pitcher_appearances['game_date'] < injury_date)
            ]
            
            # Label these as positive cases
            for _, appearance in pre_injury_appearances.iterrows():
                labeled_data.append({
                    'pitcher_id': pitcher_id,
                    'pitcher_name': injury['pitcher_name'],
                    'game_date': appearance['game_date'],
                    'appearance_id': appearance['id'],
                    'label_injury_within_21d': 1,
                    'injury_date': injury_date,
                    'injury_type': injury['injury_type'],
                    'injury_location': injury['injury_location'],
                    'injury_severity': injury['severity'],
                    'confidence_score': injury['confidence_score']
                })
        
        # Create negative cases (healthy periods)
        for _, injury in injuries_df.iterrows():
            pitcher_id = injury['pitcher_id']
            injury_date = injury['injury_date']
            
            pitcher_appearances = workload_df[workload_df['pitcher_id'] == pitcher_id].copy()
            pitcher_appearances = pitcher_appearances.sort_values('game_date')
            
            if len(pitcher_appearances) == 0:
                continue
            
            # Find healthy periods (at least 60 days before injury)
            healthy_start = injury_date - timedelta(days=120)
            healthy_end = injury_date - timedelta(days=60)
            
            healthy_appearances = pitcher_appearances[
                (pitcher_appearances['game_date'] >= healthy_start) & 
                (pitcher_appearances['game_date'] <= healthy_end)
            ]
            
            # Sample healthy appearances (to balance dataset)
            if len(healthy_appearances) > 0:
                sample_size = min(len(healthy_appearances), 3)  # Max 3 healthy samples per injury
                healthy_sample = healthy_appearances.sample(n=sample_size, random_state=42)
                
                for _, appearance in healthy_sample.iterrows():
                    labeled_data.append({
                        'pitcher_id': pitcher_id,
                        'pitcher_name': injury['pitcher_name'],
                        'game_date': appearance['game_date'],
                        'appearance_id': appearance['id'],
                        'label_injury_within_21d': 0,
                        'injury_date': None,
                        'injury_type': None,
                        'injury_location': None,
                        'injury_severity': None,
                        'confidence_score': 0.0
                    })
        
        labeled_df = pd.DataFrame(labeled_data)
        logger.info(f"Created {len(labeled_df)} labeled samples ({labeled_df['label_injury_within_21d'].sum()} positive, {len(labeled_df) - labeled_df['label_injury_within_21d'].sum()} negative)")
        
        return labeled_df
    
    def engineer_features(self, labeled_df: pd.DataFrame, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for full-scale testing"""
        logger.info("Engineering features for full-scale testing")
        
        # Convert dates
        labeled_df['game_date'] = pd.to_datetime(labeled_df['game_date'])
        pitches_df['game_date'] = pd.to_datetime(pitches_df['game_date'])
        
        # Initialize feature columns
        feature_data = []
        
        for _, row in labeled_df.iterrows():
            pitcher_id = row['pitcher_id']
            game_date = row['game_date']
            
            # Get pitcher's pitch data
            pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id].copy()
            
            # Calculate rolling window features
            features = self._calculate_rolling_features(pitcher_pitches, game_date)
            
            # Add basic features
            features.update({
                'pitcher_id': pitcher_id,
                'game_date': game_date,
                'label_injury_within_21d': row['label_injury_within_21d'],
                'injury_type': row['injury_type'],
                'injury_location': row['injury_location'],
                'injury_severity': row['injury_severity'],
                'confidence_score': row['confidence_score']
            })
            
            feature_data.append(features)
        
        feature_df = pd.DataFrame(feature_data)
        
        # Fill missing values
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_columns] = feature_df[numeric_columns].fillna(0)
        
        logger.info(f"Engineered {len(feature_df)} feature samples with {len(numeric_columns)} numeric features")
        
        return feature_df
    
    def _calculate_rolling_features(self, pitcher_pitches: pd.DataFrame, game_date: datetime) -> Dict:
        """Calculate rolling window features for a specific game date"""
        features = {}
        
        # Filter pitches before the game date
        historical_pitches = pitcher_pitches[pitcher_pitches['game_date'] < game_date].copy()
        
        if len(historical_pitches) == 0:
            # Return default values if no historical data
            return self._get_default_features()
        
        # Calculate rolling windows
        windows = [7, 14, 30, 60]
        
        for window in windows:
            window_start = game_date - timedelta(days=window)
            window_pitches = historical_pitches[historical_pitches['game_date'] >= window_start]
            
            if len(window_pitches) > 0:
                # Pitch count features
                features[f'pitch_count_{window}d'] = len(window_pitches)
                features[f'games_pitched_{window}d'] = window_pitches['game_date'].nunique()
                
                # Velocity features
                if 'release_speed' in window_pitches.columns:
                    features[f'avg_velocity_{window}d'] = window_pitches['release_speed'].mean()
                    features[f'max_velocity_{window}d'] = window_pitches['release_speed'].max()
                    features[f'velocity_std_{window}d'] = window_pitches['release_speed'].std()
                
                # Spin rate features
                if 'release_spin_rate' in window_pitches.columns:
                    features[f'avg_spin_rate_{window}d'] = window_pitches['release_spin_rate'].mean()
                    features[f'spin_rate_std_{window}d'] = window_pitches['release_spin_rate'].std()
                
                # Workload intensity
                features[f'pitches_per_game_{window}d'] = len(window_pitches) / window_pitches['game_date'].nunique() if window_pitches['game_date'].nunique() > 0 else 0
                
                # Rest days
                if len(window_pitches) > 1:
                    game_dates = sorted(window_pitches['game_date'].unique())
                    rest_days = [(game_dates[i] - game_dates[i-1]).days for i in range(1, len(game_dates))]
                    features[f'avg_rest_days_{window}d'] = np.mean(rest_days) if rest_days else 0
                    features[f'min_rest_days_{window}d'] = min(rest_days) if rest_days else 0
                else:
                    features[f'avg_rest_days_{window}d'] = 0
                    features[f'min_rest_days_{window}d'] = 0
            else:
                # No data in window
                features[f'pitch_count_{window}d'] = 0
                features[f'games_pitched_{window}d'] = 0
                features[f'avg_velocity_{window}d'] = 0
                features[f'max_velocity_{window}d'] = 0
                features[f'velocity_std_{window}d'] = 0
                features[f'avg_spin_rate_{window}d'] = 0
                features[f'spin_rate_std_{window}d'] = 0
                features[f'pitches_per_game_{window}d'] = 0
                features[f'avg_rest_days_{window}d'] = 0
                features[f'min_rest_days_{window}d'] = 0
        
        # Calculate velocity decline features
        if 'avg_velocity_7d' in features and 'avg_velocity_30d' in features:
            features['vel_decline_7d_vs_30d'] = features['avg_velocity_7d'] - features['avg_velocity_30d']
        
        # Calculate workload trends
        if 'pitch_count_7d' in features and 'pitch_count_30d' in features:
            features['workload_increase_7d_vs_30d'] = features['pitch_count_7d'] - (features['pitch_count_30d'] / 4)
        
        return features
    
    def _get_default_features(self) -> Dict:
        """Return default feature values when no historical data is available"""
        return {
            'pitch_count_7d': 0, 'games_pitched_7d': 0, 'avg_velocity_7d': 0, 'max_velocity_7d': 0,
            'velocity_std_7d': 0, 'avg_spin_rate_7d': 0, 'spin_rate_std_7d': 0, 'pitches_per_game_7d': 0,
            'avg_rest_days_7d': 0, 'min_rest_days_7d': 0,
            'pitch_count_14d': 0, 'games_pitched_14d': 0, 'avg_velocity_14d': 0, 'max_velocity_14d': 0,
            'velocity_std_14d': 0, 'avg_spin_rate_14d': 0, 'spin_rate_std_14d': 0, 'pitches_per_game_14d': 0,
            'avg_rest_days_14d': 0, 'min_rest_days_14d': 0,
            'pitch_count_30d': 0, 'games_pitched_30d': 0, 'avg_velocity_30d': 0, 'max_velocity_30d': 0,
            'velocity_std_30d': 0, 'avg_spin_rate_30d': 0, 'spin_rate_std_30d': 0, 'pitches_per_game_30d': 0,
            'avg_rest_days_30d': 0, 'min_rest_days_30d': 0,
            'pitch_count_60d': 0, 'games_pitched_60d': 0, 'avg_velocity_60d': 0, 'max_velocity_60d': 0,
            'velocity_std_60d': 0, 'avg_spin_rate_60d': 0, 'spin_rate_std_60d': 0, 'pitches_per_game_60d': 0,
            'avg_rest_days_60d': 0, 'min_rest_days_60d': 0,
            'vel_decline_7d_vs_30d': 0, 'workload_increase_7d_vs_30d': 0
        }
    
    def train_full_scale_model(self, feature_df: pd.DataFrame) -> Dict:
        """Train a full-scale model with enhanced data"""
        logger.info("Training full-scale model with enhanced injury data")
        
        # Prepare features
        feature_columns = [col for col in feature_df.columns if col not in [
            'pitcher_id', 'game_date', 'label_injury_within_21d', 'injury_type', 
            'injury_location', 'injury_severity', 'confidence_score'
        ]]
        
        X = feature_df[feature_columns]
        y = feature_df['label_injury_within_21d']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights
        class_weights = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=class_weights,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision_val = precision_score(y_test, y_pred)
        recall_val = recall_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model': model,
            'feature_importance': feature_importance,
            'metrics': {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'precision': precision_val,
                'recall': recall_val,
                'f1_score': f1_val
            },
            'feature_columns': feature_columns,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
        
        logger.info(f"Full-scale model trained successfully")
        logger.info(f"ROC-AUC: {roc_auc:.3f}, PR-AUC: {pr_auc:.3f}")
        logger.info(f"Precision: {precision_val:.3f}, Recall: {recall_val:.3f}, F1: {f1_val:.3f}")
        
        return results
    
    def analyze_results(self, results: Dict, feature_df: pd.DataFrame):
        """Analyze and visualize full-scale test results"""
        logger.info("Analyzing full-scale test results")
        
        metrics = results['metrics']
        feature_importance = results['feature_importance']
        
        print("\n" + "="*80)
        print("FULL-SCALE MODEL TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 MODEL PERFORMANCE METRICS:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        
        print(f"\n📈 DATASET STATISTICS:")
        print(f"  Total Samples: {len(feature_df)}")
        print(f"  Positive Cases: {feature_df['label_injury_within_21d'].sum()}")
        print(f"  Negative Cases: {len(feature_df) - feature_df['label_injury_within_21d'].sum()}")
        print(f"  Positive Rate: {(feature_df['label_injury_within_21d'].sum() / len(feature_df)) * 100:.1f}%")
        
        print(f"\n🏥 INJURY ANALYSIS:")
        injury_types = feature_df['injury_type'].value_counts()
        print(f"  Injury Types: {dict(injury_types)}")
        
        injury_locations = feature_df['injury_location'].value_counts()
        print(f"  Injury Locations: {dict(injury_locations)}")
        
        print(f"\n🔍 TOP FEATURES:")
        top_features = feature_importance.head(10)
        for _, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Create visualizations
        self._create_visualizations(results, feature_df)
        
        # Save results
        self._save_results(results, feature_df)
    
    def _create_visualizations(self, results: Dict, feature_df: pd.DataFrame):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Full-Scale Model Test Results', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        axes[0, 0].plot(fpr, tpr, label=f'ROC-AUC = {results["metrics"]["roc_auc"]:.3f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. PR Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
        axes[0, 1].plot(recall, precision, label=f'PR-AUC = {results["metrics"]["pr_auc"]:.3f}')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Feature Importance
        top_features = results['feature_importance'].head(10)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 10 Feature Importance')
        
        # 4. Prediction Distribution
        axes[1, 1].hist(results['y_pred_proba'], bins=20, alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Distribution')
        
        plt.tight_layout()
        plt.savefig('full_scale_test_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualizations saved as 'full_scale_test_results.png'")
    
    def _save_results(self, results: Dict, feature_df: pd.DataFrame):
        """Save test results and model"""
        # Save model
        with open('full_scale_model.pkl', 'wb') as f:
            pickle.dump(results['model'], f)
        
        # Save results summary
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': results['metrics'],
            'dataset_stats': {
                'total_samples': len(feature_df),
                'positive_cases': feature_df['label_injury_within_21d'].sum(),
                'negative_cases': len(feature_df) - feature_df['label_injury_within_21d'].sum(),
                'positive_rate': (feature_df['label_injury_within_21d'].sum() / len(feature_df)) * 100
            },
            'top_features': results['feature_importance'].head(10).to_dict('records')
        }
        
        import json
        with open('full_scale_test_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  Model: full_scale_model.pkl")
        print(f"  Summary: full_scale_test_results.json")
    
    def run_full_scale_test(self):
        """Run complete full-scale model testing"""
        logger.info("Starting full-scale model testing")
        
        try:
            # Load data
            injuries_df, workload_df, pitches_df, pitchers_df = self.load_enhanced_data()
            
            if len(injuries_df) == 0:
                logger.error("No injury data available for testing")
                return
            
            # Create labels
            labeled_df = self.create_injury_labels(injuries_df, workload_df)
            
            if len(labeled_df) == 0:
                logger.error("No labeled data created")
                return
            
            # Engineer features
            feature_df = self.engineer_features(labeled_df, pitches_df)
            
            # Train model
            results = self.train_full_scale_model(feature_df)
            
            # Analyze results
            self.analyze_results(results, feature_df)
            
            logger.info("Full-scale model testing completed successfully")
            
        except Exception as e:
            logger.error(f"Error in full-scale testing: {e}")
            raise

def main():
    """Main execution function"""
    tester = FullScaleModelTester()
    tester.run_full_scale_test()

if __name__ == "__main__":
    main()
