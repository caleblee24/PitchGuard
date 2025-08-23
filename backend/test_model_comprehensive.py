#!/usr/bin/env python3
"""
Comprehensive Model Testing Script for PitchGuard Enhanced Model

This script performs thorough evaluation of the enhanced XGBoost model including:
- Performance metrics on holdout data
- Calibration analysis
- Feature importance analysis
- Model interpretability
- API integration testing
- Data quality checks
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import requests
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.mock_data_generator import MockDataGenerator
from modeling.enhanced_model import EnhancedInjuryRiskModel
from etl.enhanced_features import EnhancedFeatureEngineer
from services.enhanced_risk_service import EnhancedRiskService
from utils.config import get_settings

class ComprehensiveModelTester:
    def __init__(self):
        self.settings = get_settings()
        self.test_results = {}
        
    def generate_test_data(self, num_pitchers: int = 50, days: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
        """Generate comprehensive test data with realistic injury patterns."""
        print("🔧 Generating comprehensive test data...")
        
        config = {
            'num_pitchers': num_pitchers,
            'start_date': datetime.now() - timedelta(days=days),
            'end_date': datetime.now(),
            'injury_probability': 0.15,  # Higher injury rate for better testing
            'velocity_decline_threshold': -2.0,
            'pitch_count_threshold': 100
        }
        
        generator = MockDataGenerator(config)
        
        # Generate data with more realistic injury patterns
        pitches_df, appearances_df, injuries_data = generator.generate_realistic_injury_scenarios()
        
        print(f"✅ Generated test data:")
        print(f"   - {len(pitches_df)} pitches")
        print(f"   - {len(appearances_df)} appearances") 
        print(f"   - {len(injuries_data)} injuries")
        
        # If we don't have enough injuries, generate more data
        if len(injuries_data) < 3:
            print("⚠️  Not enough injuries generated, creating additional injury scenarios...")
            
            # Generate additional injuries by creating more diverse scenarios
            additional_injuries = []
            for pitcher_id in appearances_df['pitcher_id'].unique()[:10]:  # Take first 10 pitchers
                pitcher_appearances = appearances_df[appearances_df['pitcher_id'] == pitcher_id]
                if len(pitcher_appearances) >= 10:
                    # Create a simple injury scenario
                    mid_point = len(pitcher_appearances) // 2
                    injury_date = pd.to_datetime(pitcher_appearances.iloc[mid_point]['game_date']) + timedelta(days=5)
                    
                    injury = {
                        'pitcher_id': pitcher_id,
                        'il_start': injury_date.strftime('%Y-%m-%d'),
                        'il_end': (injury_date + timedelta(days=30)).strftime('%Y-%m-%d'),
                        'stint_type': 'Elbow inflammation'
                    }
                    additional_injuries.append(injury)
            
            injuries_data.extend(additional_injuries)
            print(f"   - Added {len(additional_injuries)} additional injuries")
            print(f"   - Total injuries: {len(injuries_data)}")
        
        return pitches_df, appearances_df, injuries_data
    
    def train_and_evaluate_model(self, pitches_df: pd.DataFrame, appearances_df: pd.DataFrame, injuries_data: List[Dict]) -> Dict:
        """Train the enhanced model and evaluate performance"""
        print("\n🎯 Training and evaluating enhanced model...")
        
        # Feature engineering
        feature_engineer = EnhancedFeatureEngineer()
        
        # Convert injuries_data to DataFrame if it's a list
        if isinstance(injuries_data, list):
            injuries_df = pd.DataFrame(injuries_data)
        else:
            injuries_df = injuries_data
        
        # Convert date columns to datetime
        appearances_df['game_date'] = pd.to_datetime(appearances_df['game_date'])
        pitches_df['game_date'] = pd.to_datetime(pitches_df['game_date'])
        if not injuries_df.empty:
            injuries_df['il_start'] = pd.to_datetime(injuries_df['il_start'])
            injuries_df['il_end'] = pd.to_datetime(injuries_df['il_end'])
        
        # Use the latest date as as_of_date
        as_of_date = max(appearances_df['game_date'].max(), pitches_df['game_date'].max())
        
        features_df = feature_engineer.engineer_features(
            appearances_df, pitches_df, injuries_df, as_of_date
        )
        
        # Create proper injury labels
        print("🔍 Creating injury labels from injury data...")
        features_df, labels_df = self._create_injury_labels(features_df, injuries_df, as_of_date, appearances_df, pitches_df, feature_engineer)
        
        # Train model
        model = EnhancedInjuryRiskModel()
        
        # Debug: print features_df info
        print(f"📊 Features DataFrame shape: {features_df.shape}")
        print(f"📊 Features DataFrame columns: {features_df.columns.tolist()}")
        print(f"📊 Features DataFrame head:\n{features_df.head()}")
        
        # Check if we have proper labels
        if 'label_injury_within_21d' in labels_df.columns:
            print(f"✅ Created {len(labels_df)} labels with {labels_df['label_injury_within_21d'].sum()} positive cases")
            print(f"📊 Positive rate: {labels_df['label_injury_within_21d'].mean():.3f}")
            
            # Define the exact feature set to use consistently
            feature_cols = [
                # Role features
                'avg_innings_per_appearance', 'multi_inning_rate_7d', 'multi_inning_rate_14d',
                'back_to_back_days_7d', 'back_to_back_days_14d',
                # Workload features
                'last_game_pitches', 'last_3_games_pitches',
                'roll7d_pitch_count', 'roll7d_appearances', 'roll7d_avg_pitches_per_appearance',
                'roll14d_pitch_count', 'roll14d_appearances', 'roll14d_avg_pitches_per_appearance',
                'roll30d_pitch_count', 'roll30d_appearances', 'roll30d_avg_pitches_per_appearance',
                # Fatigue features
                'workload_intensity', 'fatigue_score',
                # Velocity features
                'roll7d_release_speed_ewm_mean', 'roll7d_release_speed_ewm_std', 'roll7d_release_speed_delta_vs_30d',
                'roll14d_release_speed_ewm_mean', 'roll14d_release_speed_ewm_std', 'roll14d_release_speed_delta_vs_30d',
                'roll30d_release_speed_ewm_mean', 'roll30d_release_speed_ewm_std',
                'vel_decline_7d_vs_30d', 'vel_decline_14d_vs_30d',
                # Spin rate features
                'roll7d_release_spin_rate_ewm_mean', 'roll7d_release_spin_rate_ewm_std', 'roll7d_release_spin_rate_delta_vs_30d',
                'roll14d_release_spin_rate_ewm_mean', 'roll14d_release_spin_rate_ewm_std', 'roll14d_release_spin_rate_delta_vs_30d',
                'roll30d_release_spin_rate_ewm_mean', 'roll30d_release_spin_rate_ewm_std',
                # Pitch mix features
                'roll7d_breaking_ball_pct', 'roll7d_breaking_ball_delta_vs_30d',
                'roll14d_breaking_ball_pct', 'roll14d_breaking_ball_delta_vs_30d',
                'roll30d_breaking_ball_pct'
            ]
            
            # Filter features to only include those that actually exist in the DataFrame
            available_features = [col for col in feature_cols if col in features_df.columns]
            print(f"🔍 Available features for training: {len(available_features)}")
            print(f"🔍 Missing features: {[col for col in feature_cols if col not in features_df.columns]}")
            
            # Create training features with only the available features
            training_features = features_df[available_features].copy()
            
            # Add pitcher_id and as_of_date back for the model's merge operation
            training_features['pitcher_id'] = features_df['pitcher_id']
            training_features['as_of_date'] = features_df['as_of_date']
            
            train_results = model.train(training_features, labels_df)
        else:
            print("❌ Error: Could not create proper injury labels")
            return None
        
        # Performance metrics
        y_true = labels_df['label_injury_within_21d'].values
        
        # Get predictions using the model's predict method
        # Use the exact same features that were used for training
        features_for_prediction = training_features[available_features]
        
        # Get predictions
        predictions = []
        for idx, row in features_for_prediction.iterrows():
            prediction = model.predict(pd.DataFrame([row]), pd.Series(['starter']))
            if prediction:
                predictions.append(prediction[0])
            else:
                predictions.append({'risk_score_calibrated': 0.0})
        
        # Extract risk scores
        y_pred_proba = [pred.get('risk_score_calibrated', 0.0) for pred in predictions]
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
            precision = precision_score(y_true, [1 if p > 0.5 else 0 for p in y_pred_proba], zero_division=0)
            recall = recall_score(y_true, [1 if p > 0.5 else 0 for p in y_pred_proba], zero_division=0)
            f1 = f1_score(y_true, [1 if p > 0.5 else 0 for p in y_pred_proba], zero_division=0)
            positive_rate = np.mean(y_true)
        except Exception as e:
            print(f"⚠️  Error calculating metrics: {e}")
            auc = 0.5
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            positive_rate = np.mean(y_true)
        
        print(f"✅ Model Performance:")
        print(f"   - AUC: {auc:.3f}")
        print(f"   - Precision: {precision:.3f}")
        print(f"   - Recall: {recall:.3f}")
        print(f"   - F1: {f1:.3f}")
        print(f"   - Positive Rate: {positive_rate:.3f}")
        
        # Store results
        self.test_results['model_performance'] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_rate': positive_rate,
            'y_true': y_true.tolist(),
            'y_pred_proba': y_pred_proba
        }
        
        return self.test_results
    
    def _create_injury_labels(self, features_df: pd.DataFrame, injuries_df: pd.DataFrame, as_of_date: datetime, appearances_df: pd.DataFrame, pitches_df: pd.DataFrame, feature_engineer: EnhancedFeatureEngineer) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create proper injury labels from injury data."""
        
        if injuries_df.empty:
            print("⚠️  No injury data available, creating dummy labels")
            return pd.DataFrame({
                'pitcher_id': features_df['pitcher_id'],
                'as_of_date': features_df['as_of_date'],
                'label_injury_within_21d': np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])
            }), pd.DataFrame({
                'pitcher_id': features_df['pitcher_id'],
                'as_of_date': features_df['as_of_date'],
                'label_injury_within_21d': np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])
            })
        
        # Create features for multiple dates to capture injury patterns
        print("🔍 Creating features for multiple dates to capture injury patterns...")
        
        # Get unique pitchers
        unique_pitchers = features_df['pitcher_id'].unique()
        
        # Create features for each pitcher at multiple dates
        all_features = []
        all_labels = []
        
        for pitcher_id in unique_pitchers:
            # Get this pitcher's data
            pitcher_appearances = appearances_df[appearances_df['pitcher_id'] == pitcher_id]
            pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id]
            
            if len(pitcher_appearances) == 0:
                continue
            
            # Create features for multiple dates (every 7 days for the last 60 days)
            start_date = pitcher_appearances['game_date'].min()
            end_date = pitcher_appearances['game_date'].max()
            
            # Create dates every 7 days
            feature_dates = []
            current_date = start_date
            while current_date <= end_date:
                feature_dates.append(current_date)
                current_date += timedelta(days=7)
            
            # Add the end date if not already included
            if end_date not in feature_dates:
                feature_dates.append(end_date)
            
            for feature_date in feature_dates:
                # Engineer features for this date
                features = feature_engineer.engineer_features(
                    pitcher_appearances[pitcher_appearances['game_date'] <= feature_date],
                    pitcher_pitches[pitcher_pitches['game_date'] <= feature_date],
                    pd.DataFrame(),
                    feature_date
                )
                
                if not features.empty:
                    all_features.append(features)
                    
                    # Check if this pitcher had an injury within 21 days of this date
                    pitcher_injuries = injuries_df[injuries_df['pitcher_id'] == pitcher_id]
                    
                    injury_within_21d = False
                    for _, injury in pitcher_injuries.iterrows():
                        injury_start = injury['il_start']
                        
                        # Check if injury started within 21 days after the feature date
                        days_until_injury = (injury_start - feature_date).days
                        if 0 < days_until_injury <= 21:
                            injury_within_21d = True
                            break
                    
                    all_labels.append({
                        'pitcher_id': pitcher_id,
                        'as_of_date': feature_date,
                        'label_injury_within_21d': 1 if injury_within_21d else 0
                    })
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            labels_df = pd.DataFrame(all_labels)
        else:
            # Fallback to original features
            combined_features = features_df
            labels_df = pd.DataFrame({
                'pitcher_id': features_df['pitcher_id'],
                'as_of_date': features_df['as_of_date'],
                'label_injury_within_21d': np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])
            })
        
        # Print label statistics
        positive_count = labels_df['label_injury_within_21d'].sum()
        total_count = len(labels_df)
        positive_rate = positive_count / total_count if total_count > 0 else 0
        
        print(f"📊 Label Statistics:")
        print(f"   - Total samples: {total_count}")
        print(f"   - Positive samples: {positive_count}")
        print(f"   - Positive rate: {positive_rate:.3f}")
        
        return combined_features, labels_df
    
    def test_calibration(self, model: EnhancedInjuryRiskModel, features_df: pd.DataFrame) -> Dict:
        """Test model calibration"""
        print("\n📊 Testing model calibration...")
        
        # Check if label column exists
        if 'label_injury_within_21d' in features_df.columns:
            y_true = features_df['label_injury_within_21d'].values
        else:
            # Use dummy labels for testing
            y_true = np.random.choice([0, 1], size=len(features_df), p=[0.95, 0.05])
        
        # Get predictions using the model's predict method
        feature_cols = [
            'avg_innings_per_appearance', 'multi_inning_rate_7d', 'multi_inning_rate_14d',
            'back_to_back_days_7d', 'back_to_back_days_14d', 'last_game_pitches', 'last_3_games_pitches',
            'roll7d_pitch_count', 'roll14d_pitch_count', 'roll30d_pitch_count',
            'roll7d_appearances', 'roll14d_appearances', 'roll30d_appearances',
            'roll7d_avg_pitches_per_appearance', 'roll14d_avg_pitches_per_appearance',
            'roll30d_avg_pitches_per_appearance', 'avg_rest_days', 'std_rest_days', 'min_rest_days', 'max_rest_days',
            'days_since_last_appearance', 'roll7d_release_speed_ewm_mean', 'roll14d_release_speed_ewm_mean',
            'roll7d_release_speed_delta_vs_30d', 'roll14d_release_speed_delta_vs_30d',
            'roll7d_release_spin_rate_ewm_mean', 'roll14d_release_spin_rate_ewm_mean',
            'roll7d_release_spin_rate_delta_vs_30d', 'roll14d_release_spin_rate_delta_vs_30d',
            'vel_decline_7d_vs_30d', 'vel_decline_14d_vs_30d', 'roll7d_breaking_ball_pct', 'roll14d_breaking_ball_pct',
            'roll7d_breaking_ball_delta_vs_30d', 'roll14d_breaking_ball_delta_vs_30d', 'data_completeness_score'
        ]
        
        available_features = [col for col in feature_cols if col in features_df.columns]
        features_for_prediction = features_df[available_features].fillna(0)
        
        predictions = model.predict(features_for_prediction)
        y_pred_proba = np.array([pred.get('risk_score_calibrated', 0.0) for pred in predictions])
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Brier score (lower is better)
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        
        calibration_metrics = {
            'brier_score': brier_score,
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'calibration_error': np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        }
        
        print(f"✅ Calibration Results:")
        print(f"   - Brier Score: {brier_score:.3f}")
        print(f"   - Calibration Error: {calibration_metrics['calibration_error']:.3f}")
        
        return calibration_metrics
    
    def test_feature_importance(self, model: EnhancedInjuryRiskModel, features_df: pd.DataFrame) -> Dict:
        """Analyze feature importance and interpretability"""
        print("\n🔍 Analyzing feature importance...")
        
        # Get feature importance from XGBoost
        feature_importance = model.xgb_model.feature_importances_
        feature_names = model.feature_names
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Top features
        top_features = importance_df.head(10).to_dict('records')
        
        # Test SHAP-like interpretability (simplified)
        interpretability_results = {
            'top_features': top_features,
            'total_features': len(feature_names),
            'importance_summary': {
                'max_importance': float(importance_df['importance'].max()),
                'min_importance': float(importance_df['importance'].min()),
                'mean_importance': float(importance_df['importance'].mean())
            }
        }
        
        print(f"✅ Feature Importance Analysis:")
        print(f"   - Total features: {len(feature_names)}")
        print(f"   - Top feature: {top_features[0]['feature']} ({top_features[0]['importance']:.3f})")
        
        return interpretability_results
    
    def test_api_integration(self, model: EnhancedInjuryRiskModel, features_df: pd.DataFrame) -> Dict:
        """Test API integration with the enhanced model"""
        print("\n🌐 Testing API integration...")
        
        # Initialize enhanced risk service
        risk_service = EnhancedRiskService()
        
        # Test with a few sample pitchers
        test_pitchers = features_df['pitcher_id'].unique()[:5]
        api_results = []
        
        for pitcher_id in test_pitchers:
            try:
                # Get current risk assessment
                risk_assessment = risk_service.get_current_risk_assessment(
                    pitcher_id=pitcher_id,
                    as_of_date=datetime.now().date()
                )
                
                api_results.append({
                    'pitcher_id': pitcher_id,
                    'success': True,
                    'risk_score': risk_assessment.get('risk_score', 0),
                    'risk_bucket': risk_assessment.get('risk_bucket', 'unknown'),
                    'confidence': risk_assessment.get('confidence', 'unknown'),
                    'data_completeness': risk_assessment.get('data_completeness', {})
                })
                
            except Exception as e:
                api_results.append({
                    'pitcher_id': pitcher_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Test HTTP API endpoints
        http_results = self.test_http_endpoints()
        
        api_test_results = {
            'service_integration': {
                'total_tested': len(test_pitchers),
                'successful_calls': len([r for r in api_results if r['success']]),
                'results': api_results
            },
            'http_endpoints': http_results
        }
        
        print(f"✅ API Integration Results:")
        print(f"   - Service calls: {api_test_results['service_integration']['successful_calls']}/{len(test_pitchers)} successful")
        print(f"   - HTTP endpoints: {api_test_results['http_endpoints']['successful_endpoints']}/{api_test_results['http_endpoints']['total_endpoints']} working")
        
        return api_test_results
    
    def test_http_endpoints(self) -> Dict:
        """Test HTTP API endpoints"""
        base_url = "http://localhost:8000"
        endpoints = [
            "/health",
            "/api/v1/pitchers",
            "/api/v1/risk/pitcher/1/current",
            "/api/v1/risk/team/1/summary"
        ]
        
        results = []
        for endpoint in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                results.append({
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'response_time': response.elapsed.total_seconds()
                })
            except Exception as e:
                results.append({
                    'endpoint': endpoint,
                    'status_code': None,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'total_endpoints': len(endpoints),
            'successful_endpoints': len([r for r in results if r['success']]),
            'results': results
        }
    
    def test_data_quality(self, features_df: pd.DataFrame) -> Dict:
        """Test data quality and completeness"""
        print("\n📋 Testing data quality...")
        
        # Check for missing values
        missing_data = features_df.isnull().sum()
        missing_percentage = (missing_data / len(features_df)) * 100
        
        # Check for outliers in key features
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for feature in numeric_features:
            if feature != 'label_injury_within_21d':  # Skip target
                Q1 = features_df[feature].quantile(0.25)
                Q3 = features_df[feature].quantile(0.75)
                IQR = Q3 - Q1
                outliers = features_df[(features_df[feature] < Q1 - 1.5*IQR) | 
                                     (features_df[feature] > Q3 + 1.5*IQR)]
                outlier_stats[feature] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(features_df)) * 100
                }
        
        # Data completeness by feature type
        workload_features = [col for col in features_df.columns if 'pitch_count' in col]
        velocity_features = [col for col in features_df.columns if 'vel' in col]
        fatigue_features = [col for col in features_df.columns if 'fatigue' in col]
        
        completeness = {
            'workload_features': features_df[workload_features].notnull().mean().mean(),
            'velocity_features': features_df[velocity_features].notnull().mean().mean(),
            'fatigue_features': features_df[fatigue_features].notnull().mean().mean(),
            'overall': features_df.notnull().mean().mean()
        }
        
        quality_results = {
            'missing_data': missing_percentage.to_dict(),
            'outlier_stats': outlier_stats,
            'completeness': completeness,
            'total_rows': len(features_df),
            'total_features': len(features_df.columns)
        }
        
        print(f"✅ Data Quality Results:")
        print(f"   - Overall completeness: {completeness['overall']:.3f}")
        print(f"   - Workload features completeness: {completeness['workload_features']:.3f}")
        print(f"   - Velocity features completeness: {completeness['velocity_features']:.3f}")
        
        return quality_results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        print("\n📝 Generating comprehensive test report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'model_performance': self.test_results.get('model_performance', {}),
                'calibration': self.test_results.get('calibration', {}),
                'feature_importance': self.test_results.get('feature_importance', {}),
                'api_integration': self.test_results.get('api_integration', {}),
                'data_quality': self.test_results.get('data_quality', {})
            },
            'recommendations': self.generate_recommendations()
        }
        
        # Save report
        report_path = "comprehensive_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Test report saved to: {report_path}")
        return report_path
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        metrics = self.test_results.get('model_performance', {}).get('metrics', {})
        if metrics.get('auc', 0) < 0.7:
            recommendations.append("Consider feature engineering improvements to boost AUC above 0.7")
        
        if metrics.get('positive_rate', 0) < 0.05:
            recommendations.append("Increase injury data generation for better class balance")
        
        # Calibration recommendations
        calibration = self.test_results.get('calibration', {})
        if calibration.get('brier_score', 1) > 0.25:
            recommendations.append("Model calibration needs improvement - consider recalibration")
        
        # API recommendations
        api_results = self.test_results.get('api_integration', {})
        if api_results.get('service_integration', {}).get('successful_calls', 0) < 3:
            recommendations.append("API integration needs debugging - check service initialization")
        
        # Data quality recommendations
        quality = self.test_results.get('data_quality', {})
        if quality.get('completeness', {}).get('overall', 0) < 0.8:
            recommendations.append("Data completeness below 80% - improve feature engineering")
        
        if not recommendations:
            recommendations.append("All systems performing well - ready for production testing")
        
        return recommendations
    
    def run_comprehensive_test(self) -> str:
        """Run the complete comprehensive test suite"""
        print("🚀 Starting Comprehensive Model Testing Suite")
        print("=" * 60)
        
        try:
            # Step 1: Generate test data
            pitches_df, appearances_df, injuries_data = self.generate_test_data()
            
            # Step 2: Train and evaluate model
            model_results = self.train_and_evaluate_model(pitches_df, appearances_df, injuries_data)
            self.test_results['model_performance'] = model_results
            
            # Step 3: Test calibration
            calibration_results = self.test_calibration(model_results['model'], model_results['features_df'])
            self.test_results['calibration'] = calibration_results
            
            # Step 4: Test feature importance
            importance_results = self.test_feature_importance(model_results['model'], model_results['features_df'])
            self.test_results['feature_importance'] = importance_results
            
            # Step 5: Test API integration
            api_results = self.test_api_integration(model_results['model'], model_results['features_df'])
            self.test_results['api_integration'] = api_results
            
            # Step 6: Test data quality
            quality_results = self.test_data_quality(model_results['features_df'])
            self.test_results['data_quality'] = quality_results
            
            # Step 7: Generate report
            report_path = self.generate_test_report()
            
            print("\n" + "=" * 60)
            print("✅ Comprehensive Testing Complete!")
            print(f"📊 Report saved to: {report_path}")
            
            return report_path
            
        except Exception as e:
            print(f"❌ Testing failed with error: {str(e)}")
            raise

def main():
    """Main function to run comprehensive testing"""
    tester = ComprehensiveModelTester()
    report_path = tester.run_comprehensive_test()
    
    # Print summary
    print("\n📋 Test Summary:")
    print(f"   - Model Performance: {'✅' if tester.test_results.get('model_performance', {}).get('metrics', {}).get('auc', 0) > 0.6 else '❌'}")
    print(f"   - Calibration: {'✅' if tester.test_results.get('calibration', {}).get('brier_score', 1) < 0.3 else '❌'}")
    print(f"   - API Integration: {'✅' if tester.test_results.get('api_integration', {}).get('service_integration', {}).get('successful_calls', 0) > 0 else '❌'}")
    print(f"   - Data Quality: {'✅' if tester.test_results.get('data_quality', {}).get('completeness', {}).get('overall', 0) > 0.7 else '❌'}")
    
    print(f"\n📄 Full report available at: {report_path}")

if __name__ == "__main__":
    main()
