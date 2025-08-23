#!/usr/bin/env python3
"""
Analyze Enhanced Model Accuracy

This script provides a comprehensive analysis of the enhanced PitchGuard model's
accuracy metrics using real MLB data from 2022-2024.
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.connection import get_db_session
from database.models import Pitcher, Pitch, Appearance, Injury
from etl.enhanced_features import EnhancedFeatureEngineer
from modeling.enhanced_model import EnhancedInjuryRiskModel
from utils.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelAccuracyAnalyzer:
    """Analyzer for enhanced model accuracy metrics."""
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engineer = EnhancedFeatureEngineer()
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the trained enhanced model."""
        try:
            model_path = "models/enhanced_model_real_data.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Loaded enhanced model: {self.model.model_version}")
            else:
                logger.error("Enhanced model not found!")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        return True
    
    def load_test_data(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """Load test data for accuracy analysis."""
        logger.info(f"Loading test data from {start_date} to {end_date}")
        
        # Load data from database
        from sqlalchemy import create_engine, text
        
        engine = create_engine(self.settings.database_url)
        query = """
        SELECT 
            p.pitcher_id,
            p.game_date,
            p.pitch_type,
            p.release_speed,
            p.release_spin_rate,
            p.pitch_number,
            p.at_bat_id,
            a.pitches_thrown,
            a.avg_vel,
            a.avg_spin,
            a.vel_std,
            a.spin_std,
            a.outs_recorded,
            a.innings_pitched
        FROM pitches p
        LEFT JOIN appearances a ON p.pitcher_id = a.pitcher_id 
            AND p.game_date = a.game_date
        WHERE p.game_date BETWEEN :start_date AND :end_date
        ORDER BY p.pitcher_id, p.game_date, p.at_bat_id, p.pitch_number
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={'start_date': start_date, 'end_date': end_date}
            )
        
        # Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        logger.info(f"Loaded {len(df):,} pitch records")
        return df
    
    def generate_injury_labels(self, df: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
        """Generate injury labels based on appearance patterns."""
        logger.info("Generating injury labels...")
        
        # Group by pitcher and date to get daily appearance counts
        daily_appearances = df.groupby(['pitcher_id', 'game_date']).agg({
            'pitches_thrown': 'sum',
            'innings_pitched': 'sum'
        }).reset_index()
        
        # Sort by pitcher and date
        daily_appearances = daily_appearances.sort_values(['pitcher_id', 'game_date'])
        
        # Create injury labels based on gaps in appearances
        injury_labels = []
        
        for pitcher_id in daily_appearances['pitcher_id'].unique():
            pitcher_data = daily_appearances[daily_appearances['pitcher_id'] == pitcher_id].copy()
            
            # Calculate days between appearances
            pitcher_data['next_appearance'] = pitcher_data['game_date'].shift(-1)
            pitcher_data['days_between'] = (pitcher_data['next_appearance'] - pitcher_data['game_date']).dt.days
            
            # Label as injury if gap > lookback_days (indicating potential IL stint)
            pitcher_data['injury_within_21d'] = (pitcher_data['days_between'] > lookback_days).astype(int)
            
            injury_labels.append(pitcher_data)
        
        injury_df = pd.concat(injury_labels, ignore_index=True)
        
        # Merge back with original data
        result_df = df.merge(
            injury_df[['pitcher_id', 'game_date', 'injury_within_21d']],
            on=['pitcher_id', 'game_date'],
            how='left'
        )
        
        # Fill missing labels with 0
        result_df['injury_within_21d'] = result_df['injury_within_21d'].fillna(0)
        
        injury_rate = result_df['injury_within_21d'].mean()
        logger.info(f"Injury rate: {injury_rate:.3f} ({injury_rate*100:.1f}%)")
        
        return result_df
    
    def prepare_test_data(self, df: pd.DataFrame) -> tuple:
        """Prepare test data with enhanced features."""
        logger.info("Preparing test data with enhanced features...")
        
        # First, we need to prepare the data in the format expected by the feature engineer
        appearances_df = df.groupby(['pitcher_id', 'game_date']).agg({
            'pitches_thrown': 'sum',
            'avg_vel': 'mean',
            'avg_spin': 'mean',
            'vel_std': 'mean',
            'spin_std': 'mean',
            'outs_recorded': 'sum',
            'innings_pitched': 'sum'
        }).reset_index()
        
        pitches_df = df[['pitcher_id', 'game_date', 'pitch_type', 'release_speed', 
                        'release_spin_rate', 'pitch_number', 'at_bat_id']].copy()
        
        # Create empty injuries DataFrame (we'll use appearance gaps as injury proxy)
        injuries_df = pd.DataFrame(columns=['pitcher_id', 'injury_date', 'injury_type'])
        
        # Use the latest date as as_of_date
        as_of_date = df['game_date'].max()
        
        # Generate enhanced features
        features_df = self.feature_engineer.engineer_features(
            appearances_df, pitches_df, injuries_df, as_of_date
        )
        
        # Create labels (use the last appearance for each pitcher as prediction point)
        labels_df = df.groupby('pitcher_id').agg({
            'injury_within_21d': 'max',
            'game_date': 'max'
        }).reset_index()
        
        labels_df = labels_df.rename(columns={
            'game_date': 'as_of_date',
            'injury_within_21d': 'label_injury_within_21d'
        })
        
        logger.info(f"Prepared {len(features_df)} feature records")
        logger.info(f"Label distribution: {labels_df['label_injury_within_21d'].value_counts().to_dict()}")
        
        return features_df, labels_df
    
    def calculate_accuracy_metrics(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
        """Calculate comprehensive accuracy metrics."""
        logger.info("Calculating accuracy metrics...")
        
        # Split data for evaluation
        train_features, test_features, train_labels, test_labels = train_test_split(
            features_df, labels_df, test_size=0.3, random_state=42, stratify=labels_df['label_injury_within_21d']
        )
        
        # Train model on training set
        self.model.train(train_features, train_labels)
        
        # Predict on test set
        predictions = self.model.predict(test_features)
        
        # Extract predicted probabilities and convert to binary predictions
        y_true = test_labels['label_injury_within_21d'].values
        y_pred_proba = predictions['risk_score_calibrated'].values
        
        # Use 0.5 threshold for binary predictions
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # AUC metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Brier Score (calibration quality)
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['confusion_matrix'] = {
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1]
        }
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Top-K metrics
        k_values = [5, 10, 20, 50]
        top_k_metrics = {}
        
        for k in k_values:
            if k <= len(y_true):
                top_k_indices = np.argsort(y_pred_proba)[-k:]
                top_k_labels = y_true[top_k_indices]
                top_k_metrics[f'recall_at_{k}'] = np.mean(top_k_labels)
                top_k_metrics[f'precision_at_{k}'] = np.sum(top_k_labels) / k if k > 0 else 0
        
        metrics['top_k_metrics'] = top_k_metrics
        
        # Data summary
        metrics['data_summary'] = {
            'total_samples': len(y_true),
            'positive_samples': np.sum(y_true),
            'negative_samples': len(y_true) - np.sum(y_true),
            'positive_rate': np.mean(y_true),
            'test_samples': len(test_labels)
        }
        
        return metrics
    
    def print_accuracy_report(self, metrics: dict):
        """Print comprehensive accuracy report."""
        print("\n" + "=" * 80)
        print("🎯 ENHANCED PITCHGUARD MODEL ACCURACY REPORT")
        print("=" * 80)
        print(f"Model Version: {self.model.model_version}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Data Summary
        print("📊 DATA SUMMARY")
        print("-" * 40)
        summary = metrics['data_summary']
        print(f"Total Test Samples: {summary['total_samples']:,}")
        print(f"Positive Cases (Injuries): {summary['positive_samples']:,} ({summary['positive_rate']:.1%})")
        print(f"Negative Cases (No Injuries): {summary['negative_samples']:,} ({(1-summary['positive_rate']):.1%})")
        print()
        
        # Core Accuracy Metrics
        print("🎯 CORE ACCURACY METRICS")
        print("-" * 40)
        print(f"Accuracy:           {metrics['accuracy']:.3f} ({metrics['accuracy']:.1%})")
        print(f"Precision:          {metrics['precision']:.3f} ({metrics['precision']:.1%})")
        print(f"Recall (Sensitivity): {metrics['recall']:.3f} ({metrics['recall']:.1%})")
        print(f"Specificity:        {metrics['specificity']:.3f} ({metrics['specificity']:.1%})")
        print(f"F1-Score:           {metrics['f1_score']:.3f}")
        print()
        
        # AUC Metrics
        print("📈 AUC METRICS")
        print("-" * 40)
        print(f"ROC AUC:            {metrics['roc_auc']:.3f}")
        print(f"Precision-Recall AUC: {metrics['pr_auc']:.3f}")
        print(f"Brier Score:        {metrics['brier_score']:.3f}")
        print()
        
        # Confusion Matrix
        print("🔍 CONFUSION MATRIX")
        print("-" * 40)
        cm = metrics['confusion_matrix']
        print(f"True Negatives:     {cm['true_negatives']:,}")
        print(f"False Positives:    {cm['false_positives']:,}")
        print(f"False Negatives:    {cm['false_negatives']:,}")
        print(f"True Positives:     {cm['true_positives']:,}")
        print()
        
        # Top-K Metrics
        print("🏆 TOP-K PERFORMANCE")
        print("-" * 40)
        for k, recall in metrics['top_k_metrics'].items():
            if k.startswith('recall_at_'):
                k_val = k.split('_')[-1]
                precision = metrics['top_k_metrics'].get(f'precision_at_{k_val}', 0)
                print(f"Top {k_val:>2}: Recall={recall:.3f} ({recall:.1%}), Precision={precision:.3f} ({precision:.1%})")
        print()
        
        # Model Performance Assessment
        print("📋 MODEL PERFORMANCE ASSESSMENT")
        print("-" * 40)
        
        # ROC AUC interpretation
        roc_auc = metrics['roc_auc']
        if roc_auc >= 0.9:
            roc_rating = "Excellent"
        elif roc_auc >= 0.8:
            roc_rating = "Good"
        elif roc_auc >= 0.7:
            roc_rating = "Fair"
        elif roc_auc >= 0.6:
            roc_rating = "Poor"
        else:
            roc_rating = "Very Poor"
        
        print(f"ROC AUC Rating:     {roc_rating} ({roc_auc:.3f})")
        
        # Precision-Recall AUC interpretation
        pr_auc = metrics['pr_auc']
        if pr_auc >= 0.8:
            pr_rating = "Excellent"
        elif pr_auc >= 0.6:
            pr_rating = "Good"
        elif pr_auc >= 0.4:
            pr_rating = "Fair"
        else:
            pr_rating = "Poor"
        
        print(f"PR-AUC Rating:      {pr_rating} ({pr_auc:.3f})")
        
        # Calibration quality
        brier = metrics['brier_score']
        if brier <= 0.1:
            cal_rating = "Excellent"
        elif brier <= 0.2:
            cal_rating = "Good"
        elif brier <= 0.3:
            cal_rating = "Fair"
        else:
            cal_rating = "Poor"
        
        print(f"Calibration:        {cal_rating} (Brier: {brier:.3f})")
        print()
        
        # Business Impact
        print("💼 BUSINESS IMPACT ANALYSIS")
        print("-" * 40)
        
        # Calculate potential injury prevention
        total_pitchers = summary['total_samples']
        predicted_high_risk = cm['true_positives'] + cm['false_positives']
        actual_injuries = summary['positive_samples']
        
        print(f"Total Pitchers Analyzed: {total_pitchers:,}")
        print(f"Predicted High Risk: {predicted_high_risk:,} ({predicted_high_risk/total_pitchers:.1%})")
        print(f"Actual Injuries: {actual_injuries:,} ({actual_injuries/total_pitchers:.1%})")
        
        if cm['true_positives'] > 0:
            prevention_rate = cm['true_positives'] / actual_injuries
            print(f"Potential Injury Prevention: {cm['true_positives']:,} ({prevention_rate:.1%})")
        
        print()
        print("=" * 80)
        print("✅ ACCURACY ANALYSIS COMPLETE")
        print("=" * 80)
    
    def run_accuracy_analysis(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """Run complete accuracy analysis."""
        logger.info("🚀 Starting Enhanced Model Accuracy Analysis")
        logger.info("=" * 60)
        
        # Load test data
        df = self.load_test_data(start_date, end_date)
        
        # Generate injury labels
        df = self.generate_injury_labels(df)
        
        # Prepare test data
        features_df, labels_df = self.prepare_test_data(df)
        
        # Calculate accuracy metrics
        metrics = self.calculate_accuracy_metrics(features_df, labels_df)
        
        # Print comprehensive report
        self.print_accuracy_report(metrics)
        
        # Save results
        results_path = "models/accuracy_analysis_results.json"
        import json
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"✅ Accuracy analysis results saved to {results_path}")
        
        return metrics

def main():
    """Main accuracy analysis function."""
    analyzer = ModelAccuracyAnalyzer()
    
    if not analyzer.model:
        logger.error("❌ Could not load enhanced model. Please train the model first.")
        return
    
    # Run accuracy analysis on 2024 data (test set)
    metrics = analyzer.run_accuracy_analysis("2024-01-01", "2024-12-31")
    
    print(f"\n🎯 KEY ACCURACY METRICS SUMMARY:")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Precision: {metrics['precision']:.1%}")
    print(f"   Recall: {metrics['recall']:.1%}")
    print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.3f}")

if __name__ == "__main__":
    main()
