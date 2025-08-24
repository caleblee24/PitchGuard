#!/usr/bin/env python3
"""
Retrain PitchGuard model with real injury data
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealInjuryModelTrainer:
    """Train model with real injury data"""
    
    def __init__(self, db_path: str = "pitchguard.db"):
        self.db_path = db_path
        self.model = None
        self.feature_importance = None
        
    def load_real_injury_data(self) -> pd.DataFrame:
        """Load real injury data from database"""
        logger.info("Loading real injury data from database")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get injury records
        injury_query = """
        SELECT 
            pitcher_id,
            pitcher_name,
            team,
            injury_date,
            injury_type,
            injury_location,
            severity,
            source,
            confidence_score
        FROM injury_records 
        WHERE confidence_score > 0.5
        ORDER BY injury_date
        """
        
        injuries_df = pd.read_sql_query(injury_query, conn)
        
        # Get pitcher workload data
        workload_query = """
        SELECT 
            p.pitcher_id,
            p.pitcher_name,
            p.team,
            p.game_date,
            p.pitches_thrown,
            p.innings_pitched,
            p.earned_runs,
            p.walks,
            p.strikeouts,
            p.hits_allowed
        FROM appearances p
        ORDER BY p.pitcher_id, p.game_date
        """
        
        workload_df = pd.read_sql_query(workload_query, conn)
        
        # Get pitch-level data for feature engineering
        pitch_query = """
        SELECT 
            pitcher_id,
            game_date,
            pitch_type,
            release_speed,
            release_spin_rate,
            release_extension,
            pfx_x,
            pfx_z,
            plate_x,
            plate_z
        FROM pitches
        WHERE release_speed IS NOT NULL
        ORDER BY pitcher_id, game_date
        """
        
        pitches_df = pd.read_sql_query(pitch_query, conn)
        
        conn.close()
        
        logger.info(f"Loaded {len(injuries_df)} injury records")
        logger.info(f"Loaded {len(workload_df)} appearance records")
        logger.info(f"Loaded {len(pitches_df)} pitch records")
        
        return injuries_df, workload_df, pitches_df
    
    def create_injury_labels(self, injuries_df: pd.DataFrame, workload_df: pd.DataFrame) -> pd.DataFrame:
        """Create injury labels for training data"""
        logger.info("Creating injury labels for training data")
        
        # Convert dates
        injuries_df['injury_date'] = pd.to_datetime(injuries_df['injury_date'])
        workload_df['game_date'] = pd.to_datetime(workload_df['game_date'])
        
        # Create injury labels
        labeled_data = []
        
        for _, injury in injuries_df.iterrows():
            pitcher_id = injury['pitcher_id']
            injury_date = injury['injury_date']
            
            # Get pitcher's workload data before injury
            pitcher_workload = workload_df[workload_df['pitcher_id'] == pitcher_id].copy()
            pitcher_workload = pitcher_workload[pitcher_workload['game_date'] < injury_date]
            
            if len(pitcher_workload) > 0:
                # Create positive samples (appearances leading to injury)
                for _, appearance in pitcher_workload.tail(5).iterrows():  # Last 5 appearances before injury
                    labeled_data.append({
                        'pitcher_id': pitcher_id,
                        'pitcher_name': injury['pitcher_name'],
                        'team': injury['team'],
                        'game_date': appearance['game_date'],
                        'label_injury_within_21d': 1,
                        'injury_date': injury_date,
                        'injury_type': injury['injury_type'],
                        'injury_location': injury['injury_location'],
                        'severity': injury['severity'],
                        'pitches_thrown': appearance['pitches_thrown'],
                        'innings_pitched': appearance['innings_pitched'],
                        'earned_runs': appearance['earned_runs'],
                        'walks': appearance['walks'],
                        'strikeouts': appearance['strikeouts'],
                        'hits_allowed': appearance['hits_allowed']
                    })
        
        # Create negative samples (healthy appearances)
        for pitcher_id in workload_df['pitcher_id'].unique():
            pitcher_workload = workload_df[workload_df['pitcher_id'] == pitcher_id].copy()
            
            # Get injury dates for this pitcher
            pitcher_injuries = injuries_df[injuries_df['pitcher_id'] == pitcher_id]
            injury_dates = set(pitcher_injuries['injury_date'].dt.date)
            
            # Find healthy periods (no injuries within 30 days)
            for _, appearance in pitcher_workload.iterrows():
                appearance_date = appearance['game_date'].date()
                
                # Check if this appearance is far from any injury
                is_healthy = True
                for injury_date in injury_dates:
                    days_diff = abs((appearance_date - injury_date).days)
                    if days_diff <= 30:
                        is_healthy = False
                        break
                
                if is_healthy:
                    labeled_data.append({
                        'pitcher_id': pitcher_id,
                        'pitcher_name': appearance['pitcher_name'],
                        'team': appearance['team'],
                        'game_date': appearance['game_date'],
                        'label_injury_within_21d': 0,
                        'injury_date': None,
                        'injury_type': None,
                        'injury_location': None,
                        'severity': None,
                        'pitches_thrown': appearance['pitches_thrown'],
                        'innings_pitched': appearance['innings_pitched'],
                        'earned_runs': appearance['earned_runs'],
                        'walks': appearance['walks'],
                        'strikeouts': appearance['strikeouts'],
                        'hits_allowed': appearance['hits_allowed']
                    })
        
        labeled_df = pd.DataFrame(labeled_data)
        
        logger.info(f"Created {len(labeled_df)} labeled samples")
        logger.info(f"Positive samples: {labeled_df['label_injury_within_21d'].sum()}")
        logger.info(f"Negative samples: {(labeled_df['label_injury_within_21d'] == 0).sum()}")
        
        return labeled_df
    
    def engineer_features(self, labeled_df: pd.DataFrame, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for the model"""
        logger.info("Engineering features for real injury model")
        
        # Convert dates
        labeled_df['game_date'] = pd.to_datetime(labeled_df['game_date'])
        pitches_df['game_date'] = pd.to_datetime(pitches_df['game_date'])
        
        # Initialize feature columns
        feature_cols = []
        
        # Basic workload features
        labeled_df['pitches_per_inning'] = labeled_df['pitches_thrown'] / labeled_df['innings_pitched'].replace(0, 1)
        labeled_df['strikeout_rate'] = labeled_df['strikeouts'] / labeled_df['pitches_thrown'].replace(0, 1)
        labeled_df['walk_rate'] = labeled_df['walks'] / labeled_df['pitches_thrown'].replace(0, 1)
        labeled_df['hit_rate'] = labeled_df['hits_allowed'] / labeled_df['innings_pitched'].replace(0, 1)
        labeled_df['era'] = labeled_df['earned_runs'] * 9 / labeled_df['innings_pitched'].replace(0, 1)
        
        feature_cols.extend(['pitches_per_inning', 'strikeout_rate', 'walk_rate', 'hit_rate', 'era'])
        
        # Rolling window features for each pitcher
        for pitcher_id in labeled_df['pitcher_id'].unique():
            pitcher_data = labeled_df[labeled_df['pitcher_id'] == pitcher_id].sort_values('game_date')
            
            # 7-day rolling features
            pitcher_data['pitches_7d'] = pitcher_data['pitches_thrown'].rolling(window=7, min_periods=1).sum()
            pitcher_data['innings_7d'] = pitcher_data['innings_pitched'].rolling(window=7, min_periods=1).sum()
            pitcher_data['strikeouts_7d'] = pitcher_data['strikeouts'].rolling(window=7, min_periods=1).sum()
            pitcher_data['walks_7d'] = pitcher_data['walks'].rolling(window=7, min_periods=1).sum()
            
            # 14-day rolling features
            pitcher_data['pitches_14d'] = pitcher_data['pitches_thrown'].rolling(window=14, min_periods=1).sum()
            pitcher_data['innings_14d'] = pitcher_data['innings_pitched'].rolling(window=14, min_periods=1).sum()
            pitcher_data['strikeouts_14d'] = pitcher_data['strikeouts'].rolling(window=14, min_periods=1).sum()
            pitcher_data['walks_14d'] = pitcher_data['walks'].rolling(window=14, min_periods=1).sum()
            
            # 30-day rolling features
            pitcher_data['pitches_30d'] = pitcher_data['pitches_thrown'].rolling(window=30, min_periods=1).sum()
            pitcher_data['innings_30d'] = pitcher_data['innings_pitched'].rolling(window=30, min_periods=1).sum()
            pitcher_data['strikeouts_30d'] = pitcher_data['strikeouts'].rolling(window=30, min_periods=1).sum()
            pitcher_data['walks_30d'] = pitcher_data['walks'].rolling(window=30, min_periods=1).sum()
            
            # Update the main dataframe
            labeled_df.loc[labeled_df['pitcher_id'] == pitcher_id, pitcher_data.columns] = pitcher_data
        
        # Add rolling features to feature columns
        rolling_features = [
            'pitches_7d', 'innings_7d', 'strikeouts_7d', 'walks_7d',
            'pitches_14d', 'innings_14d', 'strikeouts_14d', 'walks_14d',
            'pitches_30d', 'innings_30d', 'strikeouts_30d', 'walks_30d'
        ]
        feature_cols.extend(rolling_features)
        
        # Add pitch-level features if available
        if len(pitches_df) > 0:
            for pitcher_id in labeled_df['pitcher_id'].unique():
                pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id]
                
                if len(pitcher_pitches) > 0:
                    # Calculate average velocity and spin rate for each appearance
                    for _, appearance in labeled_df[labeled_df['pitcher_id'] == pitcher_id].iterrows():
                        game_pitches = pitcher_pitches[pitcher_pitches['game_date'] == appearance['game_date']]
                        
                        if len(game_pitches) > 0:
                            labeled_df.loc[(labeled_df['pitcher_id'] == pitcher_id) & 
                                         (labeled_df['game_date'] == appearance['game_date']), 'avg_velocity'] = game_pitches['release_speed'].mean()
                            labeled_df.loc[(labeled_df['pitcher_id'] == pitcher_id) & 
                                         (labeled_df['game_date'] == appearance['game_date']), 'avg_spin_rate'] = game_pitches['release_spin_rate'].mean()
            
            feature_cols.extend(['avg_velocity', 'avg_spin_rate'])
        
        # Fill missing values
        labeled_df[feature_cols] = labeled_df[feature_cols].fillna(0)
        
        logger.info(f"Engineered {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols}")
        
        return labeled_df, feature_cols
    
    def train_model(self, labeled_df: pd.DataFrame, feature_cols: list) -> dict:
        """Train the XGBoost model"""
        logger.info("Training XGBoost model with real injury data")
        
        # Prepare data
        X = labeled_df[feature_cols]
        y = labeled_df['label_injury_within_21d']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Calculate class weights
        class_weights = {
            0: 1.0,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=class_weights[1]
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'precision': (y_pred & y_test).sum() / y_pred.sum() if y_pred.sum() > 0 else 0,
            'recall': (y_pred & y_test).sum() / y_test.sum() if y_test.sum() > 0 else 0
        }
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        metrics['pr_auc'] = auc(recall, precision)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        
        logger.info("Model training complete")
        logger.info(f"Metrics: {metrics}")
        logger.info(f"Top 5 features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        self.model = model
        self.feature_importance = feature_importance
        
        return {
            'model': model,
            'feature_importance': feature_importance,
            'metrics': metrics,
            'feature_cols': feature_cols
        }
    
    def save_model(self, model_data: dict, filename: str = "real_injury_model.pkl"):
        """Save the trained model"""
        logger.info(f"Saving model to {filename}")
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Model saved successfully")
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline with real injury data")
        
        # Load data
        injuries_df, workload_df, pitches_df = self.load_real_injury_data()
        
        if len(injuries_df) == 0:
            logger.warning("No injury data found. Please run the injury scraper first.")
            return None
        
        # Create labels
        labeled_df = self.create_injury_labels(injuries_df, workload_df)
        
        if len(labeled_df) == 0:
            logger.warning("No labeled data created. Check data quality.")
            return None
        
        # Engineer features
        labeled_df, feature_cols = self.engineer_features(labeled_df, pitches_df)
        
        # Train model
        model_data = self.train_model(labeled_df, feature_cols)
        
        # Save model
        self.save_model(model_data)
        
        return model_data

def main():
    """Main execution function"""
    trainer = RealInjuryModelTrainer()
    
    # Run full training
    model_data = trainer.run_full_training()
    
    if model_data:
        print("\n" + "="*60)
        print("REAL INJURY MODEL TRAINING RESULTS")
        print("="*60)
        print(f"Model saved successfully")
        print(f"Metrics: {model_data['metrics']}")
        print(f"Top features: {sorted(model_data['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]}")

if __name__ == "__main__":
    main()
