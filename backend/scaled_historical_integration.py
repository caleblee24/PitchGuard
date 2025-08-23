#!/usr/bin/env python3
"""
Scaled Historical Data Integration for PitchGuard
Generates 500 pitchers across 5 seasons for rare event modeling.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Tuple, Optional
import logging

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from etl.enhanced_features import EnhancedFeatureEngineer
from modeling.enhanced_model import EnhancedInjuryRiskModel
from database.connection import get_db_session_context
from database.models import Pitch, Appearance, Injury

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScaledHistoricalDataIntegrator:
    """Scaled historical data integration for rare event modeling."""
    
    def __init__(self):
        self.seasons = [2019, 2020, 2021, 2022, 2023]
        self.num_pitchers = 500
        self.expected_injuries_per_season = 50  # 10% injury rate
        
    def generate_scaled_statcast_data(self) -> pd.DataFrame:
        """Generate Statcast data for 500 pitchers across 5 seasons."""
        print(f"🔍 Generating scaled Statcast data: {self.num_pitchers} pitchers × {len(self.seasons)} seasons...")
        
        all_pitches = []
        pitcher_id_start = 10001
        
        for season in self.seasons:
            print(f"   📅 Processing season {season}...")
            
            # Season dates
            if season == 2020:  # COVID-shortened season
                start_date = f"{season}-07-23"
                end_date = f"{season}-09-27"
            else:
                start_date = f"{season}-03-28"
                end_date = f"{season}-10-01"
            
            # Generate pitchers for this season
            season_pitchers = list(range(pitcher_id_start, pitcher_id_start + self.num_pitchers))
            
            for pitcher_id in season_pitchers:
                pitcher_pitches = self._generate_realistic_statcast_data(
                    pitcher_id, start_date, end_date, season
                )
                all_pitches.extend(pitcher_pitches)
            
            pitcher_id_start += self.num_pitchers
        
        pitches_df = pd.DataFrame(all_pitches)
        print(f"   ✅ Generated {len(pitches_df)} pitches for {self.num_pitchers * len(self.seasons)} pitcher-seasons")
        
        return pitches_df
    
    def generate_scaled_injury_data(self) -> pd.DataFrame:
        """Generate injury data for 500 pitchers across 5 seasons."""
        print(f"🏥 Generating scaled injury data: {self.num_pitchers} pitchers × {len(self.seasons)} seasons...")
        
        all_injuries = []
        pitcher_id_start = 10001
        
        for season in self.seasons:
            print(f"   📅 Processing season {season}...")
            
            # Season dates
            if season == 2020:  # COVID-shortened season
                start_date = f"{season}-01-01"
                end_date = f"{season}-12-31"
            else:
                start_date = f"{season}-01-01"
                end_date = f"{season}-12-31"
            
            # Generate injuries for this season
            season_pitchers = list(range(pitcher_id_start, pitcher_id_start + self.num_pitchers))
            season_injuries = self._generate_realistic_injury_data(
                season_pitchers, start_date, end_date, season
            )
            all_injuries.extend(season_injuries)
            
            pitcher_id_start += self.num_pitchers
        
        injuries_df = pd.DataFrame(all_injuries)
        print(f"   ✅ Generated {len(injuries_df)} injury records")
        print(f"   📊 Injury types: {injuries_df['injury_type'].value_counts().to_dict()}")
        
        return injuries_df
    
    def _generate_realistic_statcast_data(
        self, 
        pitcher_id: int, 
        start_date: str, 
        end_date: str,
        season: int
    ) -> List[Dict]:
        """Generate realistic Statcast-like data for a pitcher in a specific season."""
        
        # Pitcher characteristics (consistent across seasons)
        np.random.seed(pitcher_id + season * 1000)  # Consistent per pitcher-season
        
        is_starter = np.random.random() < 0.4  # 40% starters, 60% relievers
        velocity_tier = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
        
        # Base velocity by tier
        base_velocities = {'high': 95, 'medium': 92, 'low': 88}
        base_velocity = base_velocities[velocity_tier]
        
        # Generate appearances throughout season
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if is_starter:
            # Starters: ~32 starts, every 5 days
            n_appearances = np.random.poisson(32)
            appearance_interval = 5
        else:
            # Relievers: ~65 appearances, more variable
            n_appearances = np.random.poisson(65)
            appearance_interval = 2.5
        
        appearances = []
        current_date = start + timedelta(days=np.random.randint(0, 10))
        
        # Track fatigue over season
        cumulative_pitches = 0
        
        for game_num in range(min(n_appearances, 80)):  # Cap at 80 games
            if current_date > end:
                break
                
            # Rest days (more rest = better performance)
            if game_num > 0:
                rest_days = max(1, np.random.poisson(appearance_interval))
                current_date += timedelta(days=rest_days)
            
            # Pitches per appearance
            if is_starter:
                pitches_in_game = max(60, np.random.normal(95, 20))
            else:
                pitches_in_game = max(10, np.random.normal(25, 10))
            
            pitches_in_game = int(pitches_in_game)
            cumulative_pitches += pitches_in_game
            
            # Fatigue effects (velocity decline over season)
            season_progress = game_num / n_appearances
            fatigue_factor = 1 - (season_progress * 0.02)  # 2% decline over season
            
            # Short-term fatigue (high recent workload)
            recent_workload_factor = max(0.95, 1 - (cumulative_pitches / 3000) * 0.05)
            
            current_velocity = base_velocity * fatigue_factor * recent_workload_factor
            
            # Generate pitches for this appearance
            for pitch_num in range(pitches_in_game):
                pitch_velocity = current_velocity + np.random.normal(0, 2)
                spin_rate = 2200 + (pitch_velocity - 90) * 30 + np.random.normal(0, 100)
                
                pitch_data = {
                    'pitcher_id': pitcher_id,
                    'game_date': current_date.date(),
                    'pitch_number': pitch_num + 1,
                    'at_bat_id': game_num * 100 + (pitch_num // 5),  # ~5 pitches per AB
                    'pitch_type': np.random.choice(['FF', 'SL', 'CH', 'CB', 'CU'], 
                                                 p=[0.4, 0.25, 0.15, 0.1, 0.1]),
                    'release_speed': round(pitch_velocity, 1),
                    'release_spin_rate': round(spin_rate, 0),
                    'release_pos_x': np.random.normal(0, 0.3),
                    'release_pos_z': np.random.normal(6.0, 0.2),
                    'release_extension': np.random.normal(6.2, 0.3),
                    'season': season
                }
                
                appearances.append(pitch_data)
            
            current_date += timedelta(days=1)
        
        return appearances
    
    def _generate_realistic_injury_data(
        self, 
        pitcher_ids: List[int], 
        start_date: str, 
        end_date: str,
        season: int
    ) -> List[Dict]:
        """Generate realistic MLB injury data for multiple pitchers in a season."""
        
        # MLB injury patterns from research
        injury_types = [
            'Shoulder Inflammation', 'Elbow Inflammation', 'Forearm Strain',
            'Lat Strain', 'Oblique Strain', 'Back Strain', 'Finger Injury',
            'Tommy John Surgery', 'Shoulder Impingement', 'Bicep Tendinitis'
        ]
        
        injury_probabilities = [0.15, 0.20, 0.12, 0.08, 0.10, 0.08, 0.05, 0.03, 0.12, 0.07]
        
        # Duration patterns (days on IL)
        duration_patterns = {
            'Shoulder Inflammation': (15, 45),
            'Elbow Inflammation': (15, 60),
            'Forearm Strain': (10, 30),
            'Lat Strain': (15, 45),
            'Oblique Strain': (10, 30),
            'Back Strain': (10, 25),
            'Finger Injury': (10, 21),
            'Tommy John Surgery': (365, 500),
            'Shoulder Impingement': (30, 90),
            'Bicep Tendinitis': (15, 45)
        }
        
        injuries = []
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        for pitcher_id in pitcher_ids:
            np.random.seed(pitcher_id + season * 1000)  # Consistent per pitcher-season
            
            # Injury probability varies by pitcher "profile"
            injury_prone = np.random.random() < 0.3  # 30% are injury-prone
            base_injury_rate = 0.12 if injury_prone else 0.08  # 12% vs 8% annual rate
            
            # Determine if pitcher gets injured this season
            if np.random.random() < base_injury_rate:
                # Pick injury type
                injury_type = np.random.choice(injury_types, p=injury_probabilities)
                
                # Pick injury date (more likely mid-season due to fatigue)
                season_days = (end - start).days
                injury_day = int(np.random.beta(2, 2) * season_days)  # Bell curve, peak mid-season
                injury_date = start + timedelta(days=injury_day)
                
                # Duration
                min_days, max_days = duration_patterns[injury_type]
                duration = np.random.randint(min_days, max_days + 1)
                
                # Some injuries are season-ending
                if injury_type == 'Tommy John Surgery' or np.random.random() < 0.1:
                    end_date_actual = None  # Season-ending
                else:
                    end_date_actual = injury_date + timedelta(days=duration)
                
                injuries.append({
                    'pitcher_id': pitcher_id,
                    'il_start': injury_date.date(),
                    'il_end': end_date_actual.date() if end_date_actual else None,
                    'injury_type': injury_type,
                    'stint_type': '10-day IL' if duration <= 15 else '15-day IL' if duration <= 60 else '60-day IL',
                    'season': season
                })
        
        return injuries
    
    def aggregate_to_appearances(self, pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch data to appearance level."""
        print("📊 Aggregating pitches to appearances...")
        
        appearances = pitches_df.groupby(['pitcher_id', 'game_date', 'season']).agg({
            'pitch_number': 'count',  # Total pitches
            'release_speed': ['mean', 'std'],
            'release_spin_rate': ['mean', 'std'],
            'release_pos_x': 'std',
            'release_pos_z': 'std',
            'release_extension': 'mean'
        }).round(2)
        
        # Flatten column names
        appearances.columns = [
            'pitches_thrown', 'avg_vel', 'vel_std', 'avg_spin', 'spin_std',
            'release_pos_x_std', 'release_pos_z_std', 'avg_extension'
        ]
        
        # Calculate innings pitched (rough estimate)
        appearances['innings_pitched'] = (appearances['pitches_thrown'] / 15).round(1)
        appearances['outs_recorded'] = (appearances['innings_pitched'] * 3).astype(int)
        
        appearances = appearances.reset_index()
        
        print(f"   ✅ Created {len(appearances)} appearances")
        return appearances
    
    def train_scaled_model(
        self, 
        pitches_df: pd.DataFrame, 
        appearances_df: pd.DataFrame, 
        injuries_df: pd.DataFrame
    ) -> Dict:
        """Train model on scaled historical data."""
        print("🚀 Training model on scaled historical data...")
        
        # Feature engineering
        feature_engineer = EnhancedFeatureEngineer()
        
        # Generate features for multiple dates per pitcher
        all_features = []
        all_labels = []
        
        for pitcher_id in appearances_df['pitcher_id'].unique():
            pitcher_appearances = appearances_df[appearances_df['pitcher_id'] == pitcher_id].copy()
            pitcher_pitches = pitches_df[pitches_df['pitcher_id'] == pitcher_id].copy()
            pitcher_injuries = injuries_df[injuries_df['pitcher_id'] == pitcher_id].copy()
            
            if len(pitcher_appearances) < 10:  # Need minimum appearances
                continue
            
            # Ensure date columns are datetime objects
            pitcher_appearances['game_date'] = pd.to_datetime(pitcher_appearances['game_date'])
            pitcher_pitches['game_date'] = pd.to_datetime(pitcher_pitches['game_date'])
            if not pitcher_injuries.empty:
                pitcher_injuries['il_start'] = pd.to_datetime(pitcher_injuries['il_start'])
            
            # Sort by date
            pitcher_appearances = pitcher_appearances.sort_values('game_date')
            pitcher_pitches = pitcher_pitches.sort_values('game_date')
            
            # Generate features every 2 weeks
            start_date = pitcher_appearances['game_date'].min()
            end_date = pitcher_appearances['game_date'].max()
            
            current_date = start_date + timedelta(days=14)  # Start after 2 weeks
            
            while current_date <= end_date:
                # Get historical data up to current date
                hist_appearances = pitcher_appearances[pitcher_appearances['game_date'] <= current_date]
                hist_pitches = pitcher_pitches[pitcher_pitches['game_date'] <= current_date]
                
                if len(hist_appearances) >= 5:  # Minimum for features
                    # Convert current_date to datetime if it's a date
                    if hasattr(current_date, 'date'):
                        current_datetime = current_date
                    else:
                        current_datetime = datetime.combine(current_date, datetime.min.time())
                    
                    features = feature_engineer.engineer_features(
                        hist_appearances, hist_pitches, pd.DataFrame(), current_datetime
                    )
                    
                    if not features.empty:
                        all_features.append(features)
                        
                        # Create injury label (21-day forward window)
                        injury_within_21d = False
                        future_date = current_datetime + timedelta(days=21)
                        
                        for _, injury in pitcher_injuries.iterrows():
                            injury_start = injury['il_start']  # Already converted to datetime
                            if current_datetime < injury_start <= future_date:
                                injury_within_21d = True
                                break
                        
                        all_labels.append({
                            'pitcher_id': pitcher_id,
                            'as_of_date': current_date,
                            'label_injury_within_21d': 1 if injury_within_21d else 0
                        })
                
                current_date += timedelta(days=14)  # Every 2 weeks
        
        # Combine features and labels
        combined_features = pd.concat(all_features, ignore_index=True)
        labels_df = pd.DataFrame(all_labels)
        
        print(f"   📊 Generated {len(combined_features)} feature samples")
        print(f"   📊 Positive rate: {labels_df['label_injury_within_21d'].mean():.3f}")
        
        # Train model
        # Time-based split
        sorted_labels = labels_df.sort_values('as_of_date')
        split_idx = int(len(sorted_labels) * 0.8)
        
        train_labels = sorted_labels.iloc[:split_idx]
        val_labels = sorted_labels.iloc[split_idx:]
        
        # Get corresponding features
        train_features = train_labels.merge(combined_features, on=['pitcher_id', 'as_of_date'])
        val_features = val_labels.merge(combined_features, on=['pitcher_id', 'as_of_date'])
        
        # Train simplified model (like in audit)
        feature_cols = [col for col in train_features.columns 
                       if col not in ['pitcher_id', 'as_of_date', 'label_injury_within_21d', 'role', 'data_completeness']
                       and train_features[col].dtype in ['int64', 'float64', 'float32', 'int32']]
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_labels['label_injury_within_21d'].values
        X_val = val_features[feature_cols].fillna(0)
        y_val = val_labels['label_injury_within_21d'].values
        
        # Train XGBoost
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': pos_weight
        }
        
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate
        val_probs = xgb_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        pr_auc = average_precision_score(y_val, val_probs)
        
        # Top 10% metrics
        n_top = max(1, int(len(y_val) * 0.1))
        top_indices = np.argsort(val_probs)[-n_top:]
        recall_top_10 = y_val[top_indices].mean()
        baseline = y_val.mean()
        lift = recall_top_10 / baseline if baseline > 0 else 0
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'model': xgb_model,
            'feature_names': feature_cols,
            'auc': auc,
            'pr_auc': pr_auc,
            'recall_top_10': recall_top_10,
            'lift_top_10': lift,
            'baseline': baseline,
            'n_train': len(y_train),
            'n_val': len(y_val),
            'pos_rate': baseline,
            'feature_importance': feature_importance
        }
        
        print(f"   ✅ Training complete!")
        print(f"      - AUC: {auc:.3f}")
        print(f"      - PR-AUC: {pr_auc:.3f}")
        print(f"      - Recall@Top-10%: {recall_top_10:.3f}")
        print(f"      - Lift: {lift:.1f}x")
        print(f"      - Samples: {len(y_train)} train, {len(y_val)} val")
        print(f"      - Top features: {feature_importance.head(5)['feature'].tolist()}")
        
        return results
    
    def run_scaled_integration(self):
        """Run complete scaled historical data integration pipeline."""
        print("🚀 Starting Scaled Historical Data Integration")
        print("=" * 60)
        
        # Step 1: Generate scaled Statcast data
        pitches_df = self.generate_scaled_statcast_data()
        
        # Step 2: Generate scaled injury data
        injuries_df = self.generate_scaled_injury_data()
        
        # Step 3: Aggregate to appearances
        appearances_df = self.aggregate_to_appearances(pitches_df)
        
        # Step 4: Train model
        results = self.train_scaled_model(pitches_df, appearances_df, injuries_df)
        
        print("\n🎉 Scaled Historical Data Integration Complete!")
        print("=" * 60)
        print(f"📊 Final Results:")
        print(f"   - Dataset: {results['n_train'] + results['n_val']} samples")
        print(f"   - Positive rate: {results['pos_rate']:.3f}")
        print(f"   - AUC: {results['auc']:.3f}")
        print(f"   - PR-AUC: {results['pr_auc']:.3f}")
        print(f"   - Lift@Top-10%: {results['lift_top_10']:.1f}x")
        print(f"   - Top 5 features: {results['feature_importance'].head(5)['feature'].tolist()}")
        
        return results

if __name__ == "__main__":
    integrator = ScaledHistoricalDataIntegrator()
    results = integrator.run_scaled_integration()
