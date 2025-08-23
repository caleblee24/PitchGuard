"""
Enhanced Feature Engineering for PitchGuard MVP
Implements Sprint 1 features: role-aware loads, pitch-mix deltas, EWMs, acute vs chronic indices
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering for injury risk prediction.
    Implements role-aware, pitch-mix, and temporal features.
    """
    
    def __init__(self, min_pitches_for_velo: int = 20):
        self.min_pitches_for_velo = min_pitches_for_velo
        
    def engineer_features(
        self, 
        appearances: pd.DataFrame,
        pitches: pd.DataFrame,
        injuries: pd.DataFrame,
        as_of_date: datetime
    ) -> pd.DataFrame:
        """
        Generate enhanced features for injury risk prediction.
        
        Args:
            appearances: DataFrame with appearance data
            pitches: DataFrame with pitch-level data
            injuries: DataFrame with injury data
            as_of_date: Date to compute features as of
            
        Returns:
            DataFrame with engineered features per pitcher
        """
        logger.info(f"Engineering enhanced features as of {as_of_date}")
        
        # Get unique pitchers
        pitchers = appearances['pitcher_id'].unique()
        feature_rows = []
        
        for pitcher_id in pitchers:
            try:
                features = self._compute_pitcher_features(
                    pitcher_id, appearances, pitches, injuries, as_of_date
                )
                if features is not None:
                    feature_rows.append(features)
            except Exception as e:
                logger.warning(f"Error computing features for pitcher {pitcher_id}: {e}")
                continue
        
        if not feature_rows:
            logger.warning("No features computed for any pitcher")
            return pd.DataFrame()
            
        features_df = pd.DataFrame(feature_rows)
        logger.info(f"Computed features for {len(features_df)} pitchers")
        
        return features_df
    
    def _compute_pitcher_features(
        self,
        pitcher_id: int,
        appearances: pd.DataFrame,
        pitches: pd.DataFrame,
        injuries: pd.DataFrame,
        as_of_date: datetime
    ) -> Optional[Dict]:
        """Compute all features for a single pitcher."""
        
        # Ensure date columns are datetime objects for comparison
        if 'game_date' in appearances.columns:
            appearances = appearances.copy()
            appearances['game_date'] = pd.to_datetime(appearances['game_date'])
        
        if 'game_date' in pitches.columns:
            pitches = pitches.copy()
            pitches['game_date'] = pd.to_datetime(pitches['game_date'])
        
        # Filter data for this pitcher up to as_of_date
        pitcher_appearances = appearances[
            (appearances['pitcher_id'] == pitcher_id) & 
            (appearances['game_date'] <= as_of_date)
        ].sort_values('game_date')
        
        pitcher_pitches = pitches[
            (pitches['pitcher_id'] == pitcher_id) & 
            (pitches['game_date'] <= as_of_date)
        ].sort_values(['game_date', 'pitch_number'])
        
        if len(pitcher_appearances) == 0:
            return None
            
        # Base features
        features = {
            'pitcher_id': pitcher_id,
            'as_of_date': as_of_date,
            'total_appearances': len(pitcher_appearances),
            'days_since_last_appearance': self._days_since_last_appearance(
                pitcher_appearances, as_of_date
            )
        }
        
        # Role-aware features
        features.update(self._compute_role_features(pitcher_appearances))
        
        # Acute vs chronic workload features
        features.update(self._compute_workload_features(pitcher_appearances))
        
        # Fatigue signals (velocity and spin)
        features.update(self._compute_fatigue_features(pitcher_pitches, pitcher_appearances))
        
        # Pitch mix features
        features.update(self._compute_pitch_mix_features(pitcher_pitches))
        
        # Data completeness flags
        features.update(self._compute_completeness_flags(pitcher_pitches, pitcher_appearances))
        
        return features
    
    def _days_since_last_appearance(
        self, 
        appearances: pd.DataFrame, 
        as_of_date: datetime
    ) -> int:
        """Compute days since last appearance."""
        if len(appearances) == 0:
            return 999  # Large number for pitchers with no appearances
            
        last_appearance = appearances['game_date'].max()
        return (as_of_date - last_appearance).days
    
    def _compute_role_features(self, appearances: pd.DataFrame) -> Dict:
        """Compute role-aware features."""
        if len(appearances) < 3:
            return {
                'role': 'unknown',
                'avg_innings_per_appearance': 0.0,
                'multi_inning_rate_7d': 0.0,
                'multi_inning_rate_14d': 0.0,
                'back_to_back_days_7d': 0,
                'back_to_back_days_14d': 0
            }
        
        # Determine role based on recent appearances
        recent_appearances = appearances.tail(10)
        avg_innings = recent_appearances['innings_pitched'].mean()
        
        if avg_innings >= 5.0:
            role = 'starter'
        elif avg_innings >= 1.0:
            role = 'reliever'
        else:
            role = 'closer'
        
        # Multi-inning features
        multi_inning_7d = self._count_multi_inning_appearances(appearances, days=7)
        multi_inning_14d = self._count_multi_inning_appearances(appearances, days=14)
        
        # Back-to-back features
        back_to_back_7d = self._count_back_to_back_days(appearances, days=7)
        back_to_back_14d = self._count_back_to_back_days(appearances, days=14)
        
        return {
            'role': role,
            'avg_innings_per_appearance': avg_innings,
            'multi_inning_rate_7d': multi_inning_7d / max(1, len(self._filter_recent_appearances(appearances, days=7))),
            'multi_inning_rate_14d': multi_inning_14d / max(1, len(self._filter_recent_appearances(appearances, days=14))),
            'back_to_back_days_7d': back_to_back_7d,
            'back_to_back_days_14d': back_to_back_14d
        }
    
    def _compute_workload_features(self, appearances: pd.DataFrame) -> Dict:
        """Compute workload intensity features."""
        features = {}
        
        if len(appearances) == 0:
            # Default values for no appearances
            features.update({
                'total_appearances': 0,
                'days_since_last_appearance': 999,  # High value to indicate no recent activity
                'avg_innings_per_appearance': 0.0,
                'multi_inning_rate_7d': 0.0,
                'multi_inning_rate_14d': 0.0,
                'back_to_back_days_7d': 0,
                'back_to_back_days_14d': 0,
                'last_game_pitches': 0,
                'last_3_games_pitches': 0,
                'roll7d_pitch_count': 0,
                'roll7d_appearances': 0,
                'roll7d_avg_pitches_per_appearance': 0.0,
                'roll14d_pitch_count': 0,
                'roll14d_appearances': 0,
                'roll14d_avg_pitches_per_appearance': 0.0,
                'roll30d_pitch_count': 0,
                'roll30d_appearances': 0,
                'roll30d_avg_pitches_per_appearance': 0.0,
                'workload_intensity': 0.0,
                'fatigue_score': 0.0
            })
            return features
        
        # Basic workload metrics
        features['total_appearances'] = len(appearances)
        features['days_since_last_appearance'] = (
            pd.Timestamp.now() - appearances['game_date'].max()
        ).days
        
        # Innings and multi-inning features
        if 'innings_pitched' in appearances.columns:
            features['avg_innings_per_appearance'] = appearances['innings_pitched'].mean()
            features['multi_inning_rate_7d'] = self._count_multi_inning_appearances(appearances, 7) / max(1, len(self._filter_recent_appearances(appearances, 7)))
            features['multi_inning_rate_14d'] = self._count_multi_inning_appearances(appearances, 14) / max(1, len(self._filter_recent_appearances(appearances, 14)))
        else:
            features['avg_innings_per_appearance'] = 1.0  # Default for relievers
            features['multi_inning_rate_7d'] = 0.0
            features['multi_inning_rate_14d'] = 0.0
        
        # Back-to-back appearances
        features['back_to_back_days_7d'] = self._count_back_to_back_days(appearances, 7)
        features['back_to_back_days_14d'] = self._count_back_to_back_days(appearances, 14)
        
        # Recent pitch counts
        features['last_game_pitches'] = appearances.iloc[-1]['pitches_thrown'] if len(appearances) > 0 else 0
        features['last_3_games_pitches'] = appearances.tail(3)['pitches_thrown'].sum()
        
        # Rolling workload windows
        for days in [7, 14, 30]:
            recent = self._filter_recent_appearances(appearances, days)
            features[f'roll{days}d_pitch_count'] = recent['pitches_thrown'].sum()
            features[f'roll{days}d_appearances'] = len(recent)
            features[f'roll{days}d_avg_pitches_per_appearance'] = (
                recent['pitches_thrown'].mean() if len(recent) > 0 else 0.0
            )
        
        # Workload intensity score (combination of recent pitch count and frequency)
        recent_7d_pitches = features['roll7d_pitch_count']
        recent_7d_appearances = features['roll7d_appearances']
        
        if recent_7d_appearances > 0:
            # Higher intensity for more pitches in fewer appearances
            features['workload_intensity'] = recent_7d_pitches / recent_7d_appearances
        else:
            features['workload_intensity'] = 0.0
        
        # Fatigue score (combination of workload and rest patterns)
        rest_days = features.get('avg_rest_days', 4.0)  # Default to 4 days
        workload_intensity = features['workload_intensity']
        
        # Higher fatigue for high workload and low rest
        features['fatigue_score'] = (workload_intensity * 0.7) + ((4.0 - rest_days) * 0.3)
        
        return features
    
    def _compute_fatigue_features(
        self, 
        pitches: pd.DataFrame, 
        appearances: pd.DataFrame
    ) -> Dict:
        """Compute fatigue signals from velocity and spin rate."""
        features = {}
        
        # EWM features for velocity and spin
        for metric in ['release_speed', 'release_spin_rate']:
            if len(pitches) > 0:
                # Compute EWM with different spans
                for span in [7, 14, 30]:
                    recent_pitches = self._filter_recent_pitches(pitches, days=span)
                    if len(recent_pitches) >= self.min_pitches_for_velo:
                        ewm_mean = recent_pitches[metric].ewm(span=min(span, len(recent_pitches))).mean().iloc[-1]
                        ewm_std = recent_pitches[metric].ewm(span=min(span, len(recent_pitches))).std().iloc[-1]
                        
                        features[f'roll{span}d_{metric}_ewm_mean'] = ewm_mean
                        features[f'roll{span}d_{metric}_ewm_std'] = ewm_std
                        
                        # Delta vs 30-day baseline
                        if span < 30:
                            baseline_pitches = self._filter_recent_pitches(pitches, days=30)
                            if len(baseline_pitches) >= self.min_pitches_for_velo:
                                baseline_mean = baseline_pitches[metric].mean()
                                features[f'roll{span}d_{metric}_delta_vs_30d'] = ewm_mean - baseline_mean
                            else:
                                features[f'roll{span}d_{metric}_delta_vs_30d'] = 0.0
                    else:
                        features[f'roll{span}d_{metric}_ewm_mean'] = 0.0
                        features[f'roll{span}d_{metric}_ewm_std'] = 0.0
                        features[f'roll{span}d_{metric}_delta_vs_30d'] = 0.0
            else:
                # No pitch data
                for span in [7, 14, 30]:
                    features[f'roll{span}d_{metric}_ewm_mean'] = 0.0
                    features[f'roll{span}d_{metric}_ewm_std'] = 0.0
                    features[f'roll{span}d_{metric}_delta_vs_30d'] = 0.0
        
        # Velocity decline features (key fatigue indicator)
        features['vel_decline_7d_vs_30d'] = features.get('roll7d_release_speed_delta_vs_30d', 0.0)
        features['vel_decline_14d_vs_30d'] = features.get('roll14d_release_speed_delta_vs_30d', 0.0)
        
        return features
    
    def _compute_pitch_mix_features(self, pitches: pd.DataFrame) -> Dict:
        """Compute pitch mix and breaking ball features."""
        features = {}
        
        # Define breaking ball types
        breaking_balls = ['slider', 'curveball', 'sweeper', 'cutter', 'knuckle_curve']
        
        for days in [7, 14, 30]:
            recent_pitches = self._filter_recent_pitches(pitches, days=days)
            
            if len(recent_pitches) > 0:
                # Breaking ball percentage
                breaking_ball_count = recent_pitches[
                    recent_pitches['pitch_type'].isin(breaking_balls)
                ].shape[0]
                
                features[f'roll{days}d_breaking_ball_pct'] = (
                    breaking_ball_count / len(recent_pitches)
                )
                
                # Delta vs 30-day baseline
                if days < 30:
                    baseline_pitches = self._filter_recent_pitches(pitches, days=30)
                    if len(baseline_pitches) > 0:
                        baseline_breaking_balls = baseline_pitches[
                            baseline_pitches['pitch_type'].isin(breaking_balls)
                        ].shape[0]
                        baseline_pct = baseline_breaking_balls / len(baseline_pitches)
                        features[f'roll{days}d_breaking_ball_delta_vs_30d'] = (
                            features[f'roll{days}d_breaking_ball_pct'] - baseline_pct
                        )
                    else:
                        features[f'roll{days}d_breaking_ball_delta_vs_30d'] = 0.0
            else:
                features[f'roll{days}d_breaking_ball_pct'] = 0.0
                features[f'roll{days}d_breaking_ball_delta_vs_30d'] = 0.0
        
        return features
    
    def _compute_rest_features(self, appearances: pd.DataFrame) -> Dict:
        """Compute rest day features."""
        features = {}
        
        if len(appearances) < 2:
            features['avg_rest_days'] = 0.0
            features['std_rest_days'] = 0.0
            features['min_rest_days'] = 0.0
            features['max_rest_days'] = 0.0
            return features
        
        # Calculate rest days between appearances
        rest_days = []
        for i in range(1, len(appearances)):
            rest = (appearances.iloc[i]['game_date'] - appearances.iloc[i-1]['game_date']).days
            rest_days.append(rest)
        
        if rest_days:
            features['avg_rest_days'] = np.mean(rest_days)
            features['std_rest_days'] = np.std(rest_days)
            features['min_rest_days'] = np.min(rest_days)
            features['max_rest_days'] = np.max(rest_days)
        else:
            features['avg_rest_days'] = 0.0
            features['std_rest_days'] = 0.0
            features['min_rest_days'] = 0.0
            features['max_rest_days'] = 0.0
        
        return features
    
    def _compute_completeness_flags(
        self, 
        pitches: pd.DataFrame, 
        appearances: pd.DataFrame
    ) -> Dict:
        """Compute data completeness flags."""
        flags = {}
        
        # Velocity data completeness
        for days in [7, 14, 30]:
            recent_pitches = self._filter_recent_pitches(pitches, days=days)
            flags[f'has_velo_data_{days}d'] = len(recent_pitches) >= self.min_pitches_for_velo
            flags[f'has_spin_data_{days}d'] = len(recent_pitches) >= self.min_pitches_for_velo
        
        # Appearance data completeness
        flags['has_recent_appearances'] = len(appearances) > 0
        flags['has_multiple_appearances'] = len(appearances) >= 3
        
        # Overall completeness score
        completeness_score = sum(flags.values()) / len(flags)
        flags['data_completeness_score'] = completeness_score
        
        # Categorical completeness
        if completeness_score >= 0.8:
            flags['data_completeness'] = 'high'
        elif completeness_score >= 0.5:
            flags['data_completeness'] = 'medium'
        else:
            flags['data_completeness'] = 'low'
        
        return flags
    
    def _filter_recent_appearances(self, appearances: pd.DataFrame, days: int) -> pd.DataFrame:
        """Filter appearances to recent N days."""
        if len(appearances) == 0:
            return appearances
        
        cutoff_date = appearances['game_date'].max() - timedelta(days=days)
        return appearances[appearances['game_date'] >= cutoff_date]
    
    def _filter_recent_pitches(self, pitches: pd.DataFrame, days: int) -> pd.DataFrame:
        """Filter pitches to recent N days."""
        if len(pitches) == 0:
            return pitches
        
        cutoff_date = pitches['game_date'].max() - timedelta(days=days)
        return pitches[pitches['game_date'] >= cutoff_date]
    
    def _count_multi_inning_appearances(self, appearances: pd.DataFrame, days: int) -> int:
        """Count multi-inning appearances in recent days."""
        recent = self._filter_recent_appearances(appearances, days)
        return len(recent[recent['innings_pitched'] >= 2.0])
    
    def _count_back_to_back_days(self, appearances: pd.DataFrame, days: int) -> int:
        """Count back-to-back day appearances in recent days."""
        recent = self._filter_recent_appearances(appearances, days)
        if len(recent) < 2:
            return 0
        
        back_to_back_count = 0
        for i in range(1, len(recent)):
            rest_days = (recent.iloc[i]['game_date'] - recent.iloc[i-1]['game_date']).days
            if rest_days == 1:
                back_to_back_count += 1
        
        return back_to_back_count
