"""
Enhanced XGBoost Model for PitchGuard MVP
Implements Sprint 1: XGBoost + isotonic calibration + subgroup calibration + enriched payload
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)

class EnhancedInjuryRiskModel:
    """
    Enhanced injury risk prediction model with XGBoost and proper calibration.
    """
    
    def __init__(
        self,
        model_version: str = "pitchguard_xgb_v1",
        random_state: int = 42
    ):
        self.model_version = model_version
        self.random_state = random_state
        
        # Model components
        self.xgb_model = None
        self.scaler = StandardScaler()
        
        # Calibration components
        self.calibrator_overall = None
        self.calibrator_starters = None
        self.calibrator_relievers = None
        
        # Feature metadata
        self.feature_names = []
        self.feature_importance = {}
        
        # Model metadata
        self.training_metrics = {}
        self.calibration_metrics = {}
        
    def train(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the enhanced injury risk model.
        
        Args:
            features_df: DataFrame with engineered features
            labels_df: DataFrame with injury labels
            test_size: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics and model info
        """
        logger.info(f"Training enhanced model version: {self.model_version}")
        
        # Prepare training data
        X, y, groups = self._prepare_training_data(features_df, labels_df)
        
        # Time-based split (no future leakage)
        train_idx, val_idx = self._time_based_split(X, test_size)
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train XGBoost model
        self._train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Train calibration models
        self._train_calibration(X_val_scaled, y_val, groups_val)
        
        # Evaluate model
        metrics = self._evaluate_model(X_val_scaled, y_val, groups_val)
        
        # Store training metrics
        self.training_metrics = metrics
        
        logger.info(f"Training completed. AUC: {metrics['auc']:.3f}, PR-AUC: {metrics['pr_auc']:.3f}")
        
        return metrics
    
    def predict(
        self,
        features: pd.DataFrame,
        pitcher_roles: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Make predictions with enriched payload.
        
        Args:
            features: DataFrame with features for prediction
            pitcher_roles: Series with pitcher roles for subgroup calibration
            
        Returns:
            Dictionary with enriched prediction payload
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get raw predictions
        raw_probs = self.xgb_model.predict_proba(features_scaled)[:, 1]
        
        # Apply calibration based on role
        calibrated_probs = self._apply_calibration(raw_probs, pitcher_roles)
        
        # Generate enriched payload
        payloads = []
        for i, (_, row) in enumerate(features.iterrows()):
            payload = self._generate_enriched_payload(
                row, calibrated_probs[i], raw_probs[i], pitcher_roles.iloc[i] if pitcher_roles is not None else None
            )
            payloads.append(payload)
        
        return payloads
    
    def _prepare_training_data(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data with proper feature selection."""
        
        # Merge features and labels
        merged = features_df.merge(
            labels_df[['pitcher_id', 'as_of_date', 'label_injury_within_21d']],
            on=['pitcher_id', 'as_of_date'],
            how='inner'
        )
        
        # Select features for training
        feature_cols = [
            # Role features
            'avg_innings_per_appearance', 'multi_inning_rate_7d', 'multi_inning_rate_14d',
            'back_to_back_days_7d', 'back_to_back_days_14d',
            
            # Workload features
            'last_game_pitches', 'last_3_games_pitches',
            'roll7d_pitch_count', 'roll14d_pitch_count', 'roll30d_pitch_count',
            'roll7d_appearances', 'roll14d_appearances', 'roll30d_appearances',
            'roll7d_avg_pitches_per_appearance', 'roll14d_avg_pitches_per_appearance',
            'roll30d_avg_pitches_per_appearance',
            
            # Rest features
            'avg_rest_days', 'std_rest_days', 'min_rest_days', 'max_rest_days',
            'days_since_last_appearance',
            
            # Fatigue features
            'roll7d_release_speed_ewm_mean', 'roll14d_release_speed_ewm_mean',
            'roll7d_release_speed_delta_vs_30d', 'roll14d_release_speed_delta_vs_30d',
            'roll7d_release_spin_rate_ewm_mean', 'roll14d_release_spin_rate_ewm_mean',
            'roll7d_release_spin_rate_delta_vs_30d', 'roll14d_release_spin_rate_delta_vs_30d',
            'vel_decline_7d_vs_30d', 'vel_decline_14d_vs_30d',
            
            # Pitch mix features
            'roll7d_breaking_ball_pct', 'roll14d_breaking_ball_pct',
            'roll7d_breaking_ball_delta_vs_30d', 'roll14d_breaking_ball_delta_vs_30d',
            
            # Completeness features
            'data_completeness_score'
        ]
        
        # Filter to available features
        available_features = [col for col in feature_cols if col in merged.columns]
        
        # Fill missing values
        X = merged[available_features].fillna(0)
        y = merged['label_injury_within_21d']
        groups = merged['pitcher_id']  # For group-aware evaluation
        
        logger.info(f"Prepared training data: {len(X)} samples, {len(available_features)} features")
        
        return X, y, groups
    
    def _time_based_split(
        self,
        X: pd.DataFrame,
        test_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform time-based split to avoid future leakage."""
        
        # Sort by as_of_date (assuming it's in the index or we can infer order)
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))
        
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, n_samples)
        
        return train_idx, val_idx
    
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series
    ):
        """Train XGBoost model with class weights."""
        
        # Calculate class weights for imbalance
        class_weights = self._calculate_class_weights(y_train)
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'scale_pos_weight': class_weights['positive'] / class_weights['negative']
        }
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Store feature importance
        importance = self.xgb_model.feature_importances_
        self.feature_importance = dict(zip(self.feature_names, importance))
        
        logger.info("XGBoost model trained successfully")
    
    def _train_calibration(
        self,
        X_val: np.ndarray,
        y_val: pd.Series,
        groups_val: pd.Series
    ):
        """Train calibration models for overall and subgroups."""
        
        # Get raw predictions
        raw_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        
        # Overall calibration
        self.calibrator_overall = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_overall.fit(raw_probs, y_val)
        
        # Subgroup calibration (starters vs relievers)
        if hasattr(groups_val, 'index') and 'role' in groups_val.index:
            starter_mask = groups_val['role'] == 'starter'
            reliever_mask = groups_val['role'] == 'reliever'
            
            if starter_mask.sum() > 10:
                self.calibrator_starters = IsotonicRegression(out_of_bounds='clip')
                self.calibrator_starters.fit(raw_probs[starter_mask], y_val[starter_mask])
            
            if reliever_mask.sum() > 10:
                self.calibrator_relievers = IsotonicRegression(out_of_bounds='clip')
                self.calibrator_relievers.fit(raw_probs[reliever_mask], y_val[reliever_mask])
        
        logger.info("Calibration models trained successfully")
    
    def _apply_calibration(
        self,
        raw_probs: np.ndarray,
        pitcher_roles: Optional[pd.Series]
    ) -> np.ndarray:
        """Apply appropriate calibration based on pitcher role."""
        
        calibrated_probs = np.copy(raw_probs)
        
        # Apply subgroup calibration if available
        if pitcher_roles is not None:
            for i, role in enumerate(pitcher_roles):
                if role == 'starter' and self.calibrator_starters is not None:
                    calibrated_probs[i] = self.calibrator_starters.predict([raw_probs[i]])[0]
                elif role == 'reliever' and self.calibrator_relievers is not None:
                    calibrated_probs[i] = self.calibrator_relievers.predict([raw_probs[i]])[0]
                else:
                    # Fall back to overall calibration
                    calibrated_probs[i] = self.calibrator_overall.predict([raw_probs[i]])[0]
        else:
            # Apply overall calibration
            calibrated_probs = self.calibrator_overall.predict(raw_probs)
        
        return calibrated_probs
    
    def _generate_enriched_payload(
        self,
        features: pd.Series,
        calibrated_prob: float,
        raw_prob: float,
        role: Optional[str]
    ) -> Dict[str, Any]:
        """Generate enriched prediction payload."""
        
        # Risk bucket based on percentile
        if calibrated_prob < 0.1:
            risk_bucket = 'low'
        elif calibrated_prob < 0.3:
            risk_bucket = 'medium'
        else:
            risk_bucket = 'high'
        
        # Confidence based on data completeness
        completeness = features.get('data_completeness_score', 0.0)
        if completeness >= 0.8:
            confidence = 'high'
        elif completeness >= 0.5:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Top contributors
        contributors = self._get_top_contributors(features, top_k=3)
        
        # Cohort percentile (simplified - could be enhanced with actual cohort data)
        cohort_percentile = self._estimate_cohort_percentile(calibrated_prob, role)
        
        # Recommended actions
        recommended_actions = self._generate_recommended_actions(features, calibrated_prob)
        
        # Data completeness map
        completeness_map = self._get_completeness_map(features)
        
        return {
            'model_version': self.model_version,
            'risk_score_calibrated': float(calibrated_prob),
            'risk_score_raw': float(raw_prob),
            'risk_bucket': risk_bucket,
            'confidence': confidence,
            'contributors': contributors,
            'cohort_percentile': cohort_percentile,
            'recommended_actions': recommended_actions,
            'data_completeness': completeness_map
        }
    
    def _get_top_contributors(
        self,
        features: pd.Series,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Get top contributing features with explanations."""
        
        # Calculate feature contributions (simplified SHAP-like approach)
        contributions = []
        
        for feature_name in self.feature_names:
            if feature_name in features and feature_name in self.feature_importance:
                value = features[feature_name]
                importance = self.feature_importance[feature_name]
                
                # Determine direction (simplified)
                direction = self._get_feature_direction(feature_name, value)
                
                contributions.append({
                    'name': feature_name,
                    'value': float(value),
                    'importance': float(importance),
                    'direction': direction,
                    'percentile': self._estimate_percentile(value, feature_name)
                })
        
        # Sort by importance and take top k
        contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        return contributions[:top_k]
    
    def _get_feature_direction(self, feature_name: str, value: float) -> str:
        """Determine if feature increases or decreases risk."""
        
        # Simplified logic based on feature names
        if 'vel_decline' in feature_name or 'pitch_count' in feature_name:
            return 'increases' if value > 0 else 'decreases'
        elif 'rest_days' in feature_name:
            return 'decreases' if value > 0 else 'increases'
        elif 'breaking_ball' in feature_name:
            return 'increases' if value > 0 else 'decreases'
        else:
            return 'neutral'
    
    def _estimate_percentile(self, value: float, feature_name: str) -> float:
        """Estimate percentile of feature value (simplified)."""
        # This would be enhanced with actual distribution data
        return 50.0  # Placeholder
    
    def _estimate_cohort_percentile(self, risk_score: float, role: Optional[str]) -> float:
        """Estimate risk percentile within cohort (simplified)."""
        # This would be enhanced with actual cohort data
        return risk_score * 100  # Placeholder
    
    def _generate_recommended_actions(
        self,
        features: pd.Series,
        risk_score: float
    ) -> List[str]:
        """Generate recommended actions based on features and risk."""
        
        actions = []
        
        # High pitch count recommendations
        if features.get('last_game_pitches', 0) > 100:
            actions.append("Consider reduced pitch count in next outing")
        
        if features.get('roll7d_pitch_count', 0) > 300:
            actions.append("Monitor cumulative workload - consider extra rest day")
        
        # Velocity decline recommendations
        if features.get('vel_decline_7d_vs_30d', 0) < -2.0:
            actions.append("Velocity decline detected - consider mechanical review")
        
        # Rest day recommendations
        if features.get('avg_rest_days', 0) < 3.0:
            actions.append("Consider increasing rest between appearances")
        
        # Breaking ball recommendations
        if features.get('roll7d_breaking_ball_pct', 0) > 0.6:
            actions.append("High breaking ball usage - consider pitch mix adjustment")
        
        # Default recommendation for high risk
        if risk_score > 0.3 and not actions:
            actions.append("Elevated risk signals - monitor closely and consider workload reduction")
        
        return actions[:3]  # Limit to top 3 recommendations
    
    def _get_completeness_map(self, features: pd.Series) -> Dict[str, bool]:
        """Get data completeness map for each feature family."""
        
        completeness_map = {}
        
        # Check completeness for different feature families
        feature_families = {
            'velocity_data': ['roll7d_release_speed_ewm_mean', 'roll14d_release_speed_ewm_mean'],
            'spin_data': ['roll7d_release_spin_rate_ewm_mean', 'roll14d_release_spin_rate_ewm_mean'],
            'workload_data': ['roll7d_pitch_count', 'roll14d_pitch_count', 'roll30d_pitch_count'],
            'rest_data': ['avg_rest_days', 'std_rest_days'],
            'pitch_mix_data': ['roll7d_breaking_ball_pct', 'roll14d_breaking_ball_pct']
        }
        
        for family, feature_list in feature_families.items():
            available_features = [f for f in feature_list if f in features]
            completeness_map[family] = len(available_features) > 0
        
        return completeness_map
    
    def _calculate_class_weights(self, y: pd.Series) -> Dict[str, float]:
        """Calculate class weights for imbalanced data."""
        
        class_counts = y.value_counts()
        total_samples = len(y)
        
        weights = {
            'positive': total_samples / (2 * class_counts.get(1, 1)),
            'negative': total_samples / (2 * class_counts.get(0, 1))
        }
        
        return weights
    
    def _evaluate_model(
        self,
        X_val: np.ndarray,
        y_val: pd.Series,
        groups_val: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        
        # Get predictions
        raw_probs = self.xgb_model.predict_proba(X_val)[:, 1]
        calibrated_probs = self._apply_calibration(raw_probs, None)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_val, calibrated_probs),
            'pr_auc': average_precision_score(y_val, calibrated_probs),
            'brier_score': brier_score_loss(y_val, calibrated_probs)
        }
        
        # Calculate recall at top 10%
        n_top = max(1, int(len(y_val) * 0.1))
        top_indices = np.argsort(calibrated_probs)[-n_top:]
        recall_top_10 = y_val.iloc[top_indices].mean()
        metrics['recall_top_10'] = recall_top_10
        
        # Calculate lift at top 10%
        baseline_rate = y_val.mean()
        metrics['lift_top_10'] = recall_top_10 / baseline_rate if baseline_rate > 0 else 0
        
        logger.info(f"Model evaluation completed: {metrics}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        
        model_data = {
            'model_version': self.model_version,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'calibrator_overall': self.calibrator_overall,
            'calibrator_starters': self.calibrator_starters,
            'calibrator_relievers': self.calibrator_relievers,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        
        model_data = joblib.load(filepath)
        
        self.model_version = model_data['model_version']
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.calibrator_overall = model_data['calibrator_overall']
        self.calibrator_starters = model_data['calibrator_starters']
        self.calibrator_relievers = model_data['calibrator_relievers']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.training_metrics = model_data['training_metrics']
        
        logger.info(f"Model loaded from {filepath}")
