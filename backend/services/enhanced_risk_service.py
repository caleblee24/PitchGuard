"""
Enhanced Risk Assessment Service for PitchGuard MVP.
Provides injury risk predictions using the enhanced XGBoost model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import logging
import os
import sys

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.enhanced_features import EnhancedFeatureEngineer
from modeling.enhanced_model import EnhancedInjuryRiskModel

logger = logging.getLogger(__name__)

class EnhancedRiskService:
    """Enhanced service for calculating injury risk assessments."""
    
    def __init__(self):
        self.model_version = "pitchguard_xgb_v1_2025-08-20"
        self.model = None
        self.feature_engineer = EnhancedFeatureEngineer()
        
        # Try to load the trained model
        self._load_model()
    
    def _load_model(self):
        """Load the trained enhanced model."""
        try:
            # Try to load the new real data model first
            model_path = "models/enhanced_model_real_data.joblib"
            if os.path.exists(model_path):
                import joblib
                self.model = joblib.load(model_path)
                logger.info(f"Loaded enhanced real data model: {self.model.model_version}")
            else:
                # Fallback to old model path
                model_path = "models/enhanced_injury_risk_model.pkl"
                if os.path.exists(model_path):
                    self.model = EnhancedInjuryRiskModel()
                    self.model.load_model(model_path)
                    logger.info(f"Loaded enhanced model: {self.model.model_version}")
                else:
                    logger.warning("Enhanced model not found. Using fallback mock model.")
                    self.model = None
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            self.model = None
    
    def calculate_risk_score(
        self, 
        pitcher_id: int, 
        as_of_date: datetime,
        appearances_data: Optional[pd.DataFrame] = None,
        pitches_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate injury risk score for a pitcher using enhanced model.
        
        Args:
            pitcher_id: ID of the pitcher
            as_of_date: Date to calculate risk as of
            appearances_data: Optional appearances data
            pitches_data: Optional pitches data
            
        Returns:
            Risk score between 0 and 1
        """
        if self.model is not None and appearances_data is not None and pitches_data is not None:
            try:
                # Use enhanced model
                return self._calculate_enhanced_risk_score(
                    pitcher_id, as_of_date, appearances_data, pitches_data
                )
            except Exception as e:
                logger.error(f"Error in enhanced risk calculation: {e}")
                # Fall back to mock implementation
        
        # Fallback mock implementation
        return self._calculate_mock_risk_score(pitcher_id, as_of_date)
    
    def _calculate_enhanced_risk_score(
        self,
        pitcher_id: int,
        as_of_date: datetime,
        appearances_data: pd.DataFrame,
        pitches_data: pd.DataFrame
    ) -> float:
        """Calculate risk score using the enhanced model."""
        
        # Filter data for this pitcher
        pitcher_appearances = appearances_data[
            (appearances_data['pitcher_id'] == pitcher_id) &
            (appearances_data['game_date'] <= as_of_date)
        ]
        
        pitcher_pitches = pitches_data[
            (pitches_data['pitcher_id'] == pitcher_id) &
            (pitches_data['game_date'] <= as_of_date)
        ]
        
        if len(pitcher_appearances) == 0:
            logger.warning(f"No appearance data for pitcher {pitcher_id}")
            return 0.1  # Default low risk
        
        # Engineer features
        features = self.feature_engineer.engineer_features(
            pitcher_appearances, pitcher_pitches, pd.DataFrame(), as_of_date
        )
        
        if features.empty:
            logger.warning(f"No features computed for pitcher {pitcher_id}")
            return 0.1  # Default low risk
        
        # Make prediction
        prediction = self.model.predict(features)
        
        if prediction:
            return prediction[0]['risk_score_calibrated']
        else:
            return 0.1  # Default low risk
    
    def _calculate_mock_risk_score(self, pitcher_id: int, as_of_date: datetime) -> float:
        """Calculate mock risk score for fallback."""
        
        # Simple heuristic based on pitcher ID and date
        base_risk = 0.1 + (pitcher_id % 100) / 1000.0
        
        # Add some temporal variation
        days_since_epoch = (as_of_date - datetime(2024, 1, 1)).days
        temporal_factor = 0.05 * np.sin(days_since_epoch / 30.0)
        
        risk_score = max(0.0, min(1.0, base_risk + temporal_factor))
        
        logger.info(f"Calculated mock risk score {risk_score:.3f} for pitcher {pitcher_id}")
        
        return risk_score
    
    def get_current_risk_assessment(
        self,
        pitcher_id: int,
        as_of_date: Optional[datetime] = None,
        appearances_data: Optional[pd.DataFrame] = None,
        pitches_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk assessment for a pitcher.
        
        Args:
            pitcher_id: ID of the pitcher
            as_of_date: Date to assess risk as of (defaults to current date)
            appearances_data: Optional appearances data
            pitches_data: Optional pitches data
            
        Returns:
            Dictionary with enhanced risk assessment details
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Ensure as_of_date is a datetime object
        if isinstance(as_of_date, date):
            as_of_date = datetime.combine(as_of_date, datetime.min.time())
        
        if self.model is not None and appearances_data is not None and pitches_data is not None:
            try:
                # Use enhanced model for full assessment
                return self._get_enhanced_risk_assessment(
                    pitcher_id, as_of_date, appearances_data, pitches_data
                )
            except Exception as e:
                logger.error(f"Error in enhanced risk assessment: {e}")
                # Fall back to mock implementation
        
        # Fallback mock implementation
        return self._get_mock_risk_assessment(pitcher_id, as_of_date)
    
    def _get_enhanced_risk_assessment(
        self,
        pitcher_id: int,
        as_of_date: datetime,
        appearances_data: pd.DataFrame,
        pitches_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get enhanced risk assessment using the trained model."""
        
        # Filter data for this pitcher
        # Ensure game_date columns are datetime objects for comparison
        if 'game_date' in appearances_data.columns:
            appearances_data = appearances_data.copy()
            appearances_data['game_date'] = pd.to_datetime(appearances_data['game_date'])
        
        if 'game_date' in pitches_data.columns:
            pitches_data = pitches_data.copy()
            pitches_data['game_date'] = pd.to_datetime(pitches_data['game_date'])
        
        pitcher_appearances = appearances_data[
            (appearances_data['pitcher_id'] == pitcher_id) &
            (appearances_data['game_date'] <= as_of_date)
        ]
        
        pitcher_pitches = pitches_data[
            (pitches_data['pitcher_id'] == pitcher_id) &
            (pitches_data['game_date'] <= as_of_date)
        ]
        
        if len(pitcher_appearances) == 0:
            logger.warning(f"No appearance data for pitcher {pitcher_id}")
            return self._get_mock_risk_assessment(pitcher_id, as_of_date)
        
        # Engineer features
        features = self.feature_engineer.engineer_features(
            pitcher_appearances, pitcher_pitches, pd.DataFrame(), as_of_date
        )
        
        if features.empty:
            logger.warning(f"No features computed for pitcher {pitcher_id}")
            return self._get_mock_risk_assessment(pitcher_id, as_of_date)
        
        # Get pitcher role for subgroup calibration
        pitcher_role = features.iloc[0].get('role', 'unknown') if not features.empty else 'unknown'
        
        # Make prediction
        prediction = self.model.predict(features, pd.Series([pitcher_role]))
        
        if prediction:
            # Return enriched payload
            return {
                "pitcher_id": pitcher_id,
                "as_of_date": as_of_date.isoformat(),
                **prediction[0]  # Include all enriched fields
            }
        else:
            return self._get_mock_risk_assessment(pitcher_id, as_of_date)
    
    def _get_mock_risk_assessment(self, pitcher_id: int, as_of_date: datetime) -> Dict[str, Any]:
        """Get mock risk assessment for fallback."""
        
        # Calculate risk score
        risk_score = self._calculate_mock_risk_score(pitcher_id, as_of_date)
        
        # Determine risk level
        if risk_score < 0.2:
            risk_level = "low"
        elif risk_score < 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Mock feature contributions
        features = self._calculate_mock_features(pitcher_id, as_of_date)
        
        # Get top contributors
        top_contributors = self._get_top_contributors(features)
        
        return {
            "pitcher_id": pitcher_id,
            "as_of_date": as_of_date.isoformat(),
            "risk_score_calibrated": risk_score,
            "risk_bucket": risk_level,
            "model_version": "pitchguard_mock_fallback",
            "confidence": "low",
            "contributors": top_contributors,
            "cohort_percentile": risk_score * 100,
            "recommended_actions": ["Monitor workload", "Consider rest day"],
            "data_completeness": {
                "velocity_data": True,
                "workload_data": True,
                "rest_data": True
            }
        }
    
    def _calculate_mock_features(self, pitcher_id: int, as_of_date: datetime) -> Dict:
        """Calculate mock features for demonstration."""
        
        # Mock feature values based on pitcher ID
        features = {
            "roll7d_pitch_count": 150 + (pitcher_id % 50),
            "roll14d_pitch_count": 300 + (pitcher_id % 100),
            "avg_velocity_7d": 92.5 + (pitcher_id % 10) / 10.0,
            "velocity_decline_7d": -1.2 + (pitcher_id % 5) / 10.0,
            "rest_days": 4 + (pitcher_id % 3),
            "breaking_ball_pct": 0.4 + (pitcher_id % 30) / 100.0
        }
        
        return features
    
    def _get_top_contributors(self, features: Dict) -> List[Dict]:
        """Get top contributing features to risk score."""
        
        # Mock feature importance
        importance_map = {
            "roll7d_pitch_count": 0.3,
            "velocity_decline_7d": 0.25,
            "rest_days": 0.2,
            "breaking_ball_pct": 0.15,
            "avg_velocity_7d": 0.1
        }
        
        contributors = []
        for feature_name, value in features.items():
            if feature_name in importance_map:
                # Determine direction
                if "decline" in feature_name or "pitch_count" in feature_name:
                    direction = "increases" if value > 0 else "decreases"
                elif "rest_days" in feature_name:
                    direction = "decreases" if value > 0 else "increases"
                else:
                    direction = "neutral"
                
                contributors.append({
                    "name": feature_name,
                    "value": value,
                    "importance": importance_map[feature_name],
                    "direction": direction,
                    "percentile": 50.0
                })
        
        # Sort by importance and return top 3
        contributors.sort(key=lambda x: x["importance"], reverse=True)
        return contributors[:3]

# Create instance for use in API
enhanced_risk_service = EnhancedRiskService()
