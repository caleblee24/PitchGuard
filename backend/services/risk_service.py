"""
Risk assessment service for injury risk prediction.
"""

from sqlalchemy.orm import Session
from datetime import date, timedelta
import logging
from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

from database.models import Pitcher, FeatureSnapshot, Appearance, Injury
from api.schemas import RiskLevel, RiskFactor
from utils.config import settings

logger = logging.getLogger(__name__)

class RiskAssessmentService:
    """Service for assessing pitcher injury risk."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained injury risk model."""
        try:
            if os.path.exists(settings.model_path):
                model_data = joblib.load(settings.model_path)
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names', [])
                logger.info(f"Loaded model from {settings.model_path}")
            else:
                logger.warning(f"Model not found at {settings.model_path}, using mock model")
                self._create_mock_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock model for development purposes."""
        self.model = MockRiskModel()
        self.feature_names = [
            'roll21d_pitch_count', 'roll21d_avg_vel', 'roll21d_vel_decline',
            'roll7d_pitch_count', 'roll7d_vel_decline', 'vel_trend_slope',
            'days_since_last_appearance', 'workload_intensity', 'fatigue_score'
        ]
        logger.info("Using mock risk model for development")
    
    def assess_risk(
        self,
        pitcher_id: int,
        as_of_date: date,
        db: Session
    ) -> Dict:
        """
        Assess injury risk for a pitcher on a specific date.
        
        Returns dict with risk_level, risk_score, confidence, etc.
        """
        
        # Get or create feature snapshot for this date
        features = self._get_feature_snapshot(pitcher_id, as_of_date, db)
        
        if not features:
            return self._create_no_data_response(pitcher_id, as_of_date, db)
        
        return self.assess_risk_from_features(features, db)
    
    def assess_risk_from_features(
        self,
        features: FeatureSnapshot,
        db: Session
    ) -> Dict:
        """Assess risk from a feature snapshot."""
        
        # Extract feature values
        feature_values = self._extract_features(features)
        
        # Get risk score from model
        risk_score, confidence = self._predict_risk(feature_values)
        
        # Determine risk level
        risk_level = self._categorize_risk(risk_score)
        
        # Get risk factors and explanations
        risk_factors = self._get_risk_factors(features, feature_values)
        primary_concerns = self._get_primary_concerns(risk_factors)
        recommendations = self._get_recommendations(risk_level, primary_concerns)
        
        # Data quality assessment
        data_completeness = features.data_completeness or "medium"
        days_of_data = self._calculate_days_of_data(features, db)
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "primary_concerns": primary_concerns,
            "recommendations": recommendations,
            "data_completeness": data_completeness,
            "days_of_data": days_of_data,
            "last_appearance": self._get_last_appearance_date(features.pitcher_id, db)
        }
    
    def _get_feature_snapshot(
        self,
        pitcher_id: int,
        as_of_date: date,
        db: Session
    ) -> Optional[FeatureSnapshot]:
        """Get or create feature snapshot for the given date."""
        
        # Try to find existing snapshot
        snapshot = db.query(FeatureSnapshot)\
            .filter(
                FeatureSnapshot.pitcher_id == pitcher_id,
                FeatureSnapshot.as_of_date == as_of_date
            ).first()
        
        if snapshot:
            return snapshot
        
        # If no snapshot exists, try to create one from recent data
        logger.info(f"Creating feature snapshot for pitcher {pitcher_id} on {as_of_date}")
        return self._create_feature_snapshot(pitcher_id, as_of_date, db)
    
    def _create_feature_snapshot(
        self,
        pitcher_id: int,
        as_of_date: date,
        db: Session
    ) -> Optional[FeatureSnapshot]:
        """Create a feature snapshot from appearance data."""
        
        # Get appearances up to the as_of_date
        appearances = db.query(Appearance)\
            .filter(
                Appearance.pitcher_id == pitcher_id,
                Appearance.game_date <= as_of_date
            )\
            .order_by(Appearance.game_date.desc())\
            .limit(50).all()  # Get last 50 appearances
        
        if len(appearances) < 5:  # Need minimum data
            return None
        
        # Calculate rolling features
        features = self._calculate_rolling_features(appearances, as_of_date)
        
        # Create snapshot
        snapshot = FeatureSnapshot(
            pitcher_id=pitcher_id,
            as_of_date=as_of_date,
            **features
        )
        
        # Don't persist for now (would need transaction management)
        return snapshot
    
    def _calculate_rolling_features(
        self,
        appearances: List[Appearance],
        as_of_date: date
    ) -> Dict:
        """Calculate rolling features from appearances."""
        
        # Sort appearances by date (most recent first)
        appearances = sorted(appearances, key=lambda x: x.game_date, reverse=True)
        
        # Filter appearances within rolling windows
        window_21d = [a for a in appearances if (as_of_date - a.game_date).days <= 21]
        window_7d = [a for a in appearances if (as_of_date - a.game_date).days <= 7]
        window_3d = [a for a in appearances if (as_of_date - a.game_date).days <= 3]
        
        features = {}
        
        # 21-day rolling features
        if window_21d:
            features['roll21d_pitch_count'] = sum(a.pitches_thrown for a in window_21d)
            features['roll21d_avg_vel'] = np.mean([a.avg_vel for a in window_21d])
            features['roll21d_appearances'] = len(window_21d)
            
            # Velocity decline
            if len(window_21d) >= 2:
                velocities = [a.avg_vel for a in window_21d]
                features['roll21d_vel_decline'] = velocities[0] - velocities[-1]
            else:
                features['roll21d_vel_decline'] = 0.0
        
        # 7-day rolling features
        if window_7d:
            features['roll7d_pitch_count'] = sum(a.pitches_thrown for a in window_7d)
            features['roll7d_avg_vel'] = np.mean([a.avg_vel for a in window_7d])
            features['roll7d_appearances'] = len(window_7d)
            
            if len(window_7d) >= 2:
                velocities = [a.avg_vel for a in window_7d]
                features['roll7d_vel_decline'] = velocities[0] - velocities[-1]
            else:
                features['roll7d_vel_decline'] = 0.0
        
        # 3-day rolling features
        if window_3d:
            features['roll3d_pitch_count'] = sum(a.pitches_thrown for a in window_3d)
            features['roll3d_avg_vel'] = np.mean([a.avg_vel for a in window_3d])
            features['roll3d_appearances'] = len(window_3d)
        
        # Trend calculations
        if len(appearances) >= 5:
            recent_vels = [a.avg_vel for a in appearances[:5]]
            dates = [(as_of_date - a.game_date).days for a in appearances[:5]]
            
            # Simple linear trend
            if len(set(dates)) > 1:  # Need variation in dates
                vel_trend = np.polyfit(dates, recent_vels, 1)[0]
                features['vel_trend_slope'] = vel_trend
            else:
                features['vel_trend_slope'] = 0.0
        
        # Recovery and workload patterns
        if appearances:
            features['days_since_last_appearance'] = (as_of_date - appearances[0].game_date).days
            
            # Simple workload intensity (recent vs typical)
            recent_workload = features.get('roll7d_pitch_count', 0)
            features['workload_intensity'] = min(recent_workload / 150.0, 2.0)  # Normalize
            
            # Simple fatigue score
            features['fatigue_score'] = features['workload_intensity'] * 0.7 + \
                                      (1.0 if features.get('roll21d_vel_decline', 0) > 1.0 else 0.0) * 0.3
        
        # Set defaults for missing features
        default_features = {
            'roll21d_pitch_count': 0, 'roll21d_avg_vel': 0.0, 'roll21d_vel_decline': 0.0,
            'roll7d_pitch_count': 0, 'roll7d_vel_decline': 0.0, 'vel_trend_slope': 0.0,
            'days_since_last_appearance': 7, 'workload_intensity': 0.5, 'fatigue_score': 0.3,
            'data_completeness': 'medium' if len(appearances) >= 10 else 'low'
        }
        
        for key, default_value in default_features.items():
            if key not in features:
                features[key] = default_value
        
        return features
    
    def _extract_features(self, features: FeatureSnapshot) -> List[float]:
        """Extract feature values in the correct order for the model."""
        
        feature_values = []
        for feature_name in self.feature_names:
            value = getattr(features, feature_name, 0.0)
            if value is None:
                value = 0.0
            feature_values.append(float(value))
        
        return feature_values
    
    def _predict_risk(self, feature_values: List[float]) -> tuple[float, float]:
        """Get risk prediction from the model."""
        
        if not self.model:
            return 0.3, 0.7  # Default moderate risk
        
        try:
            # Reshape for sklearn
            X = np.array([feature_values])
            
            # Get probability
            risk_prob = self.model.predict_proba(X)[0][1]  # Probability of positive class
            
            # Mock confidence based on feature completeness
            confidence = 0.8 if len([v for v in feature_values if v != 0]) >= 5 else 0.6
            
            return float(risk_prob), float(confidence)
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return 0.3, 0.5
    
    def _categorize_risk(self, risk_score: float) -> RiskLevel:
        """Categorize risk score into risk levels."""
        
        if risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_risk_factors(
        self,
        features: FeatureSnapshot,
        feature_values: List[float]
    ) -> List[RiskFactor]:
        """Get list of risk factors with importance scores."""
        
        risk_factors = []
        
        # High workload
        if features.roll21d_pitch_count and features.roll21d_pitch_count > 300:
            risk_factors.append(RiskFactor(
                factor="High 21-day pitch count",
                value=features.roll21d_pitch_count,
                importance=0.8,
                description=f"Thrown {features.roll21d_pitch_count} pitches in last 21 days"
            ))
        
        # Velocity decline
        if features.roll21d_vel_decline and features.roll21d_vel_decline > 1.0:
            risk_factors.append(RiskFactor(
                factor="Velocity decline",
                value=features.roll21d_vel_decline,
                importance=0.9,
                description=f"Velocity down {features.roll21d_vel_decline:.1f} MPH over 21 days"
            ))
        
        # Short rest
        if features.days_since_last_appearance and features.days_since_last_appearance < 2:
            risk_factors.append(RiskFactor(
                factor="Insufficient rest",
                value=features.days_since_last_appearance,
                importance=0.7,
                description=f"Only {features.days_since_last_appearance} days since last appearance"
            ))
        
        # High fatigue score
        if features.fatigue_score and features.fatigue_score > 0.7:
            risk_factors.append(RiskFactor(
                factor="High fatigue score",
                value=features.fatigue_score,
                importance=0.8,
                description=f"Fatigue score of {features.fatigue_score:.2f} indicates high stress"
            ))
        
        return risk_factors
    
    def _get_primary_concerns(self, risk_factors: List[RiskFactor]) -> List[str]:
        """Get primary concerns from risk factors."""
        
        # Sort by importance and take top concerns
        sorted_factors = sorted(risk_factors, key=lambda x: x.importance, reverse=True)
        return [factor.factor for factor in sorted_factors[:3]]
    
    def _get_recommendations(self, risk_level: RiskLevel, concerns: List[str]) -> List[str]:
        """Get recommendations based on risk level and concerns."""
        
        recommendations = []
        
        if risk_level == RiskLevel.HIGH:
            recommendations.append("Consider rest day or reduced workload")
            recommendations.append("Monitor velocity closely in next appearance")
            if "High 21-day pitch count" in concerns:
                recommendations.append("Limit pitch count in next outing")
        
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Continue monitoring workload patterns")
            if "Velocity decline" in concerns:
                recommendations.append("Track velocity trend over next few outings")
        
        else:
            recommendations.append("Maintain current workload management")
        
        return recommendations
    
    def _calculate_days_of_data(self, features: FeatureSnapshot, db: Session) -> int:
        """Calculate how many days of data we have for this pitcher."""
        
        first_appearance = db.query(Appearance)\
            .filter(Appearance.pitcher_id == features.pitcher_id)\
            .order_by(Appearance.game_date)\
            .first()
        
        if first_appearance:
            return (features.as_of_date - first_appearance.game_date).days
        
        return 0
    
    def _get_last_appearance_date(self, pitcher_id: int, db: Session) -> Optional[date]:
        """Get the date of the pitcher's last appearance."""
        
        last_appearance = db.query(Appearance)\
            .filter(Appearance.pitcher_id == pitcher_id)\
            .order_by(Appearance.game_date.desc())\
            .first()
        
        return last_appearance.game_date if last_appearance else None
    
    def _create_no_data_response(
        self,
        pitcher_id: int,
        as_of_date: date,
        db: Session
    ) -> Dict:
        """Create response when insufficient data is available."""
        
        return {
            "risk_level": RiskLevel.MEDIUM,
            "risk_score": 0.5,
            "confidence": 0.3,
            "risk_factors": [],
            "primary_concerns": ["Insufficient data for assessment"],
            "recommendations": ["Need more appearance data for accurate assessment"],
            "data_completeness": "low",
            "days_of_data": 0,
            "last_appearance": self._get_last_appearance_date(pitcher_id, db)
        }


class MockRiskModel:
    """Mock model for development purposes."""
    
    def predict_proba(self, X):
        """Mock prediction based on simple heuristics."""
        
        # X is array of feature values
        features = X[0] if len(X) > 0 else []
        
        # Simple heuristic: higher workload and velocity decline = higher risk
        risk_score = 0.3  # Base risk
        
        if len(features) >= 9:
            # High pitch count increases risk
            if features[0] > 250:  # roll21d_pitch_count
                risk_score += 0.2
            
            # Velocity decline increases risk
            if features[2] > 1.0:  # roll21d_vel_decline
                risk_score += 0.3
            
            # High fatigue score increases risk
            if features[8] > 0.7:  # fatigue_score
                risk_score += 0.2
        
        risk_score = min(0.9, max(0.1, risk_score))
        
        return np.array([[1 - risk_score, risk_score]])
