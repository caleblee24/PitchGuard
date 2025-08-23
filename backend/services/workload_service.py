"""
Workload analysis service for pitcher workload monitoring.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, timedelta
import logging
from typing import List, Dict
import numpy as np

from database.models import Pitcher, Appearance
from api.schemas import TrendDirection
from utils.config import settings

logger = logging.getLogger(__name__)

class WorkloadService:
    """Service for workload analysis and monitoring."""
    
    def __init__(self):
        self.baseline_stats = self._load_baseline_stats()
    
    def _load_baseline_stats(self) -> Dict:
        """Load baseline workload statistics for comparison."""
        return {
            'starter': {
                'avg_pitches_per_game': 95,
                'avg_rest_days': 4,
                'max_safe_pitches_21d': 350
            },
            'reliever': {
                'avg_pitches_per_game': 25,
                'avg_rest_days': 2,
                'max_safe_pitches_21d': 200
            }
        }
    
    def calculate_trend(self, values: List[float]) -> TrendDirection:
        """Calculate trend direction from a series of values."""
        
        if len(values) < 3:
            return TrendDirection.STABLE
        
        # Use linear regression to detect trend
        x = np.arange(len(values))
        
        try:
            slope = np.polyfit(x, values, 1)[0]
            
            # Threshold for considering a trend significant
            threshold = np.std(values) * 0.1 if len(values) > 1 else 0.1
            
            if slope > threshold:
                return TrendDirection.INCREASING
            elif slope < -threshold:
                return TrendDirection.DECREASING
            else:
                return TrendDirection.STABLE
                
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            return TrendDirection.STABLE
    
    def assess_workload_intensity(
        self,
        pitcher_id: int,
        recent_pitches: int,
        avg_rest_days: float,
        db: Session
    ) -> str:
        """Assess workload intensity compared to role baseline."""
        
        # Get pitcher role
        pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
        if not pitcher:
            return "normal"
        
        role = pitcher.role.lower()
        baseline = self.baseline_stats.get(role, self.baseline_stats['starter'])
        
        # Calculate intensity factors
        pitch_intensity = recent_pitches / baseline['avg_pitches_per_game'] / 7  # Normalize to daily
        rest_intensity = baseline['avg_rest_days'] / max(avg_rest_days, 0.5)  # Inverse: less rest = higher intensity
        
        # Combined intensity score
        intensity_score = (pitch_intensity * 0.7) + (rest_intensity * 0.3)
        
        if intensity_score > 1.5:
            return "high"
        elif intensity_score > 1.2:
            return "elevated"
        else:
            return "normal"
    
    def compare_to_role(
        self,
        pitcher_id: int,
        total_pitches: int,
        appearances: int,
        db: Session
    ) -> Dict:
        """Compare pitcher's workload to others in the same role."""
        
        # Get pitcher role
        pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
        if not pitcher:
            return {"error": "Pitcher not found"}
        
        role = pitcher.role.lower()
        
        # Get workload statistics for all pitchers in the same role
        role_stats = db.query(
            func.avg(func.sum(Appearance.pitches_thrown)).label('avg_total_pitches'),
            func.avg(func.count(Appearance.id)).label('avg_appearances'),
            func.percentile_cont(0.25).within_group(func.sum(Appearance.pitches_thrown)).label('p25_pitches'),
            func.percentile_cont(0.50).within_group(func.sum(Appearance.pitches_thrown)).label('p50_pitches'),
            func.percentile_cont(0.75).within_group(func.sum(Appearance.pitches_thrown)).label('p75_pitches'),
            func.percentile_cont(0.90).within_group(func.sum(Appearance.pitches_thrown)).label('p90_pitches')
        ).join(Pitcher).filter(
            Pitcher.role == role,
            Appearance.game_date >= date.today() - timedelta(days=21)
        ).group_by(Appearance.pitcher_id).first()
        
        if not role_stats:
            # Fallback to baseline stats
            baseline = self.baseline_stats[role]
            return {
                "role": role,
                "pitcher_pitches": total_pitches,
                "role_average": baseline['avg_pitches_per_game'] * 7,  # Weekly estimate
                "percentile": 50,  # Default to median
                "comparison": "average"
            }
        
        # Calculate percentile
        percentile = self._calculate_percentile(
            total_pitches,
            [role_stats.p25_pitches, role_stats.p50_pitches, role_stats.p75_pitches, role_stats.p90_pitches]
        )
        
        # Determine comparison category
        if percentile >= 90:
            comparison = "very high"
        elif percentile >= 75:
            comparison = "high"
        elif percentile >= 25:
            comparison = "average"
        else:
            comparison = "low"
        
        return {
            "role": role,
            "pitcher_pitches": total_pitches,
            "pitcher_appearances": appearances,
            "role_average_pitches": round(float(role_stats.avg_total_pitches or 0), 1),
            "role_average_appearances": round(float(role_stats.avg_appearances or 0), 1),
            "percentile": percentile,
            "comparison": comparison
        }
    
    def _calculate_percentile(self, value: float, percentile_values: List[float]) -> int:
        """Calculate which percentile a value falls into."""
        
        if not percentile_values or len(percentile_values) < 4:
            return 50  # Default to median
        
        p25, p50, p75, p90 = percentile_values
        
        if value >= p90:
            return 90
        elif value >= p75:
            return 75
        elif value >= p50:
            return 50
        elif value >= p25:
            return 25
        else:
            return 10
    
    def get_workload_recommendations(
        self,
        pitcher_id: int,
        current_workload: Dict,
        db: Session
    ) -> List[str]:
        """Get workload management recommendations."""
        
        recommendations = []
        
        # Get pitcher info
        pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
        if not pitcher:
            return ["Unable to assess - pitcher not found"]
        
        role = pitcher.role.lower()
        baseline = self.baseline_stats[role]
        
        # Analyze workload factors
        total_pitches = current_workload.get('total_pitches', 0)
        appearances = current_workload.get('appearances', 0)
        avg_rest = current_workload.get('avg_rest_days', 0)
        
        # High pitch count recommendations
        if total_pitches > baseline['max_safe_pitches_21d']:
            recommendations.append(f"High 21-day pitch count ({total_pitches}). Consider reducing workload.")
        
        # Rest recommendations
        if avg_rest < baseline['avg_rest_days'] * 0.7:
            recommendations.append(f"Below average rest ({avg_rest:.1f} days). Schedule additional rest.")
        
        # Role-specific recommendations
        if role == 'starter':
            if appearances > 5:  # More than 1 start per week on average
                recommendations.append("High start frequency. Monitor for fatigue.")
        else:  # reliever
            if appearances > 10:  # More than every other day
                recommendations.append("High appearance frequency. Consider limiting usage.")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Workload within normal parameters. Continue current management.")
        
        return recommendations
    
    def calculate_workload_score(
        self,
        pitcher_id: int,
        workload_data: Dict,
        db: Session
    ) -> Dict:
        """Calculate overall workload stress score."""
        
        # Get pitcher role
        pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
        if not pitcher:
            return {"error": "Pitcher not found"}
        
        role = pitcher.role.lower()
        baseline = self.baseline_stats[role]
        
        # Calculate component scores (0-1 scale)
        pitch_score = min(workload_data.get('total_pitches', 0) / baseline['max_safe_pitches_21d'], 1.0)
        
        rest_score = max(0, 1 - (workload_data.get('avg_rest_days', baseline['avg_rest_days']) / baseline['avg_rest_days']))
        
        frequency_score = min(workload_data.get('appearances', 0) / (21 / baseline['avg_rest_days']), 1.0)
        
        # Weighted composite score
        composite_score = (pitch_score * 0.5) + (rest_score * 0.3) + (frequency_score * 0.2)
        
        # Categorize score
        if composite_score >= 0.8:
            category = "very high"
            color = "red"
        elif composite_score >= 0.6:
            category = "high"
            color = "orange"
        elif composite_score >= 0.4:
            category = "moderate"
            color = "yellow"
        else:
            category = "low"
            color = "green"
        
        return {
            "composite_score": round(composite_score, 2),
            "category": category,
            "color": color,
            "components": {
                "pitch_volume": round(pitch_score, 2),
                "rest_deficit": round(rest_score, 2),
                "frequency": round(frequency_score, 2)
            }
        }
