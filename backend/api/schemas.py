"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from enum import Enum

class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TrendDirection(str, Enum):
    """Trend direction enumeration."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"

# Base response model
class BaseResponse(BaseModel):
    """Base response model with success indicator."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Pitcher schemas
class PitcherResponse(BaseModel):
    """Basic pitcher information response."""
    pitcher_id: int
    name: str
    team: str
    role: str
    last_appearance: Optional[date]
    current_risk_level: RiskLevel
    current_risk_score: float = Field(ge=0.0, le=1.0)
    season_appearances: int
    recent_velocity: Optional[float]
    velocity_trend: TrendDirection

class AppearanceInfo(BaseModel):
    """Appearance information."""
    date: str
    pitches: int
    velocity: float
    spin_rate: int
    innings: float

class InjuryInfo(BaseModel):
    """Injury information."""
    start_date: str
    end_date: Optional[str]
    type: str

class PitcherDetailResponse(BaseModel):
    """Detailed pitcher information response."""
    pitcher_id: int
    name: str
    team: str
    role: str
    baseline_velocity: Optional[float]
    baseline_spin_rate: Optional[int]
    injury_risk_profile: Optional[str]
    
    # Current risk assessment
    current_risk_level: RiskLevel
    current_risk_score: float = Field(ge=0.0, le=1.0)
    risk_factors: List[str]
    
    # Season statistics
    season_appearances: int
    season_pitches: int
    season_avg_velocity: float
    season_innings: float
    
    # Recent performance
    recent_appearances: List[AppearanceInfo]
    injury_history: List[InjuryInfo]
    
    # Trends
    workload_trend: TrendDirection
    velocity_trend: TrendDirection
    last_updated: datetime

# Risk assessment schemas
class RiskAssessmentRequest(BaseModel):
    """Risk assessment request."""
    pitcher_id: int
    as_of_date: date = Field(default_factory=date.today)

class RiskFactor(BaseModel):
    """Individual risk factor."""
    factor: str
    value: float
    importance: float = Field(ge=0.0, le=1.0)
    description: str

class RiskAssessmentResponse(BaseResponse):
    """Risk assessment response."""
    pitcher_id: int
    as_of_date: date
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Risk factors and explanations
    risk_factors: List[RiskFactor]
    primary_concerns: List[str]
    recommendations: List[str]
    
    # Data quality
    data_completeness: str
    days_of_data: int
    last_appearance: Optional[date]

# Workload schemas
class WorkloadDataPoint(BaseModel):
    """Single workload data point."""
    date: date
    pitches: int
    velocity: float
    spin_rate: int
    rest_days: int
    innings: float

class WorkloadTimeSeriesResponse(BaseResponse):
    """Workload time series response."""
    pitcher_id: int
    start_date: date
    end_date: date
    data_points: List[WorkloadDataPoint]
    
    # Summary statistics
    total_pitches: int
    avg_pitches_per_game: float
    avg_velocity: float
    avg_rest_days: float
    
    # Trends
    pitch_count_trend: TrendDirection
    velocity_trend: TrendDirection
    workload_intensity: str  # "normal", "elevated", "high"

# Team overview schemas
class TeamRiskSummary(BaseModel):
    """Team risk summary."""
    high_risk: int
    medium_risk: int
    low_risk: int
    total_pitchers: int

class TeamOverviewResponse(BaseResponse):
    """Team overview response."""
    team_id: str
    risk_summary: TeamRiskSummary
    total_pitches: int
    avg_pitches_per_game: float
    total_appearances: int
    recent_appearances: int
    last_updated: datetime

# Model information schemas
class ModelFeature(BaseModel):
    """Model feature information."""
    name: str
    importance: float
    description: str

class CurrentModelResponse(BaseResponse):
    """Current model information response."""
    model_name: str
    model_version: str
    trained_at: datetime
    
    # Performance metrics
    auc_roc: float
    precision_at_top_10: float
    recall_at_top_10: float
    brier_score: float
    calibration_error: float
    
    # Training data info
    training_samples: int
    positive_samples: int
    training_date_range: Dict[str, date]
    
    # Feature information
    features: List[ModelFeature]

# Error schemas
class ErrorDetail(BaseModel):
    """Error detail information."""
    code: int
    message: str
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: ErrorDetail

# Health check schemas
class HealthCheckResponse(BaseResponse):
    """Health check response."""
    status: str
    version: str

class DetailedHealthCheck(BaseModel):
    """Detailed health check."""
    status: str
    message: Optional[str]

class DetailedHealthCheckResponse(BaseResponse):
    """Detailed health check response."""
    status: str
    version: str
    checks: Dict[str, DetailedHealthCheck]
