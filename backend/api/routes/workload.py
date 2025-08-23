"""
Workload monitoring API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date, datetime, timedelta
import logging
from typing import Optional

from database.connection import get_db_session
from database.models import Pitcher, Appearance, FeatureSnapshot
from api.schemas import WorkloadTimeSeriesResponse, WorkloadDataPoint, TrendDirection
from services.workload_service import WorkloadService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize workload service
workload_service = WorkloadService()

@router.get("/workload/pitcher/{pitcher_id}", response_model=WorkloadTimeSeriesResponse)
async def get_pitcher_workload(
    pitcher_id: int,
    start_date: Optional[date] = Query(None, description="Start date (defaults to 30 days ago)"),
    end_date: Optional[date] = Query(None, description="End date (defaults to today)"),
    db: Session = Depends(get_db_session)
):
    """Get workload time series for a specific pitcher."""
    
    # Validate pitcher exists
    pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    # Set default date range
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Get appearances in date range
    appearances = db.query(Appearance)\
        .filter(
            Appearance.pitcher_id == pitcher_id,
            Appearance.game_date >= start_date,
            Appearance.game_date <= end_date
        )\
        .order_by(Appearance.game_date)\
        .all()
    
    if not appearances:
        return WorkloadTimeSeriesResponse(
            pitcher_id=pitcher_id,
            start_date=start_date,
            end_date=end_date,
            data_points=[],
            total_pitches=0,
            avg_pitches_per_game=0.0,
            avg_velocity=0.0,
            avg_rest_days=0.0,
            pitch_count_trend=TrendDirection.STABLE,
            velocity_trend=TrendDirection.STABLE,
            workload_intensity="normal"
        )
    
    # Convert appearances to workload data points
    data_points = []
    prev_date = None
    
    for appearance in appearances:
        # Calculate rest days
        rest_days = 0
        if prev_date:
            rest_days = (appearance.game_date - prev_date).days - 1
        
        data_points.append(WorkloadDataPoint(
            date=appearance.game_date,
            pitches=appearance.pitches_thrown,
            velocity=appearance.avg_vel,
            spin_rate=appearance.avg_spin,
            rest_days=rest_days,
            innings=appearance.innings_pitched
        ))
        
        prev_date = appearance.game_date
    
    # Calculate summary statistics
    total_pitches = sum(dp.pitches for dp in data_points)
    avg_pitches_per_game = total_pitches / len(data_points) if data_points else 0
    avg_velocity = sum(dp.velocity for dp in data_points) / len(data_points) if data_points else 0
    avg_rest_days = sum(dp.rest_days for dp in data_points[1:]) / max(1, len(data_points) - 1) if len(data_points) > 1 else 0
    
    # Calculate trends
    pitch_count_trend = workload_service.calculate_trend([dp.pitches for dp in data_points])
    velocity_trend = workload_service.calculate_trend([dp.velocity for dp in data_points])
    
    # Determine workload intensity
    workload_intensity = workload_service.assess_workload_intensity(
        pitcher_id=pitcher_id,
        recent_pitches=total_pitches,
        avg_rest_days=avg_rest_days,
        db=db
    )
    
    return WorkloadTimeSeriesResponse(
        pitcher_id=pitcher_id,
        start_date=start_date,
        end_date=end_date,
        data_points=data_points,
        total_pitches=total_pitches,
        avg_pitches_per_game=round(avg_pitches_per_game, 1),
        avg_velocity=round(avg_velocity, 1),
        avg_rest_days=round(avg_rest_days, 1),
        pitch_count_trend=pitch_count_trend,
        velocity_trend=velocity_trend,
        workload_intensity=workload_intensity
    )

@router.get("/workload/pitcher/{pitcher_id}/summary")
async def get_pitcher_workload_summary(
    pitcher_id: int,
    days: int = Query(21, ge=7, le=90, description="Number of days to analyze"),
    db: Session = Depends(get_db_session)
):
    """Get workload summary for a pitcher over a specific period."""
    
    # Validate pitcher exists
    pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Get workload statistics
    workload_stats = db.query(
        func.count(Appearance.id).label('appearances'),
        func.sum(Appearance.pitches_thrown).label('total_pitches'),
        func.avg(Appearance.pitches_thrown).label('avg_pitches'),
        func.max(Appearance.pitches_thrown).label('max_pitches'),
        func.avg(Appearance.avg_vel).label('avg_velocity'),
        func.sum(Appearance.innings_pitched).label('total_innings')
    ).filter(
        Appearance.pitcher_id == pitcher_id,
        Appearance.game_date >= start_date,
        Appearance.game_date <= end_date
    ).first()
    
    # Get latest feature snapshot for additional metrics
    latest_features = db.query(FeatureSnapshot)\
        .filter(FeatureSnapshot.pitcher_id == pitcher_id)\
        .order_by(FeatureSnapshot.as_of_date.desc())\
        .first()
    
    # Calculate workload percentiles compared to role
    role_comparison = workload_service.compare_to_role(
        pitcher_id=pitcher_id,
        total_pitches=int(workload_stats.total_pitches or 0),
        appearances=workload_stats.appearances or 0,
        db=db
    )
    
    return {
        "success": True,
        "pitcher_id": pitcher_id,
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        
        # Raw statistics
        "appearances": workload_stats.appearances or 0,
        "total_pitches": int(workload_stats.total_pitches or 0),
        "avg_pitches_per_game": round(float(workload_stats.avg_pitches or 0), 1),
        "max_pitches_in_game": workload_stats.max_pitches or 0,
        "avg_velocity": round(float(workload_stats.avg_velocity or 0), 1),
        "total_innings": round(float(workload_stats.total_innings or 0), 1),
        
        # Derived metrics
        "pitches_per_day": round((workload_stats.total_pitches or 0) / days, 1),
        "rest_days_avg": round(days / max(1, workload_stats.appearances or 1) - 1, 1),
        
        # Rolling metrics from features
        "rolling_metrics": {
            "roll21d_pitch_count": latest_features.roll21d_pitch_count if latest_features else None,
            "roll7d_pitch_count": latest_features.roll7d_pitch_count if latest_features else None,
            "roll3d_pitch_count": latest_features.roll3d_pitch_count if latest_features else None,
            "workload_intensity": latest_features.workload_intensity if latest_features else None,
            "fatigue_score": latest_features.fatigue_score if latest_features else None
        },
        
        # Comparisons
        "role_comparison": role_comparison,
        
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/workload/team/{team_id}/summary")
async def get_team_workload_summary(
    team_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    db: Session = Depends(get_db_session)
):
    """Get workload summary for all pitchers on a team."""
    
    # Get all pitchers for the team
    pitchers = db.query(Pitcher).filter(Pitcher.team == team_id.upper()).all()
    
    if not pitchers:
        raise HTTPException(status_code=404, detail="Team not found")
    
    pitcher_ids = [p.pitcher_id for p in pitchers]
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Get team workload statistics
    team_stats = db.query(
        func.count(Appearance.id).label('total_appearances'),
        func.sum(Appearance.pitches_thrown).label('total_pitches'),
        func.avg(Appearance.pitches_thrown).label('avg_pitches'),
        func.count(func.distinct(Appearance.pitcher_id)).label('active_pitchers')
    ).filter(
        Appearance.pitcher_id.in_(pitcher_ids),
        Appearance.game_date >= start_date,
        Appearance.game_date <= end_date
    ).first()
    
    # Get individual pitcher summaries
    pitcher_summaries = []
    for pitcher in pitchers:
        pitcher_stats = db.query(
            func.count(Appearance.id).label('appearances'),
            func.sum(Appearance.pitches_thrown).label('pitches'),
            func.avg(Appearance.avg_vel).label('avg_vel')
        ).filter(
            Appearance.pitcher_id == pitcher.pitcher_id,
            Appearance.game_date >= start_date,
            Appearance.game_date <= end_date
        ).first()
        
        if pitcher_stats.appearances and pitcher_stats.appearances > 0:
            pitcher_summaries.append({
                "pitcher_id": pitcher.pitcher_id,
                "name": pitcher.name,
                "role": pitcher.role,
                "appearances": pitcher_stats.appearances,
                "pitches": int(pitcher_stats.pitches or 0),
                "avg_velocity": round(float(pitcher_stats.avg_vel or 0), 1)
            })
    
    # Sort by total pitches (descending)
    pitcher_summaries.sort(key=lambda x: x["pitches"], reverse=True)
    
    return {
        "success": True,
        "team_id": team_id.upper(),
        "period_days": days,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        
        # Team statistics
        "total_appearances": team_stats.total_appearances or 0,
        "total_pitches": int(team_stats.total_pitches or 0),
        "avg_pitches_per_game": round(float(team_stats.avg_pitches or 0), 1),
        "active_pitchers": team_stats.active_pitchers or 0,
        "total_pitchers": len(pitchers),
        
        # Individual pitcher summaries
        "pitcher_summaries": pitcher_summaries,
        
        "timestamp": datetime.utcnow().isoformat()
    }
