"""
Pitcher-related API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, date
from typing import List, Optional
import logging

from database.connection import get_db_session
from database.models import Pitcher, Appearance, Injury, FeatureSnapshot
from api.schemas import PitcherResponse, PitcherDetailResponse, TeamOverviewResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/pitchers", response_model=List[PitcherResponse])
async def get_pitchers(
    team: Optional[str] = Query(None, description="Filter by team"),
    role: Optional[str] = Query(None, description="Filter by role (starter/reliever)"),
    limit: int = Query(50, ge=1, le=100, description="Number of pitchers to return"),
    offset: int = Query(0, ge=0, description="Number of pitchers to skip"),
    db: Session = Depends(get_db_session)
):
    """Get list of pitchers with optional filtering."""
    
    query = db.query(Pitcher)
    
    # Apply filters
    if team:
        query = query.filter(Pitcher.team == team.upper())
    
    if role:
        query = query.filter(Pitcher.role == role.lower())
    
    # Apply pagination
    pitchers = query.offset(offset).limit(limit).all()
    
    # Convert to response format
    pitcher_responses = []
    for pitcher in pitchers:
        # Get latest appearance for current stats
        latest_appearance = db.query(Appearance)\
            .filter(Appearance.pitcher_id == pitcher.pitcher_id)\
            .order_by(Appearance.game_date.desc())\
            .first()
        
        # Get latest risk assessment
        latest_risk = db.query(FeatureSnapshot)\
            .filter(FeatureSnapshot.pitcher_id == pitcher.pitcher_id)\
            .order_by(FeatureSnapshot.as_of_date.desc())\
            .first()
        
        pitcher_responses.append(PitcherResponse(
            pitcher_id=pitcher.pitcher_id,
            name=pitcher.name,
            team=pitcher.team,
            role=pitcher.role,
            last_appearance=latest_appearance.game_date if latest_appearance else None,
            current_risk_level="medium",  # TODO: Calculate from model
            current_risk_score=0.3,  # TODO: Calculate from model
            season_appearances=db.query(Appearance)
                .filter(Appearance.pitcher_id == pitcher.pitcher_id)
                .count(),
            recent_velocity=latest_appearance.avg_vel if latest_appearance else None,
            velocity_trend="stable"  # TODO: Calculate trend
        ))
    
    return pitcher_responses

@router.get("/pitchers/{pitcher_id}", response_model=PitcherDetailResponse)
async def get_pitcher_detail(
    pitcher_id: int,
    db: Session = Depends(get_db_session)
):
    """Get detailed information for a specific pitcher."""
    
    # Get pitcher
    pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    # Get recent appearances (last 30 days)
    recent_appearances = db.query(Appearance)\
        .filter(Appearance.pitcher_id == pitcher_id)\
        .order_by(Appearance.game_date.desc())\
        .limit(20).all()
    
    # Get injury history
    injuries = db.query(Injury)\
        .filter(Injury.pitcher_id == pitcher_id)\
        .order_by(Injury.il_start.desc())\
        .all()
    
    # Get latest risk assessment
    latest_risk = db.query(FeatureSnapshot)\
        .filter(FeatureSnapshot.pitcher_id == pitcher_id)\
        .order_by(FeatureSnapshot.as_of_date.desc())\
        .first()
    
    # Calculate season stats
    season_stats = db.query(
        func.count(Appearance.id).label('appearances'),
        func.sum(Appearance.pitches_thrown).label('total_pitches'),
        func.avg(Appearance.avg_vel).label('avg_velocity'),
        func.sum(Appearance.innings_pitched).label('total_innings')
    ).filter(Appearance.pitcher_id == pitcher_id).first()
    
    return PitcherDetailResponse(
        pitcher_id=pitcher.pitcher_id,
        name=pitcher.name,
        team=pitcher.team,
        role=pitcher.role,
        baseline_velocity=pitcher.baseline_velocity,
        baseline_spin_rate=pitcher.baseline_spin_rate,
        injury_risk_profile=pitcher.injury_risk_profile,
        
        # Current risk assessment
        current_risk_level="medium",  # TODO: Calculate from model
        current_risk_score=0.3,  # TODO: Calculate from model
        risk_factors=["High recent workload", "Velocity decline"],  # TODO: From model
        
        # Season statistics
        season_appearances=season_stats.appearances or 0,
        season_pitches=int(season_stats.total_pitches or 0),
        season_avg_velocity=round(float(season_stats.avg_velocity or 0), 1),
        season_innings=round(float(season_stats.total_innings or 0), 1),
        
        # Recent performance
        recent_appearances=[
            {
                "date": app.game_date.isoformat(),
                "pitches": app.pitches_thrown,
                "velocity": app.avg_vel,
                "spin_rate": app.avg_spin,
                "innings": app.innings_pitched
            }
            for app in recent_appearances
        ],
        
        # Injury history
        injury_history=[
            {
                "start_date": injury.il_start.isoformat(),
                "end_date": injury.il_end.isoformat() if injury.il_end else None,
                "type": injury.stint_type
            }
            for injury in injuries
        ],
        
        # Workload trends (TODO: Calculate from features)
        workload_trend="increasing",
        velocity_trend="declining",
        last_updated=latest_risk.created_at if latest_risk else datetime.utcnow()
    )

@router.get("/teams/{team_id}/overview", response_model=TeamOverviewResponse)
async def get_team_overview(
    team_id: str,
    db: Session = Depends(get_db_session)
):
    """Get team overview with pitcher risk summaries."""
    
    # Get all pitchers for the team
    pitchers = db.query(Pitcher).filter(Pitcher.team == team_id.upper()).all()
    
    if not pitchers:
        raise HTTPException(status_code=404, detail="Team not found")
    
    pitcher_ids = [p.pitcher_id for p in pitchers]
    
    # Get risk summaries for all team pitchers
    risk_summary = {
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
        "total_pitchers": len(pitchers)
    }
    
    # TODO: Calculate actual risk levels from model
    # For now, use mock distribution
    risk_summary["high_risk"] = len(pitchers) // 10 or 1
    risk_summary["medium_risk"] = len(pitchers) // 3
    risk_summary["low_risk"] = len(pitchers) - risk_summary["high_risk"] - risk_summary["medium_risk"]
    
    # Get team workload statistics
    team_workload = db.query(
        func.sum(Appearance.pitches_thrown).label('total_pitches'),
        func.avg(Appearance.pitches_thrown).label('avg_pitches_per_game'),
        func.count(Appearance.id).label('total_appearances')
    ).filter(Appearance.pitcher_id.in_(pitcher_ids)).first()
    
    # Get recent team activity (last 7 days)
    recent_date = date.today()
    recent_appearances = db.query(Appearance)\
        .filter(
            Appearance.pitcher_id.in_(pitcher_ids),
            Appearance.game_date >= recent_date
        ).count()
    
    return TeamOverviewResponse(
        team_id=team_id.upper(),
        risk_summary=risk_summary,
        total_pitches=int(team_workload.total_pitches or 0),
        avg_pitches_per_game=round(float(team_workload.avg_pitches_per_game or 0), 1),
        total_appearances=team_workload.total_appearances or 0,
        recent_appearances=recent_appearances,
        last_updated=datetime.utcnow()
    )
