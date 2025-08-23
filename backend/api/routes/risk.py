"""
Risk assessment API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import date, datetime
import logging
from typing import Optional

from database.connection import get_db_session
from database.models import Pitcher, FeatureSnapshot, Appearance
from api.schemas import RiskAssessmentRequest, RiskAssessmentResponse, RiskFactor, RiskLevel
from services.enhanced_risk_service import enhanced_risk_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/risk/pitcher", response_model=RiskAssessmentResponse)
async def assess_pitcher_risk(
    request: RiskAssessmentRequest,
    db: Session = Depends(get_db_session)
):
    """Assess injury risk for a specific pitcher on a given date."""
    
    # Validate pitcher exists
    pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == request.pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    try:
        # Convert date to datetime for enhanced service
        as_of_datetime = datetime.combine(request.as_of_date, datetime.min.time())
        
        # Get risk assessment from enhanced service
        risk_assessment = enhanced_risk_service.get_current_risk_assessment(
            pitcher_id=request.pitcher_id,
            as_of_date=as_of_datetime
        )
        
        # Map enhanced response to API schema
        # Convert confidence from string to float
        confidence_str = risk_assessment.get("confidence", "medium")
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence_float = confidence_map.get(confidence_str, 0.6)
        
        # Convert data_completeness from dict to string
        data_completeness_dict = risk_assessment.get("data_completeness", {})
        if isinstance(data_completeness_dict, dict):
            # Calculate overall completeness based on available data types
            completeness_values = list(data_completeness_dict.values())
            if completeness_values:
                avg_completeness = sum(completeness_values) / len(completeness_values)
                if avg_completeness >= 0.8:
                    data_completeness_str = "high"
                elif avg_completeness >= 0.5:
                    data_completeness_str = "medium"
                else:
                    data_completeness_str = "low"
            else:
                data_completeness_str = "low"
        else:
            data_completeness_str = str(data_completeness_dict)
        
        return RiskAssessmentResponse(
            pitcher_id=request.pitcher_id,
            as_of_date=request.as_of_date,
            risk_level=RiskLevel(risk_assessment.get("risk_bucket", "low")),
            risk_score=risk_assessment.get("risk_score_calibrated", 0.0),
            confidence=confidence_float,
            risk_factors=[
                RiskFactor(
                    factor=contributor.get("name", "unknown"),
                    value=contributor.get("value", 0.0),
                    importance=contributor.get("importance", 0.0),
                    description=f"{contributor.get('name', 'Unknown factor')} shows {contributor.get('direction', 'neutral')} impact"
                )
                for contributor in risk_assessment.get("contributors", [])[:3]  # Top 3
            ],
            primary_concerns=risk_assessment.get("recommended_actions", []),
            recommendations=risk_assessment.get("recommended_actions", []),
            data_completeness=data_completeness_str,
            days_of_data=risk_assessment.get("days_of_data", 0),
            last_appearance=risk_assessment.get("last_appearance")
        )
        
    except Exception as e:
        logger.error(f"Risk assessment failed for pitcher {request.pitcher_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(e)}"
        )

@router.get("/risk/pitcher/{pitcher_id}/current", response_model=RiskAssessmentResponse)
async def get_current_pitcher_risk(
    pitcher_id: int,
    db: Session = Depends(get_db_session)
):
    """Get current risk assessment for a pitcher (as of today)."""
    
    request = RiskAssessmentRequest(
        pitcher_id=pitcher_id,
        as_of_date=date.today()
    )
    
    return await assess_pitcher_risk(request, db)

@router.get("/risk/pitcher/{pitcher_id}/history")
async def get_pitcher_risk_history(
    pitcher_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db_session)
):
    """Get historical risk assessments for a pitcher."""
    
    # Validate pitcher exists
    pitcher = db.query(Pitcher).filter(Pitcher.pitcher_id == pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail="Pitcher not found")
    
    # Set default date range if not provided
    if not end_date:
        end_date = date.today()
    if not start_date:
        # Default to last 30 days
        start_date = date.fromordinal(end_date.toordinal() - 30)
    
    # Get feature snapshots for the date range
    feature_snapshots = db.query(FeatureSnapshot)\
        .filter(
            FeatureSnapshot.pitcher_id == pitcher_id,
            FeatureSnapshot.as_of_date >= start_date,
            FeatureSnapshot.as_of_date <= end_date
        )\
        .order_by(FeatureSnapshot.as_of_date)\
        .all()
    
    # Calculate risk scores for each snapshot
    risk_history = []
    for snapshot in feature_snapshots:
        try:
            # Convert date to datetime for enhanced service
            as_of_datetime = datetime.combine(snapshot.as_of_date, datetime.min.time())
            
            risk_assessment = enhanced_risk_service.get_current_risk_assessment(
                pitcher_id=pitcher_id,
                as_of_date=as_of_datetime
            )
            
            risk_history.append({
                "date": snapshot.as_of_date.isoformat(),
                "risk_level": risk_assessment.get("risk_bucket", "low"),
                "risk_score": risk_assessment.get("risk_score_calibrated", 0.0),
                "confidence": risk_assessment.get("confidence", "medium"),
                "primary_concern": risk_assessment.get("recommended_actions", [None])[0]
            })
        except Exception as e:
            logger.warning(f"Could not calculate risk for {snapshot.as_of_date}: {e}")
    
    return {
        "success": True,
        "pitcher_id": pitcher_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "risk_history": risk_history,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/risk/team/{team_id}/summary")
async def get_team_risk_summary(
    team_id: str,
    as_of_date: Optional[date] = None,
    db: Session = Depends(get_db_session)
):
    """Get risk summary for all pitchers on a team."""
    
    if not as_of_date:
        as_of_date = date.today()
    
    # Get all pitchers for the team
    pitchers = db.query(Pitcher).filter(Pitcher.team == team_id.upper()).all()
    
    if not pitchers:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Assess risk for each pitcher
    risk_summary = {
        "high_risk": [],
        "medium_risk": [],
        "low_risk": [],
        "no_data": []
    }
    
    for pitcher in pitchers:
        try:
            # Convert date to datetime for enhanced service
            as_of_datetime = datetime.combine(as_of_date, datetime.min.time())
            
            risk_assessment = enhanced_risk_service.get_current_risk_assessment(
                pitcher_id=pitcher.pitcher_id,
                as_of_date=as_of_datetime
            )
            
            pitcher_info = {
                "pitcher_id": pitcher.pitcher_id,
                "name": pitcher.name,
                "role": pitcher.role,
                "risk_score": risk_assessment.get("risk_score_calibrated", 0.0),
                "primary_concern": risk_assessment.get("recommended_actions", [None])[0]
            }
            
            risk_bucket = risk_assessment.get("risk_bucket", "low")
            if risk_bucket == "high":
                risk_summary["high_risk"].append(pitcher_info)
            elif risk_bucket == "medium":
                risk_summary["medium_risk"].append(pitcher_info)
            else:
                risk_summary["low_risk"].append(pitcher_info)
                
        except Exception as e:
            logger.warning(f"Could not assess risk for pitcher {pitcher.pitcher_id}: {e}")
            risk_summary["no_data"].append({
                "pitcher_id": pitcher.pitcher_id,
                "name": pitcher.name,
                "role": pitcher.role,
                "error": str(e)
            })
    
    # Sort each risk level by risk score (descending)
    for level in ["high_risk", "medium_risk", "low_risk"]:
        risk_summary[level].sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    
    return {
        "success": True,
        "team_id": team_id.upper(),
        "as_of_date": as_of_date.isoformat(),
        "risk_summary": risk_summary,
        "total_pitchers": len(pitchers),
        "timestamp": datetime.utcnow().isoformat()
    }
