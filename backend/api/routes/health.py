"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from database.connection import get_db_session
from utils.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "success": True,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db_session)):
    """Detailed health check with database and system status."""
    
    health_status = {
        "success": True,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Database health check
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        health_status["success"] = False
        health_status["status"] = "unhealthy"
    
    # System resource check
    if HAS_PSUTIL:
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_status["checks"]["system"] = {
                "status": "healthy",
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_percent": psutil.cpu_percent(interval=1)
            }
            
            # Alert if resources are critically low
            if memory.percent > 90 or disk.percent > 90:
                health_status["checks"]["system"]["status"] = "warning"
                health_status["checks"]["system"]["message"] = "High resource usage"
                
        except Exception as e:
            health_status["checks"]["system"] = {
                "status": "unknown",
                "message": f"Could not check system resources: {str(e)}"
            }
    else:
        health_status["checks"]["system"] = {
            "status": "unavailable",
            "message": "psutil not available"
        }
    
    # Configuration check
    health_status["checks"]["config"] = {
        "status": "healthy",
        "database_url_configured": bool(settings.database_url),
        "model_path_configured": bool(settings.model_path),
        "debug_mode": settings.api_debug
    }
    
    return health_status

@router.get("/readiness")
async def readiness_check(db: Session = Depends(get_db_session)):
    """Kubernetes-style readiness check."""
    
    try:
        # Check database connection
        db.execute(text("SELECT 1"))
        
        # Check that required tables exist
        result = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='pitchers'")
        ).fetchone()
        
        if not result:
            raise Exception("Required tables not found")
        
        return {
            "success": True,
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "status": "not_ready",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/liveness")
async def liveness_check():
    """Kubernetes-style liveness check."""
    return {
        "success": True,
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
