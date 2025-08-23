"""
Database connection and session management.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from contextlib import contextmanager
from typing import Generator

from utils.config import settings
from database.models import Base

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.api_debug,  # Log SQL queries in debug mode
    # Use SQLite for development if PostgreSQL not available
    poolclass=StaticPool if "sqlite" in settings.database_url else None,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    Used with FastAPI Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Use for standalone database operations.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def init_database():
    """Initialize database with tables and basic setup."""
    logger.info(f"Initializing database: {settings.database_url}")
    
    try:
        # Create tables
        create_tables()
        
        # Test connection
        with get_db_session_context() as db:
            db.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# For development: Use SQLite if PostgreSQL not available
def get_development_database_url() -> str:
    """Get development database URL (SQLite fallback)."""
    try:
        # Try to connect to PostgreSQL
        test_engine = create_engine(settings.database_url)
        test_engine.connect()
        test_engine.dispose()
        return settings.database_url
    except Exception:
        logger.warning("PostgreSQL not available, using SQLite for development")
        return "sqlite:///./pitchguard_dev.db"

# Override database URL for development if needed
if settings.api_debug:
    try:
        # Test PostgreSQL connection
        test_engine = create_engine(settings.database_url)
        test_engine.connect()
        test_engine.dispose()
    except Exception:
        # Fall back to SQLite for development
        settings.database_url = "sqlite:///./pitchguard_dev.db"
        logger.info("Using SQLite for development")
        
        # Recreate engine with SQLite
        engine = create_engine(
            settings.database_url,
            echo=settings.api_debug,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
