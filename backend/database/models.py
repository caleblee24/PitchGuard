"""
SQLAlchemy models for PitchGuard database.
"""

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Boolean, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, date
from typing import Optional

Base = declarative_base()

class Pitcher(Base):
    """Pitcher profile information."""
    __tablename__ = "pitchers"
    
    pitcher_id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team = Column(String(10), nullable=False)
    role = Column(String(20), nullable=False)  # 'starter' or 'reliever'
    baseline_velocity = Column(Float)
    baseline_spin_rate = Column(Integer)
    injury_risk_profile = Column(String(20))  # 'low', 'medium', 'high'
    
    # Relationships
    pitches = relationship("Pitch", back_populates="pitcher")
    appearances = relationship("Appearance", back_populates="pitcher")
    injuries = relationship("Injury", back_populates="pitcher")
    feature_snapshots = relationship("FeatureSnapshot", back_populates="pitcher")

class Pitch(Base):
    """Individual pitch data."""
    __tablename__ = "pitches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_date = Column(Date, nullable=False)
    pitcher_id = Column(Integer, ForeignKey("pitchers.pitcher_id"), nullable=False)
    pitch_type = Column(String(10), nullable=False)
    release_speed = Column(Float, nullable=False)
    release_spin_rate = Column(Integer, nullable=False)
    pitch_number = Column(Integer, nullable=False)
    at_bat_id = Column(Integer, nullable=False)
    
    # Relationships
    pitcher = relationship("Pitcher", back_populates="pitches")
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_pitcher_date", "pitcher_id", "game_date"),
        Index("idx_game_date", "game_date"),
    )

class Appearance(Base):
    """Game appearance aggregated data."""
    __tablename__ = "appearances"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_date = Column(Date, nullable=False)
    pitcher_id = Column(Integer, ForeignKey("pitchers.pitcher_id"), nullable=False)
    pitches_thrown = Column(Integer, nullable=False)
    avg_vel = Column(Float, nullable=False)
    avg_spin = Column(Integer, nullable=False)
    vel_std = Column(Float, nullable=False)
    spin_std = Column(Integer, nullable=False)
    outs_recorded = Column(Integer, nullable=False)
    innings_pitched = Column(Float, nullable=False)
    
    # Relationships
    pitcher = relationship("Pitcher", back_populates="appearances")
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_pitcher_date_app", "pitcher_id", "game_date"),
        Index("idx_game_date_app", "game_date"),
    )

class Injury(Base):
    """Injury records."""
    __tablename__ = "injuries"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pitcher_id = Column(Integer, ForeignKey("pitchers.pitcher_id"), nullable=False)
    il_start = Column(Date, nullable=False)
    il_end = Column(Date, nullable=True)
    stint_type = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    pitcher = relationship("Pitcher", back_populates="injuries")
    
    # Indexes
    __table_args__ = (
        Index("idx_pitcher_injury", "pitcher_id", "il_start"),
        Index("idx_il_start", "il_start"),
    )

class FeatureSnapshot(Base):
    """Rolling feature snapshots for model training and inference."""
    __tablename__ = "feature_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pitcher_id = Column(Integer, ForeignKey("pitchers.pitcher_id"), nullable=False)
    as_of_date = Column(Date, nullable=False)
    
    # Rolling workload features (21-day windows)
    roll21d_pitch_count = Column(Integer)
    roll21d_avg_vel = Column(Float)
    roll21d_vel_decline = Column(Float)
    roll21d_avg_spin = Column(Float)
    roll21d_spin_decline = Column(Float)
    roll21d_appearances = Column(Integer)
    roll21d_rest_days_avg = Column(Float)
    roll21d_rest_days_std = Column(Float)
    
    # Rolling workload features (7-day windows)
    roll7d_pitch_count = Column(Integer)
    roll7d_avg_vel = Column(Float)
    roll7d_vel_decline = Column(Float)
    roll7d_avg_spin = Column(Float)
    roll7d_spin_decline = Column(Float)
    roll7d_appearances = Column(Integer)
    
    # Rolling workload features (3-day windows)
    roll3d_pitch_count = Column(Integer)
    roll3d_avg_vel = Column(Float)
    roll3d_appearances = Column(Integer)
    
    # Velocity and spin trends
    vel_trend_slope = Column(Float)
    spin_trend_slope = Column(Float)
    vel_consistency = Column(Float)  # Inverse of std dev
    spin_consistency = Column(Float)
    
    # Recovery patterns
    days_since_last_appearance = Column(Integer)
    workload_intensity = Column(Float)  # Recent workload vs baseline
    fatigue_score = Column(Float)  # Composite fatigue indicator
    
    # Labels for training
    label_injury_within_21d = Column(Boolean)
    label_injury_within_14d = Column(Boolean)
    label_injury_within_7d = Column(Boolean)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    data_completeness = Column(String(20))  # 'high', 'medium', 'low'
    
    # Relationships
    pitcher = relationship("Pitcher", back_populates="feature_snapshots")
    
    # Indexes for model training and inference
    __table_args__ = (
        Index("idx_pitcher_asof", "pitcher_id", "as_of_date"),
        Index("idx_asof_date", "as_of_date"),
        Index("idx_injury_labels", "label_injury_within_21d"),
    )

class ModelRegistry(Base):
    """Model training metadata and versioning."""
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_path = Column(String(500), nullable=False)
    
    # Training metadata
    trained_at = Column(DateTime, default=func.now())
    training_data_start = Column(Date, nullable=False)
    training_data_end = Column(Date, nullable=False)
    num_training_samples = Column(Integer, nullable=False)
    num_positive_samples = Column(Integer, nullable=False)
    
    # Performance metrics
    auc_roc = Column(Float)
    precision_at_top_10 = Column(Float)
    recall_at_top_10 = Column(Float)
    brier_score = Column(Float)
    calibration_error = Column(Float)
    
    # Feature importance (JSON string)
    feature_importance = Column(Text)
    feature_names = Column(Text)
    
    # Model configuration
    model_config = Column(Text)  # JSON string
    
    # Status
    is_active = Column(Boolean, default=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_model_version", "model_name", "model_version"),
        Index("idx_active_model", "is_active"),
    )
