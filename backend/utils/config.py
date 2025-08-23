"""
Configuration management for PitchGuard backend.
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database configuration
    database_url: str = "postgresql://localhost/pitchguard"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = True
    
    # Model configuration
    model_path: str = "./models/injury_risk_model.joblib"
    feature_config_path: str = "./config/feature_config.json"
    
    # Data configuration
    mock_data_path: str = "../data/mock"
    demo_data_path: str = "../data/demo"
    
    # Operational configuration
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    # Feature engineering parameters
    rolling_window_days: int = 21
    min_appearances_for_features: int = 5
    
    # Model parameters
    injury_window_days: int = 21
    min_training_samples: int = 100
    
    class Config:
        env_prefix = "PITCHGUARD_"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_database_url() -> str:
    """Get the database URL for SQLAlchemy."""
    return settings.database_url

def get_mock_data_path() -> str:
    """Get the path to mock data."""
    return settings.mock_data_path

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
