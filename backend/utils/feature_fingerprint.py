#!/usr/bin/env python3
"""
Feature Fingerprint System

Ensures training/serving parity by generating and validating feature fingerprints.
Prevents feature name mismatches that could cause model failures.
"""

import hashlib
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureFingerprint:
    """Feature fingerprint system for training/serving parity."""
    
    def __init__(self, feature_names: List[str], version: str = "2.0"):
        self.feature_names = sorted(feature_names)
        self.version = version
        self.fingerprint = self._generate_fingerprint()
        
    def _generate_fingerprint(self) -> str:
        """Generate SHA256 hash of sorted feature names."""
        feature_string = "|".join(self.feature_names)
        return hashlib.sha256(feature_string.encode()).hexdigest()
    
    def get_fingerprint(self) -> str:
        """Get the feature fingerprint."""
        return self.fingerprint
    
    def get_feature_names(self) -> List[str]:
        """Get the sorted feature names."""
        return self.feature_names.copy()
    
    def validate_fingerprint(self, other_fingerprint: str) -> bool:
        """Validate against another fingerprint."""
        return self.fingerprint == other_fingerprint
    
    def save_fingerprint(self, filepath: str) -> None:
        """Save fingerprint to file."""
        fingerprint_data = {
            "version": self.version,
            "feature_count": len(self.feature_names),
            "fingerprint": self.fingerprint,
            "feature_names": self.feature_names,
            "timestamp": str(pd.Timestamp.now())
        }
        
        with open(filepath, 'w') as f:
            json.dump(fingerprint_data, f, indent=2)
        
        logger.info(f"Saved feature fingerprint to {filepath}")
    
    @classmethod
    def load_fingerprint(cls, filepath: str) -> 'FeatureFingerprint':
        """Load fingerprint from file."""
        with open(filepath, 'r') as f:
            fingerprint_data = json.load(f)
        
        return cls(
            feature_names=fingerprint_data["feature_names"],
            version=fingerprint_data["version"]
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "version": self.version,
            "feature_count": len(self.feature_names),
            "fingerprint": self.fingerprint,
            "feature_names": self.feature_names
        }

def generate_feature_fingerprint(feature_names: List[str], version: str = "2.0") -> str:
    """Generate feature fingerprint from feature names."""
    fingerprint = FeatureFingerprint(feature_names, version)
    return fingerprint.get_fingerprint()

def validate_feature_fingerprint(expected_fingerprint: str, actual_fingerprint: str) -> bool:
    """Validate feature fingerprints match."""
    return expected_fingerprint == actual_fingerprint

def save_model_fingerprint(feature_names: List[str], model_path: str, version: str = "2.0") -> None:
    """Save feature fingerprint with model artifact."""
    fingerprint = FeatureFingerprint(feature_names, version)
    
    # Create models directory if it doesn't exist
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fingerprint alongside model
    fingerprint_path = model_dir / "feature_fingerprint.json"
    fingerprint.save_fingerprint(str(fingerprint_path))
    
    logger.info(f"Saved model feature fingerprint: {fingerprint_path}")

def load_model_fingerprint(model_path: str) -> Optional[FeatureFingerprint]:
    """Load feature fingerprint from model directory."""
    model_dir = Path(model_path).parent
    fingerprint_path = model_dir / "feature_fingerprint.json"
    
    if not fingerprint_path.exists():
        logger.warning(f"Feature fingerprint not found: {fingerprint_path}")
        return None
    
    try:
        return FeatureFingerprint.load_fingerprint(str(fingerprint_path))
    except Exception as e:
        logger.error(f"Error loading feature fingerprint: {e}")
        return None

def validate_serving_features(model_path: str, serving_features: List[str]) -> bool:
    """Validate serving features against model fingerprint."""
    model_fingerprint = load_model_fingerprint(model_path)
    
    if model_fingerprint is None:
        logger.error("Could not load model fingerprint")
        return False
    
    serving_fingerprint = FeatureFingerprint(serving_features)
    
    if not model_fingerprint.validate_fingerprint(serving_fingerprint.get_fingerprint()):
        logger.error("Feature fingerprint mismatch!")
        logger.error(f"Expected: {model_fingerprint.get_fingerprint()}")
        logger.error(f"Actual: {serving_fingerprint.get_fingerprint()}")
        logger.error(f"Expected features: {model_fingerprint.get_feature_names()}")
        logger.error(f"Actual features: {serving_fingerprint.get_feature_names()}")
        return False
    
    logger.info("Feature fingerprint validation passed")
    return True

# Standard feature set for PitchGuard Enhanced Model v2.0
STANDARD_FEATURES = [
    # Workload Features (12)
    "pitch_count_7d", "pitch_count_14d", "pitch_count_30d", "avg_pitches_per_appearance",
    "rest_days", "avg_rest_days", "short_rest_appearances",
    "workload_intensity_7d", "workload_intensity_14d", "high_workload_days",
    "pitch_count_7d_missing", "pitch_count_14d_missing",
    
    # Velocity Features (8)
    "avg_velocity_7d", "avg_velocity_14d", "velocity_decline_7d", "velocity_decline_14d",
    "velocity_std_7d", "velocity_std_14d", "velocity_trend",
    "vel_7d_missing", "vel_14d_missing", "vel_decline_missing",
    
    # Spin Rate Features (6)
    "avg_spin_rate_7d", "avg_spin_rate_14d", "spin_rate_decline_7d",
    "spin_rate_std_7d", "spin_rate_trend", "spin_velocity_ratio",
    "spin_7d_missing", "spin_14d_missing", "spin_decline_missing",
    
    # Pitch Mix Features (6)
    "fastball_pct", "breaking_pct", "off_speed_pct",
    "pitch_mix_change_7d", "pitch_mix_change_14d", "pitch_mix_volatility",
    "fastball_pct_missing", "breaking_pct_missing", "off_speed_pct_missing"
]

def get_standard_fingerprint() -> str:
    """Get the standard feature fingerprint for v2.0."""
    return generate_feature_fingerprint(STANDARD_FEATURES, "2.0")

def validate_standard_features(feature_names: List[str]) -> bool:
    """Validate against standard feature set."""
    standard_fingerprint = get_standard_fingerprint()
    actual_fingerprint = generate_feature_fingerprint(feature_names)
    return validate_feature_fingerprint(standard_fingerprint, actual_fingerprint)

if __name__ == "__main__":
    # Test the fingerprint system
    test_features = ["pitch_count_7d", "avg_velocity_7d", "rest_days"]
    fingerprint = FeatureFingerprint(test_features)
    
    print(f"Feature count: {len(fingerprint.get_feature_names())}")
    print(f"Fingerprint: {fingerprint.get_fingerprint()}")
    print(f"Standard fingerprint: {get_standard_fingerprint()}")
