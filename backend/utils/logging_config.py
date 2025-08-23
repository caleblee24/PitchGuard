"""
Logging configuration for PitchGuard backend.
"""

import logging
import logging.config
from typing import Any, Dict
import json
from datetime import datetime

def setup_logging():
    """Setup logging configuration."""
    
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def get_logger(name: str) -> Any:
    """Get a logger instance."""
    return logging.getLogger(name)

class APILogger:
    """API request/response logger."""
    
    def __init__(self):
        self.logger = get_logger("api")
    
    def log_request(self, method: str, path: str, params: Dict = None):
        """Log API request."""
        self.logger.info(f"API Request: {method} {path} {params or {}}")
    
    def log_response(self, method: str, path: str, status_code: int, duration_ms: float):
        """Log API response."""
        self.logger.info(f"API Response: {method} {path} {status_code} ({duration_ms}ms)")
    
    def log_error(self, method: str, path: str, error: str, status_code: int = 500):
        """Log API error."""
        self.logger.error(f"API Error: {method} {path} {status_code} - {error}")
