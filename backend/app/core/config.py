"""Configuration management for Aegis backend."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Backend configuration."""
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_title: str = "Aegis Cognitive Defense API"
    
    # Database
    db_path: str = os.getenv("DB_PATH", "./aegis.db")
    
    # WebSocket
    ws_heartbeat: int = 30  # seconds
    ws_max_clients: int = 100
    
    # Radar
    radar_scan_interval: float = float(os.getenv("RADAR_SCAN_INTERVAL", "0.5"))
    radar_refresh_rate: int = int(os.getenv("RADAR_REFRESH_RATE", "30"))
    
    # Detection
    detection_threshold: float = float(os.getenv("DETECTION_THRESHOLD", "0.65"))
    max_targets: int = int(os.getenv("MAX_TARGETS", "50"))
    
    # Tracking
    tracking_max_age: int = int(os.getenv("TRACKING_MAX_AGE", "30"))
    tracking_min_hits: int = int(os.getenv("TRACKING_MIN_HITS", "3"))
    
    # Threat Assessment
    threat_threshold_high: float = float(os.getenv("THREAT_THRESHOLD_HIGH", "0.75"))
    threat_threshold_critical: float = float(os.getenv("THREAT_THRESHOLD_CRITICAL", "0.90"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "./logs")
    
    # Models
    model_device: str = os.getenv("MODEL_DEVICE", "cuda")
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models")
    
    # Security
    jwt_secret: str = os.getenv("JWT_SECRET", "dev-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # Debug
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"


def get_config() -> Config:
    """Get current configuration."""
    return Config()
