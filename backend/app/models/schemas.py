"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────
class ThreatLevel(str, Enum):
    """Threat level classification."""
    CRITICAL = "CRITICAL"      # > 0.90
    HIGH = "HIGH"              # 0.75-0.90
    MEDIUM = "MEDIUM"          # 0.50-0.75
    LOW = "LOW"                # < 0.50


class TargetType(str, Enum):
    """Target classification."""
    DRONE = "DRONE"
    AIRCRAFT = "AIRCRAFT"
    BIRD = "BIRD"
    HELICOPTER = "HELICOPTER"
    MISSILE = "MISSILE"
    CLUTTER = "CLUTTER"
    UNKNOWN = "UNKNOWN"


# ─── Auth ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ─── Radar ────────────────────────────────────────────────────────────────────
class RadarScan(BaseModel):
    """Radar scan result."""
    scan_id: str
    timestamp: datetime
    frame_count: int
    targets_detected: int
    signal_strength: float = Field(0.0, ge=0.0, le=1.0)
    noise_level: float = Field(0.0, ge=0.0, le=1.0)


class RadarTarget(BaseModel):
    """Radar target detection."""
    id: str
    range_m: float
    bearing_deg: float
    velocity_mps: float
    rcs_dbsm: float
    signal_strength: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime


# ─── Detection ────────────────────────────────────────────────────────────────
class DetectionResult(BaseModel):
    """AI detection result."""
    target_id: str
    target_type: TargetType
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    features: Dict[str, Any] = {}
    timestamp: datetime


# ─── Tracking ────────────────────────────────────────────────────────────────
class TrackedTarget(BaseModel):
    """Tracked target state."""
    track_id: str
    target_type: TargetType
    position: Dict[str, float]      # {x, y, z} or {range, bearing, altitude}
    velocity: Dict[str, float]      # velocity vector
    hits: int                        # number of hits
    age: int                        # frames since first detection
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    last_update: datetime


# ─── Threat Assessment ────────────────────────────────────────────────────────
class Threat(BaseModel):
    """Threat assessment result."""
    track_id: str
    threat_level: ThreatLevel
    threat_score: float = Field(0.0, ge=0.0, le=1.0)
    target_type: TargetType
    position: Dict[str, float]
    velocity: Dict[str, float]
    time_to_impact_s: Optional[float] = None
    intercept_point: Optional[Dict[str, float]] = None
    timestamp: datetime
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ─── EW Response ────────────────────────────────────────────────────────────
class EWSignal(BaseModel):
    """Electronic Warfare signal detection."""
    signal_id: str
    freq_mhz: float
    power_dbm: float
    signal_type: str
    timestamp: datetime


class EWResponse(BaseModel):
    """EW countermeasure response."""
    response_id: str
    signal_id: str
    response_type: str     # JAMMING, SPOOFING, DECEPTION, etc
    frequency_mhz: float
    power_dbm: float
    duration_ms: float
    timestamp: datetime


# ─── System Health ────────────────────────────────────────────────────────────
class SystemHealthStatus(BaseModel):
    """System health status."""
    status: str             # OPERATIONAL, DEGRADED, FAILED
    uptime_hours: float
    radar_status: str
    detection_model_status: str
    tracking_status: str
    database_status: str
    last_scan_time: Optional[datetime] = None
    targets_tracked: int = 0
    alerts_active: int = 0
    timestamp: datetime


# ─── Dashboard ────────────────────────────────────────────────────────────────
class DashboardMetrics(BaseModel):
    """Dashboard metrics summary."""
    total_scans: int
    total_targets_detected: int
    active_tracks: int
    critical_threats: int
    high_threats: int
    avg_detection_confidence: float
    system_health: SystemHealthStatus
    timestamp: datetime


# ─── Events & Alerts ────────────────────────────────────────────────────────────
class SystemEvent(BaseModel):
    """System event log entry."""
    event_id: str
    event_type: str
    severity: str           # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None


class Alert(BaseModel):
    """System alert."""
    alert_id: str
    alert_type: str         # THREAT_DETECTED, ANOMALY, SYSTEM_ERROR, etc
    severity: str           # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    resolved: bool = False
    data: Optional[Dict[str, Any]] = None


# ─── WebSocket ────────────────────────────────────────────────────────────────
class WSRadarFrame(BaseModel):
    """WebSocket radar frame broadcast."""
    frame_id: str
    timestamp: datetime
    targets: List[RadarTarget]
    tracked_targets: List[TrackedTarget]
    threats: List[Threat]
    scan_status: str


class WSSystemStatus(BaseModel):
    """WebSocket system status broadcast."""
    timestamp: datetime
    health: SystemHealthStatus
    metrics: DashboardMetrics
    events: List[SystemEvent]
    alerts: List[Alert]


# ─── Batch Operations ────────────────────────────────────────────────────────────
class BatchRadarData(BaseModel):
    """Batch radar scan data."""
    scans: List[RadarScan]
    targets: List[RadarTarget]
    export_format: str = "json"    # json, csv, parquet


class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis."""
    data_path: str
    analysis_type: str             # DETECTION, TRACKING, THREAT_ASSESSMENT
    parameters: Optional[Dict[str, Any]] = None
