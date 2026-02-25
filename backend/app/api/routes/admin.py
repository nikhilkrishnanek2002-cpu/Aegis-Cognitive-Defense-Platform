"""Admin panel routes: user management, system health, config."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

router = APIRouter(prefix="/api/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    """User creation request."""
    username: str
    password: str
    role: str = "viewer"


class UpdateRoleRequest(BaseModel):
    """Role update request."""
    role: str


class UserResponse(BaseModel):
    """User response."""
    username: str
    role: str


# Mock user storage (in production: replace with backend.app.services.user_service)
USERS_DB = {
    "admin": {"username": "admin", "role": "admin", "password": "admin123"},
    "operator": {"username": "operator", "role": "operator", "password": "operator123"}
}


async def require_admin(token: str = None):
    """Check if user has admin role."""
    # In production: validate JWT token and check role
    if not token or token != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return {"username": "admin", "role": "admin"}


@router.get("/users", response_model=List[UserResponse])
async def get_users(user: dict = Depends(require_admin)):
    """Get all users."""
    return [
        UserResponse(username=u["username"], role=u["role"])
        for u in USERS_DB.values()
    ]


@router.post("/users", response_model=dict)
async def create_user(body: CreateUserRequest, user: dict = Depends(require_admin)):
    """Create a new user."""
    if body.username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User {body.username} already exists"
        )
    
    USERS_DB[body.username] = {
        "username": body.username,
        "password": body.password,
        "role": body.role
    }
    
    return {"message": f"User {body.username} created with role {body.role}"}


@router.delete("/users/{username}", response_model=dict)
async def delete_user(username: str, user: dict = Depends(require_admin)):
    """Delete a user."""
    if username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {username} not found"
        )
    
    if username == "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete admin user"
        )
    
    del USERS_DB[username]
    return {"message": f"User {username} deleted"}


@router.patch("/users/{username}/role", response_model=dict)
async def update_user_role(username: str, body: UpdateRoleRequest, user: dict = Depends(require_admin)):
    """Update user role."""
    if username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {username} not found"
        )
    
    USERS_DB[username]["role"] = body.role
    return {"message": f"User {username} role updated to {body.role}"}


@router.get("/health", response_model=dict)
async def get_system_health(user: dict = Depends(require_admin)):
    """Get system health information."""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": 0,
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "disk_percent": 0.0,
        "db_connected": True,  # Mock database status
        "rtlsdr_available": False,  # RTL-SDR not available in demo mode
        "kafka_available": False  # Kafka not required in demo mode
    }
    
    if HAS_PSUTIL:
        import psutil
        from datetime import datetime
        
        try:
            health_data["timestamp"] = datetime.utcnow().isoformat()
            health_data["cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
            health_data["memory_percent"] = float(psutil.virtual_memory().percent)
            health_data["disk_percent"] = float(psutil.disk_usage("/").percent)
            health_data["uptime_seconds"] = int(datetime.utcnow().timestamp() - psutil.boot_time())
            
            # Determine health status based on metrics
            if health_data["memory_percent"] > 90 or health_data["cpu_percent"] > 95:
                health_data["status"] = "warning"
            elif health_data["memory_percent"] > 95:
                health_data["status"] = "critical"
            else:
                health_data["status"] = "healthy"
        except Exception as e:
            health_data["error"] = str(e)
    
    return health_data


@router.get("/metrics", response_model=dict)
async def get_admin_metrics(user: dict = Depends(require_admin)):
    """Get admin-level metrics."""
    try:
        from app.engine import _controller
        from app.services.radar_service import get_radar_service
        from app.services.detection_service import get_detection_service
        from app.services.tracking_service import get_tracking_service
        from app.services.threat_service import get_threat_service
        
        radar_svc = get_radar_service()
        detection_svc = get_detection_service()
        tracking_svc = get_tracking_service()
        threat_svc = get_threat_service()
        
        # Get active tracks and threats
        active_tracks = await tracking_svc.get_active_tracks() if hasattr(tracking_svc, 'get_active_tracks') else []
        critical_threats = await threat_svc.get_critical_threats() if hasattr(threat_svc, 'get_critical_threats') else []
        
        pipeline_status = await _controller.get_status() if _controller else {"running": False, "cycle_count": 0}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "operational",
            "users_count": len(USERS_DB),
            "roles": list(set(u["role"] for u in USERS_DB.values())),
            "pipeline": {
                "running": pipeline_status.get("running", False),
                "cycle_count": pipeline_status.get("cycle_count", 0),
                "uptime_seconds": pipeline_status.get("uptime_seconds", 0)
            },
            "radar": {
                "scan_count": radar_svc.scan_count if radar_svc else 0
            },
            "detections": {
                "count": detection_svc.detection_count if detection_svc else 0
            },
            "tracking": {
                "active_tracks": len(active_tracks) if active_tracks else 0
            },
            "threats": {
                "critical_count": len(critical_threats) if critical_threats else 0
            }
        }
    except Exception as e:
        # Fallback response
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "unknown",
            "users_count": len(USERS_DB),
            "roles": list(set(u["role"] for u in USERS_DB.values())),
            "error": str(e)
        }


@router.get("/dashboard")
async def get_admin_dashboard(user: dict = Depends(require_admin)):
    """Get comprehensive admin dashboard data."""
    try:
        from app.engine import _controller
        from app.services.radar_service import get_radar_service
        from app.services.detection_service import get_detection_service
        from app.services.tracking_service import get_tracking_service
        from app.services.threat_service import get_threat_service
        from app.core.performance import timer
        
        radar_svc = get_radar_service()
        detection_svc = get_detection_service()
        tracking_svc = get_tracking_service()
        threat_svc = get_threat_service()
        
        active_tracks = await tracking_svc.get_active_tracks() if hasattr(tracking_svc, 'get_active_tracks') else []
        critical_threats = await threat_svc.get_critical_threats() if hasattr(threat_svc, 'get_critical_threats') else []
        pipeline_status = await _controller.get_status() if _controller else {"running": False, "cycle_count": 0}
        perf_stats = timer.get_all_stats()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "status": "operational",
                "uptime_seconds": pipeline_status.get("uptime_seconds", 0)
            },
            "pipeline": {
                "running": pipeline_status.get("running", False),
                "cycle_count": pipeline_status.get("cycle_count", 0),
                "initialization_status": pipeline_status.get("initialization_status", "UNKNOWN")
            },
            "metrics": {
                "radar_scans": radar_svc.scan_count if radar_svc else 0,
                "total_detections": detection_svc.detection_count if detection_svc else 0,
                "active_tracks": len(active_tracks) if active_tracks else 0,
                "critical_threats": len(critical_threats) if critical_threats else 0
            },
            "performance": {
                "avg_cycle_ms": perf_stats.get("total_cycle", {}).get("avg", 0),
                "latest_cycle_ms": perf_stats.get("total_cycle", {}).get("latest", 0)
            },
            "users": len(USERS_DB)
        }
    except Exception as e:
        # Fallback dashboard
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {"status": "unknown"},
            "pipeline": {"running": False, "cycle_count": 0},
            "metrics": {
                "radar_scans": 0,
                "total_detections": 0,
                "active_tracks": 0,
                "critical_threats": 0
            },
            "error": str(e)
        }

