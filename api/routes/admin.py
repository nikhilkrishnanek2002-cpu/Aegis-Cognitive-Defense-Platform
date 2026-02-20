"""
Admin panel routes: user management, system health, config.
All endpoints require admin role.
"""
import os
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth_utils import require_admin
from src.user_manager import list_users, create_user, delete_user, update_user_role

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.rtl_sdr_receiver import HAS_RTLSDR
from src.stream_bus import HAS_KAFKA

router = APIRouter(prefix="/api/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"


class UpdateRoleRequest(BaseModel):
    role: str


@router.get("/users")
async def get_users(_: dict = Depends(require_admin)):
    users = list_users()
    return [{"username": u[0], "role": u[1]} for u in users]


@router.post("/users")
async def add_user(body: CreateUserRequest, _: dict = Depends(require_admin)):
    create_user(body.username, body.password, body.role)
    return {"message": f"User {body.username} created"}


@router.delete("/users/{username}")
async def remove_user(username: str, _: dict = Depends(require_admin)):
    delete_user(username)
    return {"message": f"User {username} deleted"}


@router.patch("/users/{username}/role")
async def change_role(username: str, body: UpdateRoleRequest, _: dict = Depends(require_admin)):
    update_user_role(username, body.role)
    return {"message": f"Role updated for {username}"}


@router.get("/health")
async def system_health(_: dict = Depends(require_admin)):
    cpu = psutil.cpu_percent() if HAS_PSUTIL else None
    mem = psutil.virtual_memory().percent if HAS_PSUTIL else None
    db_ok = os.path.exists("results/users.db")
    return {
        "cpu_percent": cpu,
        "memory_percent": mem,
        "db_connected": db_ok,
        "rtlsdr_available": HAS_RTLSDR,
        "kafka_available": HAS_KAFKA,
    }
