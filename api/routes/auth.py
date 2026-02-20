"""
Authentication routes: login and register.
Wraps the existing src.auth module with JWT token generation.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.auth import authenticate
from src.auth import register_user
from api.auth_utils import create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"


@router.post("/login")
async def login(body: LoginRequest):
    ok, role = authenticate(body.username, body.password)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": body.username, "role": role})
    return {"access_token": token, "token_type": "bearer", "role": role, "username": body.username}


@router.post("/register")
async def register(body: RegisterRequest):
    success, msg = register_user(body.username, body.password, body.role)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": "Registration successful"}
