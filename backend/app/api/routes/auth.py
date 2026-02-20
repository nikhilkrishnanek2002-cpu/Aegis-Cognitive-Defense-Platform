"""Authentication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta
try:
    from jose import jwt, JWTError
except ImportError:
    import jwt
    JWTError = jwt.InvalidTokenError
from app.models.schemas import LoginRequest, TokenResponse
from app.core.config import get_config, Config

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Simple user store (in production: use database)
USERS = {
    "admin": "admin123",
    "operator": "operator123"
}


def create_access_token(username: str, config: Config = None) -> tuple[str, int]:
    """Create JWT access token."""
    if config is None:
        config = get_config()
    
    expires_at = datetime.utcnow() + timedelta(hours=config.jwt_expiration_hours)
    
    payload = {
        "sub": username,
        "exp": expires_at,
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(
        payload,
        config.jwt_secret,
        algorithm=config.jwt_algorithm
    )
    
    return token, int((expires_at - datetime.utcnow()).total_seconds())


def verify_token(token: str) -> str:
    """Verify JWT token and return username."""
    config = get_config()
    
    try:
        payload = jwt.decode(
            token,
            config.jwt_secret,
            algorithms=[config.jwt_algorithm]
        )
        username = payload.get("sub")
        
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        return username
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login endpoint - returns JWT token."""
    
    # Verify credentials
    if request.username not in USERS or USERS[request.username] != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create token
    token, expires_in = create_access_token(request.username)
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in
    )


@router.post("/register")
async def register(request: LoginRequest):
    """Register new user."""
    
    if request.username in USERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )
    
    # Store user (in production: hash password)
    USERS[request.username] = request.password
    
    token, expires_in = create_access_token(request.username)
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in
    )


@router.get("/me")
async def get_current_user(authorization: str = None):
    """Get current user from token."""
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization header"
        )
    
    # Extract token from "Bearer <token>"
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    username = verify_token(token)
    
    return {
        "username": username,
        "role": "admin" if username == "admin" else "operator"
    }


@router.post("/refresh")
async def refresh_token(authorization: str = None):
    """Refresh access token."""
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authorization header"
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise ValueError()
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )
    
    username = verify_token(token)
    token, expires_in = create_access_token(username)
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in
    )

