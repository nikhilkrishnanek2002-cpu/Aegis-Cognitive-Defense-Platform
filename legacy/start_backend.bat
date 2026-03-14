@echo off
REM ============================================================================
REM AEGIS COGNITIVE DEFENSE PLATFORM - FULL SIMULATION MODE
REM QUICK START GUIDE (Windows)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo AEGIS COGNITIVE DEFENSE PLATFORM - FULL SIMULATION MODE STARTUP
echo ========================================================================
echo.

REM ============================================================================
REM STEP 1: Verify Setup
REM ============================================================================
echo [STEP 1/5] Verifying setup...

if not exist "backend" (
    echo ERROR: backend\ directory not found. Are you in the project root?
    pause
    exit /b 1
)

if not exist "backend\app\main.py" (
    echo ERROR: backend\app\main.py not found
    pause
    exit /b 1
)

echo [OK] Project structure verified
echo.

REM ============================================================================
REM STEP 2: Check Python Environment
REM ============================================================================
echo [STEP 2/5] Checking Python environment...

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.8+
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] %PYTHON_VERSION% found

REM Check required packages
for %%p in (fastapi uvicorn numpy pydantic psutil) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        echo Installing %%p...
        pip install %%p
    ) else (
        echo [OK] %%p installed
    )
)
echo.

REM ============================================================================
REM STEP 3: Configure Simulation Mode
REM ============================================================================
echo [STEP 3/5] Configuring simulation mode...
set RADAR_SIMULATOR=true
echo [OK] RADAR_SIMULATOR=true
echo.

REM ============================================================================
REM STEP 4: Create Logs Directory
REM ============================================================================
echo [STEP 4/5] Setting up logs directory...

if not exist "backend\logs" mkdir backend\logs
echo [OK] Logs directory ready at backend\logs\
echo.

REM ============================================================================
REM STEP 5: Start Backend
REM ============================================================================
echo [STEP 5/5] Starting FastAPI backend...
echo.
echo ========================================================================
echo STARTING BACKEND SERVER
echo ========================================================================
echo.
echo [INFO] Backend will be available at:
echo   HTTP: http://localhost:8000
echo   WebSocket: ws://localhost:8000/ws/stream
echo   Docs: http://localhost:8000/docs
echo.
echo [INFO] Key endpoints:
echo   GET /health                    - Health check
echo   GET /api/metrics/live          - Current cycle metrics
echo   GET /api/metrics/live/history  - Metrics for graphs
echo   GET /api/logs/pipeline         - Pipeline logs
echo   GET /api/admin/dashboard       - Admin dashboard
echo   WS /ws/stream                  - Live radar data stream
echo.
echo [INFO] Expected in 5 seconds:
echo   - Radar targets being generated
echo   - Detections appearing in logs
echo   - Metrics recording
echo   - WebSocket clients receiving data
echo.
echo ========================================================================
echo.

cd backend
set PYTHONUNBUFFERED=1
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
