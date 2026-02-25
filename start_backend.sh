#!/bin/bash
# ============================================================================
# AEGIS COGNITIVE DEFENSE PLATFORM - FULL SIMULATION MODE
# QUICK START GUIDE
# ============================================================================

# Exit on any error
set -e

echo "========================================================================"
echo "AEGIS COGNITIVE DEFENSE PLATFORM - FULL SIMULATION MODE STARTUP"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: Verify Setup
# ============================================================================
echo -e "${BLUE}[STEP 1/5]${NC} Verifying setup..."
if [ ! -d "backend" ]; then
    echo "ERROR: backend/ directory not found. Are you in the project root?"
    exit 1
fi

if [ ! -f "backend/app/main.py" ]; then
    echo "ERROR: backend/app/main.py not found"
    exit 1
fi

echo -e "${GREEN}✓${NC} Project structure verified"
echo ""

# ============================================================================
# STEP 2: Check Python Environment
# ============================================================================
echo -e "${BLUE}[STEP 2/5]${NC} Checking Python environment..."

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Check required packages
for package in fastapi uvicorn numpy pydantic psutil; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $package installed"
    else
        echo -e "${YELLOW}⚠${NC} Installing $package..."
        pip install $package
    fi
done
echo ""

# ============================================================================
# STEP 3: Configure Simulation Mode
# ============================================================================
echo -e "${BLUE}[STEP 3/5]${NC} Configuring simulation mode..."
export RADAR_SIMULATOR=true
echo -e "${GREEN}✓${NC} RADAR_SIMULATOR=true"
echo ""

# ============================================================================
# STEP 4: Create Logs Directory
# ============================================================================
echo -e "${BLUE}[STEP 4/5]${NC} Setting up logs directory..."
mkdir -p backend/logs
echo -e "${GREEN}✓${NC} Logs directory ready at backend/logs/"
echo ""

# ============================================================================
# STEP 5: Start Backend
# ============================================================================
echo -e "${BLUE}[STEP 5/5]${NC} Starting FastAPI backend..."
echo ""
echo "========================================================================"
echo "STARTING BACKEND SERVER"
echo "========================================================================"
echo ""
echo -e "${YELLOW}ℹ Backend will be available at:${NC}"
echo "  HTTP: http://localhost:8000"
echo "  WebSocket: ws://localhost:8000/ws/stream"
echo "  Docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}ℹ Key endpoints:${NC}"
echo "  GET /health                    - Health check"
echo "  GET /api/metrics/live          - Current cycle metrics"
echo "  GET /api/metrics/live/history  - Metrics for graphs"
echo "  GET /api/logs/pipeline         - Pipeline logs"
echo "  GET /api/admin/dashboard       - Admin dashboard"
echo "  WS /ws/stream                  - Live radar data stream"
echo ""
echo -e "${YELLOW}ℹ Expected in 5 seconds:${NC}"
echo "  - Radar targets being generated"
echo "  - Detections appearing in logs"
echo "  - Metrics recording"
echo "  - WebSocket clients receiving data"
echo ""
echo "========================================================================"
echo ""

cd backend
export PYTHONUNBUFFERED=1
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
