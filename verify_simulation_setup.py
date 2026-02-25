#!/usr/bin/env python3
"""
Aegis Simulation Mode - Startup Verification Script
Run this BEFORE starting the backend to ensure all components are ready.
"""

import sys
import os
from pathlib import Path

def check_file_exists(path: str, name: str) -> bool:
    """Check if critical file exists."""
    if os.path.exists(path):
        print(f"  ✓ {name}")
        return True
    else:
        print(f"  ✗ MISSING: {name} at {path}")
        return False

def main():
    print("\n" + "=" * 70)
    print("AEGIS SIMULATION MODE - STARTUP VERIFICATION")
    print("=" * 70)
    
    backend_dir = Path("backend/app")
    all_good = True
    
    # Check critical files
    print("\n[1/5] Checking core files...")
    files_to_check = [
        ("backend/app/main.py", "Main FastAPI app"),
        ("backend/app/services/radar_simulator.py", "Radar simulator"),
        ("backend/app/core/metrics_store.py", "Metrics store"),
        ("backend/app/engine/pipeline.py", "Detection pipeline"),
        ("backend/app/engine/controller.py", "Pipeline controller"),
    ]
    
    for path, name in files_to_check:
        if not check_file_exists(path, name):
            all_good = False
    
    # Check Python version
    print("\n[2/5] Checking Python version...")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info >= (3, 8):
        print(f"  ✓ Python {py_version}")
    else:
        print(f"  ✗ Python {py_version} - requires 3.8+")
        all_good = False
    
    # Check required packages
    print("\n[3/5] Checking required packages...")
    required_packages = [
        "fastapi", "uvicorn", "numpy", "pydantic", "psutil"
    ]
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ MISSING: {pkg} (install with: pip install {pkg})")
            all_good = False
    
    # Check environment
    print("\n[4/5] Checking environment...")
    print(f"  ✓ Current working directory: {os.getcwd()}")
    
    if os.path.exists("backend"):
        print("  ✓ backend/ directory found")
    else:
        print("  ✗ backend/ directory NOT found")
        all_good = False
    
    if os.path.exists("frontend"):
        print("  ✓ frontend/ directory found")
    else:
        print("  ✗ frontend/ directory NOT found (optional)")
    
    # Check config
    print("\n[5/5] Checking configuration...")
    radar_sim = os.getenv("RADAR_SIMULATOR", "true")
    print(f"  ℹ RADAR_SIMULATOR={radar_sim}")
    if radar_sim.lower() in ["true", "1", "yes"]:
        print("  ✓ Simulation mode ENABLED (good for testing)")
    else:
        print("  ⚠ Simulation mode DISABLED (requires real radar hardware)")
    
    # Final status
    print("\n" + "=" * 70)
    if all_good:
        print("✓ ALL CHECKS PASSED - READY TO START")
        print("\nNext steps:")
        print("  1. cd backend")
        print("  2. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("\nThen in another terminal:")
        print("  1. cd frontend")
        print("  2. npm run dev")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - SEE ABOVE")
        print("\nPlease fix the issues and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
