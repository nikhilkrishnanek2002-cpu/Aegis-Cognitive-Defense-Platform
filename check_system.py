#!/usr/bin/env python3
"""
Quick verification script to check if Aegis platform is ready to run.
"""
import sys
import os
import subprocess

def check_module(name, package_name=None, optional=False):
    """Check if a module can be imported."""
    try:
        __import__(name)
        return True, "✅"
    except ImportError:
        if optional:
            return False, "⚠️ "
        return False, "❌"

def main():
    print("=" * 60)
    print("Aegis Cognitive Defense Platform - System Check")
    print("=" * 60)
    print()

    # Core dependencies
    print("Core Dependencies (Required for API):")
    core_deps = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("jose", "python-jose[cryptography]"),
        ("passlib", "passlib[bcrypt]"),
        ("multipart", "python-multipart"),
    ]

    core_ok = True
    for mod, pkg in core_deps:
        ok, icon = check_module(mod)
        print(f"  {icon} {pkg}")
        if not ok:
            core_ok = False

    print()

    # Optional dependencies
    print("Optional Dependencies:")
    opt_deps = [
        ("torch", "torch (for AI features)", True),
        ("yaml", "pyyaml (for config)", True),
        ("streamlit", "streamlit (for streamlit app)", True),
        ("plotly", "plotly (for visualizations)", True),
    ]

    for mod, desc, opt in opt_deps:
        ok, icon = check_module(mod, optional=opt)
        print(f"  {icon} {desc}")

    print()

    # Check npm/node
    print("Frontend Dependencies:")
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.decode().strip()
            print(f"  ✅ npm v{version}")

            # Check node_modules
            frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
            node_modules = os.path.join(frontend_dir, "node_modules")
            if os.path.isdir(node_modules):
                print(f"  ✅ node_modules installed")
            else:
                print(f"  ⚠️  node_modules not installed (run: cd frontend && npm install)")
        else:
            print(f"  ⚠️  npm found but not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"  ⚠️  npm not found (Node.js not installed)")

    print()

    # Test API imports
    print("API Module Tests:")
    try:
        from api.main import app
        print("  ✅ API main module")

        from api.routes import auth, radar, tracks, ew, admin, metrics, visualizations
        print("  ✅ All API routes")

        from src.signal_generator import generate_radar_signal
        from src.detection import detect_targets_from_raw
        print("  ✅ Radar processing modules")

    except Exception as e:
        print(f"  ❌ Import error: {e}")
        core_ok = False

    print()
    print("=" * 60)

    if core_ok:
        print("✅ System Ready!")
        print()
        print("To start the platform:")
        print("  python launcher.py")
        print()
        print("Or start components separately:")
        print("  API:      python -m uvicorn api.main:app --port 8000")
        print("  Frontend: cd frontend && npm run dev")
        return 0
    else:
        print("❌ System Not Ready")
        print()
        print("Install missing dependencies:")
        print("  pip install fastapi uvicorn numpy scipy")
        print("  pip install python-jose[cryptography] passlib[bcrypt] python-multipart")
        print()
        print("Optional (for AI features):")
        print("  pip install torch")
        return 1

if __name__ == "__main__":
    sys.exit(main())

