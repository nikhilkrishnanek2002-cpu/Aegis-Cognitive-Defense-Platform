#!/usr/bin/env python3
"""
Aegis Cognitive Defense Platform - Development Launcher (LOCAL ONLY)
Start backend API server + React frontend in one command.

Usage: python dev_local.py

Press Ctrl+C to stop both servers.
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# Colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

ROOT = Path(__file__).parent
BACKEND_API = "main_api:app"
FRONTEND_DIR = ROOT / "frontend"

processes = []


def print_banner():
    print(f"""{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════════╗
║  🛰️  AEGIS COGNITIVE DEFENSE PLATFORM v2.0                  ║
║      AI-Enabled Photonic Radar System (LOCAL DEV MODE)       ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def step(msg):
    print(f"  {CYAN}→{RESET} {msg}")


def ok(msg):
    print(f"  {GREEN}✅ {msg}{RESET}")


def warn(msg):
    print(f"  {YELLOW}⚠️  {msg}{RESET}")


def error(msg):
    print(f"  {RED}❌ {msg}{RESET}")


def cleanup(sig=None, frame=None):
    """Gracefully stop all processes."""
    print(f"\n{YELLOW}🛑 Stopping servers...{RESET}")
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    print(f"{GREEN}✅ Stopped.\n{RESET}")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def check_command(cmd):
    """Check if command exists."""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    print_banner()
    
    # Check Python dependencies
    step("Checking Python dependencies...")
    required_packages = ["fastapi", "uvicorn", "numpy", "scipy"]
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
            ok(f"Found {pkg}")
        except ImportError:
            missing.append(pkg)
            warn(f"Missing {pkg}")
    
    if missing:
        error(f"Missing packages: {', '.join(missing)}")
        print(f"\n  Install with:")
        print(f"    pip install -r requirements.txt\n")
        sys.exit(1)
    
    # Check Node.js for frontend
    step("Checking Node.js...")
    if not check_command("node"):
        warn("Node.js/npm not found - frontend will not start")
        print(f"  Install from: https://nodejs.org/\n")
        has_frontend = False
    else:
        ok("Node.js found")
        has_frontend = True
    
    # Check frontend dependencies
    if has_frontend:
        node_modules = FRONTEND_DIR / "node_modules"
        package_json = FRONTEND_DIR / "package.json"
        
        if not node_modules.exists():
            if package_json.exists():
                step("Installing frontend dependencies (this may take a minute)...")
                try:
                    subprocess.run(
                        ["npm", "install"],
                        cwd=FRONTEND_DIR,
                        capture_output=True,
                        timeout=120
                    )
                    ok("Frontend dependencies installed")
                except Exception as e:
                    warn(f"Could not install frontend deps: {e}")
                    has_frontend = False
            else:
                warn("package.json not found - skipping frontend")
                has_frontend = False
    
    print()
    
    # Start Backend API
    step("Starting Backend API on http://localhost:8000...")
    try:
        backend_proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                BACKEND_API,
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(("Backend API", backend_proc))
        ok("Backend API started")
    except Exception as e:
        error(f"Failed to start backend: {e}")
        sys.exit(1)
    
    # Wait for API to be ready
    time.sleep(3)
    
    # Start Frontend
    if has_frontend:
        step("Starting React frontend on http://localhost:3000...")
        try:
            frontend_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(FRONTEND_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            processes.append(("Frontend", frontend_proc))
            ok("Frontend started")
        except Exception as e:
            warn(f"Could not start frontend: {e}")
    else:
        warn("Skipping frontend (no Node.js)")
    
    # Print info
    print()
    print("=" * 70)
    print(f"{GREEN}{BOLD}✅ AEGIS PLATFORM IS RUNNING{RESET}")
    print("=" * 70)
    print(f"  🌐 Dashboard:   {CYAN}http://localhost:3000{RESET}")
    print(f"  📚 API Docs:    {CYAN}http://localhost:8000/docs{RESET}")
    print(f"  🔌 WebSocket:   {CYAN}ws://localhost:8000/ws/stream{RESET}")
    print(f"  🔑 Login:       admin / admin123")
    print("=" * 70)
    print(f"\n{YELLOW}Press Ctrl+C to stop servers.{RESET}\n")
    
    # Keep process alive and monitor
    try:
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    error(f"{name} stopped unexpectedly (exit code: {proc.returncode})")
                    # Cleanup and exit
                    cleanup()
    
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
