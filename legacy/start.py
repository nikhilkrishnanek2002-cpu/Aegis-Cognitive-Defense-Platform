#!/usr/bin/env python3
"""
Simple launcher - starts servers without installing dependencies.
Use this if you already have dependencies installed or if auto-install fails.
"""
import subprocess
import sys
import os
import time
import signal

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

processes = []

def cleanup(sig=None, frame=None):
    """Stop all processes."""
    print("\n🛑 Stopping servers...")
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    print("✅ Stopped.\n")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("🚀 Starting Aegis Cognitive Defense Platform\n")

# Start API
print("Starting API server on http://localhost:8000")
api_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=ROOT
)
processes.append(("API", api_proc))

time.sleep(3)

# Start Frontend if available
if os.path.isdir(FRONTEND_DIR) and os.path.isdir(os.path.join(FRONTEND_DIR, "node_modules")):
    print("Starting React frontend on http://localhost:3000")
    try:
        react_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=FRONTEND_DIR
        )
        processes.append(("React", react_proc))
    except Exception as e:
        print(f"⚠️  Could not start frontend: {e}")

print("\n" + "="*60)
print("✅ Servers running!")
print("  🌐 Dashboard: http://localhost:3000")
print("  📚 API Docs:  http://localhost:8000/docs")
print("  🔑 Login:     admin / admin123")
print("="*60)
print("\nPress Ctrl+C to stop.\n")

# Keep alive
try:
    while True:
        time.sleep(1)
        # Check if API died
        if api_proc.poll() is not None:
            print("❌ API server stopped unexpectedly")
            break
except KeyboardInterrupt:
    pass

cleanup()

