#!/usr/bin/env python3
"""
Aegis Cognitive Defense Platform — One-Command Launcher
Run with: python launcher.py
"""
import subprocess
import sys
import os
import time
import signal
import webbrowser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
NODE_MODULES = os.path.join(FRONTEND_DIR, "node_modules")

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner():
    print(f"""
{CYAN}{BOLD}  ╔══════════════════════════════════════════════════╗
  ║   🛰️  AEGIS COGNITIVE DEFENSE PLATFORM v2.0     ║
  ║      AI-Enabled Photonic Radar System            ║
  ╚══════════════════════════════════════════════════╝{RESET}
""")


def step(msg):  print(f"  {CYAN}→{RESET} {msg}")
def ok(msg):    print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠️  {msg}{RESET}")
def error(msg): print(f"  {RED}❌ {msg}{RESET}")


def check_python_deps():
    step("Checking Python dependencies...")
    required = {
        "fastapi": "fastapi", "uvicorn": "uvicorn",
        "torch": "torch", "numpy": "numpy",
        "scipy": "scipy", "jose": "python-jose[cryptography]",
        "passlib": "passlib[bcrypt]", "multipart": "python-multipart",
    }
    missing = [pkg for imp, pkg in required.items() if not _can_import(imp)]
    if missing:
        warn(f"Installing: {', '.join(missing)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + missing,
            check=True,
        )
    ok("Python dependencies ready.")


def _can_import(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def check_npm_deps():
    if not os.path.isdir(FRONTEND_DIR):
        return False
    if not os.path.isdir(NODE_MODULES):
        step("Running npm install (first time only, ~30s)...")
        r = subprocess.run(["npm", "install"], cwd=FRONTEND_DIR,
                           capture_output=True)
        if r.returncode != 0:
            error("npm install failed. Is Node.js 18+ installed?")
            print(r.stderr.decode())
            return False
        ok("npm packages installed.")
    else:
        ok("npm packages ready.")
    return True


def wait_for_port(port, timeout=30):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def launch():
    banner()

    check_python_deps()
    has_frontend = check_npm_deps()
    print()

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            step("Running on CPU")
    except ImportError:
        pass

    print(f"\n  {BOLD}Starting servers...{RESET}\n")

    processes = []

    # ── FastAPI ───────────────────────────────────────────────────────────────
    step("Starting FastAPI backend   → http://localhost:8000")
    api_log = open(os.path.join(ROOT, "api_server.log"), "w")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=api_log,
        stderr=api_log,
    )
    processes.append(("FastAPI", api_proc, api_log))

    if wait_for_port(8000, timeout=20):
        ok("FastAPI backend ready    → http://localhost:8000")
        ok("Swagger API Docs         → http://localhost:8000/docs")
    else:
        warn("FastAPI taking long to start — check api_server.log")

    print()

    # ── React ─────────────────────────────────────────────────────────────────
    react_port = 3000
    if has_frontend:
        step(f"Starting React frontend  → http://localhost:{react_port}")
        react_log = open(os.path.join(ROOT, "react_dev.log"), "w")
        react_proc = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(react_port), "--strictPort"],
            cwd=FRONTEND_DIR,
            stdin=subprocess.DEVNULL,
            stdout=react_log,
            stderr=react_log,
        )
        processes.append(("React", react_proc, react_log))

        if wait_for_port(react_port, timeout=20):
            ok(f"React dashboard ready    → http://localhost:{react_port}")
        else:
            # Check if it started on a different port
            for alt in [3001, 3002, 3003]:
                if wait_for_port(alt, timeout=2):
                    react_port = alt
                    warn(f"React started on alternate port {react_port}")
                    break
            else:
                warn("React taking long to start — check react_dev.log")

    print()
    print(f"  {GREEN}{BOLD}{'━'*50}{RESET}")
    print(f"  {BOLD}  🌐 Dashboard : http://localhost:{react_port}{RESET}")
    print(f"  {BOLD}  📚 API Docs  : http://localhost:8000/docs{RESET}")
    print(f"  {BOLD}  🔑 Login     : admin / admin123{RESET}")
    print(f"  {GREEN}{BOLD}{'━'*50}{RESET}")
    print(f"\n  Press {BOLD}Ctrl+C{RESET} to stop all servers.\n")

    # Auto-open browser
    time.sleep(1)
    try:
        webbrowser.open(f"http://localhost:{react_port}")
    except Exception:
        pass

    # ── Shutdown handler ──────────────────────────────────────────────────────
    def shutdown(sig=None, frame=None):
        print(f"\n\n  {YELLOW}Stopping servers...{RESET}")
        for name, proc, log in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        time.sleep(1)
        for name, proc, log in processes:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                log.close()
            except Exception:
                pass
        print(f"  {GREEN}Done. Goodbye!{RESET}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Keep alive: poll processes ────────────────────────────────────────────
    printed_error = set()
    while True:
        for name, proc, log in list(processes):
            rc = proc.poll()
            if rc is not None and name not in printed_error:
                error(f"{name} server stopped (exit code {rc})")
                log_path = os.path.join(ROOT, f"{'api_server' if name=='FastAPI' else 'react_dev'}.log")
                print(f"  Last log lines from {log_path}:")
                try:
                    with open(log_path) as f:
                        lines = f.readlines()
                    for l in lines[-15:]:
                        print(f"    {l.rstrip()}")
                except Exception:
                    pass
                printed_error.add(name)
        time.sleep(2)


if __name__ == "__main__":
    launch()