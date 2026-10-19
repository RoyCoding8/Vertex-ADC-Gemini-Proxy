"""Start and stop the local proxy from a small terminal menu."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "tmp" / "logs" / "proxy.log"


class ProxyProcess:
    def __init__(self, project=None, location="global", port=8765):
        self.port = port
        self.arguments = ["--location", location, "--port", str(port)]
        if project:
            self.arguments += ["--project", project]
        self.process = None

    def running(self):
        return self.process is not None and self.process.poll() is None

    def health(self):
        try:
            with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{self.port}/healthz", timeout=1) as response:
                return json.load(response)
        except (URLError, TimeoutError, ValueError, OSError):
            return None

    def start(self):
        if self.running():
            return "Already running."
        if self.health() is not None:
            return "A proxy is already running outside this TUI. Stop that instance first."
        LOG.parent.mkdir(parents=True, exist_ok=True)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with LOG.open("ab") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "proxy.py"), *self.arguments],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=output, **options,
            )
        return "Starting proxy."

    def stop(self):
        if not self.running():
            self.process = None
            return "Stopped."
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode and self.running():
                raise RuntimeError(result.stderr.decode(errors="replace").strip())
        else:
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        self.process = None
        return "Stopped."

    def status(self):
        if self.running():
            if health := self.health():
                return f"Running | {health['project']} | {health['location']}"
            return "Starting | waiting for ADC and listener"
        if self.process is not None and self.process.returncode:
            return f"Exited with code {self.process.returncode}. Press L to read the log."
        if self.health():
            return "Running outside this TUI"
        return "Stopped"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    proxy = ProxyProcess(args.project, args.location, args.port)
    try:
        print(proxy.start())
        while True:
            print(f"\nVertex ADC proxy | http://127.0.0.1:{args.port}")
            print(proxy.status())
            choice = input("[S]tart  [T] Stop  [R]estart  [L]og  [Q]uit  [Enter] Refresh > ").strip().lower()
            if choice in {"q", "quit"}:
                break
            if choice in {"s", "start"}:
                print(proxy.start())
            elif choice in {"t", "stop"}:
                print(proxy.stop())
            elif choice in {"r", "restart"}:
                print(proxy.stop())
                print(proxy.start())
            elif choice in {"l", "log"}:
                print("\n".join(LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]) if LOG.exists() else "No log yet.")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        print(proxy.stop())


if __name__ == "__main__":
    main()
