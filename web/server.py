from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = REPO_ROOT / "final"
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from unconstrained_runtime import UnconstrainedWheelBotRuntime  # noqa: E402


class RuntimeThread:
    def __init__(self, mode: str = "balanced", payload_mass: float = 0.4):
        xml_path = FINAL_DIR / "final.xml"
        self.runtime = UnconstrainedWheelBotRuntime(
            xml_path=xml_path,
            payload_mass_kg=float(payload_mass),
            mode=mode,
        )
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.paused = False
        self.thread = threading.Thread(target=self._loop, name="wheelbot-runtime", daemon=True)
        self.thread.start()

    def _loop(self):
        next_tick = time.perf_counter()
        while not self.stop_event.is_set():
            with self.lock:
                dt = float(self.runtime.model.opt.timestep)
                if not self.paused and not self.runtime.failed:
                    self.runtime.step()
            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.010))
            else:
                next_tick = time.perf_counter()

    def get_state(self) -> dict:
        with self.lock:
            s = self.runtime.get_state()
            s["paused"] = bool(self.paused)
            return s

    def reset(self, payload_mass_kg: float | None = None, mode: str | None = None) -> dict:
        with self.lock:
            self.runtime.reset(payload_mass_kg=payload_mass_kg, mode=mode)
            self.paused = False
            s = self.runtime.get_state()
            s["paused"] = False
            return s

    def set_paused(self, paused: bool) -> dict:
        with self.lock:
            self.paused = bool(paused)
            s = self.runtime.get_state()
            s["paused"] = self.paused
            if self.paused and not s["failed"]:
                s["status"] = "Paused"
            return s

    def shutdown(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)


class AppServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, runtime_thread: RuntimeThread):
        super().__init__(server_address, handler_cls)
        self.runtime_thread = runtime_thread


class ApiStaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        sys.stdout.write(f"{self.log_date_time_string()} - {fmt % args}\n")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._write_json({"ok": True, "service": "unconstrained-wheelbot-runtime"})
            return
        if path == "/api/state":
            self._write_json(self.server.runtime_thread.get_state())
            return
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/reset":
            payload = self._read_json()
            mass = payload.get("payload_mass_kg", None)
            mode = payload.get("mode", None)
            try:
                s = self.server.runtime_thread.reset(payload_mass_kg=mass, mode=mode)
            except Exception as exc:  # noqa: BLE001
                self._write_json({"ok": False, "error": str(exc)}, status=400)
                return
            self._write_json(s)
            return
        if path == "/api/pause":
            payload = self._read_json()
            paused_value = payload.get("paused", None)
            if paused_value is None:
                paused_value = not bool(self.server.runtime_thread.get_state().get("paused", False))
            s = self.server.runtime_thread.set_paused(bool(paused_value))
            self._write_json(s)
            return
        self._write_json({"ok": False, "error": "Unknown API endpoint"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="Unconstrained wheel-bot web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--mode", choices=["balanced", "aggressive", "soft"], default="balanced")
    parser.add_argument("--payload-mass", type=float, default=0.4)
    args = parser.parse_args()

    runtime_thread = RuntimeThread(mode=args.mode, payload_mass=float(args.payload_mass))
    server = AppServer((args.host, args.port), ApiStaticHandler, runtime_thread)
    print(f"Serving on http://{args.host}:{args.port}/web/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime_thread.shutdown()


if __name__ == "__main__":
    main()
