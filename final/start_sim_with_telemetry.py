from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def _terminate_child(proc: subprocess.Popen, name: str, timeout_s: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        print(f"[launcher] stopping {name} (pid={proc.pid})")
        proc.terminate()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[launcher] force-killing {name} (pid={proc.pid})")
        proc.kill()
        proc.wait(timeout=timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch MuJoCo sim and telemetry plotter together.",
    )
    parser.add_argument("--mode", choices=["smooth", "robust"], default="smooth")
    parser.add_argument("--udp-port", type=int, default=9871)
    parser.add_argument("--window-s", type=float, default=12.0)
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=0.6,
        help="Delay between starting plotter and sim.",
    )
    parser.add_argument(
        "--sim-extra",
        type=str,
        default="",
        help='Extra args appended to final.py (example: "--controller-family current_dob").',
    )
    parser.add_argument(
        "--plotter-extra",
        type=str,
        default="",
        help='Extra args appended to telemetry_plotter.py.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable

    plotter_cmd = [
        py,
        str(root / "telemetry_plotter.py"),
        "--source",
        "udp",
        "--udp-port",
        str(args.udp_port),
        "--window-s",
        str(args.window_s),
    ]
    sim_cmd = [
        py,
        str(root / "final.py"),
        "--mode",
        args.mode,
        "--telemetry",
        "--telemetry-transport",
        "udp",
        "--telemetry-udp-host",
        "127.0.0.1",
        "--telemetry-udp-port",
        str(args.udp_port),
    ]
    if args.plotter_extra.strip():
        plotter_cmd.extend(shlex.split(args.plotter_extra))
    if args.sim_extra.strip():
        sim_cmd.extend(shlex.split(args.sim_extra))

    print("[launcher] plotter:", " ".join(shlex.quote(c) for c in plotter_cmd))
    print("[launcher] sim    :", " ".join(shlex.quote(c) for c in sim_cmd))

    plotter_proc: subprocess.Popen | None = None
    sim_proc: subprocess.Popen | None = None
    try:
        plotter_proc = subprocess.Popen(plotter_cmd, cwd=str(root))
        if args.start_delay_s > 0.0:
            time.sleep(args.start_delay_s)
        sim_proc = subprocess.Popen(sim_cmd, cwd=str(root))

        while True:
            plotter_rc = plotter_proc.poll()
            sim_rc = sim_proc.poll()
            if plotter_rc is not None:
                print(f"[launcher] plotter exited with code {plotter_rc}")
                _terminate_child(sim_proc, "sim")
                return int(plotter_rc)
            if sim_rc is not None:
                print(f"[launcher] sim exited with code {sim_rc}")
                _terminate_child(plotter_proc, "plotter")
                return int(sim_rc)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[launcher] Ctrl+C received")
        return 130
    finally:
        if sim_proc is not None:
            _terminate_child(sim_proc, "sim")
        if plotter_proc is not None:
            _terminate_child(plotter_proc, "plotter")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
