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
        print(f"[play-all] stopping {name} (pid={proc.pid})")
        proc.terminate()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"[play-all] force-killing {name} (pid={proc.pid})")
        proc.kill()
        proc.wait(timeout=timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch MuJoCo sim + telemetry (and optional HIL bridge) together.",
    )
    parser.add_argument("--mode", choices=["smooth", "robust"], default="smooth")
    parser.add_argument("--udp-port", type=int, default=9871)
    parser.add_argument("--tune-port", type=int, default=9881, help="UDP port for live tuning slider commands.")
    parser.add_argument("--window-s", type=float, default=12.0)
    parser.add_argument("--start-delay-s", type=float, default=0.6, help="Delay between each process launch.")
    parser.add_argument(
        "--disturbance-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable randomized disturbance rejection test in the MuJoCo sim process.",
    )
    parser.add_argument(
        "--sim-tuning-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the tuning slider panel in telemetry_plotter for sim-first tuning.",
    )
    parser.add_argument(
        "--trajectory-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run trajectory_following_benchmark.py and exit (single-flag benchmark mode).",
    )
    parser.add_argument(
        "--trajectory-benchmark-extra",
        type=str,
        default="",
        help="Extra args appended to trajectory_following_benchmark.py when --trajectory-benchmark is enabled.",
    )
    parser.add_argument(
        "--sim-trajectory-profile",
        choices=["none", "step_x", "line_sine"],
        default="none",
        help="Trajectory reference profile for live MuJoCo sim tracking.",
    )
    parser.add_argument("--sim-trajectory-warmup-s", type=float, default=1.0)
    parser.add_argument("--sim-trajectory-step-m", type=float, default=0.18)
    parser.add_argument("--sim-trajectory-amp-m", type=float, default=0.22)
    parser.add_argument("--sim-trajectory-period-s", type=float, default=6.0)
    parser.add_argument("--sim-trajectory-x-bias-m", type=float, default=0.0)
    parser.add_argument("--sim-trajectory-y-bias-m", type=float, default=0.0)

    parser.add_argument(
        "--with-hil",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also launch hil_bridge.py.",
    )
    parser.add_argument("--esp32-ip", type=str, default=None, help="Required when --with-hil.")
    parser.add_argument(
        "--hil-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch HIL bridge with plot window.",
    )

    parser.add_argument("--sim-extra", type=str, default="", help="Extra args appended to final.py.")
    parser.add_argument(
        "--plotter-extra",
        type=str,
        default="",
        help="Extra args appended to telemetry_plotter.py.",
    )
    parser.add_argument("--hil-extra", type=str, default="", help="Extra args appended to hil_bridge.py.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    py = sys.executable

    if args.trajectory_benchmark:
        bench_cmd = [py, str(root / "trajectory_following_benchmark.py")]
        if args.trajectory_benchmark_extra.strip():
            bench_cmd.extend(shlex.split(args.trajectory_benchmark_extra))
        print("[play-all] benchmark:", " ".join(shlex.quote(c) for c in bench_cmd))
        return int(subprocess.call(bench_cmd, cwd=str(root)))

    if args.with_hil and not args.esp32_ip:
        print("[play-all] error: --esp32-ip is required when --with-hil is enabled.")
        return 2

    plotter_cmd = [
        py,
        str(root / "telemetry_plotter.py"),
        "--source",
        "udp",
        "--udp-port",
        str(args.udp_port),
        "--window-s",
        str(args.window_s),
        "--tune-target-host",
        "127.0.0.1",
        "--tune-target-port",
        str(args.tune_port),
    ]
    if args.sim_tuning_panel:
        plotter_cmd.append("--tuning-panel")
    else:
        plotter_cmd.append("--no-tuning-panel")
    if args.plotter_extra.strip():
        plotter_cmd.extend(shlex.split(args.plotter_extra))

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
        "--live-tuning",
        "--live-tuning-udp-bind",
        "127.0.0.1",
        "--live-tuning-udp-port",
        str(args.tune_port),
        "--trajectory-profile",
        str(args.sim_trajectory_profile),
        "--trajectory-warmup-s",
        str(args.sim_trajectory_warmup_s),
        "--trajectory-step-m",
        str(args.sim_trajectory_step_m),
        "--trajectory-amp-m",
        str(args.sim_trajectory_amp_m),
        "--trajectory-period-s",
        str(args.sim_trajectory_period_s),
        "--trajectory-x-bias-m",
        str(args.sim_trajectory_x_bias_m),
        "--trajectory-y-bias-m",
        str(args.sim_trajectory_y_bias_m),
    ]
    if args.disturbance_test:
        sim_cmd.append("--disturbance-rejection-test")
    if args.sim_extra.strip():
        sim_cmd.extend(shlex.split(args.sim_extra))

    hil_cmd: list[str] | None = None
    if args.with_hil:
        hil_cmd = [
            py,
            str(root / "hil_bridge.py"),
            "--esp32-ip",
            args.esp32_ip,
        ]
        if args.hil_plot:
            hil_cmd.append("--plot")
        else:
            hil_cmd.append("--no-plot")
        if args.hil_extra.strip():
            hil_cmd.extend(shlex.split(args.hil_extra))

    print("[play-all] plotter:", " ".join(shlex.quote(c) for c in plotter_cmd))
    print("[play-all] sim    :", " ".join(shlex.quote(c) for c in sim_cmd))
    if hil_cmd is not None:
        print("[play-all] hil    :", " ".join(shlex.quote(c) for c in hil_cmd))

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        processes.append(("plotter", subprocess.Popen(plotter_cmd, cwd=str(root))))
        if args.start_delay_s > 0.0:
            time.sleep(args.start_delay_s)
        processes.append(("sim", subprocess.Popen(sim_cmd, cwd=str(root))))
        if hil_cmd is not None:
            if args.start_delay_s > 0.0:
                time.sleep(args.start_delay_s)
            processes.append(("hil", subprocess.Popen(hil_cmd, cwd=str(root))))

        while True:
            for name, proc in processes:
                rc = proc.poll()
                if rc is not None:
                    print(f"[play-all] {name} exited with code {rc}")
                    for stop_name, stop_proc in processes:
                        if stop_proc is not proc:
                            _terminate_child(stop_proc, stop_name)
                    return int(rc)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[play-all] Ctrl+C received")
        return 130
    finally:
        for name, proc in reversed(processes):
            _terminate_child(proc, name)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
